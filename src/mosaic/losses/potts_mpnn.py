"""Canonical or state-selective PottsMPNN sequence recovery in Mosaic.

Default: sum all substate probabilities of an amino acid before recovery.
State selection removes probability mass WITHOUT renormalizing it. Target
state conditioning is independent of binder-state acceptance. This is a
stop-gradient reconstruction loss, not a binding-energy or pH predictor.
Target identities follow an explicit weighted mixture of independent
canonical distributions. Average binder probabilities after conditioning;
the energy model never reweights the prescribed target identities.
"""
from collections.abc import Mapping, Sequence
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm, TOKENS
from mosaic.pottsmpnn import PottsMPNN, gibbs_marginals

CANONICAL_NAMES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
NAME_TO_AA = dict(zip(CANONICAL_NAMES, TOKENS))


def state_projection(token_names: Sequence[str], selections: Mapping[str, Sequence[str]] | None = None):
    """Build [V,20] acceptance matrix in Mosaic order from explicit metadata.

    ``selections={'H': ('HIS-P',)}`` replaces the default histidine family;
    other identities retain their entire family. UNK always contributes zero.
    """
    token_names = tuple(token_names)
    if len(set(token_names)) != len(token_names):
        raise ValueError("Duplicate model tokens")
    projection = np.zeros((len(token_names), 20), np.float32)
    for i, name in enumerate(token_names):
        if name == "UNK":
            continue
        parent = name.split("-")[0]
        if parent not in NAME_TO_AA or ("-" in name and parent not in ("HIS", "ASP", "GLU")):
            raise ValueError(f"Unknown token family: {name}")
        projection[i, TOKENS.index(NAME_TO_AA[parent])] = 1
    if np.any(projection.sum(0) == 0):
        raise ValueError("Model vocabulary is missing canonical amino acids")
    for aa, accepted in (selections or {}).items():
        if aa not in TOKENS or isinstance(accepted, str) or not accepted:
            raise ValueError(f"Invalid selection for {aa}: use a nonempty sequence of token names")
        column = TOKENS.index(aa)
        for name in accepted:
            if name not in token_names or projection[token_names.index(name), column] != 1:
                raise ValueError(f"{name} is not a substate of {aa}")
        projection[:, column] = 0
        for name in accepted:
            projection[token_names.index(name), column] = 1
    return jnp.asarray(projection)


def recovery_from_probabilities(sequence, probabilities, projection, design_mask=None):
    """Negative masked mean identity recovery; no substate renormalization."""
    if sequence.ndim != 2 or sequence.shape[-1] != 20:
        raise ValueError("Expected binder probabilities [L,20] in Mosaic order")
    if probabilities.ndim != 2 or probabilities.shape[0] != sequence.shape[0]:
        raise ValueError("Expected Potts probabilities [L,V]")
    expected = (probabilities.shape[-1], 20)
    if projection.shape not in (expected, (sequence.shape[0], *expected)):
        raise ValueError("Acceptance projection must be [V,20] or [L,V,20]")
    probabilities = jax.lax.stop_gradient(probabilities)
    accepted = (probabilities @ projection if projection.ndim == 2
                else jnp.einsum("lv,lva->la", probabilities, projection))
    mask = jnp.ones(sequence.shape[0]) if design_mask is None else jnp.asarray(design_mask, dtype=sequence.dtype)
    if mask.shape != (sequence.shape[0],):
        raise ValueError("design_mask must have shape [L]")
    denominator = jnp.maximum(mask.sum(), 1)
    recovery = jnp.sum(jnp.sum(sequence * accepted, -1) * mask) / denominator
    return -recovery, {"sequence_recovery": recovery,
                       "accepted_mass": jnp.sum(accepted.sum(-1) * mask) / denominator}


def validate_target_distribution(probabilities, weights, names):
    """Validate a mixture of independent canonical target distributions."""
    p = np.asarray(probabilities, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    names = tuple(names)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1] < 1 or p.shape[2] != 20:
        raise ValueError("Target probabilities must have shape [components,target_length,20]")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0):
        raise ValueError("Target probability rows must be finite, nonnegative and sum to one")
    if w.shape != (p.shape[0],) or not np.isfinite(w).all() or np.any(w <= 0) or not np.isclose(w.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("Target weights must be positive, finite and sum to one")
    if len(names) != len(w) or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Supply one unique nonempty name per target component")
    return p, w, names


class TargetDistribution(eqx.Module):
    """Q(t) = sum_k weights[k] * product_j probabilities[k,j,t_j].

    One-hot components describe whole sequences; a soft component describes
    independently distributed identities. Canonical identities are drawn
    before choosing allowed protonation substates, preserving Q exactly.
    """
    probabilities: object
    weights: object
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, *, probabilities, weights, names):
        p, w, names = validate_target_distribution(probabilities, weights, names)
        self.probabilities, self.weights, self.names = jnp.asarray(p), jnp.asarray(w), names

    @classmethod
    def independent(cls, probabilities, *, name="target"):
        return cls(probabilities=np.asarray(probabilities)[None], weights=[1.0], names=[name])

    @classmethod
    def from_sequences(cls, sequences: Mapping[str, str], weights):
        if not sequences or any(not s or any(a not in TOKENS for a in s) for s in sequences.values()):
            raise ValueError("Supply named canonical target sequences without gaps")
        if len({len(s) for s in sequences.values()}) != 1:
            raise ValueError("Target sequences must share aligned positions")
        p = np.stack([np.eye(20, dtype=np.float32)[[TOKENS.index(a) for a in s]] for s in sequences.values()])
        return cls(probabilities=p, weights=weights, names=tuple(sequences))

    @property
    def mean_probabilities(self):
        return jnp.einsum("k,kta->ta", self.weights, self.probabilities)

    def sample(self, key, *, num_chains):
        """Stratify chains across components; keep mixture weights explicit."""
        components = len(self.names)
        if isinstance(num_chains, bool) or not isinstance(num_chains, int) or num_chains < components:
            raise ValueError("num_chains must be an integer >= target component count")
        assignment = np.arange(num_chains) % components
        counts = np.bincount(assignment, minlength=components)
        component = jnp.asarray(assignment)
        p = self.probabilities[component]
        identities = jax.random.categorical(key, jnp.where(p > 0, jnp.log(p), -jnp.inf), axis=-1)
        chain_weights = self.weights[component] / jnp.asarray(counts)[component]
        return identities, chain_weights, component


def acceptance_projection(token_names, selections):
    """Canonical-family filters, globally or as a position-specific matrix."""
    canonical = np.asarray(state_projection(token_names))
    if selections is None or isinstance(selections, Mapping):
        return state_projection(token_names, selections)
    p = np.asarray(selections)
    if (p.ndim != 3 or p.shape[1:] != canonical.shape or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > canonical) or np.any(p.sum(1) == 0)):
        raise ValueError("Invalid position-specific acceptance projection")
    return jnp.asarray(p)


def validate_output_target(observed, expected):
    if (not np.isfinite(observed).all() or np.any(observed < 0)
            or not np.allclose(observed, expected, atol=1e-5, rtol=0)):
        raise ValueError("Potts target distribution must match the structure model's target probabilities")


class PottsMPNNSequenceRecovery(LossTerm):
    mpnn: PottsMPNN
    target: TargetDistribution
    projection: object
    canonical_projection: object
    target_projection: object
    temperature: float = eqx.field(static=True)
    num_chains: int = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    num_sweeps: int = eqx.field(static=True)
    bias: object
    design_mask: object
    scaffold_aa_idx: object

    def __init__(self, mpnn, *, target: TargetDistribution, temperature=1.0,
                 num_chains=16, burn_in=5, num_sweeps=5, binder_states=None,
                 target_states=None, bias=None, design_mask=None, scaffold_aa_idx=None):
        """Expected binder recovery under an explicit target distribution.

        Both state selectors filter canonical families, e.g. {'H': ('HIS-P',)};
        a [position,V,20] matrix allows local selections. Target selectors
        restrict substates conditional on identity; they never change Q(t).
        Binder selectors accept probability mass without renormalization.
        """
        if not isinstance(target, TargetDistribution):
            raise TypeError("target must be a TargetDistribution")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name, value, minimum in (("num_chains", num_chains, len(target.names)), ("burn_in", burn_in, 0), ("num_sweeps", num_sweeps, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.mpnn, self.target = mpnn, target
        self.canonical_projection = state_projection(mpnn.token_names)
        self.projection = acceptance_projection(mpnn.token_names, binder_states)
        self.target_projection = acceptance_projection(mpnn.token_names, target_states)
        target_projection = np.asarray(self.target_projection)
        if np.any((target_projection != 0) & (target_projection != 1)):
            raise ValueError("Target state filters must contain only zero or one")
        if self.target_projection.ndim == 3 and self.target_projection.shape[0] != target.probabilities.shape[1]:
            raise ValueError("Target state projection length mismatch")
        self.temperature, self.num_chains, self.burn_in, self.num_sweeps = float(temperature), num_chains, burn_in, num_sweeps
        self.bias = None if bias is None else jnp.asarray(bias)
        self.design_mask = None if design_mask is None else jnp.asarray(design_mask, bool)
        self.scaffold_aa_idx = None if scaffold_aa_idx is None else jnp.asarray(scaffold_aa_idx, jnp.int32)
        if (self.design_mask is None) != (self.scaffold_aa_idx is None):
            raise ValueError("Supply design_mask and scaffold_aa_idx together")
        if self.scaffold_aa_idx is not None:
            scaffold = np.asarray(scaffold_aa_idx)
            if scaffold.shape != self.design_mask.shape or np.any(scaffold < 0) or np.any(scaffold >= 20):
                raise ValueError("Invalid canonical scaffold indices")

    def chain_probabilities(self, sequence, output, *, key):
        length = sequence.shape[0]
        full = output.full_sequence
        target_length = self.target.probabilities.shape[1]
        if full.shape != (length + target_length, 20):
            raise ValueError("Expected binder-first full_sequence matching target length")
        if self.design_mask is not None and self.design_mask.shape != (length,):
            raise ValueError("Binder design mask length mismatch")
        if self.projection.ndim == 3 and self.projection.shape[0] != length:
            raise ValueError("Binder state projection length mismatch")
        jax.debug.callback(validate_output_target, full[length:], self.target.mean_probabilities)
        encoder_key, target_key, sampler_key = jax.random.split(key, 3)
        identities, weights, components = self.target.sample(target_key, num_chains=self.num_chains)
        vocab = len(self.mpnn.token_names)
        known = self.canonical_projection.sum(-1) > 0
        allowed = jnp.broadcast_to(known, (self.num_chains, full.shape[0], vocab))
        identity_probabilities = jax.nn.one_hot(identities, 20)
        target_allowed = (identity_probabilities @ self.target_projection.T if self.target_projection.ndim == 2
                          else jnp.einsum("cta,tva->ctv", identity_probabilities, self.target_projection)) > 0
        allowed = allowed.at[:, length:].set(target_allowed)
        if self.design_mask is not None:
            scaffold_allowed = self.canonical_projection[:, self.scaffold_aa_idx].T > 0
            allowed = allowed.at[:, :length].set(jnp.where(self.design_mask[None, :, None], allowed[:, :length], scaffold_allowed[None]))
        token_bias = jnp.zeros((full.shape[0], vocab))
        if self.bias is not None:
            if self.bias.shape != (length, 20):
                raise ValueError("Canonical binder bias must be [L,20]")
            token_bias = token_bias.at[:length].set(self.bias @ self.canonical_projection.T)
        table, edges = self.mpnn.encode(output.backbone_coordinates, output.residue_idx, output.asym_id, key=encoder_key)
        probabilities = gibbs_marginals(table, edges, allowed, token_bias, key=sampler_key,
            temperature=self.temperature, burn_in=self.burn_in, num_sweeps=self.num_sweeps)
        return probabilities[:, :length], weights, components

    def __call__(self, sequence, output, key):
        chains, weights, components = self.chain_probabilities(sequence, output, key=key)
        probabilities = jnp.einsum("c,clv->lv", weights, chains)
        value, aux = recovery_from_probabilities(sequence, probabilities, self.projection, self.design_mask)
        _, canonical = recovery_from_probabilities(sequence, probabilities, self.canonical_projection, self.design_mask)
        aux["canonical_recovery"] = canonical["sequence_recovery"]
        aux["target_sequence_recovery"] = {}
        for component, name in enumerate(self.target.names):
            selected = (components == component).astype(chains.dtype)
            component_probabilities = jnp.einsum("c,clv->lv", selected / selected.sum(), chains)
            _, diagnostic = recovery_from_probabilities(sequence, component_probabilities, self.projection, self.design_mask)
            aux["target_sequence_recovery"][name] = diagnostic["sequence_recovery"]
        return value, aux
