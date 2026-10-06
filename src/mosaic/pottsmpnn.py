"""JAX inference for the released Proton-PottsMPNN v6 encoder.

Ports the encoder-only evaluation path from christian-creator/ProtonPottsMPNN
at 09682abfa7d20e0abcdeea0490b7a4b1c190aee3. PyTorch is used only when
loading the original checkpoint; converted NPZ loading, encoding and
conditional inference are JAX.
The Hamiltonian sums directed edges, including a diagonal self-field once.
Upstream copyright notices are retained in POTTS_MPNN_LICENSES.txt.
"""
from pathlib import Path
import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Released v6 metadata, in model order (not an inferred alphabetical order).
V6_TOKENS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "UNK", "HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D",
    "ASP-A", "GLU-P", "GLU-D", "GLU-A",
)
V6_CHECKPOINT_SHA256 = "a39872250c0b8eb4c0b4cb6472a6edad0d6f6134a50c28b611386e83c592fe7b"


class PottsMPNN(eqx.Module):
    weights: dict
    token_names: tuple[str, ...] = eqx.field(static=True)
    checkpoint_sha256: str = eqx.field(static=True)
    num_neighbors: int = eqx.field(static=True, default=48)
    backbone_noise: float = eqx.field(static=True, default=0.0)

    def _linear(self, name, x):
        value = x @ self.weights[name + ".weight"].T
        return value + self.weights.get(name + ".bias", 0.0)

    def _norm(self, name, x):
        mean = x.mean(-1, keepdims=True)
        variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return ((x - mean) * jax.lax.rsqrt(variance + 1e-5)
                * self.weights[name + ".weight"] + self.weights[name + ".bias"])

    def encode(self, backbone, residue_idx, chain_id, *, key):
        """Return [L,K,30,30] tables and [L,K] edges for complete N/CA/C/O.

        Protein residues must be unpadded with distinct CA positions. Chain
        identity is explicit; residue indices may restart within each chain.
        All arrays follow the binder-first Mosaic structure-output ordering.
        """
        if backbone.ndim != 3 or backbone.shape[1:] != (4, 3):
            raise ValueError("Expected complete protein backbone [L,4,3]")
        length = backbone.shape[0]
        if length < 1 or residue_idx.shape != (length,) or chain_id.shape != (length,):
            raise ValueError("Invalid protein index/chain arrays")
        x = jax.lax.stop_gradient(backbone)
        if self.backbone_noise:
            x = x + self.backbone_noise * jax.random.normal(key, x.shape)
        n, ca, c = x[:, 0], x[:, 1], x[:, 2]
        b1, b2 = n - ca, c - ca
        cb = ca + 0.58273431 * jnp.cross(b1, b2) - 0.56802827 * b1 - 0.54067466 * b2
        atoms = jnp.concatenate((x, cb[:, None]), axis=1)
        distance = jnp.sqrt(jnp.sum((ca[:, None] - ca[None]) ** 2, axis=-1) + 1e-6)
        # Self must be first even in degenerate/overlapping backbone outputs.
        distance = distance.at[jnp.arange(length), jnp.arange(length)].set(0.0)
        _, edges = jax.lax.top_k(-distance, min(length, self.num_neighbors))
        d = jnp.sqrt(jnp.sum(
            (atoms[:, None, :, None] - atoms[edges][:, :, None, :]) ** 2, axis=-1) + 1e-6)
        rbf = jnp.exp(-((d[..., None] - jnp.linspace(2.0, 22.0, 16)) / 1.25) ** 2)
        offset = residue_idx[:, None] - residue_idx[edges]
        positional = jnp.where(chain_id[:, None] == chain_id[edges],
                               jnp.clip(offset + 32, 0, 64), 65)
        prefix = "graph_featurization_module."
        pos = self._linear(prefix + "positional_embedding.embed_positional_features",
                           jax.nn.one_hot(positional, 66))
        raw = jnp.concatenate((pos, rbf.reshape(length, edges.shape[1], 400)), axis=-1)
        h_e = self._linear("W_e", self._norm(prefix + "edge_norm",
                                            self._linear(prefix + "edge_embedding", raw)))
        h_v = jnp.zeros((length, 128), dtype=h_e.dtype)
        for layer in range(3):
            p = f"encoder_layers.{layer}."
            def message(v, e, first, second, third):
                joined = jnp.concatenate((jnp.broadcast_to(v[:, None], e.shape), e, v[edges]), -1)
                m = jax.nn.gelu(self._linear(p + first, joined), approximate=False)
                m = jax.nn.gelu(self._linear(p + second, m), approximate=False)
                return self._linear(p + third, m)
            h_v = self._norm(p + "norm1", h_v + message(h_v, h_e, "W1", "W2", "W3").sum(1) / 30)
            ff = jax.nn.gelu(self._linear(p + "dense.W_in", h_v), approximate=False)
            h_v = self._norm(p + "norm2", h_v + self._linear(p + "dense.W_out", ff))
            h_e = self._norm(p + "norm3", h_e + message(h_v, h_e, "W11", "W12", "W13"))
        table = self._linear("etab_out", h_e).reshape(length, edges.shape[1], 30, 30)
        table = table.at[:, 0].set(table[:, 0] * jnp.eye(30))
        reverse = edges[edges] == jnp.arange(length)[:, None, None]
        reverse_slot = reverse.argmax(-1)
        transposed = table[edges, reverse_slot].swapaxes(-1, -2)
        return jnp.where(reverse.any(-1)[..., None, None], (table + transposed) / 2, table), edges


def load_potts_mpnn(checkpoint: str | Path, *, backbone_noise=0.0) -> PottsMPNN:
    """Load the supported, hash-verified v6 release. No checkpoint downloads.

    Only this trusted release is accepted: its alphabet and architecture are
    not stored completely in the checkpoint. Reject other files rather than
    assigning guessed metadata or silently loading an incompatible head.
    """
    path = Path(checkpoint)
    if not np.isfinite(backbone_noise) or backbone_noise < 0:
        raise ValueError("backbone_noise must be finite and nonnegative")
    if path.suffix == ".npz":
        # Converted once at image build time; inference never imports Torch.
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive["source_sha256"].item()) != V6_CHECKPOINT_SHA256
                    or tuple(archive["token_names"].tolist()) != V6_TOKENS):
                raise ValueError("Unsupported converted PottsMPNN checkpoint metadata")
            weights = {name: jnp.asarray(archive[name], dtype=jnp.float32)
                       for name in archive.files if name not in ("source_sha256", "token_names")}
        if weights["etab_out.weight"].shape != (900, 128):
            raise ValueError("Only the released edge-head architecture is supported")
        return PottsMPNN(weights, V6_TOKENS, V6_CHECKPOINT_SHA256,
                         backbone_noise=float(backbone_noise))
    import torch
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != V6_CHECKPOINT_SHA256:
        raise ValueError("Unsupported PottsMPNN checkpoint SHA-256")
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_data.get("train_cfg", {}).get("extended_vocab") != "v6":
        raise ValueError("Expected released v6 token metadata")
    weights = {name: jnp.asarray(value.detach().cpu().numpy(), dtype=jnp.float32)
               for name, value in checkpoint_data["model"].items()
               if name.startswith(("graph_featurization_module.", "encoder_layers.", "W_e.", "etab_out."))}
    if weights["etab_out.weight"].shape != (900, 128):
        raise ValueError("Only the released edge-head architecture is supported")
    return PottsMPNN(weights, V6_TOKENS, digest, backbone_noise=float(backbone_noise))


def incoming_edges(edges):
    """Padded adjacency retaining every incoming directed nonself edge.

    A kNN graph can have more than K incoming edges. L-1 is the safe capacity;
    truncating incoming edges to K changes the upstream Hamiltonian.
    """
    length, neighbors = edges.shape
    source = jnp.repeat(jnp.arange(length), neighbors - 1)
    slot = jnp.tile(jnp.arange(1, neighbors), length)
    target = edges[:, 1:].reshape(-1)
    if neighbors == 1:
        return jnp.zeros((length, 1), jnp.int32), jnp.zeros((length, 1), jnp.int32), jnp.zeros((length, 1), bool)
    order = jnp.argsort(target, stable=True)
    counts = jnp.bincount(target, length=length)
    starts = jnp.cumsum(counts) - counts
    rank = jnp.arange(length - 1)[None, :]
    positions = jnp.minimum(starts[:, None] + rank, target.size - 1)
    return source[order[positions]], slot[order[positions]], rank < counts[:, None]


def candidate_energies(table, edges, states, position, incoming):
    """Exact single-site energy, up to a candidate-independent constant."""
    src, slot, valid = incoming
    outgoing = table[position, jnp.arange(1, edges.shape[1]), :, states[edges[position, 1:]]].sum(0)
    entering = table[src[position], slot[position], states[src[position]], :]
    return jnp.diag(table[position, 0]) + outgoing + jnp.sum(entering * valid[position, :, None], axis=0)


def gibbs_marginals(table, edges, allowed, token_bias, *, key, temperature,
                    burn_in=5, num_sweeps=5):
    """Approximate extended-token marginals by sequential randomized Gibbs.

    Identities/states are clamped per chain via ``allowed [C,L,V]``. Average the exact
    conditional probabilities (Rao-Blackwell estimates) during retained
    sweeps. Finite chains are approximate; temperature scales model energy,
    and is not physical pH. Bias is added to conditional log probabilities.
    Return [C,L,V] marginals; the caller applies target-distribution weights.
    """
    incoming = incoming_edges(edges)
    chains, length, vocab = allowed.shape
    keys = jax.random.split(key, chains)
    def chain(chain_key, chain_allowed):
        initial_key, chain_key = jax.random.split(chain_key)
        states = jax.random.categorical(initial_key, jnp.where(chain_allowed, 0.0, -jnp.inf))
        accumulator = jnp.zeros((length, vocab), dtype=table.dtype)
        def sweep(carry, sweep_idx):
            states, accumulator, rng = carry
            rng, order_key, update_key = jax.random.split(rng, 3)
            order = jax.random.permutation(order_key, length)
            def update(carry, position):
                states, probabilities = carry
                energy = candidate_energies(table, edges, states, position, incoming)
                logits = jnp.where(chain_allowed[position], -energy / temperature + token_bias[position], -jnp.inf)
                conditional = jax.nn.softmax(logits)
                state = jax.random.categorical(jax.random.fold_in(update_key, position), logits)
                return (states.at[position].set(state), probabilities.at[position].set(conditional)), None
            (states, probabilities), _ = jax.lax.scan(update, (states, jnp.zeros_like(accumulator)), order)
            accumulator = accumulator + jnp.where(sweep_idx >= burn_in, probabilities, 0.0)
            return (states, accumulator, rng), None
        (_, accumulator, _), _ = jax.lax.scan(sweep, (states, accumulator, chain_key),
                                               jnp.arange(burn_in + num_sweeps))
        return accumulator / num_sweeps
    return jax.lax.stop_gradient(jax.vmap(chain)(keys, allowed))
