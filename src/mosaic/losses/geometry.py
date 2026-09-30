"""Differentiable Cα–Cα, backbone frame, cysteine side-chain, and His/His-Arg H-bond geometry losses (Boltz-2)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from boltz.data.const import ref_atoms
from jaxtyping import Array, Float, Int

from ..common import LossTerm
from .structure_prediction import AbstractStructureOutput


def backbone_frames_n_ca_c(
    n: Float[Array, "N 3"],
    ca: Float[Array, "N 3"],
    c: Float[Array, "N 3"],
) -> Float[Array, "N 3 3"]:
    v1 = n - ca
    v2 = c - ca
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    c_hat = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    e2 = jnp.cross(e1, c_hat)
    e2 = e2 / (jnp.linalg.norm(e2, axis=-1, keepdims=True) + 1e-8)
    e3 = jnp.cross(e1, e2)
    e3 = e3 / (jnp.linalg.norm(e3, axis=-1, keepdims=True) + 1e-8)
    return jnp.stack([e1, e2, e3], axis=-1)


def backbone_frames_from_output(
    output: AbstractStructureOutput,
) -> Float[Array, "N 3 3"]:
    bb = output.backbone_coordinates
    return backbone_frames_n_ca_c(bb[:, 0], bb[:, 1], bb[:, 2])


def dihedral_angle(
    p0: Float[Array, "3"],
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    p3: Float[Array, "3"],
) -> Float[Array, ""]:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (jnp.linalg.norm(b1) + 1e-8)
    v = b0 - jnp.dot(b0, b1n) * b1n
    w = b2 - jnp.dot(b2, b1n) * b1n
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1n, v), w)
    return jnp.arctan2(y, x)


# Amber14 ff14SB optima for a protonated histidine N–H (HID/HIE/HIP):
#   Bond protein-H – protein-NA: length 0.101 nm = 1.01 A
#     (thermax/assets/forcefields/protein.ff14SB.xml:2880)
#   Angles CR–NA–H / CW–NA–H / CC–NA–H: 2.09439 rad = 120 deg
#     (protein.ff14SB.xml:3067-3069 and equivalents).
# At fixed heavy atoms the minimum of the bond + equal angle terms is analytic:
# H sits r_eq from N along the in-ring-plane exterior bisector, i.e. -(u1+u2)/|u1+u2|.
HIS_DONOR_H_BOND_A = 1.01
HIS_DONOR_H_ANGLE_RAD = 2.0943951023931953

# Offsets into boltz.data.const.ref_atoms['HIS']
#   ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']
HIS_CG = 5
HIS_ND1 = 6
HIS_CD2 = 7
HIS_CE1 = 8
HIS_NE2 = 9


def _project_his_donor_h(
    n_pos: Float[Array, "3"],
    c1_pos: Float[Array, "3"],
    c2_pos: Float[Array, "3"],
    r_eq_A: float = HIS_DONOR_H_BOND_A,
) -> Float[Array, "3"]:
    """Core exterior-bisector projector shared by ND1/NE2 variants."""
    u1 = c1_pos - n_pos
    u1 = u1 / (jnp.linalg.norm(u1) + 1e-8)
    u2 = c2_pos - n_pos
    u2 = u2 / (jnp.linalg.norm(u2) + 1e-8)
    h_dir = -(u1 + u2)
    h_dir = h_dir / (jnp.linalg.norm(h_dir) + 1e-8)
    r = jnp.asarray(r_eq_A, dtype=n_pos.dtype)
    return n_pos + r * h_dir


def project_his_donor_h_ne2(
    ne2_pos: Float[Array, "3"],
    ce1_pos: Float[Array, "3"],
    cd2_pos: Float[Array, "3"],
    r_eq_A: float = HIS_DONOR_H_BOND_A,
) -> Float[Array, "3"]:
    """Idealized HE2 position for a protonated NE2 (HIE/HIP-like).

    Args:
        ne2_pos: donor nitrogen NE2 xyz in Angstrom.
        ce1_pos: ring-carbon neighbour CE1 xyz in Angstrom.
        cd2_pos: ring-carbon neighbour CD2 xyz in Angstrom.
        r_eq_A: Amber14 H–NA equilibrium bond length in Angstrom (1.01).

    Returns:
        xyz of the virtual HE2, differentiable w.r.t. all three inputs.
    """
    return _project_his_donor_h(ne2_pos, ce1_pos, cd2_pos, r_eq_A)


def project_his_donor_h_nd1(
    nd1_pos: Float[Array, "3"],
    cg_pos: Float[Array, "3"],
    ce1_pos: Float[Array, "3"],
    r_eq_A: float = HIS_DONOR_H_BOND_A,
) -> Float[Array, "3"]:
    """Idealized HD1 position for a protonated ND1 (HID/HIP-like).

    Args:
        nd1_pos: donor nitrogen ND1 xyz in Angstrom.
        cg_pos: ring-carbon neighbour CG xyz in Angstrom.
        ce1_pos: ring-carbon neighbour CE1 xyz in Angstrom.
        r_eq_A: Amber14 H–NA equilibrium bond length in Angstrom (1.01).

    Returns:
        xyz of the virtual HD1, differentiable w.r.t. all three inputs.
    """
    return _project_his_donor_h(nd1_pos, cg_pos, ce1_pos, r_eq_A)


# Rosetta ref2015 optimum for a H-bond donated TO His ND1
# (hbacc_IMD, RING_HYBRID): the donor H sits AH_DIS from ND1 along
# (ND1 - midpoint(CG, CE1)), i.e. B-A-H linear (180 deg) with D-H...A
# linear. AH_DIS = 1.92 A is the minimum of hbpoly_ahdist_aHIS_dHIS.
ROSETTA_HIS_HIS_AH_DIS_A = 1.92


def project_his_acceptor_nd1_h(
    nd1_pos: Float[Array, "3"],
    cg_pos: Float[Array, "3"],
    ce1_pos: Float[Array, "3"],
    ah_dis_A: float = ROSETTA_HIS_HIS_AH_DIS_A,
) -> Float[Array, "3"]:
    """Ideal donor-H position for H-bonding TO an accepting His ND1.

    Returns the xyz where a donor proton should sit at the Rosetta
    ref2015 optimum -- NOT a covalent N-H (cf. the 1.01 A covalent
    projectors `project_his_donor_h_nd1` / `project_his_donor_h_ne2`).

    Geometry: |H - ND1| = ``ah_dis_A`` along (ND1 - 0.5 * (CG + CE1)).
    Differentiable w.r.t. all three inputs.
    """
    pb = 0.5 * (cg_pos + ce1_pos)
    ba = nd1_pos - pb
    ba = ba / (jnp.linalg.norm(ba) + 1e-8)
    r = jnp.asarray(ah_dis_A, dtype=nd1_pos.dtype)
    return nd1_pos + r * ba


# Rosetta ref2015 H-bond geometry for His -> Asp/Glu (hbacc_CXL):
#   database/scoring/score_functions/hbonds/ref2015_params/HBEval.csv
#     hbdon_IMD and hbdon_IME both select the aASP_dHIS distance fit.
#   ref2015_params/HBPoly1D.csv, row 20:
#     hbpoly_ahdist_aASP_dHIS_9gt3_hesmooth_min1p6 has its minimum at
#     1.84999956 A (derivative roots within the polynomial's domain).
#   source/src/core/scoring/hbonds/hbonds_geom.cc:
#     get_hb_acc_chem_type maps both Asp/Glu to hbacc_CXL;
#     bah_chi_compute_energy_sp2 minimizes at BAH=120 deg, chi=0 or pi.
# Source: https://github.com/RosettaCommons/rosetta (main, 2026-09-30).
# This is an H-bond-term optimum for a His donor, not a full-score minimum
# or a universal donor-independent distance (e.g. Arg uses ~2.04 A).
ROSETTA_CARBOXYLATE_HIS_AH_DIS_A = 1.85


def project_carboxylate_acceptor_h(
    o_pos: Float[Array, "3"],
    c_pos: Float[Array, "3"],
    other_o_pos: Float[Array, "3"],
    ah_dis_A: float = ROSETTA_CARBOXYLATE_HIS_AH_DIS_A,
) -> Float[Array, "3"]:
    """Shared donor-H target forked by both Asp/Glu carboxylate oxygens.

    Supply (OD1, CG, OD2) or (OD2, CG, OD1) for Asp; supply
    (OE1, CD, OE2) or (OE2, CD, OE1) for Glu. Inputs are atom xyz
    coordinates in Angstrom, matching the His projector interface.

    H lies in the O-C-other_O plane on the perpendicular bisector of O-O,
    on the opposite side of O-O from C, ``ah_dis_A`` from BOTH oxygens.
    Swapping the oxygen inputs leaves the target unchanged. For equal C-O
    lengths this also lies on the outward O-C-O angle bisector; using the
    perpendicular bisector preserves equal O-H distances for unequal C-O
    lengths. The resulting C-O-H angles are not constrained to 120 degrees.

    This is a bifurcated geometric target, not the isolated single-oxygen
    Rosetta angular optimum. Only the default O-H distance (1.85 A) is
    taken from the ref2015 His-to-carboxylate distance fit. The target is
    a donor proton, not a covalently attached O-H; both oxygens are
    treated as deprotonated carboxylate acceptors.

    Requires distinct, non-collinear heavy atoms and ``ah_dis_A`` greater
    than half the O-O separation. No equidistant target exists when the
    separation exceeds twice that distance (the square root returns NaN).
    Differentiable w.r.t. all coordinates for valid geometry.
    """
    midpoint = 0.5 * (o_pos + other_o_pos)
    oo = other_o_pos - o_pos
    axis = oo / jnp.linalg.norm(oo)
    outward = midpoint - c_pos
    outward = outward - jnp.dot(outward, axis) * axis
    outward = outward / jnp.linalg.norm(outward)
    r = jnp.asarray(ah_dis_A, dtype=o_pos.dtype)
    height = jnp.sqrt(r * r - 0.25 * jnp.dot(oo, oo))
    return midpoint + height * outward


# Rosetta ref2015 His -> peptide backbone carbonyl (hbacc_PBA):
#   database/scoring/score_functions/hbonds/ref2015_params/HBEval.csv
#     maps hbdon_IMD/IME -> hbacc_PBA to the aGLY_dHIS distance fit,
#     for both seq_sep_PM1 and seq_sep_other (not restricted to Gly).
#   ref2015_params/HBPoly1D.csv, row 28:
#     hbpoly_ahdist_aGLY_dHIS_9gt3_hesmooth_min1p6 minimizes at
#     1.90999774 A (derivative roots within the polynomial's domain).
#   source/src/core/scoring/hbonds/hbonds_geom.cc:
#     backbone ABase2 is CA; bah_chi_compute_energy_sp2 has minima
#     at BAH=120 deg and chi=0 or pi in the CA-C-O plane.
# Source: https://github.com/RosettaCommons/rosetta (main, 2026-09-30).
ROSETTA_PEPTIDE_CARBONYL_HIS_AH_DIS_A = 1.91


def project_peptide_carbonyl_acceptor_h(
    o_pos: Float[Array, "3"],
    c_pos: Float[Array, "3"],
    ca_pos: Float[Array, "3"],
    ah_dis_A: float = ROSETTA_PEPTIDE_CARBONYL_HIS_AH_DIS_A,
) -> Float[Array, "3"]:
    """Ideal donor-H position for a peptide backbone carbonyl oxygen.

    Supply O, C, and CA xyz coordinates from the same residue, in
    Angstrom. The CA-C-O plane defines the local peptide plane; neither
    the same residue's N nor the next residue is needed. This applies to
    any residue's peptide carbonyl, not a terminal carboxylate or side
    chain carbonyl.

    Returns H at ``ah_dis_A`` from O, with angle(C, O, H) = 120 degrees,
    in the CA-C-O plane on the branch pointing away from CA. Rosetta's
    sp2 term permits both in-plane branches; we select one to return a
    single position. The 1.91 A default is the ref2015 His-donor distance
    fit minimum, not a donor-independent or full-score optimum. The
    returned point is a donor-H target, not a covalent O-H proton.

    Requires distinct, non-collinear heavy atoms and positive distance.
    Differentiable w.r.t. all input coordinates for valid geometry.
    """
    axis = c_pos - o_pos
    axis = axis / (jnp.linalg.norm(axis) + 1e-8)
    to_ca = ca_pos - o_pos
    perp = to_ca - jnp.dot(to_ca, axis) * axis
    perp = perp / (jnp.linalg.norm(perp) + 1e-8)
    h_dir = -0.5 * axis - 0.8660254037844386 * perp
    h_dir = h_dir / (jnp.linalg.norm(h_dir) + 1e-8)
    r = jnp.asarray(ah_dis_A, dtype=o_pos.dtype)
    return o_pos + r * h_dir


# Amber14 ff14SB optima for an Arg guanidinium N-H (NE/NH1/NH2, type N2):
#   Bond H-N2: 1.01 A (protein.ff14SB.xml, same H-N entry as His NA).
#   Angles CA-N2-H = H-N2-H = 120 deg; CZ is CA-type, so e.g.
#   CZ-NH1-HH11 = HH11-NH1-HH12 = 120 deg; N2-CA-N2 = 120 deg and the
#   CA-''-N2-N2 improper (k=43.9) lock NE/CZ/NH1/NH2 coplanar.
# Hence each pair-facing donor H sits 1.01 A from its N, in the guanidinium
# plane, 120 deg from N->CZ toward the acceptor side.
ARG_DONOR_H_BOND_A = 1.01

# Offsets into boltz.data.const.ref_atoms['ARG']
#   ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']
ARG_NE = 7
ARG_CZ = 8
ARG_NH1 = 9
ARG_NH2 = 10


def project_arg_donor_h(
    n_pos: Float[Array, "3"],
    cz_pos: Float[Array, "3"],
    toward_pos: Float[Array, "3"],
    r_eq_A: float = ARG_DONOR_H_BOND_A,
) -> Float[Array, "3"]:
    """Idealized pair-facing H for an Arg guanidinium nitrogen.

    Args:
        n_pos: donor nitrogen (NE, NH1 or NH2) xyz in Angstrom.
        cz_pos: guanidinium carbon CZ xyz in Angstrom (defines the
            in-plane 120-deg cone around the N->CZ axis).
        toward_pos: point on the acceptor side (e.g. the His acceptor
            ideal-H); selects the in-plane 120-deg branch facing the
            acceptor. Differentiable selection via soft sign -- exact
            branch flip at the perpendicular bisector is measure-zero.
        r_eq_A: Amber14 H-N2 equilibrium bond length in Angstrom (1.01).

    Returns:
        xyz of the virtual H with |H - N| = ``r_eq_A``,
        angle(CZ, N, H) = 120 deg, H coplanar with (N, CZ, toward).
        Differentiable w.r.t. all three inputs.
    """
    axis = cz_pos - n_pos
    axis = axis / (jnp.linalg.norm(axis) + 1e-8)
    # In-plane component of (toward - N) orthogonal to the N->CZ axis.
    t = toward_pos - n_pos
    perp = t - jnp.dot(t, axis) * axis
    n_perp = jnp.linalg.norm(perp) + 1e-8
    perp = perp / n_perp
    # H at 120 deg from N->CZ toward the acceptor side:
    # cos120 = -0.5, sin120 = sqrt(3)/2.
    h_dir = -0.5 * axis + 0.8660254037844386 * perp
    h_dir = h_dir / (jnp.linalg.norm(h_dir) + 1e-8)
    r = jnp.asarray(r_eq_A, dtype=n_pos.dtype)
    return n_pos + r * h_dir


def _first_atom_indices(feats0) -> Int[Array, "Ntok"]:
    return jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        feats0["atom_to_token"].T
    )


class HisCarboxylateHydrogenBondLoss(LossTerm):
    """Harmonic coincidence of His HD1 and a forked Asp/Glu acceptor target.

    Energy is ``0.5 * sum((H_HD1 - H_acceptor)**2)`` in A^2; multiply
    this term by a spring constant in a loss combination. There is no
    dead zone or best-pair selection. The acceptor uses the side-chain
    carboxylate (not a terminal backbone carboxylate).

    For distorted predictions with squared fork height below 0.01 A^2,
    a C1 height continuation and squared feasibility penalty extend the
    objective outside the exact projector's domain. Ordinary carboxylates
    use the exact fork geometry and pure harmonic coincidence energy.

    Indices are absolute Boltz token indices. These tokens MUST be fixed
    and featurized as HIS and ``acceptor_residue`` respectively, with the
    standard contiguous ref_atoms layout. Unknown-token features cannot
    be used for these side-chain projections.
    """

    his_token_index: int = eqx.field(static=True)
    acceptor_token_index: int = eqx.field(static=True)
    acceptor_residue: str = eqx.field(static=True, default="ASP")

    def __post_init__(self):
        if self.acceptor_residue not in ("ASP", "GLU"):
            raise ValueError("acceptor_residue must be ASP or GLU")
        if min(self.his_token_index, self.acceptor_token_index) < 0:
            raise ValueError("Token indices must be nonnegative")
        if self.his_token_index == self.acceptor_token_index:
            raise ValueError("His and carboxylate must be distinct tokens")

    def __call__(self, sequence, output: AbstractStructureOutput, key):
        if output.structure_coordinates is None or output.features is None:
            raise ValueError("His-carboxylate loss requires atom coordinates and features")
        first = _first_atom_indices(jax.tree.map(lambda x: x[0], output.features))
        if max(self.his_token_index, self.acceptor_token_index) >= first.shape[0]:
            raise ValueError("His/carboxylate token index outside feature tokens")
        x = output.structure_coordinates[0]
        his_base = first[self.his_token_index]
        acc_base = first[self.acceptor_token_index]
        his_atoms = ref_atoms["HIS"]
        acc_atoms = ref_atoms[self.acceptor_residue]
        h_donor = project_his_donor_h_nd1(
            *(x[his_base + his_atoms.index(name)] for name in ("ND1", "CG", "CE1"))
        )
        names = ("OD1", "CG", "OD2") if self.acceptor_residue == "ASP" else ("OE1", "CD", "OE2")
        o1, carbon, o2 = (x[acc_base + acc_atoms.index(name)] for name in names)
        midpoint = 0.5 * (o1 + o2)
        oo = o2 - o1
        axis = oo / jnp.sqrt(jnp.sum(oo * oo) + 1e-12)
        outward = midpoint - carbon
        outward = outward - jnp.dot(outward, axis) * axis
        outward = outward / jnp.sqrt(jnp.sum(outward * outward) + 1e-12)
        height_squared = ROSETTA_CARBOXYLATE_HIS_AH_DIS_A**2 - 0.25 * jnp.sum(oo * oo)
        # C1 continuation of sqrt below a 0.1 A height: predicted diffusion
        # geometries need not yet admit two 1.85 A O-H distances. Valid
        # carboxylates use the exact construction; invalid ones remain finite
        # and receive a restoring penalty instead of a silent NaN/zero loss.
        eps = 0.01
        height = jnp.where(height_squared >= eps,
                           jnp.sqrt(jnp.maximum(height_squared, eps)),
                           jnp.sqrt(eps) * jnp.exp(jnp.minimum((height_squared - eps) / (2 * eps), 0.0)))
        h_acceptor = midpoint + height * outward
        feasibility = jax.nn.relu(eps - height_squared) ** 2
        delta = h_donor - h_acceptor
        squared_gap = jnp.sum(delta * delta)
        loss = 0.5 * squared_gap + feasibility
        return loss, {
            "his_carboxylate_loss": loss,
            "his_carboxylate_gap_squared_A2": squared_gap,
            "carboxylate_feasibility_penalty": feasibility,
            "his_hd1_xyz": h_donor,
            "carboxylate_target_xyz": h_acceptor,
        }


class HisHisHydrogenBondLoss(LossTerm):
    """Satisfaction loss for at least one NE2-donor to ND1-acceptor H-bond.

    For each donor His, the covalent HE2 is projected with
    ``project_his_donor_h_ne2`` (Amber14 1.01 A exterior bisector). For each
    acceptor His, the ideal donor-H position is projected with
    ``project_his_acceptor_nd1_h`` (Rosetta ref2015 1.92 A B-A-H linear).
    At an ideal His(NE2-H)...His(ND1) H-bond the two virtual Hs coincide.

    Hinge form: ``relu(d_best - target)^2`` where ``d_best`` is the minimum
    virtual-H separation over all donor x acceptor pairs (self pairs
    excluded). One satisfied pair silences the term; extra pairs are free.

    Self pairs (donor token == acceptor token) are masked with +inf before
    the min. With no valid pair the term is inactive (0).

    Requires Boltz-2 output with ``features``. Token indices must be
    featurized as HIS (10 heavy atoms) so the ``HIS_*`` offsets apply --
    i.e. fix those positions to H in the binder scaffold (cf.
    ``SetPositions`` / ``ScaffoldBinderSequence``); with an all-X (UNK,
    5-atom) featurization ``base + 9`` would read into the next token.
    """

    donor_token_indices: Int[Array, "S"] = eqx.field(converter=jnp.asarray)
    acceptor_token_indices: Int[Array, "T"] = eqx.field(converter=jnp.asarray)
    target_distance_A: float = 0.5
    exclude_self_pairs: bool = True

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        sc = getattr(output, "structure_coordinates", None)
        feats = getattr(output, "features", None)
        if sc is None or feats is None:
            z = jnp.array(0.0)
            return z, {"his_hbond_inactive": jnp.array(1.0)}
        if self.donor_token_indices.shape[0] == 0 or self.acceptor_token_indices.shape[0] == 0:
            z = jnp.array(0.0, dtype=jnp.float32)
            return z, {"his_hbond_inactive": jnp.array(1.0)}

        assert ref_atoms["HIS"] == [
            "N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2",
        ]
        feats0 = jax.tree.map(lambda x: x[0], feats)
        first_idx = _first_atom_indices(feats0)
        x = sc[0]

        def donor_h_for_tok(ti: Int[Array, ""]):
            base = first_idx[ti]
            return project_his_donor_h_ne2(
                x[base + HIS_NE2], x[base + HIS_CE1], x[base + HIS_CD2]
            )

        def acceptor_h_for_tok(tj: Int[Array, ""]):
            base = first_idx[tj]
            return project_his_acceptor_nd1_h(
                x[base + HIS_ND1], x[base + HIS_CG], x[base + HIS_CE1]
            )

        h_don = jax.vmap(donor_h_for_tok)(self.donor_token_indices)
        h_acc = jax.vmap(acceptor_h_for_tok)(self.acceptor_token_indices)
        diff = h_don[:, None, :] - h_acc[None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)

        if self.exclude_self_pairs:
            self_mask = (
                self.donor_token_indices[:, None]
                == self.acceptor_token_indices[None, :]
            )
            dists = jnp.where(self_mask, jnp.inf, dists)

        d_best = dists.min()
        tgt = jnp.asarray(self.target_distance_A, dtype=d_best.dtype)
        hinge = jax.nn.relu(d_best - tgt)
        loss = hinge * hinge
        inactive = ~jnp.isfinite(d_best)
        loss = jnp.where(inactive, jnp.zeros((), dtype=d_best.dtype), loss)
        return loss, {
            "his_hbond_loss": loss,
            "his_hbond_d_best": jnp.where(
                inactive, jnp.zeros((), dtype=d_best.dtype), d_best
            ),
            "his_hbond_inactive": inactive.astype(jnp.float32),
        }


class HisArgHydrogenBondLoss(LossTerm):
    """Satisfaction loss for a bidentate Arg fork donating to a His ND1.

    Acceptor side: the His ND1 ideal donor-H position from
    ``project_his_acceptor_nd1_h`` (Rosetta ref2015 1.92 A B-A-H linear).
    Donor side: the two adjacent-N Arg pairs -- (NE, NH1) and (NH1, NH2) --
    each projected with ``project_arg_donor_h`` (Amber14 1.01 A, in-plane,
    120 deg from N->CZ) toward the acceptor ideal-H. At an ideal bidentate
    Arg(N-H)2...His(ND1) geometry both Arg virtual Hs coincide with the
    acceptor ideal-H, so the pair score is the mean of the two separations:

        s[p, j] = mean(|H_arg[p,0] - H_acc_ideal[j]|,
                       |H_arg[p,1] - H_acc_ideal[j]|),  p in {NE-NH1, NH1-NH2}
        d_best = min_{p, j} s[p, j];  loss = relu(d_best - target)^2

    One satisfied fork silences the term; extra forks are free. Arg and His
    live on different tokens here (binder-binder core network or either
    side of the interface), so no self-pair mask; with no valid pair the
    term is inactive (0).

    Requires Boltz-2 output with ``features``. Arg token indices must be
    featurized as ARG (11 heavy atoms, ``ARG_*`` offsets) and His as HIS
    (``HIS_*`` offsets) -- fix those positions in the binder scaffold.
    """

    arg_token_indices: Int[Array, "R"] = eqx.field(converter=jnp.asarray)
    his_token_indices: Int[Array, "S"] = eqx.field(converter=jnp.asarray)
    target_distance_A: float = 0.5

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        sc = getattr(output, "structure_coordinates", None)
        feats = getattr(output, "features", None)
        if sc is None or feats is None:
            z = jnp.array(0.0)
            return z, {"his_arg_hbond_inactive": jnp.array(1.0)}
        if self.arg_token_indices.shape[0] == 0 or self.his_token_indices.shape[0] == 0:
            z = jnp.array(0.0, dtype=jnp.float32)
            return z, {"his_arg_hbond_inactive": jnp.array(1.0)}

        assert ref_atoms["ARG"] == [
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
        ]
        assert ref_atoms["HIS"] == [
            "N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2",
        ]
        feats0 = jax.tree.map(lambda x: x[0], feats)
        first_idx = _first_atom_indices(feats0)
        x = sc[0]

        def acceptor_h_for_tok(tj: Int[Array, ""]):
            base = first_idx[tj]
            return project_his_acceptor_nd1_h(
                x[base + HIS_ND1], x[base + HIS_CG], x[base + HIS_CE1]
            )

        h_acc = jax.vmap(acceptor_h_for_tok)(self.his_token_indices)

        def pair_score_for_arg(ti: Int[Array, ""]):
            base = first_idx[ti]
            ne = x[base + ARG_NE]
            cz = x[base + ARG_CZ]
            nh1 = x[base + ARG_NH1]
            nh2 = x[base + ARG_NH2]
            # (NE, NH1) and (NH1, NH2) forks; each N projects toward
            # each acceptor ideal-H, so toward has shape (S, 3).
            h_ne = jax.vmap(lambda t: project_arg_donor_h(ne, cz, t))(h_acc)
            h_nh1 = jax.vmap(lambda t: project_arg_donor_h(nh1, cz, t))(h_acc)
            h_nh2 = jax.vmap(lambda t: project_arg_donor_h(nh2, cz, t))(h_acc)
            s_ne_nh1 = 0.5 * (
                jnp.linalg.norm(h_ne - h_acc, axis=-1)
                + jnp.linalg.norm(h_nh1 - h_acc, axis=-1)
            )
            s_nh1_nh2 = 0.5 * (
                jnp.linalg.norm(h_nh1 - h_acc, axis=-1)
                + jnp.linalg.norm(h_nh2 - h_acc, axis=-1)
            )
            return jnp.stack([s_ne_nh1, s_nh1_nh2], axis=0)

        scores = jax.vmap(pair_score_for_arg)(self.arg_token_indices)
        d_best = scores.min()
        tgt = jnp.asarray(self.target_distance_A, dtype=d_best.dtype)
        hinge = jax.nn.relu(d_best - tgt)
        loss = hinge * hinge
        inactive = ~jnp.isfinite(d_best)
        loss = jnp.where(inactive, jnp.zeros((), dtype=d_best.dtype), loss)
        return loss, {
            "his_arg_hbond_loss": loss,
            "his_arg_hbond_d_best": jnp.where(
                inactive, jnp.zeros((), dtype=d_best.dtype), d_best
            ),
            "his_arg_hbond_inactive": inactive.astype(jnp.float32),
        }


class SelectedTokenCentroidLoss(LossTerm):
    """Satisfaction pull of selected binder tokens toward the binder centroid.

    Burial proxy: ``relu(d_worst - r0)^2`` where ``d_worst`` is the largest
    |CA_i - centroid| over the selected token indices (hinge: once all
    selected tokens are inside ``r0``, the term is silent; worse stragglers
    beyond ``r0`` still pay quadratically). The centroid is taken over the
    binder (``sequence.shape[0]`` rows of ``output.backbone_coordinates``),
    not the target. Rotation/translation invariant, O(S).

    Note this constrains only the listed tokens -- other His elsewhere are
    free. ``target_radius_A = 0`` reduces to a pure ``d_worst^2`` penalty.
    """

    token_indices: Int[Array, "S"] = eqx.field(converter=jnp.asarray)
    target_radius_A: float = 0.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        if self.token_indices.shape[0] == 0:
            z = jnp.array(0.0)
            return z, {"centroid_inactive": jnp.array(1.0)}
        binder_len = sequence.shape[0]
        ca = output.backbone_coordinates[:binder_len, 1, :]
        centroid = ca.mean(axis=0)
        sel = ca[self.token_indices]
        d = jnp.linalg.norm(sel - centroid[None, :], axis=-1)
        d_worst = d.max()
        r0 = jnp.asarray(self.target_radius_A, dtype=d.dtype)
        excess = jax.nn.relu(d_worst - r0)
        loss = excess * excess
        return loss, {
            "centroid_loss": loss,
            "centroid_d_worst": d_worst,
            "centroid_d_mean": d.mean(),
            "centroid_radius": jnp.linalg.norm(
                ca - centroid[None, :], axis=-1
            ).mean(),
        }


class PairwiseCADistanceLoss(LossTerm):
    pairs_i: Int[Array, "P"] = eqx.field(converter=jnp.asarray)
    pairs_j: Int[Array, "P"] = eqx.field(converter=jnp.asarray)
    target_distances: Float[Array, "P"] = eqx.field(converter=jnp.asarray)

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        ca = output.backbone_coordinates[:, 1, :]
        d = jnp.linalg.norm(ca[self.pairs_i] - ca[self.pairs_j], axis=-1)
        sq = jnp.mean((d - self.target_distances) ** 2)
        return sq, {"ca_ca_geom_loss": sq, "ca_ca_d_pred_mean": jnp.mean(d)}


class PairwiseFrameOrientationLoss(LossTerm):
    pairs_i: Int[Array, "P"] = eqx.field(converter=jnp.asarray)
    pairs_j: Int[Array, "P"] = eqx.field(converter=jnp.asarray)
    rotation_targets: Float[Array, "P 3 3"] = eqx.field(converter=jnp.asarray)

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        R = backbone_frames_from_output(output)
        Ri = R[self.pairs_i]
        Rj = R[self.pairs_j]
        R_rel = jnp.matmul(jnp.swapaxes(Ri, -1, -2), Rj)
        err = jnp.mean((R_rel - self.rotation_targets) ** 2)
        return err, {"frame_orient_loss": err}


class CysteineSidechainGeometryLoss(LossTerm):
    """χ1 (N–Cα–Cβ–SG), optional SG–SG distance; requires Boltz-2 output with ``features``."""

    cys_token_indices: Int[Array, "S"] = eqx.field(converter=jnp.asarray)
    sg_pairs_i: Int[Array, "Q"] = eqx.field(converter=jnp.asarray)
    sg_pairs_j: Int[Array, "Q"] = eqx.field(converter=jnp.asarray)
    sg_pair_target_dist: Float[Array, "Q"] = eqx.field(converter=jnp.asarray)
    chi1_target_rad: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        sc = getattr(output, "structure_coordinates", None)
        feats = getattr(output, "features", None)
        if sc is None or feats is None:
            z = jnp.array(0.0)
            return z, {"cys_geom_inactive": jnp.array(1.0)}

        assert ref_atoms["CYS"][:6] == ["N", "CA", "C", "O", "CB", "SG"]
        i_n, i_ca, i_cb, i_sg = 0, 1, 4, 5
        feats0 = jax.tree.map(lambda x: x[0], feats)
        first_idx = _first_atom_indices(feats0)
        x = sc[0]

        def chi1_for_tok(ti: Int[Array, ""]):
            base = first_idx[ti]
            return dihedral_angle(
                x[base + i_n],
                x[base + i_ca],
                x[base + i_cb],
                x[base + i_sg],
            )

        chi_loss = jnp.array(0.0, dtype=x.dtype)
        if (
            self.chi1_target_rad is not None
            and self.cys_token_indices.shape[0] > 0
        ):
            tgt = jnp.array(self.chi1_target_rad, dtype=x.dtype)
            chi_vals = jax.vmap(chi1_for_tok)(self.cys_token_indices)
            dchi = chi_vals - tgt
            chi_loss = jnp.mean(dchi**2)

        sg_loss = jnp.array(0.0, dtype=x.dtype)
        if self.sg_pairs_i.shape[0] > 0:

            def sg_pos(ti: Int[Array, ""]):
                return x[first_idx[ti] + i_sg]

            a = jax.vmap(sg_pos)(self.sg_pairs_i)
            b = jax.vmap(sg_pos)(self.sg_pairs_j)
            dist = jnp.linalg.norm(a - b, axis=-1)
            sg_loss = jnp.mean((dist - self.sg_pair_target_dist) ** 2)

        total = chi_loss + sg_loss
        return total, {
            "cys_sidechain_geom": total,
            "cys_chi1_loss": chi_loss,
            "cys_sg_dist_loss": sg_loss,
        }
