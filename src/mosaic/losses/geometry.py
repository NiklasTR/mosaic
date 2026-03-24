"""Differentiable Cα–Cα, backbone frame, and cysteine side-chain geometry losses (Boltz-2)."""

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


def _first_atom_indices(feats0) -> Int[Array, "Ntok"]:
    return jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        feats0["atom_to_token"].T
    )


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
