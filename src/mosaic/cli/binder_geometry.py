"""Build geometry loss terms from Hydra / OmegaConf for binder_design."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mosaic.losses.geometry import (
    CysteineSidechainGeometryLoss,
    PairwiseCADistanceLoss,
    PairwiseFrameOrientationLoss,
)


def geometry_terms_from_cfg(geom: DictConfig | dict | None, binder_len: int):
    """Return list of (weight, LossTerm) for non-zero weights."""
    if geom is None:
        return []
    g = OmegaConf.to_container(geom, resolve=True)
    if not isinstance(g, dict):
        return []
    out: list = []

    w_ca = float(g.get("ca_ca_loss_weight", 0.0) or 0.0)
    pairs = g.get("ca_ca_pairs") or []
    if w_ca != 0.0 and pairs:
        ii, jj, dd = [], [], []
        for p in pairs:
            i, j = int(p["i"]), int(p["j"])
            d = float(p["target_distance"])
            if not (0 <= i < binder_len and 0 <= j < binder_len):
                raise ValueError(f"geometry.ca_ca_pairs index out of range: {i},{j} vs L={binder_len}")
            ii.append(i)
            jj.append(j)
            dd.append(d)
        term = PairwiseCADistanceLoss(
            pairs_i=jnp.array(ii, dtype=jnp.int32),
            pairs_j=jnp.array(jj, dtype=jnp.int32),
            target_distances=jnp.array(dd, dtype=jnp.float32),
        )
        out.append((w_ca, term))

    w_fr = float(g.get("frame_orient_loss_weight", 0.0) or 0.0)
    fpairs = g.get("frame_pairs") or []
    if w_fr != 0.0 and fpairs:
        ii, jj = [], []
        R_list = []
        for p in fpairs:
            i, j = int(p["i"]), int(p["j"])
            if not (0 <= i < binder_len and 0 <= j < binder_len):
                raise ValueError(
                    f"geometry.frame_pairs index out of range: {i},{j} vs L={binder_len}"
                )
            ii.append(i)
            jj.append(j)
            rt = p.get("rotation_target")
            if rt is None:
                R = np.eye(3, dtype=np.float32)
            else:
                flat = np.array(list(rt), dtype=np.float32).reshape(9)
                R = flat.reshape(3, 3)
            R_list.append(R)
        R_stack = jnp.array(np.stack(R_list, axis=0), dtype=jnp.float32)
        term = PairwiseFrameOrientationLoss(
            pairs_i=jnp.array(ii, dtype=jnp.int32),
            pairs_j=jnp.array(jj, dtype=jnp.int32),
            rotation_targets=R_stack,
        )
        out.append((w_fr, term))

    w_cys = float(g.get("cys_sidechain_loss_weight", 0.0) or 0.0)
    cy = g.get("cysteine") or {}
    if w_cys != 0.0 and cy:
        idx = cy.get("indices") or []
        chi_deg = cy.get("chi1_target_deg")
        chi_rad = None if chi_deg is None else math.radians(float(chi_deg))
        sg_list = cy.get("sg_sg_pairs") or []
        if not idx and not sg_list:
            raise ValueError(
                "geometry.cys_sidechain_loss_weight > 0 requires cysteine.indices "
                "and/or cysteine.sg_sg_pairs"
            )
        if chi_rad is not None and not idx:
            raise ValueError(
                "geometry.cysteine.chi1_target_deg requires cysteine.indices"
            )
        si, sj, sd = [], [], []
        for p in sg_list:
            si.append(int(p["i"]))
            sj.append(int(p["j"]))
            sd.append(float(p["target_distance"]))
        term = CysteineSidechainGeometryLoss(
            cys_token_indices=jnp.array(idx, dtype=jnp.int32),
            sg_pairs_i=jnp.array(si, dtype=jnp.int32),
            sg_pairs_j=jnp.array(sj, dtype=jnp.int32),
            sg_pair_target_dist=jnp.array(sd, dtype=jnp.float32),
            chi1_target_rad=chi_rad,
        )
        out.append((w_cys, term))

    return out
