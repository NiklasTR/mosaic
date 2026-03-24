import jax
import jax.numpy as jnp
import pytest

from mosaic.binder_scaffold import parse_binder_sequence_string, resolve_binder_sequence
from mosaic.losses.boltz2 import set_binder_sequence
from mosaic.losses.geometry import (
    PairwiseCADistanceLoss,
    backbone_frames_n_ca_c,
    backbone_frames_from_output,
)


def test_resolve_binder_all_x() -> None:
    s, m = resolve_binder_sequence(4, None)
    assert s == "XXXX"
    assert m.all()


def test_parse_scaffold_cxxc() -> None:
    s, m = parse_binder_sequence_string("CXXC")
    assert s == "CXXC"
    assert list(m) == [False, True, True, False]


def test_set_binder_sequence_respects_design_mask() -> None:
    L = 4
    res = jnp.zeros((1, L, 33), dtype=jnp.float32)
    res = res.at[0, :, 2:22].set(jnp.eye(20)[:L])
    msa = jnp.zeros((1, 2, L, 33), dtype=jnp.float32)
    msa = msa.at[0, 0, :, 2:22].set(jnp.eye(20)[:L])
    prof = jnp.zeros((1, L, 33), dtype=jnp.float32)
    prof = prof.at[0, :, 2:22].set(jnp.eye(20)[:L])
    feats = {"res_type": res, "msa": msa, "profile": prof}
    new_seq = jnp.zeros((L, 20), dtype=jnp.float32)
    new_seq = new_seq.at[:, 0].set(1.0)
    mask = jnp.array([True, False, False, True], dtype=jnp.bool_)
    out = set_binder_sequence(new_seq, feats, design_mask=mask)
    assert jnp.allclose(out["res_type"][0, 1], feats["res_type"][0, 1])
    assert jnp.allclose(out["res_type"][0, 2], feats["res_type"][0, 2])
    zp0 = jnp.pad(new_seq[0], (2, 11))
    assert jnp.allclose(out["res_type"][0, 0], zp0)


def test_backbone_frames_orthonormal() -> None:
    n = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    ca = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
    c = jnp.array([[2.0, 0.0, 1.0], [3.0, 1.0, 0.0]], dtype=jnp.float32)
    R = backbone_frames_n_ca_c(n, ca, c)
    for i in range(2):
        M = R[i]
        assert float(jnp.max(jnp.abs(M.T @ M - jnp.eye(3)))) < 1e-4


def test_pairwise_ca_distance_loss() -> None:
    bb = jnp.zeros((3, 4, 3), dtype=jnp.float32)
    bb = bb.at[:, 1, :].set(
        jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=jnp.float32)
    )

    class _Out:
        backbone_coordinates = bb

    out = _Out()
    term = PairwiseCADistanceLoss(
        pairs_i=jnp.array([0], dtype=jnp.int32),
        pairs_j=jnp.array([1], dtype=jnp.int32),
        target_distances=jnp.array([3.0], dtype=jnp.float32),
    )
    seq = jnp.zeros((3, 20))
    v, aux = term(seq, out, key=jax.random.key(0))
    assert float(v) == pytest.approx(0.0, abs=1e-4)
    assert float(aux["ca_ca_geom_loss"]) < 1e-3


def test_backbone_frames_from_mock_output() -> None:
    class _O:
        backbone_coordinates = jnp.zeros((2, 4, 3), dtype=jnp.float32)

    _O.backbone_coordinates = _O.backbone_coordinates.at[:, 0, 0].set(-1.0)
    _O.backbone_coordinates = _O.backbone_coordinates.at[:, 1, 0].set(0.0)
    _O.backbone_coordinates = _O.backbone_coordinates.at[:, 2, 0].set(1.0)
    R = backbone_frames_from_output(_O())
    assert R.shape == (2, 3, 3)
