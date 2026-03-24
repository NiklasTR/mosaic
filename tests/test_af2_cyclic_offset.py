"""Regression tests for Halo-aligned cyclic binder pairwise offsets in mosaic AF2."""

import numpy as np

from mosaic.models.af2 import cyclic_pairwise_offset, make_af_features
from mosaic.structure_prediction import TargetChain


def _halo_reference_cyclic_offset(length: int, offset_type: int) -> np.ndarray:
    """Copied from Halo ``cycle0.process._add_cyclic_offset`` (inner ``cyclic_offset``)."""

    i = np.arange(length)
    ij = np.stack([i, i + length], -1)
    offset = i[:, None] - i[None, :]
    c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))

    if offset_type == 1:
        c_offset = c_offset
    elif offset_type >= 2:
        a = c_offset < np.abs(offset)
        c_offset[a] = -c_offset[a]
    if offset_type == 3:
        idx = np.abs(c_offset) > 2
        c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
    return (c_offset * np.sign(offset)).astype(np.float32)


def test_cyclic_pairwise_offset_matches_halo_reference() -> None:
    for length in (3, 5, 8):
        for offset_type in (1, 2, 3):
            got = cyclic_pairwise_offset(length, offset_type)
            want = _halo_reference_cyclic_offset(length, offset_type)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_make_af_features_relative_position_offset_always_present() -> None:
    chains = [
        TargetChain(sequence="GGG", use_msa=False),
        TargetChain(sequence="AAAA", use_msa=False),
    ]
    feats = make_af_features(chains, cyclic_binder=False)
    ro = feats["relative_position_offset"]
    assert ro.shape == (7, 7)
    idx = np.concatenate([np.arange(3), np.arange(4)]).astype(np.float32)
    want = idx[:, None] - idx[None, :]
    np.testing.assert_array_equal(ro, want)


def test_make_af_features_cyclic_binder_patches_binder_block_only() -> None:
    chains = [
        TargetChain(sequence="GGG", use_msa=False),
        TargetChain(sequence="AAAA", use_msa=False),
    ]
    linear = make_af_features(chains, cyclic_binder=False)
    cyclic = make_af_features(chains, cyclic_binder=True, cyclic_offset_type=2)
    b = 3
    assert not np.allclose(
        cyclic["relative_position_offset"][:b, :b],
        linear["relative_position_offset"][:b, :b],
    )
    np.testing.assert_array_equal(
        cyclic["relative_position_offset"][b:, b:],
        linear["relative_position_offset"][b:, b:],
    )
    np.testing.assert_array_equal(
        cyclic["relative_position_offset"][:b, b:],
        linear["relative_position_offset"][:b, b:],
    )
    np.testing.assert_array_equal(
        cyclic["relative_position_offset"][b:, :b],
        linear["relative_position_offset"][b:, :b],
    )
