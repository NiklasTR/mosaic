"""Binder sequence scaffold: fixed one-letter residues in YAML + X = designable."""

from __future__ import annotations

import numpy as np

from mosaic.common import TOKENS

_ALLOWED = frozenset(TOKENS) | {"X"}


def parse_binder_sequence_string(
    binder_sequence: str,
) -> tuple[str, np.ndarray]:
    s = binder_sequence.strip().upper()
    if not s:
        raise ValueError("binder_sequence must be non-empty")
    for i, ch in enumerate(s):
        if ch not in _ALLOWED:
            raise ValueError(
                f"binder_sequence position {i}: invalid residue {ch!r}; "
                f"use one of {sorted(_ALLOWED)}"
            )
    design_mask = np.array([c == "X" for c in s], dtype=np.bool_)
    if not np.any(design_mask):
        raise ValueError(
            "binder_sequence must contain at least one X (designable position)"
        )
    return s, design_mask


def resolve_binder_sequence(
    binder_length: int | None,
    binder_sequence: str | None,
) -> tuple[str, np.ndarray]:
    """Return (full_sequence, design_mask). If binder_sequence is None, use X*binder_length."""
    if binder_sequence is not None and str(binder_sequence).strip():
        seq, mask = parse_binder_sequence_string(str(binder_sequence))
        if binder_length is not None and int(binder_length) != len(seq):
            raise ValueError(
                f"binder_length={binder_length} but binder_sequence has length {len(seq)}"
            )
        return seq, mask
    if binder_length is None:
        raise ValueError("Provide binder_length or binder_sequence")
    L = int(binder_length)
    if L < 1:
        raise ValueError("binder_length must be >= 1")
    return "X" * L, np.ones(L, dtype=np.bool_)
