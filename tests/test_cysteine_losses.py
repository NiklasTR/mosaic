"""Unit tests for cysteine count structure losses."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from mosaic.common import TOKENS
from mosaic.losses import structure_prediction as sp


def test_expected_cysteine_count_loss_zero_when_matches() -> None:
    cys = TOKENS.index("C")
    seq = jnp.zeros((4, 20))
    seq = seq.at[:, cys].set(0.5)
    term = sp.ExpectedCysteineCountLoss(target_expected_cys=2.0)
    v, aux = term(seq, MagicMock(), key=jax.random.key(0))
    assert float(v) == pytest.approx(0.0, abs=1e-5)
    assert float(aux["expected_cys"]) == pytest.approx(2.0, abs=1e-5)


def test_expected_cysteine_count_loss_positive_when_off() -> None:
    cys = TOKENS.index("C")
    seq = jnp.zeros((4, 20))
    seq = seq.at[:, cys].set(0.25)
    term = sp.ExpectedCysteineCountLoss(target_expected_cys=2.0)
    v, aux = term(seq, MagicMock(), key=jax.random.key(1))
    assert float(v) > 0
    assert float(aux["expected_cys"]) == pytest.approx(1.0, abs=1e-5)
