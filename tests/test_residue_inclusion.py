import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import ExpectedResidueCountLoss


def inclusion(sequence):
    key = jax.random.key(0)
    return sum(ExpectedResidueCountLoss(residue=group, target_expected_count=1., at_least=True)(
        sequence, None, key=key)[0] for group in ("H", "DE"))


@pytest.mark.parametrize("sequence,expected", [
    ("ADAH", 0.), ("AEAH", 0.), ("HAAE", 0.), ("DHHEDE", 0.),
    ("AAAA", 2.), ("AHAA", 1.), ("ADEA", 1.),
])
def test_position_free_inclusion(sequence, expected):
    p = jax.nn.one_hot(jnp.array([TOKENS.index(aa) for aa in sequence]), 20)
    assert float(jax.jit(inclusion)(p)) == pytest.approx(expected)
    if expected == 0.:
        np.testing.assert_array_equal(jax.grad(inclusion)(p), 0.)


def test_acidic_group_shares_count_and_gradient():
    p = jnp.zeros((2, 20)).at[:, TOKENS.index("A")].set(0.75)
    p = p.at[0, TOKENS.index("D")].set(0.25).at[1, TOKENS.index("E")].set(0.25)
    term = ExpectedResidueCountLoss(residue="DE", target_expected_count=1., at_least=True)
    value, aux = term(p, None, key=jax.random.key(0))
    assert float(value) == pytest.approx(0.25)
    assert float(aux["expected_DE"]) == pytest.approx(0.5)
    gradient = jax.grad(lambda x: term(x, None, key=jax.random.key(0))[0])(p)
    np.testing.assert_allclose(gradient[:, [TOKENS.index("D"), TOKENS.index("E")]], -1.)
    np.testing.assert_array_equal(gradient[:, TOKENS.index("A")], 0.)


@pytest.mark.parametrize("group", ["", "DD", "DX"])
def test_invalid_group_rejected(group):
    with pytest.raises(ValueError):
        ExpectedResidueCountLoss(residue=group)
