from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from boltz.data.const import ref_atoms

from mosaic.losses.geometry import (
    HisCarboxylateHydrogenBondLoss,
    project_carboxylate_acceptor_h,
    project_his_donor_h_nd1,
)


@pytest.mark.parametrize("acceptor", ["ASP", "GLU"])
def test_harmonic_gap_and_force(acceptor):
    """Known rigid translations must yield k/2*d^2 and restoring force -k*d."""
    acc_names, his_names = ref_atoms[acceptor], ref_atoms["HIS"]
    split = len(acc_names)
    x = jnp.zeros((split + len(his_names), 3))
    names = ("OD1", "CG", "OD2") if acceptor == "ASP" else ("OE1", "CD", "OE2")
    for name, point in zip(names, ([0., 1., 0.], [-0.6, 0., 0.], [0., -1., 0.])):
        x = x.at[acc_names.index(name)].set(jnp.array(point))
    target = project_carboxylate_acceptor_h(*(x[acc_names.index(n)] for n in names))
    n, c1, c2 = map(jnp.asarray, ([0., 0., 0.], [-0.5, 0.866, 0.], [-0.5, -0.866, 0.]))
    shift = target - project_his_donor_h_nd1(n, c1, c2)
    for name, point in zip(("ND1", "CG", "CE1"), (n, c1, c2)):
        x = x.at[split + his_names.index(name)].set(point + shift)
    mapping = jax.nn.one_hot(jnp.array([0] * split + [1] * len(his_names)), 2)[None]
    term = HisCarboxylateHydrogenBondLoss(1, 0, acceptor)

    def energy(delta):
        coordinates = x.at[split:].add(delta)
        output = SimpleNamespace(structure_coordinates=coordinates[None],
                                 features={"atom_to_token": mapping})
        return term(jnp.zeros((2, 20)), output, jax.random.key(0))[0]

    zero = jnp.zeros(3)
    np.testing.assert_allclose(energy(zero), 0., atol=1e-10)
    np.testing.assert_allclose(jax.grad(energy)(zero), zero, atol=1e-6)
    displacement = jnp.array([2., -1., 0.5])
    np.testing.assert_allclose(jax.jit(energy)(displacement), 2.625, atol=1e-6)
    np.testing.assert_allclose(jax.grad(energy)(displacement), displacement, atol=1e-6)
    assert float(energy(-0.1 * displacement + displacement)) < float(energy(displacement))
    # Early diffusion samples can exceed the fork's geometric domain.
    x = x.at[acc_names.index(names[0]), 1].set(3.0)
    x = x.at[acc_names.index(names[2]), 1].set(-3.0)
    assert np.isfinite(float(energy(zero)))
    assert np.isfinite(np.asarray(jax.grad(energy)(zero))).all()


def test_invalid_pair_rejected():
    with pytest.raises(ValueError):
        HisCarboxylateHydrogenBondLoss(0, 0)
    with pytest.raises(ValueError):
        HisCarboxylateHydrogenBondLoss(1, 0, "ASN")
