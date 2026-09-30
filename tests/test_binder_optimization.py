"""Binder CLI dispatch and trajectory export without loading protein models."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mosaic.cli import binder_optimization as dispatch
from mosaic.common import LossTerm
from mosaic.optimizers import simplex_APGM


class ToyLoss(LossTerm):
    def __call__(self, x, *, key):
        value = (x * jnp.arange(x.shape[-1])).sum()
        return value, {"toy": value}


def config():
    return OmegaConf.load(Path(__file__).parents[1] / "configs/binder_design.yaml").optimizer


def test_default_dispatch_preserves_legacy_calls_and_incumbent_restarts():
    opt = config()
    assert opt.method == "three_phase"
    x = jnp.array([[0.1, 0.3, 0.6], [0.4, 0.4, 0.2]])
    key = jax.random.key(12)
    expected = x
    for stage in (1, 2, 3):
        opt[f"phase{stage}_n_steps"] = 2
        _, expected = simplex_APGM(
            loss_function=ToyLoss(), x=expected if stage == 1 else jnp.log(expected + 1e-5),
            n_steps=2, stepsize=opt[f"phase{stage}_stepsize_factor"] * np.sqrt(2),
            momentum=opt[f"phase{stage}_momentum"], scale=opt[f"phase{stage}_scale"],
            logspace=stage != 1, max_gradient_norm=opt.max_gradient_norm,
            key=jax.random.fold_in(key, 810 + stage),
        )
    actual = dispatch.optimize_binder_sequence(loss=ToyLoss(), x=x, optimizer=opt, key=key)
    np.testing.assert_array_equal(actual, expected)
    del opt["method"]  # Older resolved configs retain the original behavior.
    np.testing.assert_array_equal(
        dispatch.optimize_binder_sequence(loss=ToyLoss(), x=x, optimizer=opt, key=key), expected,
    )


def test_fisher_dispatch_returns_terminal_state_and_resolves_defaults(monkeypatch):
    opt = config()
    opt.method = "fisher"
    x = jnp.full((16, 19), 1 / 19)
    key = jax.random.key(4)
    final, best = x.at[:, 0].add(0.01), x.at[:, 1].add(0.01)
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return final, best

    monkeypatch.setattr(dispatch, "fisher_mirror_descent", fake)
    actual = dispatch.optimize_binder_sequence(loss=ToyLoss(), x=x, optimizer=opt, key=key)
    np.testing.assert_array_equal(actual, final)
    assert seen["stepsize"] == pytest.approx(15.2)
    assert seen["stepsize_end"] == pytest.approx(2.8)
    assert seen["entropy_coefficient_end"] == pytest.approx(1 / 7)
    assert seen["n_steps"] == 165
    np.testing.assert_array_equal(jax.random.key_data(seen["key"]), jax.random.key_data(jax.random.fold_in(key, 811)))


@pytest.mark.parametrize("n_steps", [0, 4])
def test_fisher_trajectory_export_has_replay_state_and_numeric_aux(tmp_path, n_steps):
    opt = config()
    opt.method = "fisher"
    opt.fisher.n_steps = n_steps
    opt.fisher.save_trajectory = True
    x = jnp.array([[0.2, 0.3, 0.5], [0., 1., 0.]])
    mask = jnp.array([True, False])
    path = tmp_path / "trajectory.npz"
    final = dispatch.optimize_binder_sequence(
        loss=ToyLoss(), x=x, optimizer=opt, key=jax.random.key(0),
        design_mask=mask, trajectory_path=path,
    )
    with np.load(path, allow_pickle=False) as trace:
        for field in trace.files:
            assert trace[field].dtype.kind != "O"
        np.testing.assert_array_equal(trace["final_pssm"], final)
        np.testing.assert_array_equal(trace["design_mask"], mask)
        if n_steps:
            assert trace["logits"].shape == (n_steps, 2, 3)
            np.testing.assert_allclose(trace["loss"], (trace["pssm"] * np.arange(3)).sum((1, 2)), atol=1e-6)
            np.testing.assert_array_equal(trace["next_pssm"][-1], final)
            np.testing.assert_array_equal(trace["aux/['toy']"], trace["loss"])
            np.testing.assert_allclose(trace["pssm"][:, 1], np.tile(x[1], (n_steps, 1)))


def test_unknown_method_fails():
    opt = config()
    opt.method = "typo"
    with pytest.raises(ValueError, match="optimizer.method"):
        dispatch.optimize_binder_sequence(loss=ToyLoss(), x=jnp.full((2, 3), 1 / 3), optimizer=opt, key=jax.random.key(0))


@pytest.mark.parametrize("alphabet_size", [19, 20])
def test_full_default_fisher_schedule_on_linear_sequence_objective(alphabet_size):
    opt = config()
    opt.method = "fisher"
    x = jnp.full((16, alphabet_size), 1 / alphabet_size)
    final = dispatch.optimize_binder_sequence(
        loss=ToyLoss(), x=x, optimizer=opt, key=jax.random.key(0),
    )
    assert np.isfinite(final).all()
    np.testing.assert_allclose(final.sum(-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(final.argmax(-1), np.zeros(16))
    assert np.asarray(final[:, 0]).min() > 0.99
