"""CPU tests for the Fisher flow, using real autodiff and synthetic losses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import LossTerm
from mosaic.optimizers import _fisher_mirror_step, fisher_mirror_descent, simplex_APGM


class LinearLoss(LossTerm):
    weights: jax.Array

    def __call__(self, x, *, key):
        value = (x * self.weights).sum()
        return value, {"linear": value}


class QuadraticLoss(LossTerm):
    target: jax.Array

    def __call__(self, x, *, key):
        value = ((x - self.target) ** 2).sum()
        return value, {"quadratic": value}


class RandomLinearLoss(LossTerm):
    def __call__(self, x, *, key):
        value = (x * jax.random.normal(key, x.shape)).sum()
        return value, {"random": value}


@pytest.mark.parametrize("scale", [1.25, 1.4])
def test_matches_legacy_logspace_updates(scale):
    x = jnp.array([[0.6, 0.3, 0.1], [0.15, 0.35, 0.5]])
    loss = QuadraticLoss(jnp.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]))
    eta = 2.0
    kwargs = dict(loss_function=loss, n_steps=8, key=jax.random.key(5), max_gradient_norm=1.0)
    legacy, _ = simplex_APGM(x=jnp.log(x), stepsize=eta, scale=scale, logspace=True, **kwargs)
    final, _ = fisher_mirror_descent(
        x=x, stepsize=scale * eta, entropy_coefficient=(scale - 1) / (scale * eta), **kwargs,
    )
    np.testing.assert_allclose(final, legacy, atol=2e-6, rtol=2e-6)


def test_linear_loss_has_exact_exponentiated_solution_with_global_clipping():
    x = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    weights = jnp.array([[1.0, 4.0, -2.0], [0.0, 2.0, -3.0]])
    g = weights - weights.mean(-1, keepdims=True)
    cap = 0.4
    g = g * (cap / np.linalg.norm(g))
    final, _ = fisher_mirror_descent(
        loss_function=LinearLoss(weights), x=x, n_steps=7, stepsize=0.3,
        max_gradient_norm=cap, key=jax.random.key(0),
    )
    expected = jax.nn.softmax(jnp.log(x) - 7 * 0.3 * g, axis=-1)
    np.testing.assert_allclose(final, expected, atol=2e-7)


def test_step_has_fisher_velocity_and_is_gauge_invariant():
    z = jnp.array([[0.4, -0.3, 0.1], [-0.1, 0.5, -0.4]])
    g = jnp.array([[0.1, 0.4, -0.2], [0.3, -0.1, 0.5]])
    p = jax.nn.softmax(z, axis=-1)
    h, lam = 0.001, 0.2
    new_z = _fisher_mirror_step(z, g, h, lam)
    f = g - lam * jnp.log(p)
    velocity = -p * (f - (p * f).sum(-1, keepdims=True))
    _, actual_velocity = jax.jvp(
        lambda dt: jax.nn.softmax(_fisher_mirror_step(z, g, dt, lam)),
        (jnp.array(0.0),), (jnp.array(1.0),),
    )
    np.testing.assert_allclose(actual_velocity, velocity, atol=1e-7)
    shifted = _fisher_mirror_step(z + jnp.array([[4.0], [-7.0]]), g + 3, h, lam)
    np.testing.assert_allclose(jax.nn.softmax(shifted), jax.nn.softmax(new_z), atol=2e-7)


def test_best_and_trajectory_refer_to_evaluated_input_not_next_iterate():
    x = jnp.array([[0.5, 0.5]])
    target = jnp.array([[0.6, 0.4]])
    loss = QuadraticLoss(target)
    final, best, records = fisher_mirror_descent(
        loss_function=loss, x=x, n_steps=1, stepsize=50.0,
        key=jax.random.key(1), trajectory_fn=lambda record: record,
    )
    np.testing.assert_allclose(best, x)
    assert float(((final - target) ** 2).sum()) > float(((best - target) ** 2).sum())
    record = records[0]
    np.testing.assert_allclose(record["pssm"], x)
    np.testing.assert_allclose(record["next_pssm"], final)
    assert float(record["loss"]) == pytest.approx(float(((x - target) ** 2).sum()))
    assert record["is_best"]


def test_fixed_one_hot_rows_and_zero_initial_probabilities_remain_finite():
    x = jnp.array([[0.0, 0.4, 0.6], [0.0, 1.0, 0.0]])
    mask = jnp.array([True, False])
    final, _, records = fisher_mirror_descent(
        loss_function=LinearLoss(jnp.array([[-5., 1., 0.], [100., -100., 50.]])),
        x=x, n_steps=30, stepsize=2.0, entropy_coefficient=0.1,
        design_mask=mask, key=jax.random.key(0), trajectory_fn=lambda record: record,
    )
    for record in records:
        np.testing.assert_array_equal(record["pssm"][1], x[1])
        np.testing.assert_array_equal(record["gradient"][1], 0)
        assert np.isfinite(record["logits"]).all()
        assert np.isfinite(record["entropy"])
    assert float(records[0]["pssm"][0, 0]) > 0
    np.testing.assert_array_equal(final[1], x[1])
    np.testing.assert_allclose(final.sum(-1), 1, atol=1e-6)
    assert np.isfinite(final).all()


def test_smooth_schedule_endpoints_and_single_rng_chain():
    initial_key = jax.random.key(7)
    final, _, records = fisher_mirror_descent(
        loss_function=RandomLinearLoss(), x=jnp.full((2, 3), 1 / 3),
        n_steps=5, stepsize=0.5, stepsize_end=1.3,
        entropy_coefficient_end=0.2, entropy_schedule_power=6.0,
        key=initial_key, trajectory_fn=lambda record: record,
    )
    assert records[0]["stepsize"] == pytest.approx(0.5)
    assert records[-1]["stepsize"] == pytest.approx(1.3)
    assert records[0]["entropy_coefficient"] == 0
    assert records[-1]["entropy_coefficient"] == pytest.approx(0.2)
    assert records[2]["entropy_coefficient"] == pytest.approx(0.2 * 0.5**6)
    key = initial_key
    for index, record in enumerate(records):
        np.testing.assert_array_equal(record["key"], jax.random.key_data(key))
        if index:
            np.testing.assert_array_equal(record["pssm"], records[index - 1]["next_pssm"])
            assert record["flow_time"] == pytest.approx(sum(r["stepsize"] for r in records[:index]))
        key = jax.random.fold_in(key, 0)
    replay, _ = fisher_mirror_descent(
        loss_function=RandomLinearLoss(), x=jnp.full((2, 3), 1 / 3),
        n_steps=5, stepsize=0.5, stepsize_end=1.3,
        entropy_coefficient_end=0.2, entropy_schedule_power=6.0, key=initial_key,
    )
    np.testing.assert_array_equal(replay, final)


def test_zero_steps_does_not_evaluate_loss():
    def fail(*args, **kwargs):
        raise AssertionError("unexpected loss evaluation")

    x = jnp.array([[0.2, 0.8]])
    final, best, records = fisher_mirror_descent(
        loss_function=fail, x=x, n_steps=0, stepsize=1.0, trajectory_fn=lambda r: r,
    )
    np.testing.assert_allclose(final, x)
    np.testing.assert_array_equal(best, final)
    assert records == []


def test_softmax_underflow_does_not_destroy_logit_state():
    z = jnp.array([[100.0, -100.0]])
    assert float(jax.nn.softmax(z)[0, 1]) == 0.0
    # A later finite force can revive this probability because no log(softmax)
    # round trip is used between steps.
    updated = _fisher_mirror_step(z, jnp.array([[100., -100.]]), 1.0, 0.0)
    np.testing.assert_allclose(jax.nn.softmax(updated), [[0.5, 0.5]])


@pytest.mark.parametrize("kwargs", [
    {"n_steps": -1}, {"stepsize": 0}, {"entropy_coefficient_end": float("nan")},
    {"max_gradient_norm": -1}, {"design_mask": [True]},
    {"x": jnp.array([[0.1, -0.1], [0.5, 0.5]])},
])
def test_invalid_inputs_fail_before_oracle_call(kwargs):
    def fail(*args, **kw):
        raise AssertionError("unexpected loss evaluation")

    args = dict(loss_function=fail, x=jnp.full((2, 2), 0.5), n_steps=1, stepsize=1.)
    args.update(kwargs)
    with pytest.raises(ValueError):
        fisher_mirror_descent(**args)
