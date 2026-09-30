import logging

import equinox as eqx
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable
from mosaic.common import is_state_update, has_state_index, LossTerm, LinearCombination

import time

_LOG = logging.getLogger(__name__)
AbstractLoss = LossTerm | LinearCombination


def _print_iter(iter, aux, v):
    # first filter out anything that isn't a float or has number of dimensions > 0
    aux = eqx.filter(
        aux,
        lambda v: isinstance(v, float | str) or v.shape == (),
    )
    parts = [
        str(iter),
        f"loss: {v:0.2f}",
        *[
            f"{jax.tree_util.keystr(k, simple=True, separator='.')}:{v: 0.2f}"
            for (k, v) in jax.tree_util.tree_leaves_with_path(aux)
            if hasattr(v, "item")
            or isinstance(v, float)
            and (
                "state_index" not in jax.tree_util.keystr(k, simple=True, separator=".")
            )
        ],
    ]
    _LOG.info(" ".join(parts))


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
def _eval_loss_and_grad(
    loss_function: AbstractLoss, x, key, *, serial_evaluation = False, sample_loss = False
):
    """
    Evaluates the loss function and its gradient.

    Args:
    - loss_function: ...
    - x: soft sequence (N x 20 array with each row in the simplex)
    - key: jax random key
    - serial_evaluation: if True, evaluate each loss function in the list sequentially, to save memory
    - sample_loss: if True *and* loss is a LinearCombination, randomly sample one of the loss functions to evaluate with probability proportional to its weight.
    
    Returns:
    - ((value, aux), g): value of the loss function and auxiliary information, and gradient of the loss with respect to x

    """
    assert not (serial_evaluation and sample_loss), "serial_evaluation and sample_loss cannot both be True"

    if sample_loss:
        assert isinstance(loss_function, LinearCombination), "sample_loss can only be used with LinearCombination loss functions"
        w_total = loss_function.weights.sum()
        idx = jax.random.choice(key, len(loss_function.l), p=loss_function.weights / w_total)
        key = jax.random.fold_in(key, 0)
        return _eval_loss_and_grad(loss_function.l[idx], x, key)


    if serial_evaluation:
        assert isinstance(loss_function, LinearCombination), "serial_evaluation can only be used with LinearCombination loss functions"
        results = [
            (w, _eval_loss_and_grad(l, x, jax.random.fold_in(key, idx)))
            for (idx, (w, l)) in enumerate(zip(loss_function.weights, loss_function.l))
        ]
        v = sum(w * r[0][0] for (w, r) in results)
        aux = [r[0][1] for (w, r) in results]
        g = sum(w * r[1] for (w, r) in results)
        return (v, aux), g
       
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan = 1000000.0), aux), jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))


# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


# this function is a mess, but it's used to update stateful loss functions. see comments in mosaic/common.py
def update_states(aux, loss):
    state_index_to_update = dict(
        [
            (int(x[0].id), x[1])
            for x in jax.tree.leaves(aux, is_leaf=is_state_update)
            if is_state_update(x)
        ]
    )

    def get_modules_to_update(loss):
        return tuple(
            [
                x
                for x in jax.tree.leaves(loss, is_leaf=has_state_index)
                if has_state_index(x)
            ]
        )

    def replace_fn(module):
        return module.update_state(state_index_to_update[int(module.state_index.id)])

    return eqx.tree_at(get_modules_to_update, loss, replace_fn=replace_fn)


# def _proposal(sequence, g, temp, alphabet_size: int = 20):
#     input = jax.nn.one_hot(sequence, alphabet_size)
#     g_i_x_i = (g * input).sum(-1, keepdims=True)
#     logits = -((input * g).sum() - g_i_x_i + g) / temp
#     return jax.nn.softmax(logits), jax.nn.log_softmax(logits)

# rewrite in numpy to use float64
from scipy.special import softmax, log_softmax 
def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)


def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    alphabet_size: int = 20,
    key: None = None,
    detailed_balance: bool = False,
    fix_loss_key: bool = True,
    serial_evaluation: bool = False,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, alphabet_size), key=key_model, serial_evaluation=serial_evaluation
    )
    for iter in range(steps):
        start_time = time.time()
        ### generate a proposal

        for i in range(50):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            path_length = jax.random.randint(
                key=jax.random.key(np.random.randint(10000)),
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
            for _ in range(path_length):
                p, log_p = _proposal(proposal, g_0, proposal_temp, alphabet_size=alphabet_size)
                mut_idx = jax.random.choice(
                    key=key,
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(),
                )
                key = jax.random.fold_in(key, 0)
                position, AA = np.unravel_index(mut_idx, p.shape)
                log_q_forward += log_p[position, AA]
                mutations += [(position, AA)]
                proposal = proposal.at[position].set(AA)
            # check if proposal is same as current sequence
            if np.all(proposal == sequence):
                _LOG.info("\t %s: proposal is the same as current sequence, skipping.", i)
                #_print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
                #continue
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa) in mutations])
        _LOG.info("Proposed mutations: %s", muts)
        
        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss, jax.nn.one_hot(proposal, alphabet_size), key=key_model if fix_loss_key else key, serial_evaluation=serial_evaluation
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(prop_backward, g_1, proposal_temp, alphabet_size=alphabet_size)
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        _LOG.info(
            "iter: %s, accept %s %s %s %s %s",
            iter,
            f"{np.exp(log_acceptance_probability): 0.3f}",
            f"{v_0: 0.3f}",
            f"{v_1: 0.3f}",
            f"{log_q_forward: 0.3f}",
            f"{log_q_backward: 0.3f}",
        )
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1
        
        _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
        

        key = jax.random.fold_in(key, 0)

    return sequence


def projection_simplex(V, z=1):
    """
    From https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    """
    V = np.array(V, dtype=np.float64)
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


@jax.jit
def _fisher_mirror_step(logits, gradient, stepsize, entropy_coefficient):
    """Euler in centered logits; exponentiated gradient in probabilities."""
    updated = (1.0 + stepsize * entropy_coefficient) * logits - stepsize * gradient
    return updated - updated.mean(axis=-1, keepdims=True)


def fisher_mirror_descent(
    *,
    loss_function,
    x,
    n_steps: int,
    stepsize: float,
    stepsize_end: float | None = None,
    entropy_coefficient: float = 0.0,
    entropy_coefficient_end: float | None = None,
    entropy_schedule_power: float = 1.0,
    key=None,
    max_gradient_norm: float | None = None,
    design_mask=None,
    initial_probability_floor: float = 1e-8,
    trajectory_fn: Callable | None = None,
):
    """One uninterrupted stochastic Fisher--Rao/mirror flow on soft sequences.

    The mean-field equation is dP/dt = -R_P(G - lambda * log(P)), where
    R_P(f) = P * (f - sum(P * f, axis=-1)). Each iteration evaluates the
    existing probability-gradient oracle at P = softmax(Z), clips its global
    norm, and takes Z <- center((1 + h * lambda) * Z - h * G). There is no
    momentum, simplex projection, or restart from an incumbent.

    ``x`` is an N x K probability matrix. Only designable rows are smoothed
    once at initialization if entries fall below ``initial_probability_floor``;
    fixed rows are returned and passed to the oracle exactly as supplied.
    Zeros caused by softmax underflow do not overwrite the stored logits.

    For n_steps > 1, u = step / (n_steps - 1) and S = u**2 * (3 - 2*u).
    The stepsize interpolates its endpoints using S; the entropy coefficient
    uses S**entropy_schedule_power. A one-step run uses the starting values.
    Positive entropy coefficients encourage concentration. Endpoints default
    to their starting values, giving a constant schedule.

    Returns (final_pssm, best_evaluated_pssm). The final post-update PSSM has
    not been evaluated; there are exactly n_steps oracle calls. Best tracking
    is observational and pairs each value with its actual pre-update input.
    With zero steps both returns are the initialized PSSM.

    If supplied, trajectory_fn(record) runs once per update and its return
    values are collected as a third return value. Records contain pre- and
    post-update logits/PSSMs, the key used, processed gradient, schedules,
    pre-update loss/diagnostics and auxiliary loss outputs. This differs from
    simplex_APGM's two-argument callback. No full trajectory is retained when
    trajectory_fn is None. Loss-module state is not updated between steps.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    initial = np.asarray(x, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[0] == 0 or initial.shape[1] < 2:
        raise ValueError("x must be a nonempty N x K probability matrix with K >= 2")
    if (not np.isfinite(initial).all() or (initial < 0).any()
            or not np.allclose(initial.sum(-1), 1.0, atol=1e-6, rtol=1e-5)):
        raise ValueError("x must have finite nonnegative entries and rows summing to one")
    if design_mask is None:
        mask = np.ones(initial.shape[0], dtype=bool)
    else:
        mask = np.asarray(design_mask)
        if mask.shape != (initial.shape[0],) or mask.dtype != np.bool_:
            raise ValueError("design_mask must be a boolean vector with one entry per row")

    stepsize_end = stepsize if stepsize_end is None else stepsize_end
    entropy_coefficient_end = (
        entropy_coefficient if entropy_coefficient_end is None else entropy_coefficient_end
    )
    max_gradient_norm = np.sqrt(initial.shape[0]) if max_gradient_norm is None else max_gradient_norm
    for name, value in (
        ("stepsize", stepsize), ("stepsize_end", stepsize_end),
        ("max_gradient_norm", max_gradient_norm),
        ("entropy_schedule_power", entropy_schedule_power),
        ("initial_probability_floor", initial_probability_floor),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if initial_probability_floor >= 1.0 / initial.shape[1]:
        raise ValueError("initial_probability_floor must be less than 1 / K")
    for name, value in (
        ("entropy_coefficient", entropy_coefficient),
        ("entropy_coefficient_end", entropy_coefficient_end),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    fixed = jnp.asarray(initial)
    active = jnp.asarray(mask[:, None])
    initial_logits = jnp.log(jnp.maximum(fixed, initial_probability_floor))
    logits = jnp.where(active, initial_logits - initial_logits.mean(-1, keepdims=True), 0.0)

    def probabilities(z):
        return jnp.where(active, jax.nn.softmax(z, axis=-1), fixed)

    best_x = probabilities(logits)
    best_val = np.inf
    trajectory = []
    flow_time = 0.0
    for step in range(n_steps):
        start_time = time.time()
        u = step / (n_steps - 1) if n_steps > 1 else 0.0
        smooth = u * u * (3.0 - 2.0 * u)
        h = float(stepsize + (stepsize_end - stepsize) * smooth)
        lam = float(entropy_coefficient + (entropy_coefficient_end - entropy_coefficient)
                    * smooth ** entropy_schedule_power)
        pssm = probabilities(logits)
        (value, loss_aux), gradient = _eval_loss_and_grad(loss_function, pssm, key)
        gradient = jnp.where(active, gradient, 0.0)
        # Accumulate in float64 on the host, avoiding overflow when the oracle
        # has replaced an infinity with a large finite float32 value.
        gradient_norm = float(np.linalg.norm(np.asarray(gradient, dtype=np.float64)))
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("Fisher oracle returned a non-finite gradient")
        clip_factor = min(1.0, float(max_gradient_norm) / gradient_norm) if gradient_norm else 1.0
        gradient = gradient * clip_factor

        is_best = bool(np.isfinite(value) and value < best_val)
        if is_best:
            best_val = float(value)
            best_x = pssm

        next_logits = jnp.where(active, _fisher_mirror_step(logits, gradient, h, lam), 0.0)
        if not bool(jnp.isfinite(next_logits).all()):
            raise FloatingPointError("Fisher logits overflowed; reduce stepsize or entropy coefficient")
        next_pssm = probabilities(next_logits)
        # xlogy defines 0 * log(0) as zero, including fixed one-hot rows.
        entropy = -jax.scipy.special.xlogy(pssm, pssm).sum(-1)
        active_count = max(int(mask.sum()), 1)
        mean_entropy = (entropy * jnp.asarray(mask)).sum() / active_count
        mean_max = (pssm.max(-1) * jnp.asarray(mask)).sum() / active_count
        affinity = jnp.clip((jnp.sqrt(pssm) * jnp.sqrt(next_pssm)).sum(-1), 0.0, 1.0)
        fisher_distance = jnp.linalg.norm(2.0 * jnp.arccos(affinity) * jnp.asarray(mask))
        diagnostics = {
            "loss": value, "entropy": mean_entropy, "mean_max_probability": mean_max,
            "stepsize": h, "entropy_coefficient": lam,
            "gradient_norm": gradient_norm, "clip_factor": clip_factor,
            "fisher_distance": fisher_distance, "time": time.time() - start_time,
        }
        if trajectory_fn is not None:
            record = {
                **diagnostics, "step": step, "flow_time": flow_time,
                "key": jax.random.key_data(key), "key_impl": str(jax.random.key_impl(key)),
                "logits": logits, "pssm": pssm, "gradient": gradient,
                "next_logits": next_logits, "next_pssm": next_pssm,
                "is_best": is_best, "best_loss": best_val, "aux": loss_aux,
            }
            trajectory.append(trajectory_fn(record))
        _print_iter(step, {**diagnostics, "": loss_aux}, value)
        logits = next_logits
        flow_time += h
        key = jax.random.fold_in(key, 0)

    result = (probabilities(logits), best_x)
    return (*result, trajectory) if trajectory_fn is not None else result


def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key=None,
    max_gradient_norm: float | None = None,
    update_loss_state: bool = False,
    scale=1.0,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
    logspace: bool = False,
    serial_evaluation: bool = False,
    sample_loss: bool = False,
):
    """
    Accelerated projected gradient descent on the simplex.

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - stepsize: step size for gradient descent
    - momentum: momentum factor
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - update_loss_state: whether to update the loss function state
    - scale: proximal scaling factor for L2 regularization (or entropic regularization if logspace=True), set to > 1.0 to encourage sparsity
    - trajectory_fn: function to compute trajectory information, takes (aux, x) and returns any value.
    - logspace: whether to optimize in log space, which corresponds to a bregman proximal algorithm.

    returns:
    - x: final soft sequence after optimization
    - best_x: best soft sequence found during optimization
    - trajectory: list of trajectory information if `trajectory_fn` is provided, otherwise nothing.
    """

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[0])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

    x_prev = x

    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(
            x=v if not logspace else jax.nn.softmax(v),
            loss_function=loss_function,
            key=key,
            serial_evaluation=serial_evaluation,
            sample_loss=sample_loss,
        )

        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale * (v - stepsize * g))

        x_prev = x
        x = x_new

        if value < best_val and not np.isnan(value):
            best_val = value
            best_x = (
                x  # this isn't exactly right, because we evaluated loss at v, not x.
            )

        average_nnz = (
            (x > 0.01).sum(-1).mean()
            if not logspace
            else (jax.nn.softmax(x) > 0.01).sum(-1).mean()
        )

        # add loss and NNZ to aux
        if update_loss_state:
            loss_function = update_states(aux, loss_function)

        aux = {"loss": value, "nnz": average_nnz, "time": (time.time()-start_time), "": aux}
        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        _print_iter(
            _iter,
            eqx.filter(
                aux,
                lambda v: isinstance(v, float) or v.shape == (),
            ),
            value,
        )

    if logspace:
        x = jax.nn.softmax(x)
        best_x = jax.nn.softmax(best_x)

    if trajectory_fn is None:
        return x, best_x
    else:
        return x, best_x, trajectory
