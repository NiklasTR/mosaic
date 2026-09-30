"""Sequence optimization dispatch, independent of model loading and folding."""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.optimizers import fisher_mirror_descent, simplex_APGM

_LOG = logging.getLogger(__name__)


def _numpy_trajectory_record(record):
    """Copy replay state and numeric loss diagnostics to host memory."""
    result = {k: np.asarray(v) for k, v in record.items() if k != "aux"}
    leaves, _ = jax.tree_util.tree_flatten_with_path(record["aux"])
    for path, value in leaves:
        array = np.asarray(value)
        if array.dtype.kind in "biufc":
            result["aux/" + jax.tree_util.keystr(path)] = array
    return result


def optimize_binder_sequence(
    *, loss, x, optimizer, key, design_mask=None, trajectory_path: Path | None = None,
):
    """Return the sequence used for hard decoding by the binder-design CLI.

    ``key`` is the per-design key. The legacy method retains its three folds,
    incumbent restarts, and log smoothing. Fisher starts from fold 811 and
    returns the terminal iterate of one uninterrupted RNG/update chain.
    """
    method = str(optimizer.get("method", "three_phase"))
    if method not in ("three_phase", "fisher"):
        raise ValueError(f"Unknown optimizer.method {method!r}; expected 'three_phase' or 'fisher'")
    sqrt_l = float(np.sqrt(x.shape[0]))
    max_norm = float(optimizer["max_gradient_norm"])
    if method == "three_phase":
        pssm = x
        for stage in (1, 2, 3):
            stepsize = float(optimizer[f"phase{stage}_stepsize_factor"]) * sqrt_l
            _LOG.info(
                "simplex_APGM phase%s: n_steps=%s stepsize=%s momentum=%s scale=%s",
                stage, optimizer[f"phase{stage}_n_steps"], stepsize,
                optimizer[f"phase{stage}_momentum"], optimizer[f"phase{stage}_scale"],
            )
            _, pssm = simplex_APGM(
                loss_function=loss,
                x=pssm if stage == 1 else jnp.log(pssm + 1e-5),
                n_steps=int(optimizer[f"phase{stage}_n_steps"]),
                stepsize=stepsize,
                momentum=float(optimizer[f"phase{stage}_momentum"]),
                scale=float(optimizer[f"phase{stage}_scale"]),
                logspace=stage != 1,
                max_gradient_norm=max_norm,
                key=jax.random.fold_in(key, 810 + stage),
            )
        return pssm

    cfg = optimizer["fisher"]
    h_start = cfg.get("stepsize_start")
    h_start = 0.2 * x.shape[1] * sqrt_l if h_start is None else float(h_start)
    h_end = cfg.get("stepsize_end")
    h_end = 0.7 * sqrt_l if h_end is None else float(h_end)
    if not np.isfinite(h_end) or h_end <= 0:
        raise ValueError("optimizer.fisher.stepsize_end must be finite and positive")
    lam_end = cfg.get("entropy_coefficient_end")
    lam_end = 0.4 / h_end if lam_end is None else float(lam_end)
    save_trajectory = bool(cfg.get("save_trajectory", False))
    if save_trajectory and trajectory_path is None:
        raise ValueError("trajectory_path is required when fisher.save_trajectory is true")
    _LOG.info(
        "fisher_mirror_descent: n_steps=%s stepsize=%s->%s entropy=%s->%s power=%s",
        cfg["n_steps"], h_start, h_end, cfg["entropy_coefficient_start"],
        lam_end, cfg["entropy_schedule_power"],
    )
    result = fisher_mirror_descent(
        loss_function=loss, x=x, n_steps=int(cfg["n_steps"]),
        stepsize=h_start, stepsize_end=h_end,
        entropy_coefficient=float(cfg["entropy_coefficient_start"]),
        entropy_coefficient_end=lam_end,
        entropy_schedule_power=float(cfg["entropy_schedule_power"]),
        key=jax.random.fold_in(key, 811), max_gradient_norm=max_norm,
        design_mask=design_mask,
        trajectory_fn=_numpy_trajectory_record if save_trajectory else None,
    )
    final_pssm = result[0]
    if save_trajectory:
        records = result[2]
        arrays = {name: np.stack([r[name] for r in records]) for name in records[0]} if records else {}
        arrays.update(
            initial_pssm=np.asarray(x), final_pssm=np.asarray(final_pssm),
            best_evaluated_pssm=np.asarray(result[1]),
            design_mask=np.ones(x.shape[0], dtype=bool) if design_mask is None else np.asarray(design_mask),
        )
        np.savez_compressed(trajectory_path, **arrays)
        _LOG.info("Saved Fisher trajectory to %s", trajectory_path)
    return final_pssm
