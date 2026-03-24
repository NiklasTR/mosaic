"""Non-differentiable structure scores (external binaries, file-based metrics)."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx


class ScoreTerm(eqx.Module):
    """Like ``LossTerm`` but for post-hoc structure evaluation (no autodiff, no ``key``)."""

    def __call__(
        self, *, structure_path: Path, **kwargs
    ) -> tuple[float, dict[str, float | int]]:
        raise NotImplementedError
