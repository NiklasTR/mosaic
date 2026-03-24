"""Backward-compatible wrapper; prefer ``python -m mosaic.cli.binder_design`` (after install).

Supports ``python packages/mosaic/scripts/run_binder_design.py`` from the repo root without
an editable install by prepending this package's ``src`` layout to ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from mosaic.cli.binder_design import main

if __name__ == "__main__":
    main()
