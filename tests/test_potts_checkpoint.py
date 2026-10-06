"""The converted format loads in a JAX process without a Torch import."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mosaic.pottsmpnn import V6_CHECKPOINT_SHA256, V6_TOKENS, load_potts_mpnn


def archive(path, **changes):
    values = {"source_sha256": np.array(V6_CHECKPOINT_SHA256),
              "token_names": np.array(V6_TOKENS),
              "etab_out.weight": np.zeros((900, 128), np.float32)}
    values.update(changes)
    np.savez(path, **values)


def test_converted_loader_does_not_import_torch(tmp_path):
    path = tmp_path / "potts.npz"
    archive(path)
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Torch imported by converted loader')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from mosaic.pottsmpnn import load_potts_mpnn
model = load_potts_mpnn(sys.argv[1])
assert len(model.token_names) == 30
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True,
                   env={**os.environ, "JAX_PLATFORMS": "cpu"})


@pytest.mark.parametrize("changes", [
    {"source_sha256": np.array("wrong")},
    {"token_names": np.array(V6_TOKENS[::-1])},
    {"etab_out.weight": np.zeros((21, 128))},
])
def test_converted_loader_rejects_incompatible_metadata(tmp_path, changes):
    path = tmp_path / "potts.npz"
    archive(path, **changes)
    with pytest.raises(ValueError):
        load_potts_mpnn(path)


def test_converted_loader_rejects_invalid_noise(tmp_path):
    path = tmp_path / "potts.npz"
    archive(path)
    with pytest.raises(ValueError):
        load_potts_mpnn(path, backbone_noise=float("nan"))
