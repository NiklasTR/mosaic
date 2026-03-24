"""Tests for EvoEF/EvoEF2 binding score parsing and optional binary runs."""

import os
from pathlib import Path

import pytest

from mosaic.scoring.evoef_binding import (
    EvoEF2BindingScore,
    EvoEFBindingScore,
    parse_binding_total_from_evoef_stdout,
)


def test_parse_binding_total_last_line() -> None:
    text = """
Binding energy details between chain(s) A and chain(s) B:
----------------------------------------------------
Total                 =            -12.34
"""
    assert parse_binding_total_from_evoef_stdout(text) == pytest.approx(-12.34)


def test_parse_binding_total_multiple_pairs() -> None:
    text = """
Total                 =            1.00
junk
Total                 =            -5.50
"""
    assert parse_binding_total_from_evoef_stdout(text) == pytest.approx(-5.5)


def test_parse_binding_total_missing() -> None:
    with pytest.raises(ValueError):
        parse_binding_total_from_evoef_stdout("no total here")


@pytest.mark.skipif(
    not os.environ.get("MOSAIC_EVOEF_HOME"),
    reason="Set MOSAIC_EVOEF_HOME to packages/EvoEF for integration test.",
)
def test_evoef_binding_on_boltz_cif() -> None:
    cif = os.environ.get("MOSAIC_EVOEF_TEST_CIF")
    if not cif or not Path(cif).is_file():
        pytest.skip("Set MOSAIC_EVOEF_TEST_CIF to a two-chain .cif.")
    home = os.environ["MOSAIC_EVOEF_HOME"]
    sc = EvoEFBindingScore(package_home=home, executable=None, split=None)
    score, aux = sc(structure_path=Path(cif))
    assert isinstance(score, float)
    assert "executable_resolved" in aux


@pytest.mark.skipif(
    not os.environ.get("MOSAIC_EVOEF2_HOME"),
    reason="Set MOSAIC_EVOEF2_HOME to packages/EvoEF2 for integration test.",
)
def test_evoef2_binding_on_boltz_cif() -> None:
    cif = os.environ.get("MOSAIC_EVOEF2_TEST_CIF")
    if not cif or not Path(cif).is_file():
        pytest.skip("Set MOSAIC_EVOEF2_TEST_CIF to a two-chain .cif.")
    home = os.environ["MOSAIC_EVOEF2_HOME"]
    sc = EvoEF2BindingScore(package_home=home, executable=None, split_chains=None)
    score, aux = sc(structure_path=Path(cif))
    assert isinstance(score, float)
    assert "executable_resolved" in aux
