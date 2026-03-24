"""Tests for cysteine sequence count and Biotite disulfide detection."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # pathfinder repo root

import biotite.structure as struc
import numpy as np
import pytest

from mosaic.scoring.cysteine_disulfide import (
    CysteineDisulfideScore,
    count_cysteines_in_sequence,
    detect_disulfide_bonds_biotite,
    load_atom_array_mmcif,
    smallest_peptide_chain_id,
)


def test_count_cysteines_in_sequence() -> None:
    assert count_cysteines_in_sequence("") == 0
    assert count_cysteines_in_sequence("ACDEFG") == 1
    assert count_cysteines_in_sequence("CCC") == 3


def _minimal_disulfide_atom_array() -> struc.AtomArray:
    sg1 = np.array([0.0, 0.0, 0.0])
    sg2 = np.array([2.05, 0.0, 0.0])
    cb1 = sg1 + np.array([-1.2, 1.5, 0.0])
    cb2 = sg2 + np.array([1.2, 0.0, 1.5])
    atoms = []
    for coord, res_id, name in [
        (cb1, 1, "CB"),
        (sg1, 1, "SG"),
        (sg2, 2, "SG"),
        (cb2, 2, "CB"),
    ]:
        atoms.append(
            struc.Atom(
                coord,
                chain_id="A",
                res_id=res_id,
                res_name="CYS",
                atom_name=name,
                element="S" if name == "SG" else "C",
            )
        )
    return struc.array(atoms)


def test_detect_disulfide_bonds_biotite_one_pair() -> None:
    arr = _minimal_disulfide_atom_array()
    assert detect_disulfide_bonds_biotite(arr) == 1


def test_smallest_peptide_chain_id_on_boltz_cif() -> None:
    cif = (
        _REPO_ROOT
        / "logs/mosaic/binder_design/multiruns/2026-03-24_10-20-34/0/"
        / "boltz2_design_00001_d9cd383f00dc.cif"
    )
    if not cif.is_file():
        pytest.skip("fixture CIF not present")
    arr = load_atom_array_mmcif(cif)
    assert smallest_peptide_chain_id(arr) == "A"


def test_cysteine_disulfide_score_term() -> None:
    cif = (
        _REPO_ROOT
        / "logs/mosaic/binder_design/multiruns/2026-03-24_10-20-34/0/"
        / "boltz2_design_00001_d9cd383f00dc.cif"
    )
    if not cif.is_file():
        pytest.skip("fixture CIF not present")
    score, aux = CysteineDisulfideScore()(structure_path=cif, binder_sequence="ACDE")
    assert aux["binder_cysteine_count"] == 1
    assert "binder_disulfide_bonds" in aux
    assert float(score) == float(aux["binder_disulfide_bonds"])
