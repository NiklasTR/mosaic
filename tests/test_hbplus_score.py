"""Tests for HBPLUS-based structure scoring."""

import os
from pathlib import Path

import mosaic
import pytest

from mosaic.scoring.hbplus import (
    HBplusScore,
    count_hbonds_inter_intra_chain_from_hb2,
    parse_hbonds_from_hbplus_stdout,
    prepare_pdb_for_hbplus,
    resolve_hbplus_executable,
)


def test_parse_hbonds_from_hbplus_stdout_last_match() -> None:
    text = """Opened output file "x.hb2".
Checking for hydrogen bonds . . .
10 hydrogen bonds found.
"""
    assert parse_hbonds_from_hbplus_stdout(text) == 10


def test_parse_hbonds_from_hbplus_stdout_multiline() -> None:
    text = "5 hydrogen bonds found.\nother junk\n282 hydrogen bonds found.\n"
    assert parse_hbonds_from_hbplus_stdout(text) == 282


def test_parse_hbonds_from_hbplus_stdout_missing() -> None:
    with pytest.raises(ValueError):
        parse_hbonds_from_hbplus_stdout("no count here")


def test_resolve_hbplus_executable_by_path(tmp_path: Path) -> None:
    exe = tmp_path / "hbplus"
    exe.write_text("#!/bin/sh\necho ok\n")
    exe.chmod(0o755)
    assert resolve_hbplus_executable(str(exe), None) == str(exe.resolve())


def test_resolve_hbplus_executable_via_home(tmp_path: Path) -> None:
    exe = tmp_path / "hbplus"
    exe.write_text("#!/bin/sh\necho ok\n")
    exe.chmod(0o755)
    assert resolve_hbplus_executable("hbplus", str(tmp_path)) == str(exe.resolve())


def test_resolve_hbplus_executable_missing() -> None:
    with pytest.raises(FileNotFoundError, match="HBPLUS binary not found"):
        resolve_hbplus_executable("mosaic_nonexistent_hbplus_xyz", None)


def test_count_hbonds_inter_intra_from_hb2_three_lines() -> None:
    text = """
A0001-ALAN    A0001-ALAO    2.94 HH  -2 -1.00  -1.0 -1.00  -1.0  -1.0     1
B0001-ALAN    A0001-ALAO    0.51 HH  -2 -1.00  -1.0 -1.00  -1.0  -1.0     2
B0001-ALAN    B0001-ALAO    2.94 HH  -2 -1.00  -1.0 -1.00  -1.0  -1.0     3
"""
    inter, intra = count_hbonds_inter_intra_chain_from_hb2(text)
    assert inter == 1
    assert intra == 2


def test_count_hbonds_inter_intra_from_hb2_includes_het_cross_chain() -> None:
    text = """
A0001-ALAN    W0001-HOHO    2.80 HM  -1 -1.00  -1.0 -1.00  -1.0  -1.0     1
A0001-ALAN    B0001-ALAO    2.80 HH  -2 -1.00  -1.0 -1.00  -1.0  -1.0     2
"""
    inter, intra = count_hbonds_inter_intra_chain_from_hb2(text)
    assert inter == 2
    assert intra == 0


def test_prepare_pdb_adds_header_when_absent(tmp_path: Path) -> None:
    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId("1")
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.pos = gemmi.Position(0, 0, 0)
    res.add_atom(atom)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    raw = tmp_path / "raw.pdb"
    st.write_pdb(str(raw))
    assert not raw.read_text().lstrip().upper().startswith("HEADER")
    out = tmp_path / "for_hbplus.pdb"
    prepare_pdb_for_hbplus(raw, out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("HEADER")


def test_hbplus_score_end_to_end_packages_build(tmp_path: Path) -> None:
    """Uses ``packages/hbplus/hbplus`` when present (``make hbplus`` after unzip)."""
    pkgs = Path(mosaic.__file__).resolve().parents[3]
    hb_bin = pkgs / "hbplus" / "hbplus"
    if not hb_bin.is_file():
        pytest.skip("packages/hbplus/hbplus missing; unzip hbplus.zip under packages/ and make hbplus")

    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId("1")
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.pos = gemmi.Position(0, 0, 0)
    res.add_atom(atom)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    raw = tmp_path / "raw.pdb"
    st.write_pdb(str(raw))
    out = tmp_path / "for_hbplus.pdb"
    prepare_pdb_for_hbplus(raw, out)

    hb = HBplusScore(executable="hbplus", package_home=str(pkgs / "hbplus"))
    score, aux = hb(structure_path=out)
    assert score == aux["hbonds_inter_chain"]
    assert aux["hbonds"] == aux["hbonds_inter_chain"]
    assert aux["hbonds_hbplus_total"] >= aux["hbonds"] + aux["hbonds_intra_chain"]
    assert aux["executable_resolved"] == str(hb_bin.resolve())


def test_hbplus_packages_build_two_chains_inter_positive(tmp_path: Path) -> None:
    pkgs = Path(mosaic.__file__).resolve().parents[3]
    hb_bin = pkgs / "hbplus" / "hbplus"
    if not hb_bin.is_file():
        pytest.skip("packages/hbplus/hbplus missing; unzip hbplus.zip under packages/ and make hbplus")

    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(50, 50, 50, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    for cid, xoff in [("A", 0.0), ("B", 2.8)]:
        ch = gemmi.Chain(cid)
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId("1")
        n = gemmi.Atom()
        n.name = "N"
        n.pos = gemmi.Position(xoff, 0, 0)
        res.add_atom(n)
        o = gemmi.Atom()
        o.name = "O"
        o.pos = gemmi.Position(xoff + 2.9, 0.5, 0)
        res.add_atom(o)
        ch.add_residue(res)
        model.add_chain(ch)
    st.add_model(model)
    st.setup_entities()
    raw = tmp_path / "two.pdb"
    st.write_pdb(str(raw))
    out = tmp_path / "two_hbplus.pdb"
    prepare_pdb_for_hbplus(raw, out)

    hb = HBplusScore(executable="hbplus", package_home=str(pkgs / "hbplus"))
    score, aux = hb(structure_path=out)
    assert aux["hbonds_inter_chain"] >= 1
    assert score >= 1


@pytest.mark.skipif(
    not os.environ.get("MOSAIC_HBPLUS_EXE"),
    reason="Set MOSAIC_HBPLUS_EXE to an hbplus binary for integration test.",
)
def test_hbplus_score_on_boltz_cif_if_present() -> None:
    cif = os.environ.get("MOSAIC_HBPLUS_TEST_CIF")
    if not cif or not Path(cif).is_file():
        pytest.skip("Set MOSAIC_HBPLUS_TEST_CIF to a .cif path for integration test.")
    exe = os.environ["MOSAIC_HBPLUS_EXE"]
    hb = HBplusScore(executable=exe)
    score, aux = hb(structure_path=Path(cif))
    assert score >= 0
    assert aux["hbonds"] >= 0
