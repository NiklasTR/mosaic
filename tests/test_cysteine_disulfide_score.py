"""Gemmi-only geometry tests; all fixtures are generated locally."""
import gemmi
import pytest

from mosaic.scoring.cysteine_disulfide import (
    CysteineDisulfideScore, count_cysteines_in_sequence,
    detect_disulfide_bonds, load_structure_model, smallest_peptide_chain_id,
)


def pair(distance=2.05, twisted=True, missing_cb=False):
    chain = gemmi.Chain("A")
    for index, coords in enumerate([
        {"CA": (-2, 1, 0), "CB": (-1.2, 1.5, 0), "SG": (0, 0, 0)},
        {"CA": (distance + 2, 0, 1),
         "CB": (distance + 1.2, 0, 1.5) if twisted else (distance + 1.2, 1.5, 0),
         "SG": (distance, 0, 0)},
    ]):
        res = gemmi.Residue()
        res.name = "CYS"
        # Insertion-code variants of the same residue number stay distinct.
        res.seqid = gemmi.SeqId(1, " " if index == 0 else "A")
        for name, xyz in coords.items():
            if missing_cb and index == 1 and name == "CB":
                continue
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element("S" if name == "SG" else "C")
            atom.pos = gemmi.Position(*xyz)
            res.add_atom(atom)
        chain.add_residue(res)
    return chain


def test_sequence():
    assert count_cysteines_in_sequence("") == 0
    assert count_cysteines_in_sequence("ACDC") == 2


@pytest.mark.parametrize("distance,twisted,missing_cb,expected", [
    (2.05, True, False, 1), (2.5, True, False, 0),
    (2.05, False, False, 0), (2.05, True, True, 0),
    (2.05 - 0.05, True, False, 0), (2.05 + 0.05, True, False, 0),
])
def test_geometry(distance, twisted, missing_cb, expected):
    assert detect_disulfide_bonds(pair(distance, twisted, missing_cb)) == expected


def test_alternates_do_not_duplicate_bonds():
    chain = pair()
    for res in chain:
        atoms = [atom.clone() for atom in res]
        for atom in res:
            atom.altloc = "A"
        for atom in atoms:
            atom.altloc = "B"
            res.add_atom(atom)
    assert detect_disulfide_bonds(chain) == 1
    assert len(chain[0]) == 6  # caller's structure remains intact


@pytest.mark.parametrize("extension", ["cif", "pdb"])
def test_structure_roundtrip_and_residue_count(tmp_path, extension):
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain_b = pair()
    chain_b.name = "B"
    model.add_chain(chain_b)
    model.add_chain(pair())
    st.add_model(model)
    st.setup_entities()
    path = tmp_path / f"pair.{extension}"
    if extension == "cif":
        st.make_mmcif_document().write_file(str(path))
    else:
        st.write_pdb(str(path))
    loaded = load_structure_model(path)
    assert smallest_peptide_chain_id(loaded) == "A"
    score, aux = CysteineDisulfideScore()(structure_path=path, binder_sequence="ACC")
    assert score == 1
    assert aux["binder_cys_residues_in_structure"] == 2  # residues, not atoms
    assert aux["binder_cysteine_count"] == 2
    with pytest.raises(ValueError, match="Model"):
        load_structure_model(path, model=2)


def test_empty_model():
    assert smallest_peptide_chain_id(gemmi.Model("1")) is None
    assert detect_disulfide_bonds(gemmi.Chain("A")) == 0
