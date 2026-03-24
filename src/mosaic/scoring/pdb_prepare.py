"""Gemmi helpers: mmCIF/PDB → PDB with HEADER for external structure tools."""

from __future__ import annotations

from pathlib import Path

import gemmi


def _pdb_id_from_structure(st: gemmi.Structure) -> str:
    name = (st.name or "XXXX").strip().upper()
    return (name + "XXXX")[:4]


def header_line_for_pdb_scoring(pdb_id: str) -> str:
    pid = (pdb_id.upper() + "XXXX")[:4]
    line = list(" " * 80)
    hdr = "HEADER"
    line[0:6] = list(hdr.ljust(6))
    line[62:66] = list(pid.ljust(4))
    return "".join(line) + "\n"


def write_structure_as_pdb(st: gemmi.Structure, path: Path) -> None:
    opts = gemmi.PdbWriteOptions()
    opts.conect_records = True
    path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(path), opts)
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().upper().startswith("HEADER"):
        return
    path.write_text(header_line_for_pdb_scoring(_pdb_id_from_structure(st)) + text, encoding="utf-8")


def structure_peptide_polymers_only(st: gemmi.Structure) -> gemmi.Structure:
    """Drop waters, ligands, and non-peptide chains; keep mmCIF peptide polymers only."""
    out = gemmi.Structure()
    out.cell = st.cell
    out.spacegroup_hm = st.spacegroup_hm
    out.name = st.name
    for mi, model in enumerate(st):
        nm = gemmi.Model(mi)
        for chain in model:
            poly = chain.get_polymer()
            if poly is None:
                continue
            nc = gemmi.Chain(chain.name)
            for res in poly:
                nc.add_residue(res.clone())
            if len(nc) > 0:
                nm.add_chain(nc)
        if len(nm) > 0:
            out.add_model(nm)
    out.setup_entities()
    return out


def prepare_pdb_for_scoring(
    structure_path: Path,
    dest_pdb: Path,
    *,
    protein_only: bool = False,
) -> None:
    """Read mmCIF or PDB and write PDB with HEADER (and optional CONECT) for scoring binaries."""
    suffix = structure_path.suffix.lower()
    if suffix not in (".cif", ".mmcif", ".pdb", ".ent"):
        raise ValueError(
            f"Unsupported structure suffix {structure_path.suffix!r}; "
            "expected .cif, .mmcif, .pdb, or .ent."
        )
    st = gemmi.read_structure(str(structure_path))
    st.remove_waters()
    if protein_only:
        st = structure_peptide_polymers_only(st)
    write_structure_as_pdb(st, dest_pdb)
