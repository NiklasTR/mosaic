"""Cysteine count (sequence) and disulfide-bond count (structure, Biotite geometry)."""

from __future__ import annotations

from pathlib import Path

import biotite.structure as struc
import equinox as eqx
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure

from .base import ScoreTerm


def count_cysteines_in_sequence(seq: str) -> int:
    return sum(1 for ch in seq if ch == "C")


def _residue_count_by_ca(amino: struc.AtomArray, chain_id: str) -> int:
    sub = amino[amino.chain_id == chain_id]
    ca = sub[sub.atom_name == "CA"]
    return int(len(ca))


def smallest_peptide_chain_id(atom_array: struc.AtomArray) -> str | None:
    """Chain id with the fewest amino-acid residues (CA count); tie-break lexicographic."""
    mask = struc.filter_amino_acids(atom_array)
    amino = atom_array[mask]
    if len(amino) == 0:
        return None
    chains = np.unique(amino.chain_id.astype(str))
    ranked: list[tuple[int, str, str]] = []
    for c in chains:
        n = _residue_count_by_ca(amino, c)
        ranked.append((n, c, c))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return str(ranked[0][2])


def detect_disulfide_bonds_biotite(
    structure: struc.AtomArray,
    *,
    distance: float = 2.05,
    distance_tol: float = 0.05,
    dihedral: float = 90.0,
    dihedral_tol: float = 10.0,
) -> int:
    """Count disulfide bonds using SG distance and Cβ–Sγ–Sγ–Cβ dihedral (Biotite gallery criteria)."""
    sulfide_mask = (structure.res_name == "CYS") & (structure.atom_name == "SG")
    sg_indices = np.where(sulfide_mask)[0]
    if len(sg_indices) < 2:
        return 0

    cell_list = struc.CellList(
        structure,
        cell_size=distance + distance_tol,
        selection=sulfide_mask,
    )
    disulfide_bonds: list[tuple[int, int]] = []
    for sulfide_i in sg_indices:
        partners = cell_list.get_atoms_in_cells(coord=structure.coord[sulfide_i])
        for sulfide_j in partners:
            if sulfide_j <= sulfide_i:
                continue
            sg1 = structure[sulfide_i]
            sg2 = structure[sulfide_j]
            cb1 = structure[
                (structure.chain_id == sg1.chain_id)
                & (structure.res_id == sg1.res_id)
                & (structure.ins_code == sg1.ins_code)
                & (structure.atom_name == "CB")
            ]
            cb2 = structure[
                (structure.chain_id == sg2.chain_id)
                & (structure.res_id == sg2.res_id)
                & (structure.ins_code == sg2.ins_code)
                & (structure.atom_name == "CB")
            ]
            if cb1.array_length() == 0 or cb2.array_length() == 0:
                continue
            bond_dist = struc.distance(sg1, sg2)
            bond_dihed = float(
                np.abs(np.rad2deg(struc.dihedral(cb1[0], sg1, sg2, cb2[0])))
            )
            if (
                bond_dist > distance - distance_tol
                and bond_dist < distance + distance_tol
                and bond_dihed > dihedral - dihedral_tol
                and bond_dihed < dihedral + dihedral_tol
            ):
                bond_tuple = (int(sulfide_i), int(sulfide_j))
                if bond_tuple not in disulfide_bonds:
                    disulfide_bonds.append(bond_tuple)
    return len(disulfide_bonds)


def load_atom_array_mmcif(path: Path, *, model: int = 1) -> struc.AtomArray:
    cif = CIFFile.read(str(path))
    return get_structure(cif, model=model)


class CysteineDisulfideScore(ScoreTerm):
    """Post-fold: cysteines in the designed sequence vs disulfides on the smallest peptide chain."""

    distance: float = 2.05
    distance_tol: float = 0.05
    dihedral: float = 90.0
    dihedral_tol: float = 10.0

    def __call__(
        self,
        *,
        structure_path: Path,
        binder_sequence: str,
        **kwargs,
    ) -> tuple[float, dict[str, float | int | str]]:
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)

        n_seq_cys = count_cysteines_in_sequence(binder_sequence)
        arr = load_atom_array_mmcif(structure_path)
        chain_id = smallest_peptide_chain_id(arr)
        if chain_id is None:
            return float(0), {
                "binder_cysteine_count": n_seq_cys,
                "binder_disulfide_bonds": 0,
                "binder_chain_id": "",
                "cysteine_disulfide_score": 0.0,
            }
        chain_s = str(chain_id)
        binder = arr[arr.chain_id.astype(str) == chain_s]
        n_ss = detect_disulfide_bonds_biotite(
            binder,
            distance=self.distance,
            distance_tol=self.distance_tol,
            dihedral=self.dihedral,
            dihedral_tol=self.dihedral_tol,
        )
        cys_in_structure = int(np.sum(binder.res_name == "CYS"))
        return float(n_ss), {
            "binder_cysteine_count": n_seq_cys,
            "binder_disulfide_bonds": n_ss,
            "binder_chain_id": chain_s,
            "binder_cys_residues_in_structure": cys_in_structure,
            "cysteine_disulfide_score": float(n_ss),
        }
