"""Cysteine and disulfide counts using Gemmi structure geometry."""
from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path

import gemmi

from .base import ScoreTerm


def count_cysteines_in_sequence(seq: str) -> int:
    return seq.count("C")


def load_structure_model(path: Path, *, model: int = 1) -> gemmi.Model:
    """Read PDB/mmCIF and select a model by one-based position.

    Keep the first conformer consistently for CA counting and SG/CB geometry.
    Alternative locations must not be counted as separate bonds.
    """
    structure = gemmi.read_structure(str(path))
    if not 1 <= model <= len(structure):
        raise ValueError(f"Model {model} is outside 1..{len(structure)}: {path}")
    selected = structure[model - 1].clone()
    selected.remove_alternative_conformations()
    return selected


def smallest_peptide_chain_id(model: gemmi.Model) -> str | None:
    """Fewest amino-acid CA residues; tie-break lexicographically by chain ID."""
    counts: dict[str, int] = {}
    for chain in model:
        n = sum(
            gemmi.find_tabulated_residue(res.name).is_amino_acid()
            and any(atom.name == "CA" for atom in res)
            for res in chain
        )
        if n:
            counts[chain.name] = counts.get(chain.name, 0) + n
    return min(counts, key=lambda name: (counts[name], name)) if counts else None


def detect_disulfide_bonds(
    chain: gemmi.Chain, *, distance: float = 2.05, distance_tol: float = 0.05,
    dihedral: float = 90.0, dihedral_tol: float = 10.0,
) -> int:
    """Count unique CYS pairs using SG distance and CB–SG–SG–CB geometry.

    Retains the previous strict distance/absolute-dihedral windows. Residues
    missing SG or CB do not contribute. Normalize alternative conformations.
    """
    model = gemmi.Model("1")
    model.add_chain(chain.clone())
    model.remove_alternative_conformations()
    chain = model[0]
    cysteines = []
    for res in chain:
        if res.name != "CYS":
            continue
        atoms = {atom.name: atom.pos for atom in res}
        if "SG" in atoms and "CB" in atoms:
            cysteines.append((atoms["CB"], atoms["SG"]))
    count = 0
    for (cb1, sg1), (cb2, sg2) in combinations(cysteines, 2):
        if not distance - distance_tol < sg1.dist(sg2) < distance + distance_tol:
            continue
        angle = abs(math.degrees(gemmi.calculate_dihedral(cb1, sg1, sg2, cb2)))
        if dihedral - dihedral_tol < angle < dihedral + dihedral_tol:
            count += 1
    return count


class CysteineDisulfideScore(ScoreTerm):
    """Post-fold: sequence cysteines and disulfides on the smallest peptide chain."""

    distance: float = 2.05
    distance_tol: float = 0.05
    dihedral: float = 90.0
    dihedral_tol: float = 10.0

    def __call__(self, *, structure_path: Path, binder_sequence: str, **kwargs):
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)
        model = load_structure_model(structure_path)
        chain_id = smallest_peptide_chain_id(model)
        n_ss = n_cys = 0
        if chain_id is not None:
            binder = gemmi.Chain(chain_id)
            for chain in model:
                if chain.name == chain_id:
                    for residue in chain:
                        binder.add_residue(residue.clone())
            n_ss = detect_disulfide_bonds(
                binder, distance=self.distance, distance_tol=self.distance_tol,
                dihedral=self.dihedral, dihedral_tol=self.dihedral_tol,
            )
            n_cys = sum(res.name == "CYS" for res in binder)
        return float(n_ss), {
            "binder_cysteine_count": count_cysteines_in_sequence(binder_sequence),
            "binder_disulfide_bonds": n_ss,
            "binder_chain_id": chain_id or "",
            "binder_cys_residues_in_structure": n_cys,
            "cysteine_disulfide_score": float(n_ss),
        }
