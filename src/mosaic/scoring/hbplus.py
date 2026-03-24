"""HBPLUS hydrogen-bond count (McDonald et al.; user-supplied ``hbplus`` binary)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import equinox as eqx

from .base import ScoreTerm
from .pdb_prepare import prepare_pdb_for_scoring

_HBPLUS_COUNT_RE = re.compile(r"(\d+)\s+hydrogen bonds found\.", re.MULTILINE)
_HB2_DISTSUFFIX_RE = re.compile(r"(\d+\.\d+)\s+([A-Za-z]{2})\b")


def parse_hbonds_from_hbplus_stdout(text: str) -> int:
    matches = _HBPLUS_COUNT_RE.findall(text)
    if not matches:
        raise ValueError(
            "HBPLUS stdout did not contain 'N hydrogen bonds found.'; "
            "check the executable and input structure."
        )
    return int(matches[-1])


def _wkdir_for_hbplus(directory: Path) -> str:
    s = directory.resolve().as_posix()
    return s if s.endswith("/") else s + "/"


def resolve_hbplus_executable(
    executable: str, package_home: str | None
) -> str:
    """Return a path to the HBPLUS binary, or raise with setup hints."""
    exp = Path(executable).expanduser()
    if exp.is_file():
        return str(exp.resolve())
    if package_home is not None and str(package_home).strip():
        root = Path(str(package_home).strip()).expanduser()
        cand = root / "hbplus"
        if cand.is_file():
            return str(cand.resolve())
        if root.is_file():
            return str(root.resolve())
    w = shutil.which(executable)
    if w:
        return w
    raise FileNotFoundError(
        "HBPLUS binary not found. On clusters it is usually not on PATH: "
        "set scoring.hbplus.executable to the built binary path, or "
        "scoring.hbplus.home to the directory that contains ``hbplus`` after ``make``. "
        "Or disable with scoring.hbplus.enabled=false."
    )


def prepare_pdb_for_hbplus(structure_path: Path, dest_pdb: Path) -> None:
    """Read mmCIF or PDB with Gemmi and write a PDB suitable for HBPLUS (HEADER + optional CONECT)."""
    prepare_pdb_for_scoring(structure_path, dest_pdb, protein_only=False)


def _normalize_hbplus_chain(ch: str) -> str:
    c = ch.strip()
    if c in ("", "-"):
        return ""
    return c


def count_hbonds_inter_intra_chain_from_hb2(hb2_text: str) -> tuple[int, int]:
    """Split HBPLUS ``*.hb2`` (short format) H-bonds into inter- vs intra-chain counts.

    Uses donor / acceptor chain IDs (columns 0 and 14). Every listed bond is counted;
    the two-letter field after the D–A distance (``MM``, ``HH``, …) is only used to
    confirm the line looks like a data row (``HH`` can appear for incomplete residues,
    so it must not be treated as ``H`` = HETATM).

    Each data line has a 13-character donor field, one space, then a 13-character acceptor.
    """
    n_inter = 0
    n_intra = 0
    for line in hb2_text.splitlines():
        line = line.rstrip("\r\n")
        if len(line) < 40:
            continue
        if line.startswith(("HBPLUS", "(c)", "Citing", "<---")):
            continue
        if "Brookhaven Code" in line or "atom  resd" in line or "n    s   type" in line:
            continue
        if line[13] != " ":
            continue
        tail = line[27:]
        m = _HB2_DISTSUFFIX_RE.search(tail)
        if not m:
            continue
        dch = _normalize_hbplus_chain(line[0])
        ach = _normalize_hbplus_chain(line[14])
        if dch != ach:
            n_inter += 1
        else:
            n_intra += 1
    return n_inter, n_intra


class HBplusScore(ScoreTerm):
    """Run ``hbplus``; score is the **inter-chain** H-bond count from ``*.hb2``."""

    executable: str
    package_home: str | None = None
    extra_argv: tuple[str, ...] = ()

    def __call__(
        self, *, structure_path: Path, **_kwargs
    ) -> tuple[float, dict[str, float | int]]:
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)

        exe = resolve_hbplus_executable(self.executable, self.package_home)
        with tempfile.TemporaryDirectory(prefix="mosaic_hbplus_") as raw_td:
            td = Path(raw_td)
            wkdir = _wkdir_for_hbplus(td)
            pdb_in = td / "hbplus_input.pdb"
            prepare_pdb_for_hbplus(structure_path, pdb_in)
            pdb_s = str(pdb_in.resolve())
            cmd = [exe, "-wkdir", wkdir, pdb_s, pdb_s, *self.extra_argv]
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            out = proc.stdout or ""
            err = proc.stderr or ""
            if proc.returncode != 0:
                raise RuntimeError(
                    f"HBPLUS failed (exit {proc.returncode}): {exe!r}\n"
                    f"stderr:\n{err}\nstdout:\n{out}"
                )
            n_stdout = parse_hbonds_from_hbplus_stdout(out + err)
            hb2_path = td / f"{pdb_in.stem}.hb2"
            if not hb2_path.is_file():
                raise RuntimeError(
                    f"HBPLUS did not write expected {hb2_path.name!r} under the working "
                    f"directory; stdout/stderr:\n{out}\n{err}"
                )
            hb2_text = hb2_path.read_text(encoding="utf-8", errors="replace")
            n_inter, n_intra = count_hbonds_inter_intra_chain_from_hb2(hb2_text)
            return float(n_inter), {
                "hbonds": n_inter,
                "hbonds_inter_chain": n_inter,
                "hbonds_intra_chain": n_intra,
                "hbonds_hbplus_total": n_stdout,
                "executable_resolved": exe,
            }
