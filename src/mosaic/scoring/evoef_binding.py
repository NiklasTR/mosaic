"""EvoEF / EvoEF2 ``ComputeBinding`` scores (Huang et al.; local binary + ``library/`` layout)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import equinox as eqx

from .base import ScoreTerm
from .pdb_prepare import prepare_pdb_for_scoring

_TOTAL_RE = re.compile(r"Total\s+=\s+([-+0-9.eE]+)", re.MULTILINE)


def parse_binding_total_from_evoef_stdout(text: str) -> float:
    matches = _TOTAL_RE.findall(text)
    if not matches:
        raise ValueError(
            "EvoEF stdout did not contain a 'Total = ...' line; "
            "binding may be undefined (e.g. single chain) or the run failed."
        )
    return float(matches[-1])


def _resolve_binary(
    *,
    package_home: str | None,
    executable: str | None,
    default_exe_name: str,
) -> Path:
    if executable is not None and str(executable).strip():
        p = Path(str(executable)).expanduser()
        if p.is_file():
            return p.resolve()
    if package_home is not None and str(package_home).strip():
        p = Path(str(package_home)).expanduser() / default_exe_name
        if p.is_file():
            return p.resolve()
    w = shutil.which(default_exe_name)
    if w:
        return Path(w).resolve()
    raise FileNotFoundError(
        f"Could not find {default_exe_name}: set scoring.*.home to the package root "
        f"(containing library/) or scoring.*.executable to the binary path."
    )


class EvoEFBindingScore(ScoreTerm):
    """``EvoEF --command=ComputeBinding``; optional ``--split=part1,part2`` for multi-chain."""

    package_home: str | None
    executable: str | None
    split: str | None
    extra_argv: tuple[str, ...] = ()

    def __call__(
        self, *, structure_path: Path, **_kwargs
    ) -> tuple[float, dict[str, float | int]]:
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)
        exe = _resolve_binary(
            package_home=self.package_home,
            executable=self.executable,
            default_exe_name="EvoEF",
        )
        return _run_evoef_binding(
            exe=exe,
            structure_path=structure_path,
            split_opt_name="split",
            split_value=self.split,
            extra_argv=self.extra_argv,
        )


class EvoEF2BindingScore(ScoreTerm):
    """``EvoEF2 --command=ComputeBinding``; optional ``--split_chains=part1,part2``."""

    package_home: str | None
    executable: str | None
    split_chains: str | None
    extra_argv: tuple[str, ...] = ()

    def __call__(
        self, *, structure_path: Path, **_kwargs
    ) -> tuple[float, dict[str, float | int]]:
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)
        exe = _resolve_binary(
            package_home=self.package_home,
            executable=self.executable,
            default_exe_name="EvoEF2",
        )
        return _run_evoef_binding(
            exe=exe,
            structure_path=structure_path,
            split_opt_name="split_chains",
            split_value=self.split_chains,
            extra_argv=self.extra_argv,
        )


def _run_evoef_binding(
    *,
    exe: Path,
    structure_path: Path,
    split_opt_name: str,
    split_value: str | None,
    extra_argv: tuple[str, ...],
) -> tuple[float, dict[str, float | int]]:
    with tempfile.TemporaryDirectory(prefix="mosaic_evoef_") as raw_td:
        td = Path(raw_td)
        pdb_in = td / "complex.pdb"
        prepare_pdb_for_scoring(structure_path, pdb_in, protein_only=True)
        pdb_arg = str(pdb_in.resolve())
        cmd = [
            str(exe),
            "--command=ComputeBinding",
            f"--pdb={pdb_arg}",
        ]
        if split_value is not None and str(split_value).strip():
            cmd.append(f"--{split_opt_name}={split_value.strip()}")
        cmd.extend(extra_argv)
        proc = subprocess.run(
            cmd,
            cwd=str(td),
            check=False,
            capture_output=True,
            text=True,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        blob = out + err
        if proc.returncode != 0:
            raise RuntimeError(
                f"{exe.name} ComputeBinding failed (exit {proc.returncode}): {exe}\n"
                f"stderr:\n{err}\nstdout:\n{out}"
            )
        total = parse_binding_total_from_evoef_stdout(blob)
        return total, {
            "evoef_binding_total": total,
            "executable_resolved": str(exe),
        }
