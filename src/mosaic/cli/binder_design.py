"""Hydra entrypoint: optimize a short soft binder with Boltz-2 (joltz) against a target sequence.

Run (from ``packages/mosaic`` with mosaic installed, e.g. ``uv sync``):

  python -m mosaic.cli.binder_design launcher=local

Multirun (e.g. sweep seeds; with Submitit launcher each combo is its own job):

  python -m mosaic.cli.binder_design --multirun seed=0,1,2 launcher=aithyra-1gpu

JAX / CUDA / Boltz-2 are imported only inside ``main()`` so the Submitit driver on a
GPU-less login node does not call ``cuInit`` during module import.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import hydra
import mosaic
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(mosaic.__file__).resolve().parent.parent.parent / "configs"
_LOG = logging.getLogger(__name__)


def _pathfinder_packages_dir() -> Path:
    """``pathfinder/packages`` (sibling of ``packages/mosaic``) from installed ``mosaic`` layout."""
    return Path(mosaic.__file__).resolve().parents[3]


def _resolve_scoring_home(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((_pathfinder_packages_dir() / p).resolve())


@hydra.main(
    version_base="1.3",
    config_path=str(_CONFIG_DIR),
    config_name="binder_design",
)
def main(cfg: DictConfig) -> None:
    import hashlib
    import json
    import time

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mosaic.binder_scaffold import resolve_binder_sequence
    from mosaic.cli import binder_geometry as bg
    from mosaic.common import TOKENS
    from mosaic.losses import structure_prediction as sp
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys, NoCysScaffoldBinder, ScaffoldBinderSequence
    from mosaic.models.boltz2 import Boltz2
    from mosaic.optimizers import simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain

    def pssm20_to_sequence(pssm: jnp.ndarray) -> str:
        idx = np.asarray(jnp.argmax(pssm, axis=-1), dtype=np.int64)
        return "".join(TOKENS[int(i)] for i in idx)

    OmegaConf.resolve(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    _hc = HydraConfig.get()
    _LOG.info(
        "Starting binder_design job: output_dir=%s hydra.job.num=%s",
        out.resolve(),
        getattr(_hc.job, "num", None),
    )

    key = jax.random.key(int(cfg.seed))
    bs_cfg = cfg.get("binder_sequence")
    seq_str, design_mask_np = resolve_binder_sequence(
        int(cfg.binder_length) if cfg.get("binder_length") is not None else None,
        str(bs_cfg).strip() if bs_cfg is not None and str(bs_cfg).strip() else None,
    )
    binder_length = len(seq_str)
    design_mask_jnp = jnp.array(design_mask_np, dtype=jnp.bool_)
    all_designable = bool(np.all(design_mask_np))
    target_sequence = str(cfg.target_sequence)
    target_stem = hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()
    msa_cache = cfg.boltz2.get("msa_cache_dir")
    preprocess_out_dir = None
    if msa_cache is not None and str(msa_cache).strip():
        preprocess_out_dir = Path(str(msa_cache)).expanduser().resolve() / target_stem
    t_msa_raw = cfg.boltz2.get("target_msa_path")
    target_msa_file: Path | None = None
    if t_msa_raw is not None and str(t_msa_raw).strip():
        target_msa_file = Path(str(t_msa_raw)).expanduser().resolve()
        if not target_msa_file.is_file():
            raise FileNotFoundError(
                f"boltz2.target_msa_path is not a file: {target_msa_file}"
            )
        suf = target_msa_file.suffix.lower()
        if suf not in (".a3m", ".csv"):
            _LOG.warning(
                "boltz2.target_msa_path suffix %r is not .a3m or .csv; Boltz may reject it.",
                target_msa_file.suffix,
            )
    cyclic_b = bool(cfg.boltz2.cyclic_binder)
    target_msa = bool(cfg.boltz2.target_use_msa)

    lw_early = cfg.loss
    disulfide_pairs = int(lw_early.disulfide_pairs)
    if disulfide_pairs not in (0, 1, 2, 3):
        raise ValueError(
            f"loss.disulfide_pairs must be 0, 1, 2, or 3, got {disulfide_pairs}"
        )
    fixed_scaffold_chars = [seq_str[i] for i in range(binder_length) if not design_mask_np[i]]
    if disulfide_pairs == 0 and "C" in fixed_scaffold_chars:
        raise ValueError(
            "Fixed scaffold cysteines require loss.disulfide_pairs >= 1 (NoCys removes C from "
            "the design alphabet); use literal C in binder_sequence and disulfide_pairs > 0."
        )

    def target_chain_design() -> TargetChain:
        if target_msa_file is not None:
            return TargetChain(
                sequence=target_sequence,
                use_msa=True,
                msa_path=str(target_msa_file),
            )
        return TargetChain(
            sequence=target_sequence,
            use_msa=target_msa,
            msa_path=None,
        )

    def target_chain_with_msa() -> TargetChain:
        if target_msa_file is not None:
            return TargetChain(
                sequence=target_sequence,
                use_msa=True,
                msa_path=str(target_msa_file),
            )
        if target_msa:
            return TargetChain(sequence=target_sequence, use_msa=True, msa_path=None)
        return TargetChain(sequence=target_sequence, use_msa=False, msa_path=None)

    _LOG.info(
        "Problem setup: seed=%s binder_length=%s target_length=%s target_preprocess_stem=%s "
        "boltz2.msa_cache_dir=%s boltz2.target_msa_path=%s boltz2.cyclic_binder=%s "
        "boltz2.target_use_msa=%s boltz2.design.recycling_steps=%s boltz2.design.sampling_steps=%s "
        "boltz2.design.num_samples=%s ranking.enabled=%s boltz2_fold.enabled=%s boltz2_fold.cyclic_binder=%s "
        "runtime.max_seconds=%s loss.disulfide_pairs=%s binder_scaffold=%s",
        int(cfg.seed),
        binder_length,
        len(target_sequence),
        target_stem,
        str(preprocess_out_dir) if preprocess_out_dir else None,
        str(target_msa_file) if target_msa_file else None,
        cyclic_b,
        target_msa,
        int(cfg.boltz2.design.recycling_steps),
        int(cfg.boltz2.design.sampling_steps),
        int(cfg.boltz2.design.num_samples),
        bool(cfg.ranking.enabled),
        bool(cfg.boltz2_fold.enabled),
        bool(cfg.boltz2_fold.cyclic_binder),
        cfg.runtime.max_seconds,
        disulfide_pairs,
        seq_str if not all_designable else "all_X",
    )

    rt = cfg.runtime
    max_sec_raw = rt.max_seconds
    if max_sec_raw is None:
        budget_seconds: float | None = None
    else:
        budget_seconds = float(max_sec_raw)
        if budget_seconds <= 0:
            budget_seconds = None
    loop_until_wall = budget_seconds is not None
    _LOG.info(
        "runtime.max_seconds=%s (%s)",
        budget_seconds,
        "loop until wall" if loop_until_wall else "single design",
    )

    b2_ck = cfg.boltz2.checkpoint
    _LOG.info("Loading Boltz-2 for design (may JIT / download checkpoint on first use).")
    model = (
        Boltz2(Path(str(b2_ck)).expanduser().resolve())
        if b2_ck
        else Boltz2()
    )
    _LOG.info("Building binder+target features (Boltz-2 YAML).")
    bf_kw: dict = {
        "chains": [target_chain_design()],
        "cyclic_binder": cyclic_b,
        "preprocess_out_dir": preprocess_out_dir,
        "input_stem": target_stem,
    }
    if bs_cfg is not None and str(bs_cfg).strip():
        bf_kw["binder_sequence"] = str(bs_cfg).strip()
    else:
        bf_kw["binder_length"] = binder_length
    features, _ = model.binder_features(**bf_kw)
    scaffold_20 = jnp.array(features["res_type"][0, :binder_length, 2:22], dtype=jnp.float32)
    scaffold_aa_idx = jnp.array(
        [TOKENS.index(seq_str[i]) for i in range(binder_length)], dtype=jnp.int32
    )
    dm_float = design_mask_jnp.astype(jnp.float32)

    lw = cfg.loss
    parts: list = []

    def add_weighted(w, term) -> None:
        wf = float(w)
        if wf != 0.0:
            parts.append(wf * term)

    add_weighted(lw.binder_target_contact, sp.BinderTargetContact())
    add_weighted(lw.within_binder_contact, sp.WithinBinderContact())
    if float(lw.inverse_folding_mpnn) != 0.0:
        mpnn = load_mpnn_sol(float(cfg.boltz2.design.mpnn_backbone_noise))
        mpnn_bias = jnp.zeros((binder_length, 20))
        if disulfide_pairs == 0:
            cix = TOKENS.index("C")
            for i in range(binder_length):
                if design_mask_np[i]:
                    mpnn_bias = mpnn_bias.at[i, cix].set(-1e6)
        mpnn_dm = None if all_designable else dm_float
        mpnn_scaffold = None if all_designable else scaffold_aa_idx
        add_weighted(
            lw.inverse_folding_mpnn,
            InverseFoldingSequenceRecovery(
                mpnn,
                temp=jnp.array(float(lw.mpnn_temp)),
                bias=mpnn_bias,
                design_mask=mpnn_dm,
                scaffold_aa_idx=mpnn_scaffold,
            ),
        )
    add_weighted(lw.target_binder_pae, sp.TargetBinderPAE())
    add_weighted(lw.binder_target_pae, sp.BinderTargetPAE())
    add_weighted(lw.iptm, sp.IPTMLoss())
    add_weighted(lw.within_binder_pae, sp.WithinBinderPAE())
    add_weighted(lw.ptm_energy, sp.pTMEnergy())
    add_weighted(lw.plddt, sp.PLDDTLoss())

    if disulfide_pairs >= 1:
        w_cys = float(lw.cysteine_count_weight)
        if w_cys != 0.0:
            add_weighted(
                w_cys,
                sp.ExpectedCysteineCountLoss(
                    target_expected_cys=float(2 * disulfide_pairs),
                ),
            )

    for w_g, term in bg.geometry_terms_from_cfg(cfg.get("geometry"), binder_length):
        add_weighted(w_g, term)

    if not parts:
        raise ValueError(
            "At least one loss.* weight must be non-zero in binder_design config."
        )
    combo = parts[0]
    for p in parts[1:]:
        combo = combo + p

    _LOG.info(
        "Building multisample Boltz-2 loss (%s terms)%s, num_samples=%s",
        len(parts),
        " + NoCys" if disulfide_pairs == 0 else " (full 20 AA, cysteine allowed)",
        int(cfg.boltz2.design.num_samples),
    )
    design_ms = model.build_multisample_loss(
        loss=combo,
        features=features,
        recycling_steps=int(cfg.boltz2.design.recycling_steps),
        num_samples=int(cfg.boltz2.design.num_samples),
        sampling_steps=int(cfg.boltz2.design.sampling_steps),
        binder_design_mask=design_mask_jnp,
    )
    if disulfide_pairs == 0:
        loss_inner = (
            NoCysScaffoldBinder(design_ms, scaffold_20, design_mask_jnp)
            if not all_designable
            else NoCys(design_ms)
        )
    else:
        loss_inner = (
            ScaffoldBinderSequence(design_ms, scaffold_20, design_mask_jnp, False)
            if not all_designable
            else design_ms
        )
    loss = loss_inner

    opt = cfg.optimizer
    sqrt_l = float(np.sqrt(binder_length))
    mgn = float(opt.max_gradient_norm)

    if bool(cfg.ranking.enabled):

        @eqx.filter_jit
        def _eval_ranking(loss_fn, one_hot, k):
            return loss_fn(one_hot, key=k)
    else:
        _eval_ranking = None

    fold_ck = cfg.boltz2_fold.checkpoint
    if not fold_ck:
        fold_ck = cfg.boltz2.checkpoint
    if fold_ck == b2_ck:
        b2_for_fold = model
    else:
        b2_for_fold = (
            Boltz2(Path(str(fold_ck)).expanduser().resolve())
            if fold_ck
            else Boltz2()
        )
    cyclic_fold = bool(cfg.boltz2_fold.cyclic_binder)
    do_fold = bool(cfg.boltz2_fold.enabled)
    _sc = cfg.scoring
    if bool(_sc.hbplus.enabled) and not do_fold:
        _LOG.warning(
            "scoring.hbplus.enabled is true but boltz2_fold.enabled is false; HBPLUS will not run."
        )
    if (bool(_sc.evoef.enabled) or bool(_sc.evoef2.enabled)) and not do_fold:
        _LOG.warning(
            "scoring.evoef / scoring.evoef2 enabled but boltz2_fold.enabled is false; "
            "EvoEF scores will not run."
        )
    if bool(_sc.get("cysteine_disulfide", {}).get("enabled", False)) and not do_fold:
        _LOG.warning(
            "scoring.cysteine_disulfide.enabled is true but boltz2_fold.enabled is false; "
            "cysteine/disulfide scores will not run."
        )

    def _cfg_opt_str(node, key: str) -> str | None:
        raw = node.get(key)
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None

    t_loop = time.time()
    designs_path = out / "designs_incremental.fa"
    seq_path = out / "designed_sequence.txt"
    pssm_path = out / "pssm.npy"
    cfg_path = out / "resolved_config.yaml"
    n_done = 0
    seq = ""
    pssm = jnp.zeros((binder_length, 20))
    last_ranking_loss: float | None = None

    while True:
        if loop_until_wall and (time.time() - t_loop) >= budget_seconds:
            break
        if not loop_until_wall and n_done > 0:
            break

        ik = jax.random.fold_in(key, n_done + 7_000_000)
        k1 = jax.random.fold_in(ik, 801)
        aa_dim = 19 if disulfide_pairs == 0 else 20
        u = jax.random.uniform(
            k1, shape=(binder_length, aa_dim), minval=0.25, maxval=0.75
        )
        g0 = jax.random.gumbel(
            jax.random.fold_in(ik, 802), shape=(binder_length, aa_dim)
        )
        x0 = jax.nn.softmax(u * g0)
        if not all_designable:
            if disulfide_pairs == 0:
                cix = TOKENS.index("C")
                idx19 = jnp.array([j for j in range(20) if j != cix], dtype=jnp.int32)
                s19_fix = scaffold_20[:, idx19]
                x0 = jnp.where(
                    design_mask_jnp[:, None],
                    x0,
                    jax.lax.stop_gradient(s19_fix),
                )
            else:
                x0 = jnp.where(
                    design_mask_jnp[:, None],
                    x0,
                    jax.lax.stop_gradient(scaffold_20),
                )

        if n_done == 0:
            _LOG.info(
                "simplex_APGM phase1: n_steps=%s stepsize=%s momentum=%s scale=%s max_grad_norm=%s",
                int(opt.phase1_n_steps),
                float(opt.phase1_stepsize_factor) * sqrt_l,
                float(opt.phase1_momentum),
                float(opt.phase1_scale),
                mgn,
            )
        _, pssm = simplex_APGM(
            loss_function=loss,
            x=x0,
            n_steps=int(opt.phase1_n_steps),
            stepsize=float(opt.phase1_stepsize_factor) * sqrt_l,
            momentum=float(opt.phase1_momentum),
            scale=float(opt.phase1_scale),
            logspace=False,
            max_gradient_norm=mgn,
            key=jax.random.fold_in(ik, 811),
        )
        if n_done == 0:
            _LOG.info(
                "simplex_APGM phase2: n_steps=%s stepsize=%s scale=%s logspace=True",
                int(opt.phase2_n_steps),
                float(opt.phase2_stepsize_factor) * sqrt_l,
                float(opt.phase2_scale),
            )
        _, pssm = simplex_APGM(
            loss_function=loss,
            x=jnp.log(pssm + 1e-5),
            n_steps=int(opt.phase2_n_steps),
            stepsize=float(opt.phase2_stepsize_factor) * sqrt_l,
            momentum=float(opt.phase2_momentum),
            scale=float(opt.phase2_scale),
            logspace=True,
            max_gradient_norm=mgn,
            key=jax.random.fold_in(ik, 812),
        )
        if n_done == 0:
            _LOG.info(
                "simplex_APGM phase3: n_steps=%s stepsize=%s scale=%s logspace=True",
                int(opt.phase3_n_steps),
                float(opt.phase3_stepsize_factor) * sqrt_l,
                float(opt.phase3_scale),
            )
        _, pssm = simplex_APGM(
            loss_function=loss,
            x=jnp.log(pssm + 1e-5),
            n_steps=int(opt.phase3_n_steps),
            stepsize=float(opt.phase3_stepsize_factor) * sqrt_l,
            momentum=float(opt.phase3_momentum),
            scale=float(opt.phase3_scale),
            logspace=True,
            max_gradient_norm=mgn,
            key=jax.random.fold_in(ik, 813),
        )

        if disulfide_pairs == 0:
            pssm = NoCys.sequence(pssm)
        pssm_np = np.asarray(pssm)
        idx_hard = np.argmax(pssm_np, axis=-1).astype(np.int64)
        if not all_designable:
            for i in range(binder_length):
                if not design_mask_np[i]:
                    idx_hard[i] = TOKENS.index(seq_str[i])
        seq = "".join(TOKENS[int(i)] for i in idx_hard)
        pssm = jax.nn.one_hot(jnp.array(idx_hard, dtype=jnp.int32), 20)

        rank_score_str = "nan"
        if bool(cfg.ranking.enabled):
            rk = cfg.ranking
            rank_combo = (
                float(rk.iptm) * sp.IPTMLoss()
                + float(rk.target_binder_ipsae) * sp.TargetBinderIPSAE()
                + float(rk.binder_target_ipsae) * sp.BinderTargetIPSAE()
            )
            rank_feat, _ = model.target_only_features(
                [
                    TargetChain(sequence=seq, use_msa=False, cyclic_peptide=cyclic_b),
                    target_chain_with_msa(),
                ]
            )
            ranking_loss = model.build_multisample_loss(
                loss=rank_combo,
                features=rank_feat,
                recycling_steps=int(rk.recycling_steps),
                num_samples=int(rk.num_samples),
            )
            seq_idx = jnp.argmax(pssm, axis=-1)
            one_hot = jax.nn.one_hot(seq_idx, len(TOKENS))
            rank_key = jax.random.fold_in(ik, 900)
            assert _eval_ranking is not None
            rank_val, _rank_aux = _eval_ranking(ranking_loss, one_hot, rank_key)
            last_ranking_loss = float(rank_val)
            rank_score_str = f"{last_ranking_loss:.6f}"
            _LOG.info(
                "design #%s ranking_loss=%s seq_len=%s",
                n_done + 1,
                rank_score_str,
                len(seq),
            )

        with open(designs_path, "a", encoding="utf-8") as df:
            df.write(f">{rank_score_str}\n{seq}\n")
            df.flush()

        seq_path.write_text(seq + "\n", encoding="utf-8")
        np.save(pssm_path, np.asarray(pssm))

        if do_fold:
            seq_hash = hashlib.sha256(seq.encode("utf-8")).hexdigest()[:12]
            cif_stem = f"boltz2_design_{n_done + 1:05d}_{seq_hash}"
            _LOG.info(
                "Boltz-2 forward fold #%s: %s.cif checkpoint=%s recycling_steps=%s sampling_steps=%s",
                n_done + 1,
                cif_stem,
                fold_ck,
                int(cfg.boltz2_fold.recycling_steps),
                int(cfg.boltz2_fold.sampling_steps),
            )
            b2_feat, b2_writer = b2_for_fold.target_only_features(
                [
                    TargetChain(sequence=seq, use_msa=False, cyclic_peptide=cyclic_fold),
                    target_chain_with_msa(),
                ]
            )
            fold_key_b2 = jax.random.fold_in(key, 920 + n_done)
            b2_pred = b2_for_fold.predict(
                PSSM=None,
                features=b2_feat,
                writer=b2_writer,
                recycling_steps=int(cfg.boltz2_fold.recycling_steps),
                sampling_steps=int(cfg.boltz2_fold.sampling_steps),
                key=fold_key_b2,
            )
            b2_pred.st.name = cif_stem
            b2_cif = out / f"{cif_stem}.cif"
            b2_pred.st.make_mmcif_document().write_file(str(b2_cif))
            b2_plddt = np.asarray(b2_pred.plddt)
            bt_ipsae = float(np.asarray(b2_pred.bt_ipsae))
            tb_ipsae = float(np.asarray(b2_pred.tb_ipsae))
            b2_metrics = {
                "design_index": n_done + 1,
                "binder_sequence": seq,
                "cif_stem": cif_stem,
                "iptm": float(b2_pred.iptm),
                "bt_ipsae": bt_ipsae if math.isfinite(bt_ipsae) else None,
                "tb_ipsae": tb_ipsae if math.isfinite(tb_ipsae) else None,
                "plddt_mean": float(b2_plddt.mean()),
                "plddt_min": float(b2_plddt.min()),
                "plddt_max": float(b2_plddt.max()),
            }
            if math.isfinite(bt_ipsae) and math.isfinite(tb_ipsae):
                b2_metrics["ipsae_min"] = min(bt_ipsae, tb_ipsae)
            if last_ranking_loss is not None:
                b2_metrics["ranking_loss"] = last_ranking_loss
            b2_metrics_path = out / f"{cif_stem}_metrics.json"
            b2_metrics_path.write_text(
                json.dumps(b2_metrics, indent=2), encoding="utf-8"
            )
            _LOG.info(
                "Boltz-2 fold wrote %s (metrics %s)",
                b2_cif.resolve(),
                b2_metrics_path.resolve(),
            )

            hp = cfg.scoring.hbplus
            if bool(hp.enabled):
                from mosaic.scoring import HBplusScore

                extra = hp.get("extra_args")
                argv = tuple(str(x) for x in extra) if extra else ()
                hp_home = _resolve_scoring_home(_cfg_opt_str(hp, "home"))
                hb = HBplusScore(
                    executable=str(hp.executable),
                    package_home=hp_home,
                    extra_argv=argv,
                )
                hb_score, hb_aux = hb(structure_path=b2_cif)
                b2_metrics["hbplus_hbonds"] = int(hb_aux["hbonds"])
                b2_metrics["hbplus_score"] = hb_score
                b2_metrics["hbplus_hbonds_intra_chain"] = int(
                    hb_aux["hbonds_intra_chain"]
                )
                b2_metrics["hbplus_hbonds_total"] = int(
                    hb_aux["hbonds_hbplus_total"]
                )
                b2_metrics_path.write_text(
                    json.dumps(b2_metrics, indent=2), encoding="utf-8"
                )
                _LOG.info(
                    "HBPLUS inter-chain hbonds=%s intra_chain=%s hbplus_total=%s (%s)",
                    hb_aux["hbonds_inter_chain"],
                    hb_aux["hbonds_intra_chain"],
                    hb_aux["hbonds_hbplus_total"],
                    hb_aux.get("executable_resolved"),
                )

            ef = cfg.scoring.evoef
            if bool(ef.enabled):
                from mosaic.scoring import EvoEFBindingScore

                ex = ef.get("extra_args")
                eargv = tuple(str(x) for x in ex) if ex else ()
                ef_home = _resolve_scoring_home(_cfg_opt_str(ef, "home"))
                ev = EvoEFBindingScore(
                    package_home=ef_home,
                    executable=_cfg_opt_str(ef, "executable"),
                    split=_cfg_opt_str(ef, "split"),
                    extra_argv=eargv,
                )
                e_score, e_aux = ev(structure_path=b2_cif)
                b2_metrics["evoef_binding"] = e_score
                b2_metrics_path.write_text(
                    json.dumps(b2_metrics, indent=2), encoding="utf-8"
                )
                _LOG.info(
                    "EvoEF ComputeBinding total=%s (%s)",
                    e_score,
                    e_aux.get("executable_resolved"),
                )

            e2 = cfg.scoring.evoef2
            if bool(e2.enabled):
                from mosaic.scoring import EvoEF2BindingScore

                ex2 = e2.get("extra_args")
                e2argv = tuple(str(x) for x in ex2) if ex2 else ()
                e2_home = _resolve_scoring_home(_cfg_opt_str(e2, "home"))
                ev2 = EvoEF2BindingScore(
                    package_home=e2_home,
                    executable=_cfg_opt_str(e2, "executable"),
                    split_chains=_cfg_opt_str(e2, "split_chains"),
                    extra_argv=e2argv,
                )
                e2_score, e2_aux = ev2(structure_path=b2_cif)
                b2_metrics["evoef2_binding"] = e2_score
                b2_metrics_path.write_text(
                    json.dumps(b2_metrics, indent=2), encoding="utf-8"
                )
                _LOG.info(
                    "EvoEF2 ComputeBinding total=%s (%s)",
                    e2_score,
                    e2_aux.get("executable_resolved"),
                )

            cd = _sc.get("cysteine_disulfide")
            if cd is not None and bool(cd.enabled):
                from mosaic.scoring import CysteineDisulfideScore

                cd_score, cd_aux = CysteineDisulfideScore()(
                    structure_path=b2_cif,
                    binder_sequence=seq,
                )
                b2_metrics["binder_cysteine_count"] = int(
                    cd_aux["binder_cysteine_count"]
                )
                b2_metrics["binder_disulfide_bonds"] = int(
                    cd_aux["binder_disulfide_bonds"]
                )
                b2_metrics["cysteine_disulfide_score"] = float(cd_score)
                b2_metrics["binder_chain_id_scoring"] = str(
                    cd_aux.get("binder_chain_id", "")
                )
                if "binder_cys_residues_in_structure" in cd_aux:
                    b2_metrics["binder_cys_residues_in_structure"] = int(
                        cd_aux["binder_cys_residues_in_structure"]
                    )
                b2_metrics_path.write_text(
                    json.dumps(b2_metrics, indent=2), encoding="utf-8"
                )
                _LOG.info(
                    "Cysteine/disulfide seq_Cys=%s structure_CYS=%s SS_bonds=%s chain=%s",
                    cd_aux["binder_cysteine_count"],
                    cd_aux.get("binder_cys_residues_in_structure"),
                    cd_aux["binder_disulfide_bonds"],
                    cd_aux.get("binder_chain_id"),
                )

        n_done += 1

    if bool(cfg.ranking.enabled) and last_ranking_loss is not None:
        rank_path = out / "ranking_metrics.json"
        rank_path.write_text(
            json.dumps({"ranking_loss": last_ranking_loss}, indent=2),
            encoding="utf-8",
        )
        _LOG.info(
            "ranking_metrics.json (last design) ranking_loss=%s",
            last_ranking_loss,
        )

    _LOG.info(
        "Wrote %s design(s): incremental %s, last sequence %s, last pssm %s",
        n_done,
        designs_path.resolve(),
        seq_path.resolve(),
        pssm_path.resolve(),
    )

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg_path.write_text(
        yaml.safe_dump(flat_cfg, sort_keys=False),
        encoding="utf-8",
    )
    _LOG.info(
        "Done: designed_sequence length=%s aa written to %s",
        len(seq),
        seq_path.resolve(),
    )


if __name__ == "__main__":
    main()
