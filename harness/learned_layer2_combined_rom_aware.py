"""Learned Layer 2 — combined OpenCap + ASPset cohort + ROM-aware loss.

Agent LL build. Hybrid of:
  * Agent HH2 (`harness.learned_layer2_combined`) — combined OpenCap (9) +
    ASPset (15) cohort, masked per-frame SmoothL1 loss. Pooled |r| 0.645 →
    0.670 OpenCap-held; 0.677 ASPset-held.
  * Agent GG2 (`harness.rom_aware_layer2`) — extrema-aware loss
    (SmoothL1 + lam * |peak_pred - peak_gt| + lam * |min_pred - min_gt|),
    OpenCap-only cohort, lam=1.0. ROM tier wins +5 promotions over v17.

Hypothesis (per Cameron's LL brief): the two improvements compound on slots
where the failure mode is extrema accuracy. Combined cohort = better
generalization. ROM-aware loss = tighter LoA at L3 ridge.

# Convention alignment (carried from HH2)
ASPset -> OpenCap remap, identical to HH2:
  - hip_flexion_r:   identity
  - hip_adduction_r: identity, clamp |x|>85° -> NaN  (BUT see drop below)
  - knee_angle_r:    OC = 180 - ASP
  - ankle_angle_r:   NaN (ASPset has no foot KPs)
  - lumbar_extension: OC = ASP - 180

# Hip-adduction-r ASPset-drop (HH2's own recommendation)
HH2's `hip_adduction_r` regressed on OpenCap-held folds (−0.051 |r|) due to
ASPset's asin-based frontal-plane definition not matching OpenSim's
lumped-rotation hip_adduction. HH2 REPORT recommends "drop hip_adduction_r
from ASPset training, or use per-source target heads." Per Cameron's LL
brief: "Default to dropping unless that significantly hurts the rest."
We implement this as a CLI flag `--drop-aspset-hipadd` (default ON).

# Loss (Option 2 from GG2 + masking from HH2)
For a batch of `clips_per_step` clips:

  per_frame = SmoothL1(pred_n[mask], gt_n[mask])
  peak      = mean_{c,m: n_finite>=5} | pred_n[c,m].amax() - gt_n[c,m].amax() |
  min       = mean_{c,m: n_finite>=5} | pred_n[c,m].amin() - gt_n[c,m].amin() |
  total     = per_frame + lam * peak + lam * min

where `mask` is the finite-target mask. For ASPset clips, `mask[:, ankle]`
is always False (no foot KPs), and `mask[:, hip_add]` is forced False when
the drop flag is set. Both per-frame and extrema terms respect the same
mask, so masked metrics contribute zero gradient.

# Training
  * 25 epochs (HH2 default, also LL brief)
  * clips_per_step=4 (GG2 default)
  * AdamW lr=1e-3 wd=1e-4
  * cosine LR
  * CPU only

# 24-fold LOSO discipline preserved at L2.
"""
from __future__ import annotations

import builtins
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_combined import (
    DEPLOY_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    TemporalKeypointCNNConf,
    build_combined_cohort,
    build_windows_for_clip,
)


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer2_combined_rom_aware"
MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_rom_aware_v1.pt"
)


HIP_ADD_IDX: Final[int] = DEPLOY_METRICS.index("hip_adduction_r")  # 1


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[LL t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Per-clip tensor preparation (full-clip windows + GT + finite mask).
# -------------------------------------------------------------------------


def _flatten(x: np.ndarray) -> np.ndarray:
    """(N, T_win, K, C) -> (N, T_win, K*C)."""
    n, t, k, c = x.shape
    return x.reshape(n, t, k * c)


def prepare_clip_tensors(
    clip_recs: list[dict],
    *,
    drop_aspset_hipadd: bool = True,
) -> list[dict]:
    """Build per-clip torch tensors for the LL training loop.

    For each clip:
        x:      (T_frames, T_win, K*3) float32
        y:      (T_frames, n_metrics) float32 (NaN preserved for masked)
        finite: (T_frames, n_metrics) bool

    For ASPset clips, hip_adduction_r mask is forced False when
    drop_aspset_hipadd is True (HH2's own recommendation).
    """
    out: list[dict] = []
    n_drop_hipadd_aspset = 0
    for rec in clip_recs:
        windows, angles = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        # Per-clip finite mask.
        finite = np.isfinite(angles)
        # Drop hip_adduction_r supervision for ASPset clips if requested.
        if drop_aspset_hipadd and rec.get("source") == "aspset":
            if finite[:, HIP_ADD_IDX].any():
                n_drop_hipadd_aspset += 1
            finite[:, HIP_ADD_IDX] = False
        if not finite.any():
            continue
        x_flat = _flatten(windows)
        out.append({
            "x": torch.from_numpy(x_flat).float(),
            "y": torch.from_numpy(angles).float(),
            "finite": torch.from_numpy(finite),
            "clip_id": rec["clip"],
            "subject": rec["subject"],
            "trial": rec["trial"],
            "cam": rec["cam"],
            "source": rec.get("source", "unknown"),
        })
    if drop_aspset_hipadd:
        log(
            f"    drop_aspset_hipadd: masked hip_adduction_r in "
            f"{n_drop_hipadd_aspset} ASPset clips"
        )
    return out


# -------------------------------------------------------------------------
# Loss (HH2 masking + GG2 extrema-aware).
# -------------------------------------------------------------------------


def extrema_aware_masked_loss(
    pred_clips_n: list[torch.Tensor],
    gt_clips_n: list[torch.Tensor],
    finite_clips: list[torch.Tensor],
    *,
    lam: float = 1.0,
    smoothl1_beta: float = 1.0,
    min_finite_for_extrema: int = 5,
) -> tuple[torch.Tensor, dict]:
    """Per-frame SmoothL1 (masked) + per-(clip, metric) peak/min loss (masked).

    All inputs are in NORMALIZED angle space. NaN targets are excluded from
    both per-frame and extrema terms.
    """
    # ---- per-frame SmoothL1 pooled across batch (finite only) ----
    frame_pred_parts: list[torch.Tensor] = []
    frame_gt_parts: list[torch.Tensor] = []
    for p, g, mask in zip(pred_clips_n, gt_clips_n, finite_clips):
        if mask.any():
            frame_pred_parts.append(p[mask])
            frame_gt_parts.append(g[mask])

    if frame_pred_parts:
        all_pred = torch.cat(frame_pred_parts, dim=0)
        all_gt = torch.cat(frame_gt_parts, dim=0)
        frame_loss = nn.functional.smooth_l1_loss(
            all_pred, all_gt, beta=smoothl1_beta
        )
    else:
        # Defensive: keep grad alive.
        frame_loss = sum(p.sum() * 0.0 for p in pred_clips_n) \
            if pred_clips_n else torch.zeros((), requires_grad=True)

    # ---- per-(clip, metric) peak / min loss ----
    peak_terms: list[torch.Tensor] = []
    min_terms: list[torch.Tensor] = []
    for p, g, mask in zip(pred_clips_n, gt_clips_n, finite_clips):
        _T, M = p.shape
        for m in range(M):
            mm = mask[:, m]
            if int(mm.sum().item()) < min_finite_for_extrema:
                continue
            p_m = p[mm, m]
            g_m = g[mm, m]
            peak_terms.append(torch.abs(p_m.amax() - g_m.amax()))
            min_terms.append(torch.abs(p_m.amin() - g_m.amin()))

    if peak_terms:
        peak_loss = torch.stack(peak_terms).mean()
        min_loss = torch.stack(min_terms).mean()
    else:
        peak_loss = torch.zeros((), requires_grad=True)
        min_loss = torch.zeros((), requires_grad=True)

    total = frame_loss + lam * peak_loss + lam * min_loss
    return total, {
        "frame": float(frame_loss.detach().item()),
        "peak": float(peak_loss.detach().item()),
        "min": float(min_loss.detach().item()),
        "n_clip_extrema": len(peak_terms),
    }


# -------------------------------------------------------------------------
# Training one LOSO fold.
# -------------------------------------------------------------------------


def _compute_target_norm_from_clips(
    clip_data: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target mean/std across finite frames of all training clips."""
    y_mean = np.zeros(len(DEPLOY_METRICS), dtype=np.float32)
    y_std = np.ones(len(DEPLOY_METRICS), dtype=np.float32)
    for m in range(len(DEPLOY_METRICS)):
        vals = []
        for d in clip_data:
            y_np = d["y"].numpy()
            mm = d["finite"].numpy()[:, m]
            if mm.any():
                vals.append(y_np[mm, m])
        if vals:
            arr = np.concatenate(vals)
            y_mean[m] = float(arr.mean())
            y_std[m] = max(float(arr.std()), 1e-3)
    return y_mean, y_std


def train_one_fold_ll(
    train_clips: list[dict],
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lam: float = 1.0,
    seed: int = 0,
    drop_aspset_hipadd: bool = True,
    fold_label: str = "",
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train TemporalKeypointCNNConf with combined-cohort + ROM-aware loss."""
    rng = np.random.default_rng(seed)
    log(
        f"  [{fold_label}] preparing per-clip tensors from "
        f"{len(train_clips)} clips ..."
    )
    t_prep = time.time()
    clip_data = prepare_clip_tensors(
        train_clips, drop_aspset_hipadd=drop_aspset_hipadd
    )
    log(
        f"  [{fold_label}] prepared {len(clip_data)} usable clips "
        f"in {time.time()-t_prep:.0f}s"
    )
    if len(clip_data) < 4:
        raise RuntimeError(
            f"Too few training clips: {len(clip_data)}"
        )

    y_mean, y_std = _compute_target_norm_from_clips(clip_data)
    y_mean_t = torch.from_numpy(y_mean).float()
    y_std_t = torch.from_numpy(y_std).float()
    log(
        f"  [{fold_label}] y_mean={y_mean.round(2).tolist()} "
        f"y_std={y_std.round(2).tolist()}"
    )

    # Pre-normalize each clip's GT once; NaN -> 0 so masked terms don't taint.
    for d in clip_data:
        y_n = (d["y"] - y_mean_t) / y_std_t
        d["y_n"] = torch.where(d["finite"], y_n, torch.zeros_like(y_n))

    model = TemporalKeypointCNNConf()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"  [{fold_label}] model params={n_params:,} epochs={epochs} "
        f"clips_per_step={clips_per_step} lam={lam} "
        f"drop_aspset_hipadd={drop_aspset_hipadd}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_clips = len(clip_data)
    steps_per_epoch = max(1, n_clips // clips_per_step)
    log_hist: dict[str, list] = {
        "train_total": [], "train_frame": [],
        "train_peak": [], "train_min": [],
    }
    t_fold_start = time.time()
    last_print = time.time()

    for ep in range(epochs):
        model.train()
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} START "
            f"steps={steps_per_epoch} "
            f"fold_elapsed={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()
        perm = rng.permutation(n_clips)
        ep_total = 0.0
        ep_frame = 0.0
        ep_peak = 0.0
        ep_min = 0.0
        n_steps = 0

        for step in range(steps_per_epoch):
            start = step * clips_per_step
            picks = perm[start:start + clips_per_step]
            if len(picks) == 0:
                continue

            opt.zero_grad()
            pred_list: list[torch.Tensor] = []
            gt_list: list[torch.Tensor] = []
            mask_list: list[torch.Tensor] = []
            for ci in picks:
                d = clip_data[ci]
                p_n = model(d["x"])
                pred_list.append(p_n)
                gt_list.append(d["y_n"])
                mask_list.append(d["finite"])

            total_loss, comp = extrema_aware_masked_loss(
                pred_list, gt_list, mask_list, lam=lam,
            )
            total_loss.backward()
            opt.step()

            tloss = float(total_loss.detach().item())
            ep_total += tloss
            ep_frame += comp["frame"]
            ep_peak += comp["peak"]
            ep_min += comp["min"]
            n_steps += 1

            if (step % 5 == 0) or ((time.time() - last_print) > 12.0):
                log(
                    f"  [{fold_label}] ep {ep+1}/{epochs} step "
                    f"{step+1}/{steps_per_epoch} total={tloss:.4f} "
                    f"frame={comp['frame']:.4f} peak={comp['peak']:.4f} "
                    f"min={comp['min']:.4f} nM={comp['n_clip_extrema']}"
                )
                last_print = time.time()

        sched.step()
        ep_total /= max(n_steps, 1)
        ep_frame /= max(n_steps, 1)
        ep_peak /= max(n_steps, 1)
        ep_min /= max(n_steps, 1)
        log_hist["train_total"].append(ep_total)
        log_hist["train_frame"].append(ep_frame)
        log_hist["train_peak"].append(ep_peak)
        log_hist["train_min"].append(ep_min)
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} END "
            f"total={ep_total:.4f} frame={ep_frame:.4f} "
            f"peak={ep_peak:.4f} min={ep_min:.4f} "
            f"fold_elapsed={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()

    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std    # type: ignore[attr-defined]
    return model, log_hist


# -------------------------------------------------------------------------
# Per-fold evaluation (per-frame |r| + per-clip ROM CCC + ROM LoA).
# -------------------------------------------------------------------------


def predict_on_clip(
    model: TemporalKeypointCNNConf, clip_rec: dict
) -> np.ndarray:
    windows, _ = build_windows_for_clip(clip_rec)
    if windows.shape[0] == 0:
        return np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32)
    n, t, k, c = windows.shape
    x_flat = windows.reshape(n, t, k * c)
    x_t = torch.from_numpy(x_flat).float()
    model.eval()
    with torch.no_grad():
        p_n = model(x_t).numpy()
    y_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
    y_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
    pred = p_n * y_std + y_mean
    return pred.astype(np.float32)


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2:
        return float("nan")
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp = float(np.var(p, ddof=0))
    vo = float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom < 1e-12:
        return float("nan")
    return 2.0 * cov / denom


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.var(a) < 1e-12 or np.var(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_fold(
    model: TemporalKeypointCNNConf, held_clips: list[dict]
) -> dict:
    """Per-metric stats across the held-out clips."""
    per_metric_frame_r: dict[str, list[float]] = {
        m: [] for m in DEPLOY_METRICS
    }
    rom_pairs: dict[str, list[tuple[float, float]]] = {
        m: [] for m in DEPLOY_METRICS
    }

    for rec in held_clips:
        pred = predict_on_clip(model, rec)
        gt = rec["angles"]
        n = min(pred.shape[0], gt.shape[0])
        if n < 10:
            continue
        pred = pred[:n]
        gt = gt[:n]
        for k, metric in enumerate(DEPLOY_METRICS):
            a = pred[:, k]
            b = gt[:, k]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 30:
                continue
            ar = a[finite]
            br = b[finite]
            r = _pearson(ar, br)
            abs_r = abs(r) if np.isfinite(r) else float("nan")
            if np.isfinite(abs_r):
                per_metric_frame_r[metric].append(abs_r)
            rom_pairs[metric].append(
                (float(ar.max() - ar.min()), float(br.max() - br.min()))
            )

    per_metric_stats: dict[str, dict] = {}
    for m in DEPLOY_METRICS:
        frs = per_metric_frame_r[m]
        rl = rom_pairs[m]
        if rl:
            rp = np.array([x[0] for x in rl])
            rg = np.array([x[1] for x in rl])
            rom_ccc = _ccc(rp, rg)
            rom_mae = float(np.mean(np.abs(rp - rg)))
        else:
            rom_ccc = float("nan")
            rom_mae = float("nan")
        per_metric_stats[m] = {
            "n_clips": len(rl),
            "frame_abs_r_mean": (
                float(np.mean(frs)) if frs else float("nan")
            ),
            "rom_ccc": rom_ccc,
            "rom_mae": rom_mae,
        }

    pooled_frame = [
        r for vals in per_metric_frame_r.values() for r in vals
    ]
    pooled_rom_ccc = [
        per_metric_stats[m]["rom_ccc"]
        for m in DEPLOY_METRICS
        if np.isfinite(per_metric_stats[m]["rom_ccc"])
    ]
    return {
        "per_metric": per_metric_stats,
        "pooled_frame_abs_r": (
            float(np.mean(pooled_frame)) if pooled_frame else float("nan")
        ),
        "metric_mean_rom_ccc": (
            float(np.mean(pooled_rom_ccc))
            if pooled_rom_ccc else float("nan")
        ),
    }


# -------------------------------------------------------------------------
# LOSO driver.
# -------------------------------------------------------------------------


def run_loso(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    drop_aspset_hipadd: bool = True,
) -> dict:
    subjects = sorted(cohort.keys())
    log(f"LOSO over {len(subjects)} subjects")

    fold_results: list[dict] = []
    best_state = None
    best_y_mean = None
    best_y_std = None
    best_score = -1e9
    best_subject = None
    last_state = None
    last_y_mean = None
    last_y_std = None

    t_loso_start = time.time()
    for fold_i, held in enumerate(subjects):
        fold_label = f"fold {fold_i+1}/{len(subjects)} held={held}"
        log(
            f"=== {fold_label} START "
            f"t_loso={time.time()-t_loso_start:.0f}s ==="
        )
        train_subjects = [s for s in subjects if s != held]
        train_clips = [c for s in train_subjects for c in cohort[s]]
        held_clips = cohort[held]
        log(
            f"  [{fold_label}] train clips={len(train_clips)} "
            f"held clips={len(held_clips)}"
        )

        try:
            model, hist = train_one_fold_ll(
                train_clips,
                epochs=epochs,
                clips_per_step=clips_per_step,
                lam=lam,
                seed=fold_i,
                drop_aspset_hipadd=drop_aspset_hipadd,
                fold_label=fold_label,
            )
        except Exception as e:
            log(f"  [{fold_label}] TRAINING FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            continue

        log(f"  [{fold_label}] evaluating on held-out ...")
        ev = evaluate_fold(model, held_clips)
        log(
            f"  [{fold_label}] pooled_frame|r|="
            f"{ev['pooled_frame_abs_r']:.4f} "
            f"metric_mean_rom_ccc={ev['metric_mean_rom_ccc']:.4f}"
        )
        for m, st in ev["per_metric"].items():
            log(
                f"    {m:20s} frame|r|={st['frame_abs_r_mean']:.3f} "
                f"rom_ccc={st['rom_ccc']:.3f} "
                f"rom_mae={st['rom_mae']:.2f}deg "
                f"n={st['n_clips']}"
            )

        fold_results.append({
            "held_out": held,
            "source": "opencap" if held.startswith("opencap_") else "aspset",
            "fold_idx": fold_i,
            "n_train_clips": len(train_clips),
            "n_held_clips": len(held_clips),
            "train_hist": hist,
            "eval": ev,
        })

        last_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        last_y_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
        last_y_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
        score = ev["pooled_frame_abs_r"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_subject = held
            best_state = last_state
            best_y_mean = last_y_mean
            best_y_std = last_y_std

        _save_intermediate(
            fold_results, epochs, clips_per_step, lam, drop_aspset_hipadd,
        )
        log(
            f"=== {fold_label} END "
            f"cumulative_t={time.time()-t_loso_start:.0f}s ==="
        )

    overall = _aggregate_overall(fold_results)
    log(
        f"=== OVERALL LOSO pooled_frame|r|="
        f"{overall['pooled_frame_abs_r_mean']:.4f} "
        f"metric_mean_rom_ccc={overall['metric_mean_rom_ccc_mean']:.4f} ==="
    )
    log(
        f"  by-source: OpenCap-held |r|="
        f"{overall['by_source']['opencap']['pooled_frame_abs_r_mean']:.4f} "
        f"(n={overall['by_source']['opencap']['n_folds']}), "
        f"ASPset-held |r|="
        f"{overall['by_source']['aspset']['pooled_frame_abs_r_mean']:.4f} "
        f"(n={overall['by_source']['aspset']['n_folds']})"
    )

    save_state = best_state if best_state is not None else last_state
    save_y_mean = best_y_mean if best_y_mean is not None else last_y_mean
    save_y_std = best_y_std if best_y_std is not None else last_y_std
    if save_state is not None:
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": save_state,
            "y_mean": save_y_mean,
            "y_std": save_y_std,
            "n_keys": N_KEYS,
            "temporal_t": TEMPORAL_T,
            "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
            "metrics": list(DEPLOY_METRICS),
            "model_class": "TemporalKeypointCNNConf",
            "training": "learned_layer2_combined_rom_aware_v1",
            "lam": lam,
            "drop_aspset_hipadd": drop_aspset_hipadd,
            "best_loso_subject": best_subject,
            "best_loso_pooled_abs_r": best_score,
        }, MODEL_OUT)
        log(
            f"saved best-fold checkpoint -> {MODEL_OUT} "
            f"(held={best_subject} |r|={best_score:.3f})"
        )

    return {
        "version": "learned_layer2_combined_rom_aware_v1",
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "drop_aspset_hipadd": drop_aspset_hipadd,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "fold_results": fold_results,
        "overall": overall,
    }


def _aggregate_overall(fold_results: list[dict]) -> dict:
    per_metric_acc: dict[str, dict[str, list[float]]] = {
        m: {"frame_abs_r_mean": [], "rom_ccc": [], "rom_mae": []}
        for m in DEPLOY_METRICS
    }
    pooled_frame_per_fold = []
    pooled_rom_ccc_per_fold = []
    for fr in fold_results:
        for m, st in fr["eval"]["per_metric"].items():
            for key in per_metric_acc[m]:
                v = st.get(key, float("nan"))
                if np.isfinite(v):
                    per_metric_acc[m][key].append(v)
        if np.isfinite(fr["eval"]["pooled_frame_abs_r"]):
            pooled_frame_per_fold.append(fr["eval"]["pooled_frame_abs_r"])
        if np.isfinite(fr["eval"]["metric_mean_rom_ccc"]):
            pooled_rom_ccc_per_fold.append(fr["eval"]["metric_mean_rom_ccc"])

    by_source = {"opencap": [], "aspset": []}
    by_source_per_metric: dict[str, dict[str, list[float]]] = {
        src: {m: [] for m in DEPLOY_METRICS} for src in by_source
    }
    for fr in fold_results:
        src = fr.get("source", "unknown")
        if src not in by_source:
            continue
        p = fr["eval"]["pooled_frame_abs_r"]
        if np.isfinite(p):
            by_source[src].append(p)
        for m, st in fr["eval"]["per_metric"].items():
            v = st.get("frame_abs_r_mean")
            if v is not None and np.isfinite(v):
                by_source_per_metric[src][m].append(v)

    return {
        "per_metric": {
            m: {
                key: {
                    "n_folds": len(vals),
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": (
                        float(np.std(vals))
                        if len(vals) > 1 else float("nan")
                    ),
                }
                for key, vals in per_metric_acc[m].items()
            }
            for m in DEPLOY_METRICS
        },
        "pooled_frame_abs_r_mean": (
            float(np.mean(pooled_frame_per_fold))
            if pooled_frame_per_fold else float("nan")
        ),
        "metric_mean_rom_ccc_mean": (
            float(np.mean(pooled_rom_ccc_per_fold))
            if pooled_rom_ccc_per_fold else float("nan")
        ),
        "by_source": {
            src: {
                "n_folds": len(vals),
                "pooled_frame_abs_r_mean": (
                    float(np.mean(vals)) if vals else float("nan")
                ),
            }
            for src, vals in by_source.items()
        },
        "by_source_per_metric": {
            src: {
                m: {
                    "n_folds": len(vals),
                    "frame_abs_r_mean": (
                        float(np.mean(vals)) if vals else float("nan")
                    ),
                }
                for m, vals in d.items()
            }
            for src, d in by_source_per_metric.items()
        },
    }


def _save_intermediate(
    fold_results: list[dict],
    epochs: int,
    clips_per_step: int,
    lam: float,
    drop_aspset_hipadd: bool,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = []
    for fr in fold_results:
        ev = fr["eval"]
        slim.append({
            "held_out": fr["held_out"],
            "source": fr["source"],
            "fold_idx": fr["fold_idx"],
            "n_train_clips": fr["n_train_clips"],
            "n_held_clips": fr["n_held_clips"],
            "train_hist": fr["train_hist"],
            "eval": {
                "per_metric": ev["per_metric"],
                "pooled_frame_abs_r": ev["pooled_frame_abs_r"],
                "metric_mean_rom_ccc": ev["metric_mean_rom_ccc"],
            },
        })
    out = {
        "version": "learned_layer2_combined_rom_aware_v1",
        "intermediate": True,
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "drop_aspset_hipadd": drop_aspset_hipadd,
        "n_folds_completed": len(slim),
        "fold_results": slim,
    }
    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))


# -------------------------------------------------------------------------
# All-data training (no LOSO at L2) -- used by the Layer 3 wrapper.
# -------------------------------------------------------------------------


def train_all_data_ll(
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    drop_aspset_hipadd: bool = True,
    seed: int = 42,
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Build combined cohort, train on ALL subjects (no LOSO at L2).

    Used by `harness.layer3_retrain_on_combined_rom_aware_l2` for Phase B.
    Caller is responsible for saving the checkpoint with the right schema.
    """
    cohort = build_combined_cohort()
    if not cohort:
        raise RuntimeError("combined cohort is empty; cannot train")
    all_clips = [c for clips in cohort.values() for c in clips]
    log(
        f"all-data training: {len(cohort)} subjects, {len(all_clips)} clips"
    )
    model, hist = train_one_fold_ll(
        all_clips,
        epochs=epochs,
        clips_per_step=clips_per_step,
        lam=lam,
        seed=seed,
        drop_aspset_hipadd=drop_aspset_hipadd,
        fold_label="ALL-DATA-LL",
    )
    return model, {
        "hist": hist,
        "n_subjects": len(cohort),
        "n_clips": len(all_clips),
        "subjects": sorted(cohort.keys()),
    }


# -------------------------------------------------------------------------
# Main (LOSO eval driver).
# -------------------------------------------------------------------------


def main(
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    drop_aspset_hipadd: bool = True,
    max_subjects: int | None = None,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_combined_rom_aware (LL) START ===")
    log(
        f"epochs={epochs} clips_per_step={clips_per_step} lam={lam} "
        f"drop_aspset_hipadd={drop_aspset_hipadd} "
        f"T={TEMPORAL_T} N_KEYS={N_KEYS}"
    )

    cohort = build_combined_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return {}

    if max_subjects is not None:
        keep = sorted(cohort.keys())[:max_subjects]
        cohort = {k: cohort[k] for k in keep}
        log(f"DEBUG: limited to {len(cohort)} subjects: {keep}")

    results = run_loso(
        cohort,
        epochs=epochs,
        clips_per_step=clips_per_step,
        lam=lam,
        drop_aspset_hipadd=drop_aspset_hipadd,
    )

    out_main = OUT_DIR / "per_slot_results.json"
    out_main.write_text(json.dumps({
        "version": "learned_layer2_combined_rom_aware_v1",
        "intermediate": False,
        "epochs": results["epochs"],
        "clips_per_step": results["clips_per_step"],
        "lam": results["lam"],
        "drop_aspset_hipadd": results["drop_aspset_hipadd"],
        "subjects": results["subjects"],
        "metrics": results["metrics"],
        "overall": results["overall"],
        "fold_results_slim": [
            {
                "held_out": fr["held_out"],
                "source": fr["source"],
                "fold_idx": fr["fold_idx"],
                "n_train_clips": fr["n_train_clips"],
                "n_held_clips": fr["n_held_clips"],
                "train_hist": fr["train_hist"],
                "eval_per_metric": fr["eval"]["per_metric"],
                "eval_pooled_frame_abs_r": fr["eval"]["pooled_frame_abs_r"],
                "eval_metric_mean_rom_ccc": fr["eval"]["metric_mean_rom_ccc"],
            }
            for fr in results["fold_results"]
        ],
    }, indent=2, default=str))
    log(f"wrote summary to {out_main}")
    log("=== learned_layer2_combined_rom_aware (LL) DONE ===")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--clips-per-step", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--keep-aspset-hipadd", action="store_true",
        help="Disable the default ASPset hip_adduction_r drop",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="DEBUG: limit cohort to first N subjects",
    )
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        clips_per_step=args.clips_per_step,
        lam=args.lam,
        drop_aspset_hipadd=not args.keep_aspset_hipadd,
        max_subjects=args.max_subjects,
    )
