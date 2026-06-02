"""ROM-aware Learned Layer 2 - Agent GG2 build.

Extends Agent EE2's `TemporalKeypointCNNConf` with an extrema-aware loss so
the model is explicitly rewarded for landing peaks and valleys at the right
amplitude, not just for per-frame waveform agreement.

Why this build exists:
- Agent EE2 achieved per-frame pooled |r| = 0.645 (+0.131 vs Couro hand-eng).
- Agent FF retrained Layer 3 ridge on EE2's traces and got NET tier loss
  (3 promotions, 8 demotions). Diagnosis: ROM = max - min, ridge L3 is
  sensitive to extrema, not waveform shape. EE2's per-frame |r| gain didn't
  translate.
- Agent GG made a first attempt (lam=0.5, 25 epochs) and stalled at fold 5/9
  due to print silence. Per-fold ROM CCC numbers it produced were also low
  (mean ROM CCC ~0.05-0.23). This build uses lam=1.0 per the brief and
  tighter print discipline.

Loss (Option 2 from GG's brief, lam=1.0 default):

    loss = SmoothL1(pred_frames_n, gt_frames_n)
         + lam * mean( |peak(pred_n) - peak(gt_n)| )       (across clip, metric)
         + lam * mean( |min(pred_n)  - min(gt_n)|  )

All in normalized angle space. peak/min taken per (clip, metric) over the
clip's center-frame trajectory. torch.amax / torch.amin give differentiable
gradient to the worst-offending frames.

Batching: each gradient step processes a small number of full clips
(default 4). For each clip we forward-pass all its frames in one go, then
compute extrema per (clip, metric).

Print discipline: every 5 batches AND every 12s, whichever fires first. No
gap > 30s anywhere.

Evaluation: for each held-out subject, predict every clip; per metric
compute per-clip ROM (peak - valley), then CCC + Bland-Altman LoA across
the subject's clips. Aggregate across 9 LOSO folds.

Speedups vs EE2:
- epochs=12 (down from 25). Per-step cost is higher because we process
  full clips, but extrema gradient is denser per step.
- No internal val split (use train loss for monitoring; LOSO discipline
  preserved at fold level).
"""
from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_real_gt import (
    DEPLOY_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    TemporalKeypointCNNConf,
    build_clip_dataset,
    build_windows_for_clip,
)
from harness.smpl_layer2_poc import HALPE26_INDEX


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "rom_aware_layer2"
MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "rom_aware_layer2_v1.pt"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[GG2 t={elapsed:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Cohort build (uses EE2's build_clip_dataset directly).
# ----------------------------------------------------------------------------


def build_cohort() -> dict[str, list[dict]]:
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )
    kp_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"discovered {len(kp_files)} keypoint files in {OPENCAP_KP}")

    cohort: dict[str, list[dict]] = {}
    n_kept = 0
    n_skip = 0
    last_log = time.time()
    for i, kp_path in enumerate(kp_files):
        if (time.time() - last_log) > 15.0:
            log(
                f"  scanning {i+1}/{len(kp_files)} (kept={n_kept} skip={n_skip})"
            )
            last_log = time.time()
        rec = build_clip_dataset(kp_path, halpe_idx_arr)
        if rec is None:
            n_skip += 1
            continue
        cohort.setdefault(rec["subject"], []).append(rec)
        n_kept += 1

    log(
        f"cohort built: kept={n_kept} skip={n_skip} "
        f"subjects={sorted(cohort.keys())}"
    )
    for subj, recs in sorted(cohort.items()):
        n_frames = sum(r["kp"].shape[0] for r in recs)
        log(f"  {subj}: {len(recs)} clips, {n_frames} frames")
    return cohort


# ----------------------------------------------------------------------------
# Per-clip tensor preparation.
# ----------------------------------------------------------------------------


def _flatten(x: np.ndarray) -> np.ndarray:
    """(B, T_win, K, C) -> (B, T_win, K*C)."""
    b, t, k, c = x.shape
    return x.reshape(b, t, k * c)


def prepare_clip_tensors(clip_recs: list[dict]) -> list[dict]:
    """Build per-clip torch tensors used by the training loop.

    For each clip, produces:
        x:      (T_frames, T_win, K*3) float32 - sliding windows
        y:      (T_frames, n_metrics) float32 - per-frame GT (NaN preserved)
        finite: (T_frames, n_metrics) bool
    Skip clips with too few frames or no finite GT.
    """
    out: list[dict] = []
    for rec in clip_recs:
        windows, angles, _ = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        finite = np.isfinite(angles)
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
        })
    return out


# ----------------------------------------------------------------------------
# Loss.
# ----------------------------------------------------------------------------


def extrema_aware_loss(
    pred_clips_n: list[torch.Tensor],
    gt_clips_n: list[torch.Tensor],
    finite_clips: list[torch.Tensor],
    *,
    lam: float = 1.0,
    smoothl1_beta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Per-frame SmoothL1 + per-(clip, metric) peak/min loss.

    pred_clips_n/gt_clips_n: list of (T_clip, n_metrics) NORMALIZED tensors.
    finite_clips: list of (T_clip, n_metrics) bool masks.

    Returns total loss + dict of components for logging.
    """
    # ---- per-frame SmoothL1 pooled across clips on finite mask ----
    frame_pred_parts: list[torch.Tensor] = []
    frame_gt_parts: list[torch.Tensor] = []
    for p, g, mask in zip(pred_clips_n, gt_clips_n, finite_clips):
        if mask.any():
            frame_pred_parts.append(p[mask])
            # GT may contain NaN where mask is False; mask gives only finite
            # positions.
            frame_gt_parts.append(g[mask])

    if frame_pred_parts:
        all_pred = torch.cat(frame_pred_parts, dim=0)
        all_gt = torch.cat(frame_gt_parts, dim=0)
        frame_loss = nn.functional.smooth_l1_loss(
            all_pred, all_gt, beta=smoothl1_beta
        )
    else:
        # No finite frames in batch -- highly unusual but be safe.
        frame_loss = sum(
            p.sum() * 0.0 for p in pred_clips_n
        ) if pred_clips_n else torch.zeros((), requires_grad=True)

    # ---- per-(clip, metric) peak / min loss ----
    peak_terms: list[torch.Tensor] = []
    min_terms: list[torch.Tensor] = []
    for p, g, mask in zip(pred_clips_n, gt_clips_n, finite_clips):
        T, M = p.shape
        for m in range(M):
            mm = mask[:, m]
            if mm.sum().item() < 5:
                continue
            p_m = p[mm, m]
            g_m = g[mm, m]
            # NaN-safe: g_m is finite by mask construction.
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


# ----------------------------------------------------------------------------
# Training one LOSO fold.
# ----------------------------------------------------------------------------


def train_one_fold_romaware(
    train_clips: list[dict],
    *,
    epochs: int = 12,
    clips_per_step: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lam: float = 1.0,
    seed: int = 0,
    fold_label: str = "",
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train TemporalKeypointCNNConf with extrema-aware loss.

    Each gradient step samples `clips_per_step` random clips from the training
    pool. For each clip we forward-pass all its frames once and compute the
    extrema loss in normalized space.
    """
    rng = np.random.default_rng(seed)
    log(f"  [{fold_label}] preparing clip tensors from {len(train_clips)} clips ...")
    t_prep = time.time()
    clip_data = prepare_clip_tensors(train_clips)
    log(
        f"  [{fold_label}] prepared {len(clip_data)} usable clips "
        f"in {time.time()-t_prep:.0f}s"
    )
    if len(clip_data) < 4:
        raise RuntimeError(
            f"Too few training clips for ROM-aware build: {len(clip_data)}"
        )

    # Per-metric mean/std for normalization (over all finite frames).
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
    y_mean_t = torch.from_numpy(y_mean).float()
    y_std_t = torch.from_numpy(y_std).float()
    log(
        f"  [{fold_label}] y_mean={y_mean.round(2).tolist()} "
        f"y_std={y_std.round(2).tolist()}"
    )

    # Pre-normalize each clip's GT once; preserve NaN by zeroing them after
    # division (mask carries finiteness).
    for d in clip_data:
        y_n = (d["y"] - y_mean_t) / y_std_t
        # Replace NaN with 0 so masked terms don't taint backward.
        d["y_n"] = torch.where(d["finite"], y_n, torch.zeros_like(y_n))

    model = TemporalKeypointCNNConf()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"  [{fold_label}] model params={n_params:,} epochs={epochs} "
        f"clips_per_step={clips_per_step} lam={lam}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_clips = len(clip_data)
    steps_per_epoch = max(1, n_clips // clips_per_step)
    log_hist: dict[str, list] = {
        "train_total": [], "train_frame": [], "train_peak": [], "train_min": [],
    }
    t_fold_start = time.time()
    last_print = time.time()

    for ep in range(epochs):
        model.train()
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} START steps={steps_per_epoch} "
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

            total_loss, comp = extrema_aware_loss(
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

            # Print every 5 steps OR every >12s, whichever fires first.
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
            f"  [{fold_label}] ep {ep+1}/{epochs} END total={ep_total:.4f} "
            f"frame={ep_frame:.4f} peak={ep_peak:.4f} min={ep_min:.4f} "
            f"fold_elapsed={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()

    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std    # type: ignore[attr-defined]
    return model, log_hist


# ----------------------------------------------------------------------------
# Per-fold evaluation.
# ----------------------------------------------------------------------------


def predict_on_clip(
    model: TemporalKeypointCNNConf, clip_rec: dict
) -> np.ndarray:
    """Predict (T, n_metrics) angles for one clip."""
    windows, _, _ = build_windows_for_clip(clip_rec)
    if windows.shape[0] == 0:
        return np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32)
    x_flat = _flatten(windows)
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


def evaluate_fold_full(
    model: TemporalKeypointCNNConf, held_clips: list[dict]
) -> dict:
    """Per-frame |r| AND per-clip ROM for held-out clips."""
    per_metric_frame_r: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}
    rom_pairs: dict[str, list[tuple[float, float]]] = {
        m: [] for m in DEPLOY_METRICS
    }
    peak_pairs: dict[str, list[tuple[float, float]]] = {
        m: [] for m in DEPLOY_METRICS
    }
    valley_pairs: dict[str, list[tuple[float, float]]] = {
        m: [] for m in DEPLOY_METRICS
    }
    per_clip_records: list[dict] = []

    for rec in held_clips:
        pred = predict_on_clip(model, rec)
        gt = rec["angles"]
        n = min(pred.shape[0], gt.shape[0])
        if n < 10:
            continue
        pred = pred[:n]
        gt = gt[:n]

        clip_metrics: dict[str, dict | None] = {}
        for k, metric in enumerate(DEPLOY_METRICS):
            a = pred[:, k]
            b = gt[:, k]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 30:
                clip_metrics[metric] = None
                continue
            ar = a[finite]
            br = b[finite]
            r = _pearson(ar, br)
            abs_r = abs(r) if np.isfinite(r) else float("nan")
            p_peak = float(np.max(ar))
            p_min = float(np.min(ar))
            g_peak = float(np.max(br))
            g_min = float(np.min(br))
            clip_metrics[metric] = {
                "frame_r": r,
                "frame_abs_r": abs_r,
                "pred_peak": p_peak,
                "pred_min": p_min,
                "pred_rom": p_peak - p_min,
                "gt_peak": g_peak,
                "gt_min": g_min,
                "gt_rom": g_peak - g_min,
            }
            if np.isfinite(abs_r):
                per_metric_frame_r[metric].append(abs_r)
            rom_pairs[metric].append((p_peak - p_min, g_peak - g_min))
            peak_pairs[metric].append((p_peak, g_peak))
            valley_pairs[metric].append((p_min, g_min))

        per_clip_records.append({
            "clip": rec["clip"],
            "subject": rec["subject"],
            "trial": rec["trial"],
            "cam": rec["cam"],
            "n_frames": int(n),
            "metrics": clip_metrics,
        })

    # Aggregate per metric.
    per_metric_stats: dict[str, dict] = {}
    for m in DEPLOY_METRICS:
        frs = per_metric_frame_r[m]
        rl = rom_pairs[m]
        pl = peak_pairs[m]
        vl = valley_pairs[m]
        if rl:
            rp = np.array([x[0] for x in rl])
            rg = np.array([x[1] for x in rl])
            rom_r = _pearson(rp, rg)
            rom_ccc = _ccc(rp, rg)
            rom_mae = float(np.mean(np.abs(rp - rg)))
            diffs = rp - rg
            mean_bias = float(np.mean(diffs))
            sd_bias = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
            loa_half = (1.96 * sd_bias) if np.isfinite(sd_bias) else float("nan")
        else:
            rom_r = rom_ccc = rom_mae = mean_bias = sd_bias = loa_half = float("nan")
        if pl:
            pp = np.array([x[0] for x in pl])
            pg = np.array([x[1] for x in pl])
            peak_ccc = _ccc(pp, pg)
        else:
            peak_ccc = float("nan")
        if vl:
            vp = np.array([x[0] for x in vl])
            vg = np.array([x[1] for x in vl])
            valley_ccc = _ccc(vp, vg)
        else:
            valley_ccc = float("nan")
        per_metric_stats[m] = {
            "n_clips": len(rl),
            "frame_abs_r_mean": (
                float(np.mean(frs)) if frs else float("nan")
            ),
            "rom_r": rom_r,
            "rom_ccc": rom_ccc,
            "rom_mae": rom_mae,
            "rom_mean_bias": mean_bias,
            "rom_loa_half": loa_half,
            "peak_ccc": peak_ccc,
            "valley_ccc": valley_ccc,
        }

    pooled_frame = [r for vals in per_metric_frame_r.values() for r in vals]
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
            float(np.mean(pooled_rom_ccc)) if pooled_rom_ccc else float("nan")
        ),
        "per_clip": per_clip_records,
    }


# ----------------------------------------------------------------------------
# LOSO driver.
# ----------------------------------------------------------------------------


def run_loso_romaware(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 12,
    clips_per_step: int = 4,
    lam: float = 1.0,
) -> dict:
    subjects = sorted(cohort.keys())
    log(f"LOSO over {len(subjects)} subjects: {subjects}")

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
            model, hist = train_one_fold_romaware(
                train_clips,
                epochs=epochs,
                clips_per_step=clips_per_step,
                lam=lam,
                seed=fold_i,
                fold_label=fold_label,
            )
        except Exception as e:
            log(f"  [{fold_label}] TRAINING FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            continue

        log(f"  [{fold_label}] evaluating on held-out ...")
        ev = evaluate_fold_full(model, held_clips)
        log(
            f"  [{fold_label}] pooled_frame|r|={ev['pooled_frame_abs_r']:.4f} "
            f"metric_mean_rom_ccc={ev['metric_mean_rom_ccc']:.4f}"
        )
        for m, st in ev["per_metric"].items():
            log(
                f"    {m:20s} frame|r|={st['frame_abs_r_mean']:.3f} "
                f"rom_ccc={st['rom_ccc']:.3f} rom_r={st['rom_r']:.3f} "
                f"rom_mae={st['rom_mae']:.2f}deg "
                f"peak_ccc={st['peak_ccc']:.3f} "
                f"valley_ccc={st['valley_ccc']:.3f} n={st['n_clips']}"
            )

        fold_results.append({
            "held_out": held,
            "fold_idx": fold_i,
            "n_train_clips": len(train_clips),
            "n_held_clips": len(held_clips),
            "train_hist": hist,
            "eval": ev,
        })

        last_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        last_y_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
        last_y_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
        score = ev["metric_mean_rom_ccc"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_subject = held
            best_state = last_state
            best_y_mean = last_y_mean
            best_y_std = last_y_std

        _save_intermediate(fold_results, epochs, clips_per_step, lam)
        log(
            f"=== {fold_label} END  cumulative_t={time.time()-t_loso_start:.0f}s ==="
        )

    overall = _aggregate_overall(fold_results)
    log(
        f"=== OVERALL LOSO pooled_frame|r|="
        f"{overall['pooled_frame_abs_r_mean']:.4f} "
        f"metric-mean ROM CCC={overall['metric_mean_rom_ccc_mean']:.4f} ==="
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
            "training": "rom_aware_layer2_v1",
            "lam": lam,
            "best_loso_subject": best_subject,
            "best_metric_mean_rom_ccc": best_score,
        }, MODEL_OUT)
        log(
            f"saved checkpoint to {MODEL_OUT} "
            f"(best held={best_subject} ROM_CCC={best_score:.3f})"
        )

    return {
        "version": "rom_aware_layer2_v1",
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "fold_results": fold_results,
        "overall": overall,
    }


def _aggregate_overall(fold_results: list[dict]) -> dict:
    per_metric_acc: dict[str, dict[str, list[float]]] = {
        m: {
            "frame_abs_r_mean": [], "rom_ccc": [], "rom_r": [], "rom_mae": [],
            "rom_mean_bias": [], "rom_loa_half": [],
            "peak_ccc": [], "valley_ccc": [],
        }
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

    return {
        "per_metric": {
            m: {
                key: {
                    "n_folds": len(vals),
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": (
                        float(np.std(vals)) if len(vals) > 1 else float("nan")
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
    }


def _save_intermediate(
    fold_results: list[dict], epochs: int, clips_per_step: int, lam: float
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = []
    for fr in fold_results:
        ev = fr["eval"]
        slim.append({
            "held_out": fr["held_out"],
            "fold_idx": fr["fold_idx"],
            "n_train_clips": fr["n_train_clips"],
            "n_held_clips": fr["n_held_clips"],
            "train_hist": fr["train_hist"],
            "eval": {
                "per_metric": ev["per_metric"],
                "pooled_frame_abs_r": ev["pooled_frame_abs_r"],
                "metric_mean_rom_ccc": ev["metric_mean_rom_ccc"],
                "n_clips_evaluated": len(ev["per_clip"]),
            },
        })
    out = {
        "version": "rom_aware_layer2_v1",
        "intermediate": True,
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "n_folds_completed": len(slim),
        "fold_results": slim,
    }
    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------


def main(
    epochs: int = 12, clips_per_step: int = 4, lam: float = 1.0,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== rom_aware_layer2 (GG2) START ===")
    log(
        f"epochs={epochs} clips_per_step={clips_per_step} lam={lam} "
        f"T={TEMPORAL_T} N_KEYS={N_KEYS}"
    )

    cohort = build_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return {}

    results = run_loso_romaware(
        cohort, epochs=epochs, clips_per_step=clips_per_step, lam=lam,
    )

    out_main = OUT_DIR / "per_slot_results.json"
    out_main.write_text(json.dumps({
        "version": "rom_aware_layer2_v1",
        "intermediate": False,
        "epochs": results["epochs"],
        "clips_per_step": results["clips_per_step"],
        "lam": results["lam"],
        "subjects": results["subjects"],
        "metrics": results["metrics"],
        "overall": results["overall"],
        "fold_results_slim": [
            {
                "held_out": fr["held_out"],
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

    out_perclip = OUT_DIR / "per_clip_results.json"
    out_perclip.write_text(json.dumps({
        "version": "rom_aware_layer2_v1",
        "fold_per_clip": [
            {
                "held_out": fr["held_out"],
                "per_clip": fr["eval"]["per_clip"],
            }
            for fr in results["fold_results"]
        ],
    }, indent=2, default=str))
    log(f"wrote per-clip details to {out_perclip}")

    log("=== rom_aware_layer2 (GG2) DONE ===")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--clips-per-step", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()
    main(epochs=args.epochs, clips_per_step=args.clips_per_step, lam=args.lam)
