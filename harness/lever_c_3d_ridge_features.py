"""Lever (c) — 3D positions as Layer-3 ridge features (experiment).

The blend → Layer-3 integration report (v16) diagnosed why the view-aware blend
didn't promote slots at Layer 3:

    "Layer-3 ridges distill the trace into scalar features (ROM, at_contact,
     phased samples). When the lifter wins it smooths the trace shape; when
     it loses it adds high-frequency noise. Across the LOSO cohort the noise
     on losing clips degrades the ridge fit more than the smoothing on
     winning clips improves it."

That report listed three unexplored levers. Lever (c) is the most structural:

    "feed the lifter's 3D positions directly as features rather than
     collapsing through Layer 2 angles."

This script implements that. Instead of blending Couro and lifter angle traces
and feeding the blend into the existing v9 phased scalar features, we leave
the v9 features alone (so we don't degrade Couro-strong slots) and ADD
6 features summarizing 3D position information from VideoPose3D per clip:

    f3d_1: pelvis-to-knee Z-distance, max over clip       (forward stride depth)
    f3d_2: knee-to-knee X-distance, range over clip       (frontal knee separation)
    f3d_3: hip-hip width 3D, median over clip             (subject-size normalizer)
    f3d_4: right ankle Y range (foot lift)
    f3d_5: pelvis-shoulder Z-distance range (trunk lean)
    f3d_6: right hip Z-position SD                        (sagittal hip motion energy)

Rationale: these capture frontal-plane and depth signal the 2D-angle features
can't see, without trying to outsmart the existing scalar feature set. Six
features keeps the regularization regime compatible with ridge alpha=10.

Target slot for first experiment:
    hip_adduction_r / front_center
    Current v17: CCC 0.77, LoA half ±16.2°, bias +3.6°, n=22 subjects, 445 trials
    Gate to Good: LoA half < ±10°

The slot is in the 5 "Mid-promotability" Poor slots. Front-center is where the
blend report measured the lifter winning the most (+0.149 |r|). If 3D
features help anywhere, they should help here.

Runtime estimate (Saad machine, MPS-enabled M-series):
    - VideoPose3D inference on ~445 clips: ~10-20 min
    - LOSO ridge (22 subjects, 26 features, 445 trials): seconds
    - Total: ~20 min per slot

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.lever_c_3d_ridge_features

Outputs (only if any slot promotes):
    data/lever_c_3d_ridge/per_slot_results.json
    data/lever_c_3d_ridge/REPORT.md

Honesty constraints:
    - LOSO discipline preserved on every tier-change claim.
    - 3D features are ADDED to the v17 baseline X, not replacing anything.
      A null result means "3D features add no information beyond v9 scalars."
    - Bias is reported separately from CCC + LoA; do NOT post-hoc bias-correct
      then claim LoA improvement.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import train_v9_phased as v9
from harness import train_v12_combined as v12
from harness.biomech_validity_stats import (
    VIEW_TO_CAM,
    _loso_extract,
    _pearson,
    _ccc,
)
from harness.videopose3d import load_pretrained, lift_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
KP_DIR = DATA_ROOT / "couro_keypoints"
CHECKPOINT_PATH = REPO_ROOT / "models" / "videopose3d_h36m.bin"
OUT_DIR = DATA_ROOT / "lever_c_3d_ridge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lever_c")


# H36M-17 indices (matches harness/videopose3d.py)
H36M = {
    "pelvis": 0, "r_hip": 1, "r_knee": 2, "r_ankle": 3,
    "l_hip": 4, "l_knee": 5, "l_ankle": 6,
    "spine": 7, "neck": 8, "head": 10,
    "l_shoulder": 11, "r_shoulder": 14,
}


@dataclass(frozen=True)
class Slot:
    target: str
    view: str
    builder: str  # "v9_phased" | "v12_combined"


TARGET_SLOTS = (
    # OpenCap-only baseline (v9_phased) is an apples-to-apples comparison —
    # both baseline and lever_c train on the same n=32 OpenCap rows.
    # The v12_combined "445-row" v17 baseline includes 413 ASPset rows where
    # we don't yet have 3D features, so comparing v12_combined-with-3D-on-OC-only
    # against v12_combined-without-3D-on-all-445 is unfair. v9_phased is fair.
    Slot("hip_adduction_r", "front_center", "v9_phased"),
    Slot("hip_adduction_r", "front_oblique_right", "v9_phased"),
    Slot("knee_angle_r", "front_oblique_left", "v9_phased"),
    Slot("hip_flexion_r", "front_oblique_right", "v9_phased"),
    Slot("hip_flexion_r", "side_left", "v9_phased"),
)


def extract_3d_features(kp_2d_halpe26: np.ndarray, width: int, height: int,
                        model) -> np.ndarray:
    """Run VideoPose3D and compute 6 per-clip 3D summary features."""
    joints_3d = lift_sequence(model, kp_2d_halpe26, width, height)  # (T, 17, 3)

    pelvis = joints_3d[:, H36M["pelvis"]]
    r_hip = joints_3d[:, H36M["r_hip"]]
    l_hip = joints_3d[:, H36M["l_hip"]]
    r_knee = joints_3d[:, H36M["r_knee"]]
    l_knee = joints_3d[:, H36M["l_knee"]]
    r_ankle = joints_3d[:, H36M["r_ankle"]]
    r_shoulder = joints_3d[:, H36M["r_shoulder"]]

    f3d_1 = float(np.max(np.abs(r_knee[:, 2] - pelvis[:, 2])))           # forward stride depth (max abs)
    f3d_2 = float(np.ptp(r_knee[:, 0] - l_knee[:, 0]))                   # frontal knee separation range
    f3d_3 = float(np.median(np.linalg.norm(r_hip - l_hip, axis=1)))      # hip-hip width median (subject size proxy)
    f3d_4 = float(np.ptp(r_ankle[:, 1]))                                  # foot lift range
    f3d_5 = float(np.ptp(np.linalg.norm(r_shoulder - pelvis, axis=1)))    # trunk lean energy
    f3d_6 = float(np.std(r_hip[:, 2]))                                    # sagittal hip motion energy

    return np.array([f3d_1, f3d_2, f3d_3, f3d_4, f3d_5, f3d_6], dtype=np.float64)


def load_kp_for_clip(kp_path: Path) -> tuple[np.ndarray, int, int]:
    """Load Halpe-26 keypoint sequence + image dims from Couro keypoint JSON."""
    blob = json.loads(kp_path.read_text())
    meta = blob["video_metadata"]
    width = int(meta["width"])
    height = int(meta["height"])
    seq = blob["keypoints_sequence"]
    kp = np.zeros((len(seq), 26, 2), dtype=np.float64)
    for i, frame in enumerate(seq):
        kp[i] = np.asarray(frame["keypoints"], dtype=np.float64)
    return kp, width, height


def find_clip_kp_path(subject: str, task: str, cam: str) -> Path | None:
    candidate = KP_DIR / f"{subject}_{task}_{cam}.json"
    if candidate.exists():
        return candidate
    return None


def build_augmented_dataset(slot: Slot, model):
    """Wrap the existing builder; for OpenCap rows, append 6 3D features.

    ASPset rows get 3D features computed from their separate keypoint dir, if
    available. If not, ASPset rows are dropped (clean signal-only experiment).
    """
    cam = VIEW_TO_CAM.get(slot.view)
    if cam is None:
        raise RuntimeError(f"View {slot.view} not in VIEW_TO_CAM")

    if slot.builder == "v9_phased":
        v9.KP_DIR = KP_DIR
        X_base, y, subjects, _, _ = v9.build_dataset_v9(slot.target, cam)
        sources = ["opencap"] * len(y)
    elif slot.builder == "v12_combined":
        v9.KP_DIR = KP_DIR
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
        ds = v12.build_combined_dataset(slot.target, slot.view)
        if ds is None:
            raise RuntimeError(f"Builder returned None for {slot}")
        X_base, y, subjects, sources = ds
        X_base = np.asarray(X_base, dtype=np.float64)
    else:
        raise RuntimeError(f"Unknown builder {slot.builder}")

    # Now reconstruct per-clip identifiers so we can find each clip's keypoint
    # file. The builders' subject lists are post-prefix ("opencap_subjectN" or
    # raw ASPset subject); we only run 3D for OpenCap rows where we have
    # keypoints in couro_keypoints/.
    n = len(y)
    augment = np.full((n, 6), np.nan, dtype=np.float64)

    keep = np.zeros(n, dtype=bool)
    log.info("Computing 3D features for %d trials (slot: %s/%s)…",
             n, slot.target, slot.view)

    # Walk the OpenCap keypoint dir to align rows by index. The v9 builder
    # iterates sorted(KP_DIR.glob(f"*_{cam}.json")) and yields DJ tasks; we
    # mirror that traversal here to recover (subject, task) per row.
    rows = []
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        rows.append((subject, task, kp_path))

    if slot.builder == "v9_phased":
        opencap_rows = rows[: n]
    else:
        opencap_rows = rows[: len([s for s in sources if s == "opencap"])]

    for i, (subject, task, kp_path) in enumerate(opencap_rows):
        try:
            kp_2d, w, h = load_kp_for_clip(kp_path)
            feats = extract_3d_features(kp_2d, w, h, model)
            augment[i] = feats
            keep[i] = True
        except Exception as e:  # noqa: BLE001
            log.warning("3D feature extraction failed for %s_%s_%s: %s",
                        subject, task, cam, e)

    # For ASPset rows we leave nan and drop later — keeps the experiment
    # signal-only on OpenCap. A future extension could mirror this for ASPset.
    aspset_idxs = [i for i, s in enumerate(sources) if s != "opencap"]
    log.info("Dropping %d ASPset rows (no 3D for this experiment)", len(aspset_idxs))

    final_keep = keep & ~np.isin(np.arange(n), aspset_idxs)
    X_base = X_base[final_keep]
    augment = augment[final_keep]
    y = np.asarray(y)[final_keep]
    subjects = [s for i, s in enumerate(subjects) if final_keep[i]]

    X_aug = np.concatenate([X_base, augment], axis=1)
    log.info("Augmented dataset: %d rows × %d features (was %d)",
             len(y), X_aug.shape[1], X_base.shape[1])
    return X_base, X_aug, y, subjects


def loso_score(X: np.ndarray, y: np.ndarray, subjects: list[str],
               alpha: float = 10.0) -> dict:
    p, o, _ = _loso_extract(X, y, subjects, alpha=alpha)
    valid = np.isfinite(p) & np.isfinite(o)
    p = p[valid]; o = o[valid]
    r = _pearson(p, o)
    ccc = _ccc(p, o)
    diff = p - o
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_half = 1.96 * sd_diff
    mae = float(np.mean(np.abs(diff)))
    return {
        "n_subjects": len(set(subjects)),
        "n_trials": int(len(y)),
        "pearson_r": float(r),
        "ccc": float(ccc),
        "mean_bias": bias,
        "loa_half_width": float(loa_half),
        "mae": mae,
    }


def tier(ccc: float, loa: float) -> str:
    if ccc > 0.75 and loa < 5.0:
        return "Excellent"
    if ccc > 0.60 and loa < 10.0:
        return "Good"
    if ccc > 0.40 and loa < 15.0:
        return "Moderate"
    return "Poor"


def run() -> dict:
    log.info("Loading VideoPose3D from %s", CHECKPOINT_PATH)
    # MPS path has a known input/weight device-mismatch in lift_sequence (input
    # tensor created on CPU, model on MPS). CPU is robust and fast enough at the
    # 445-clip scale of this experiment.
    device = "cpu"
    log.info("Using device: %s", device)
    model = load_pretrained(CHECKPOINT_PATH, device=device)

    results = []
    for slot in TARGET_SLOTS:
        log.info("=== Slot %s/%s (builder %s) ===",
                 slot.target, slot.view, slot.builder)
        X_base, X_aug, y, subjects = build_augmented_dataset(slot, model)

        base_stats = loso_score(X_base, y, subjects)
        aug_stats = loso_score(X_aug, y, subjects)

        result = {
            "slot": f"{slot.target}/{slot.view}",
            "builder": slot.builder,
            "baseline_v17_scalars_only": base_stats,
            "lever_c_with_3d_features": aug_stats,
            "delta_ccc": aug_stats["ccc"] - base_stats["ccc"],
            "delta_loa_half": aug_stats["loa_half_width"] - base_stats["loa_half_width"],
            "baseline_tier": tier(base_stats["ccc"], base_stats["loa_half_width"]),
            "lever_c_tier": tier(aug_stats["ccc"], aug_stats["loa_half_width"]),
        }
        results.append(result)
        log.info("Baseline tier: %s (CCC=%.3f, LoA=%.2f°)",
                 result["baseline_tier"], base_stats["ccc"],
                 base_stats["loa_half_width"])
        log.info("Lever (c) tier: %s (CCC=%.3f, LoA=%.2f°)",
                 result["lever_c_tier"], aug_stats["ccc"],
                 aug_stats["loa_half_width"])
        log.info("Delta: CCC %+.3f, LoA %+.2f°",
                 result["delta_ccc"], result["delta_loa_half"])

    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps(
        {"version": "lever_c_v1", "slots": results}, indent=2))
    log.info("Wrote results to %s", out_path)
    return {"slots": results}


if __name__ == "__main__":
    run()
