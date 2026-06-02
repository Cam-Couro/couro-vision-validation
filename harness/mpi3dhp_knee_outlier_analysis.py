"""Investigate why S4 and S8 knee-flexion correlations collapse on
MPI-INF-3DHP rear-camera (cam 7) when the rest of the n=8 cohort lands
around r=0.79 +/- 0.12.

Outputs three deliverables under
    data/mpi_inf_3dhp_knee_outliers/
        - REPORT.md                (markdown report)
        - knee_outlier_stats.json  (raw quantitative results)
        - overlays/*.png           (DWPose overlays for visual inspection)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/"
    "research-agent/multiview-validation"
).resolve()
DATA = ROOT / "data"
DW_DIR = DATA / "mpi_inf_3dhp_dwpose"
GT_DIR = DATA / "mpi_inf_3dhp_gt"
RAW_DIR = DATA / "mpi_inf_3dhp"
OUT_DIR = DATA / "mpi_inf_3dhp_knee_outliers"
OVERLAY_DIR = OUT_DIR / "overlays"

# Halpe-26 keypoint indices (matches mpi3dhp_pred_angles.py)
KP = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8,
    "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14,
    "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "L_big_toe": 20, "R_big_toe": 21,
    "L_small_toe": 22, "R_small_toe": 23,
    "L_heel": 24, "R_heel": 25,
}

# Halpe-26 skeleton edges for overlay drawing.
SKELETON = [
    (KP["neck"], KP["head_top"]),
    (KP["neck"], KP["L_sho"]), (KP["neck"], KP["R_sho"]),
    (KP["L_sho"], KP["L_elb"]), (KP["L_elb"], KP["L_wri"]),
    (KP["R_sho"], KP["R_elb"]), (KP["R_elb"], KP["R_wri"]),
    (KP["neck"], KP["hip_center"]),
    (KP["hip_center"], KP["L_hip"]), (KP["hip_center"], KP["R_hip"]),
    (KP["L_hip"], KP["L_kne"]), (KP["L_kne"], KP["L_ank"]),
    (KP["R_hip"], KP["R_kne"]), (KP["R_kne"], KP["R_ank"]),
    (KP["L_ank"], KP["L_big_toe"]), (KP["L_ank"], KP["L_heel"]),
    (KP["R_ank"], KP["R_big_toe"]), (KP["R_ank"], KP["R_heel"]),
]

# Subjects under investigation (S4, S8 = outliers) + baselines (S1, S7).
OUTLIER_SUBJECTS = ["S4", "S8"]
BASELINE_SUBJECTS = ["S1", "S7"]
ALL_SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]

METRICS = (
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
    "hip_adduction_r", "hip_adduction_l",
    "lumbar_extension",
)


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DWPoseTrack:
    """Per-frame keypoints, scores, and frame indices for one subject/cam."""

    xy: np.ndarray            # (n_frames, 26, 2)
    conf: np.ndarray          # (n_frames, 26)
    frame_idx: np.ndarray     # (n_frames,)


def load_dwpose(subject: str) -> DWPoseTrack:
    path = DW_DIR / f"{subject}_Seq2_cam7.json"
    with open(path) as f:
        d = json.load(f)
    frames = d["keypoints_sequence"]
    n = len(frames)
    xy = np.full((n, 26, 2), np.nan)
    conf = np.zeros((n, 26))
    fidx = np.zeros(n, dtype=int)
    for i, fr in enumerate(frames):
        kps = fr["keypoints"]
        scores = fr.get("keypoint_scores", [1.0] * 26)
        m = min(26, len(kps))
        for j in range(m):
            xy[i, j, 0], xy[i, j, 1] = float(kps[j][0]), float(kps[j][1])
            conf[i, j] = float(scores[j])
        fidx[i] = int(fr["frame_idx"])
    return DWPoseTrack(xy=xy, conf=conf, frame_idx=fidx)


# ----------------------------------------------------------------------
# Angle computation (mirrors mpi3dhp_pred_angles.compute_pred_angles)
# ----------------------------------------------------------------------


def _angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.where(n1 * n2 > 1e-9, n1 * n2, 1e-9)
    cos = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_angles(track: DWPoseTrack, conf_threshold: float = 0.3) -> dict:
    xy = track.xy.copy()
    xy[track.conf < conf_threshold] = np.nan

    def p(name: str) -> np.ndarray:
        return xy[:, KP[name], :]

    R_hip, R_kne, R_ank = p("R_hip"), p("R_kne"), p("R_ank")
    L_hip, L_kne, L_ank = p("L_hip"), p("L_kne"), p("L_ank")
    R_toe, L_toe = p("R_big_toe"), p("L_big_toe")
    neck, hip_c = p("neck"), p("hip_center")

    torso = neck - hip_c
    hip_flex_r = _angle_2d(torso, R_kne - R_hip)
    hip_flex_l = _angle_2d(torso, L_kne - L_hip)

    knee_angle_r = 180.0 - _angle_2d(R_hip - R_kne, R_ank - R_kne)
    knee_angle_l = 180.0 - _angle_2d(L_hip - L_kne, L_ank - L_kne)

    ankle_angle_r = 90.0 - _angle_2d(R_kne - R_ank, R_toe - R_ank)
    ankle_angle_l = 90.0 - _angle_2d(L_kne - L_ank, L_toe - L_ank)

    hip_add_r = np.degrees(np.arctan2(
        np.abs(R_kne[:, 0] - R_hip[:, 0]),
        np.abs(R_kne[:, 1] - R_hip[:, 1]) + 1e-9,
    ))
    hip_add_l = np.degrees(np.arctan2(
        np.abs(L_kne[:, 0] - L_hip[:, 0]),
        np.abs(L_kne[:, 1] - L_hip[:, 1]) + 1e-9,
    ))
    trunk = hip_c - neck
    lumbar_ext = np.degrees(np.arctan2(
        np.abs(trunk[:, 0]),
        np.abs(trunk[:, 1]) + 1e-9,
    ))

    return {
        "hip_flexion_r": hip_flex_r, "hip_flexion_l": hip_flex_l,
        "knee_angle_r": knee_angle_r, "knee_angle_l": knee_angle_l,
        "ankle_angle_r": ankle_angle_r, "ankle_angle_l": ankle_angle_l,
        "hip_adduction_r": hip_add_r, "hip_adduction_l": hip_add_l,
        "lumbar_extension": lumbar_ext,
    }


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------


def pearson_r(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan"), int(mask.sum())
    a2 = a[mask] - a[mask].mean()
    b2 = b[mask] - b[mask].mean()
    denom = np.linalg.norm(a2) * np.linalg.norm(b2)
    if denom < 1e-9:
        return float("nan"), int(mask.sum())
    return float((a2 * b2).sum() / denom), int(mask.sum())


def correlate_subject(
    subject: str, track: DWPoseTrack, conf_threshold: float
) -> dict:
    angles = compute_angles(track, conf_threshold=conf_threshold)
    gt_path = GT_DIR / f"{subject}_Seq2_angles.json"
    with open(gt_path) as f:
        gt = json.load(f)
    out = {}
    for m in METRICS:
        p = np.asarray(angles[m])
        g_all = np.asarray(gt[m])
        g = g_all[track.frame_idx]
        r, n = pearson_r(p, g)
        out[m] = {"r": r, "n": n}
    return out


# ----------------------------------------------------------------------
# Quantitative knee-keypoint stats
# ----------------------------------------------------------------------


def knee_stats(track: DWPoseTrack) -> dict:
    """Mean conf, median conf, frac low-conf, and frame-to-frame jitter
    (px std of first differences) for L and R knee keypoints."""
    out: dict = {}
    for side, idx in (("R_kne", KP["R_kne"]), ("L_kne", KP["L_kne"])):
        conf = track.conf[:, idx]
        xy = track.xy[:, idx, :]
        # Jitter on finite, in-bounds knee positions only
        mask = np.isfinite(xy).all(axis=1) & (conf >= 0.3)
        xy_m = xy.copy()
        xy_m[~mask] = np.nan
        diffs = np.diff(xy_m, axis=0)
        speed = np.linalg.norm(diffs, axis=1)
        speed_finite = speed[np.isfinite(speed)]
        out[side] = {
            "n_frames": int(track.xy.shape[0]),
            "mean_conf": float(np.nanmean(conf)),
            "median_conf": float(np.nanmedian(conf)),
            "frac_below_0.5": float((conf < 0.5).mean()),
            "frac_below_0.3": float((conf < 0.3).mean()),
            "jitter_px_std": float(np.nanstd(speed_finite))
            if speed_finite.size > 0 else float("nan"),
            "jitter_px_p95": float(np.nanpercentile(speed_finite, 95))
            if speed_finite.size > 0 else float("nan"),
        }
    return out


# ----------------------------------------------------------------------
# Overlay rendering
# ----------------------------------------------------------------------


def video_for(subject: str) -> Path:
    return RAW_DIR / subject / "Seq2" / "imageSequence" / "video_7.avi"


def pick_overlay_indices(track: DWPoseTrack, n_picks: int) -> list[int]:
    """Sample evenly across the sequence (uniform across frame_idx)."""
    n = len(track.frame_idx)
    if n == 0:
        return []
    # Skip the first/last 5% so we avoid startup/end artifacts.
    lo = int(0.05 * n)
    hi = int(0.95 * n)
    picks = np.linspace(lo, hi - 1, n_picks).astype(int)
    return [int(track.frame_idx[i]) for i in picks]


def draw_overlay(
    frame: np.ndarray,
    xy: np.ndarray,
    conf: np.ndarray,
    annotate_title: str,
) -> np.ndarray:
    img = frame.copy()
    # Skeleton edges in cyan (low conf = darker)
    for a, b in SKELETON:
        if a >= len(xy) or b >= len(xy):
            continue
        if conf[a] < 0.2 or conf[b] < 0.2:
            continue
        pa = (int(xy[a, 0]), int(xy[a, 1]))
        pb = (int(xy[b, 0]), int(xy[b, 1]))
        cv2.line(img, pa, pb, (220, 220, 220), 2, cv2.LINE_AA)

    # All keypoints in small gray
    for j in range(len(xy)):
        if conf[j] < 0.2 or not np.isfinite(xy[j]).all():
            continue
        cv2.circle(
            img, (int(xy[j, 0]), int(xy[j, 1])), 3, (180, 180, 180), -1,
        )

    # Highlight knees and adjacent joints
    highlights = [
        (KP["L_hip"], (255, 200, 0), "LHip"),
        (KP["R_hip"], (255, 140, 0), "RHip"),
        (KP["L_kne"], (0, 255, 0), "LKne"),
        (KP["R_kne"], (0, 200, 255), "RKne"),
        (KP["L_ank"], (255, 0, 255), "LAnk"),
        (KP["R_ank"], (255, 0, 180), "RAnk"),
    ]
    for j, color, label in highlights:
        if not np.isfinite(xy[j]).all():
            continue
        x, y = int(xy[j, 0]), int(xy[j, 1])
        cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 9, (0, 0, 0), 1, cv2.LINE_AA)
        # Annotate with confidence
        cv2.putText(
            img, f"{label} {conf[j]:.2f}",
            (x + 10, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    # Title banner
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(
        img, annotate_title, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def render_overlays(
    subject: str,
    track: DWPoseTrack,
    n_picks: int,
    label: str,
) -> list[dict]:
    """Render overlays for `n_picks` frames; return overlay metadata."""
    video_path = video_for(subject)
    if not video_path.exists():
        return []
    targets = pick_overlay_indices(track, n_picks)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fidx_to_row = {int(f): i for i, f in enumerate(track.frame_idx)}
    out_meta: list[dict] = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        row = fidx_to_row.get(t)
        if row is None:
            continue
        xy = track.xy[row]
        conf = track.conf[row]
        l_conf = conf[KP["L_kne"]]
        r_conf = conf[KP["R_kne"]]
        title = (
            f"{label} {subject} cam7 frame={t} "
            f"LKneConf={l_conf:.2f} RKneConf={r_conf:.2f}"
        )
        out_img = draw_overlay(frame, xy, conf, title)
        png_name = f"{subject}_cam7_f{t:06d}.png"
        cv2.imwrite(str(OVERLAY_DIR / png_name), out_img)
        out_meta.append({
            "subject": subject,
            "frame_idx": t,
            "L_kne_conf": float(l_conf),
            "R_kne_conf": float(r_conf),
            "png": png_name,
        })
    cap.release()
    return out_meta


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


def build_report(
    stats: dict,
    base_corr: dict,
    filt_corr_by_thr: dict,
    overlays: dict,
) -> str:
    lines: list[str] = []
    lines.append("# MPI-INF-3DHP Rear-Cam Knee Outlier Investigation")
    lines.append("")
    lines.append(
        "Subjects S4 and S8 collapse on knee-flexion correlation when "
        "Agent F ran rear-cam validation (cam 7, -159 deg)."
    )
    lines.append(
        "Cohort knee r = 0.79 +/- 0.12; S4 R=0.509 L=0.590, S8 L=0.587. "
        "This report tests whether the cause is recoverable (low-confidence "
        "DWPose detections) or geometric (occlusion / foreshortening)."
    )
    lines.append("")

    # ---- Quantitative table ----
    lines.append("## 1. Knee Keypoint Quantitative Stats")
    lines.append("")
    lines.append(
        "| Subject | Side | Mean conf | Median conf | "
        "Frac < 0.5 | Frac < 0.3 | Jitter px std | Jitter px p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for subj in ALL_SUBJECTS:
        for side in ("R_kne", "L_kne"):
            s = stats[subj][side]
            lines.append(
                f"| {subj} | {side} "
                f"| {s['mean_conf']:.3f} | {s['median_conf']:.3f} "
                f"| {s['frac_below_0.5']*100:.1f}% "
                f"| {s['frac_below_0.3']*100:.1f}% "
                f"| {s['jitter_px_std']:.2f} | {s['jitter_px_p95']:.2f} |"
            )
    lines.append("")

    # Cohort summary
    def cohort_mean(side: str, field: str, exclude: list[str]) -> float:
        vals = [
            stats[s][side][field]
            for s in ALL_SUBJECTS if s not in exclude
        ]
        return float(np.mean(vals))

    lines.append("### Cohort comparison (mean across subjects)")
    lines.append("")
    lines.append(
        "| Group | R_kne mean conf | L_kne mean conf | "
        "R_kne frac<0.5 | L_kne frac<0.5 | R_kne jitter | L_kne jitter |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    outliers = ["S4", "S8"]
    lines.append(
        f"| Outliers (S4,S8) "
        f"| {cohort_mean('R_kne','mean_conf',['S1','S2','S3','S5','S6','S7']):.3f} "
        f"| {cohort_mean('L_kne','mean_conf',['S1','S2','S3','S5','S6','S7']):.3f} "
        f"| {cohort_mean('R_kne','frac_below_0.5',['S1','S2','S3','S5','S6','S7'])*100:.1f}% "
        f"| {cohort_mean('L_kne','frac_below_0.5',['S1','S2','S3','S5','S6','S7'])*100:.1f}% "
        f"| {cohort_mean('R_kne','jitter_px_std',['S1','S2','S3','S5','S6','S7']):.2f} "
        f"| {cohort_mean('L_kne','jitter_px_std',['S1','S2','S3','S5','S6','S7']):.2f} |"
    )
    lines.append(
        f"| Strong (S1,S2,S5,S6,S7) "
        f"| {cohort_mean('R_kne','mean_conf',['S3','S4','S8']):.3f} "
        f"| {cohort_mean('L_kne','mean_conf',['S3','S4','S8']):.3f} "
        f"| {cohort_mean('R_kne','frac_below_0.5',['S3','S4','S8'])*100:.1f}% "
        f"| {cohort_mean('L_kne','frac_below_0.5',['S3','S4','S8'])*100:.1f}% "
        f"| {cohort_mean('R_kne','jitter_px_std',['S3','S4','S8']):.2f} "
        f"| {cohort_mean('L_kne','jitter_px_std',['S3','S4','S8']):.2f} |"
    )
    lines.append("")

    # ---- Confidence-filter sweep ----
    lines.append("## 2. Confidence-Filter Recovery Test")
    lines.append("")
    lines.append(
        "Recompute Pearson r for knee_angle after raising the DWPose "
        "confidence floor on knee keypoints. If outlier r recovers above "
        "0.80 at a reasonable retention rate, the issue is recoverable."
    )
    lines.append("")
    lines.append("| Subject | Side | r @ 0.3 (baseline) | r @ 0.5 | r @ 0.7 |")
    lines.append("|---|---|---|---|---|")
    for subj in OUTLIER_SUBJECTS + BASELINE_SUBJECTS:
        for side in ("knee_angle_r", "knee_angle_l"):
            base_r = base_corr[subj][side]["r"]
            r05 = filt_corr_by_thr[0.5][subj][side]["r"]
            r07 = filt_corr_by_thr[0.7][subj][side]["r"]
            lines.append(
                f"| {subj} | {side} "
                f"| {base_r:.3f} | {r05:.3f} | {r07:.3f} |"
            )
    lines.append("")

    # ---- Overlay listing ----
    lines.append("## 3. Visual Overlays")
    lines.append("")
    lines.append(
        "Overlays drawn on the rear-cam (cam 7) image with knees (LKne, "
        "RKne) highlighted alongside hip and ankle. Each frame is "
        "annotated with the DWPose confidence for that joint."
    )
    lines.append("")
    for subj in OUTLIER_SUBJECTS + BASELINE_SUBJECTS:
        lines.append(f"### {subj}")
        lines.append("")
        for meta in overlays.get(subj, []):
            lines.append(
                f"- `overlays/{meta['png']}` "
                f"(frame {meta['frame_idx']}, "
                f"L conf {meta['L_kne_conf']:.2f}, "
                f"R conf {meta['R_kne_conf']:.2f})"
            )
        lines.append("")

    # ---- Verdict ----
    lines.append("## 4. Verdict & Shippability")
    lines.append("")
    lines.append(
        "See the recovery table above. Conclusions and shippability "
        "recommendation are written by the investigator after reading "
        "this report."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Load tracks for all subjects (for quantitative stats).
    tracks: dict[str, DWPoseTrack] = {}
    for s in ALL_SUBJECTS:
        tracks[s] = load_dwpose(s)
        print(f"loaded {s}: n_frames={len(tracks[s].frame_idx)}")

    # 1. Per-subject knee keypoint stats.
    stats = {s: knee_stats(tracks[s]) for s in ALL_SUBJECTS}

    # 2. Baseline correlations (conf >= 0.3) and confidence-filtered.
    base_corr: dict = {}
    filt_corr_by_thr: dict = {0.5: {}, 0.7: {}}
    for s in OUTLIER_SUBJECTS + BASELINE_SUBJECTS:
        print(f"correlating {s} ...")
        base_corr[s] = correlate_subject(s, tracks[s], 0.3)
        for thr in (0.5, 0.7):
            filt_corr_by_thr[thr][s] = correlate_subject(s, tracks[s], thr)

    # 3. Overlays.
    overlays: dict[str, list[dict]] = {}
    for s in OUTLIER_SUBJECTS:
        overlays[s] = render_overlays(s, tracks[s], n_picks=8, label="OUTLIER")
        print(f"rendered {len(overlays[s])} overlays for {s}")
    for s in BASELINE_SUBJECTS:
        overlays[s] = render_overlays(s, tracks[s], n_picks=3, label="BASELINE")
        print(f"rendered {len(overlays[s])} overlays for {s}")

    # Persist raw data.
    raw = {
        "stats": stats,
        "base_corr": base_corr,
        "filt_corr_by_thr": {
            str(k): v for k, v in filt_corr_by_thr.items()
        },
        "overlays": overlays,
    }
    with open(OUT_DIR / "knee_outlier_stats.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Report.
    md = build_report(stats, base_corr, filt_corr_by_thr, overlays)
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write(md)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
