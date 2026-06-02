"""Print human-readable summary tables from keypoint_r_summary.json."""
import json
import numpy as np

with open('/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/mpi_inf_3dhp_keypoint_r/keypoint_r_summary.json') as f:
    s = json.load(f)

# Per-keypoint table for S1 cam0/3/7
print("=== S1 per-keypoint r_combined (cam0 / cam3 / cam7) ===")
runs_s1 = {r['cam_id']: r for r in s['runs'] if r['subject'] == 1}
kps = list(runs_s1[0]['per_keypoint'].keys())
print(f"{'kp':12s} {'cam0':>8s} {'cam3':>8s} {'cam7':>8s} {'mpi_joint':>12s} {'err_cam0_px':>12s}")
for kp in kps:
    r0 = runs_s1[0]['per_keypoint'][kp]
    r3 = runs_s1[3]['per_keypoint'][kp]
    r7 = runs_s1[7]['per_keypoint'][kp]
    print(f"{kp:12s} {r0['r_combined']:8.4f} {r3['r_combined']:8.4f} {r7['r_combined']:8.4f} {r0['mpi_joint']:>12s} {r0['mean_pixel_err']:12.2f}")

print()
print("=== Cross-subject rear-cam7 r_combined per keypoint ===")
header = f"{'kp':12s} " + " ".join(f"  S{x:1d}  " for x in range(1, 9)) + "   mean  median"
print(header)
rear = {r['subject']: r for r in s['runs'] if r['cam_id'] == 7}
for kp in kps:
    vals = []
    row = f"{kp:12s} "
    for subj in range(1, 9):
        v = rear[subj]['per_keypoint'][kp]['r_combined']
        vals.append(v)
        row += f" {v:6.4f}"
    mean = float(np.nanmean(vals))
    med = float(np.nanmedian(vals))
    row += f"   {mean:6.4f} {med:6.4f}"
    print(row)

# Headlines
print()
for cam, agg in s['by_camera'].items():
    print(f"{cam}: headline mean r = {agg['headline_mean_r']:.4f}, "
          f"median = {agg['headline_median_r']:.4f}, n_cells = {agg['n_cells']}")

# Cross-subject aggregate for cam7
print()
all_cam7 = [
    v
    for r in s['runs'] if r['cam_id'] == 7
    for v in (kv['r_combined'] for kv in r['per_keypoint'].values())
    if np.isfinite(v)
]
print(f"All-subject all-keypoint cam7 mean r = {np.mean(all_cam7):.4f}, "
      f"median = {np.median(all_cam7):.4f}, n = {len(all_cam7)}")

# Mean pixel error per subject (sanity vs 10.3 MPJPE claim)
print("\n=== Mean pixel error per subject (rear cam7) ===")
for subj in range(1, 9):
    errs = [
        kv['mean_pixel_err']
        for kv in rear[subj]['per_keypoint'].values()
        if np.isfinite(kv['mean_pixel_err'])
    ]
    print(f"S{subj}: mean pixel err over kp = {np.mean(errs):.2f} px")

print()
for cam in (0, 3, 7):
    errs = [
        kv['mean_pixel_err']
        for kv in runs_s1[cam]['per_keypoint'].values()
        if np.isfinite(kv['mean_pixel_err'])
    ]
    print(f"S1 cam{cam}: mean pixel err = {np.mean(errs):.2f} px")
