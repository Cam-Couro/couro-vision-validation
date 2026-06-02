"""v9 phased — full per-(target × view) trainer, output to JSON for finalize."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
    from harness.train_v9_phased import (
        TARGETS, FEATURE_NAMES, build_dataset_v9,
    )
else:
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import VIEW_BUCKET_NAMES
    from .train_v9_phased import TARGETS, FEATURE_NAMES, build_dataset_v9

RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def main():
    out = {
        "version": "1.0",
        "produced_by": "harness.train_v9_full",
        "produced_date": "2026-05-27",
        "approach": "v9 phased (5 angles per metric across loading window)",
        "feature_names": list(FEATURE_NAMES),
        "models": {},
    }
    print(f"\n{'TARGET':<22}{'VIEW':<22}{'n':>5}{'cv_RMSE':>10}{'cv_r':>7}")
    print("-" * 70)
    for target in TARGETS:
        out["models"][target] = {}
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            view = VIEW_BUCKET_NAMES[cam]
            X, y, subj, tasks, skipped = build_dataset_v9(target, cam)
            if len(y) < 20:
                continue
            cv = loso_cv(X, y, subj, alpha=10.0)
            full = fit_ridge(X, y, alpha=10.0)
            naive_pred = X[:, 0]
            naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
            print(
                f"  {target:<20}{view:<22}{len(y):>5d}"
                f"{cv['rmse']:>9.2f}°{cv['pearson_r']:>+6.2f}"
            )
            out["models"][target][view] = {
                "camera_id": cam,
                "view_bucket": view,
                "target_metric": target,
                "n_train_trials": int(len(y)),
                "n_subjects": len(set(subj)),
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": full.feature_mean.tolist(),
                "feature_std": full.feature_std.tolist(),
                "weights_zscored": full.weights.tolist(),
                "intercept": full.bias,
                "naive_baseline_rmse": naive_rmse,
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"],
                    "mae_deg": cv["mae"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                    "loa_95_lower_deg": cv["loa"][0],
                    "loa_95_upper_deg": cv["loa"][1],
                    "pearson_r": cv["pearson_r"],
                },
            }
    out_path = RESULTS_DIR / "v9_phased_all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
