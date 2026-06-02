"""v10: Gradient boosting on the v9 phased feature set.

Ridge can only fit linear combinations of features. Gradient boosting can
capture nonlinear interactions (e.g., "phase_2 angle high AND coupling low
→ different prediction than phase_2 high AND coupling high"). For
biomechanical features that couple nonlinearly across the kinematic chain,
this should find signal ridge misses.

Risk: GBM overfits faster than ridge on small data (n=54 trials). Mitigate
with shallow trees (max_depth=3), strong L2 regularization, and early
stopping via LOSO CV.

Same training/eval protocol as ridge: LOSO CV, compare Pearson r + RMSE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.train_v9_phased import TARGETS, FEATURE_NAMES, build_dataset_v9
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
else:
    from .train_v9_phased import TARGETS, FEATURE_NAMES, build_dataset_v9
    from .train_multi_metric import VIEW_BUCKET_NAMES


RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def loso_gbm(X, y, subjects, *,
             max_depth=3, max_iter=80, l2_reg=1.0, learning_rate=0.05):
    """Leave-one-subject-out CV with HistGradientBoostingRegressor.

    Returns same dict shape as harness.train_regression_poc.loso_cv so we can
    compare apples-to-apples with ridge results.
    """
    subjects = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    fold_rmses = []
    for s in unique:
        train_mask = subjects != s
        test_mask = subjects == s
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            max_iter=max_iter,
            l2_regularization=l2_reg,
            learning_rate=learning_rate,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        all_pred[test_mask] = pred
        rmse = float(np.sqrt(np.mean((pred - y[test_mask]) ** 2)))
        fold_rmses.append((s, rmse, int(test_mask.sum())))
    overall_rmse = float(np.sqrt(np.mean((all_pred - y) ** 2)))
    overall_mae = float(np.mean(np.abs(all_pred - y)))
    overall_bias = float(np.mean(all_pred - y))
    sd_diff = float(np.std(all_pred - y, ddof=1)) if len(y) > 1 else float("nan")
    r = float(np.corrcoef(all_pred, y)[0, 1]) if y.std() > 0 else float("nan")
    return {
        "rmse": overall_rmse,
        "mae": overall_mae,
        "bias": overall_bias,
        "sd_diff": sd_diff,
        "pearson_r": r,
        "loa": (overall_bias - 1.96 * sd_diff, overall_bias + 1.96 * sd_diff),
        "fold_rmses": fold_rmses,
    }


def main():
    # Load v9 ridge results for direct comparison
    v9_path = RESULTS_DIR / "v9_phased_all_metrics.json"
    v9_r: dict = {}
    if v9_path.exists():
        with open(v9_path) as f:
            d = json.load(f)
        for tgt, td in d["models"].items():
            v9_r[tgt] = {
                v: {
                    "rmse": m["loso_cv_stats"]["rmse_deg"],
                    "r": m["loso_cv_stats"]["pearson_r"],
                }
                for v, m in td.items()
            }

    out = {
        "version": "1.0",
        "produced_by": "harness.train_v10_gbm",
        "produced_date": "2026-05-27",
        "approach": "v10 = v9 phased features + HistGradientBoosting (nonlinear)",
        "feature_names": list(FEATURE_NAMES),
        "gbm_params": {
            "max_depth": 3, "max_iter": 80, "l2_regularization": 1.0,
            "learning_rate": 0.05, "min_samples_leaf": 5,
        },
        "models": {},
    }

    print(
        f"\n{'TARGET':<22}{'VIEW':<22}{'n':>5}"
        f"{'v9_RMSE':>10}{'v9_r':>7}{'v10_RMSE':>10}{'v10_r':>7}{'Δr':>7}"
    )
    print("-" * 95)
    deploys_v10 = 0
    for target in TARGETS:
        out["models"][target] = {}
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            view = VIEW_BUCKET_NAMES[cam]
            X, y, subj, tasks, skipped = build_dataset_v9(target, cam)
            if len(y) < 20:
                continue
            cv = loso_gbm(X, y, subj)
            v9_view = v9_r.get(target, {}).get(view, {})
            v9_view_r = v9_view.get("r", float("nan"))
            v9_view_rmse = v9_view.get("rmse", float("nan"))
            d_r = cv["pearson_r"] - v9_view_r if np.isfinite(v9_view_r) else float("nan")
            if cv["pearson_r"] >= 0.10:
                deploys_v10 += 1
            print(
                f"  {target:<20}{view:<22}{len(y):>5d}"
                f"{v9_view_rmse:>9.2f}°{v9_view_r:>+6.2f}"
                f"{cv['rmse']:>9.2f}°{cv['pearson_r']:>+6.2f}{d_r:>+6.2f}"
            )

            out["models"][target][view] = {
                "camera_id": cam, "view_bucket": view, "target_metric": target,
                "n_train_trials": int(len(y)), "n_subjects": len(set(subj)),
                "feature_names": list(FEATURE_NAMES),
                "approach": "v10_gbm",
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"], "mae_deg": cv["mae"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                    "loa_95_lower_deg": cv["loa"][0],
                    "loa_95_upper_deg": cv["loa"][1],
                    "pearson_r": cv["pearson_r"],
                },
                "v9_comparison": {
                    "v9_rmse_deg": v9_view_rmse, "v9_pearson_r": v9_view_r,
                    "delta_r": d_r,
                },
            }
    print(f"\nv10 GBM deploys (r ≥ 0.10): {deploys_v10}")

    out_path = RESULTS_DIR / "v10_gbm_all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
