"""Variant: average the per-repeat ERROR (not the corrected prediction vector) across
repeats, for comparison against v1's averaged-prediction convention - to see which
convention matches this project's previously-reported Table 1 baseline numbers
(27.6mm PLAX / 34.3mm A4C)."""
import json
import os

import numpy as np

DATA_DIR = "<DATA_ROOT>/plax_curated157"
N_REPEATS = 10
SHIFT_MAX_MM = 30.0
SEED = 0

import train_plax_a4c_pointnet_curated157_bigjitter as T

train_cases, val_cases, test_cases = T.load_split()
labels = T.load_labels()
test_ds = T.PlaxA4cDataset(test_cases, labels, augment=False)
VIEWS = T.VIEWS
SCALE = 300.0

baseline_mm = {}
for v in VIEWS:
    positions = []
    for c in train_cases:
        views = labels[c]["views"]
        if v in views:
            positions.append(np.array(views[v]["position_m"], dtype=np.float64) * 1000.0)
    baseline_mm[v] = np.mean(positions, axis=0).astype(np.float32)

rng = np.random.default_rng(SEED)

by_case_errs = {v: {} for v in VIEWS}
by_case_pred_axes = {v: {} for v in VIEWS}  # for per-axis MAE: average |pred-shift - gt| per axis per repeat
gt_by_case = {v: {} for v in VIEWS}

for rep in range(N_REPEATS):
    for i in range(len(test_ds)):
        item = test_ds[i]
        pts, target, case_id = item

        mag = rng.uniform(0.0, SHIFT_MAX_MM)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        shift = (direction * mag).astype(np.float32)

        for vi, v in enumerate(VIEWS):
            pred_v = baseline_mm[v]
            unshifted_est = pred_v - shift
            gt = target[vi * 3:(vi + 1) * 3] * SCALE
            err = float(np.linalg.norm(unshifted_est - gt))
            by_case_errs[v].setdefault(case_id, []).append(err)
            by_case_pred_axes[v].setdefault(case_id, []).append(np.abs(unshifted_est - gt))
            gt_by_case[v][case_id] = gt

results = {v: {} for v in VIEWS}
for v in VIEWS:
    for case_id, errs in by_case_errs[v].items():
        mean_err = float(np.mean(errs))
        axis_mae = np.mean(by_case_pred_axes[v][case_id], axis=0).tolist()
        results[v][case_id] = {
            "err_mm": mean_err,
            "axis_abs_err_mm": axis_mae,
        }

OUT = os.path.join(DATA_DIR, "baseline_standard_shifted_eval_v2_avgerr.json")
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {OUT}")
for v in VIEWS:
    errs = [r["err_mm"] for r in results[v].values()]
    print(f"[{v}] n={len(errs)} mean={np.mean(errs):.1f}mm median={np.median(errs):.1f}mm max={np.max(errs):.1f}mm")
