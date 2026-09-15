"""Fairer baseline per ICRA reviewer feedback: predicted position = observed point
cloud's centroid + a constant offset learned as the train-set mean of
(target_local_mm - centroid_of_that_case's_own_point_cloud). Unlike the pure
mean-position baseline, this baseline DOES use the observed point cloud (via its
centroid), so it can track a global rigid-body shift applied to the input - exactly the
synthetic perturbation the standard protocol (Sec 7.5.1) injects. Reported under both
the plain (no-perturbation) protocol and the standard shift-inclusive protocol, to let
the two be read side by side as the reviewer requested.
"""
import json
import os

import numpy as np

DATA_DIR = "<DATA_ROOT>/plax_curated157"
PC_DIR = "<DATA_ROOT>/plax_full187/pointclouds_local_frame_full187"
N_REPEATS = 10
SHIFT_MAX_MM = 30.0
SEED = 0

with open(DATA_DIR + "/plax_curated157_split.json") as f:
    split = json.load(f)
with open(DATA_DIR + "/local_frame_labels_curated157.json") as f:
    labels = json.load(f)["cases"]

train_cases = split["train_cases"]
test_cases = split["test_cases"]
VIEWS = ["PLAX", "A4C"]

pc_cache = {}


def get_pc(case_id):
    if case_id not in pc_cache:
        pc_cache[case_id] = np.load(os.path.join(PC_DIR, f"{case_id}_pointcloud_local_mm.npy")).astype(np.float64)
    return pc_cache[case_id]


# 1) learn offset per view from train cases
offsets = {}
for v in VIEWS:
    diffs = []
    for c in train_cases:
        views = labels[c]["views"]
        if v not in views:
            continue
        pc = get_pc(c)
        centroid = pc.mean(axis=0)
        target_mm = np.array(views[v]["position_m"], dtype=np.float64) * 1000.0
        diffs.append(target_mm - centroid)
    offsets[v] = np.mean(diffs, axis=0)
    print(f"offset[{v}] = {offsets[v]}  (n_train_with_view={len(diffs)})")

# 2) evaluate on test cases: no-perturbation and shift-inclusive
rng = np.random.default_rng(SEED)
results = {"no_perturbation": {v: {} for v in VIEWS}, "shift_inclusive": {v: {} for v in VIEWS}}

for v in VIEWS:
    for c in test_cases:
        views = labels[c]["views"]
        if v not in views:
            continue
        pc = get_pc(c)
        gt_mm = np.array(views[v]["position_m"], dtype=np.float64) * 1000.0
        centroid = pc.mean(axis=0)

        # no perturbation
        pred_np = centroid + offsets[v]
        err_np = float(np.linalg.norm(pred_np - gt_mm))
        results["no_perturbation"][v][c] = {"pred_local_mm": pred_np.tolist(), "gt_local_mm": gt_mm.tolist(), "err_mm": err_np}

        # shift-inclusive: average unshifted estimate over N_REPEATS random shifts
        estimates = []
        for rep in range(N_REPEATS):
            mag = rng.uniform(0.0, SHIFT_MAX_MM)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            shift = direction * mag
            shifted_centroid = centroid + shift  # observing the shifted point cloud
            pred_shifted = shifted_centroid + offsets[v]
            unshifted_est = pred_shifted - shift  # standard protocol convention
            estimates.append(unshifted_est)
        avg_pred = np.mean(estimates, axis=0)
        err_shift = float(np.linalg.norm(avg_pred - gt_mm))
        results["shift_inclusive"][v][c] = {"pred_local_mm": avg_pred.tolist(), "gt_local_mm": gt_mm.tolist(), "err_mm": err_shift}

OUT = os.path.join(DATA_DIR, "centroid_offset_baseline_results.json")
with open(OUT, "w") as f:
    json.dump({"offsets": {v: offsets[v].tolist() for v in VIEWS}, "results": results}, f, indent=2)
print(f"saved {OUT}")

for protocol in ["no_perturbation", "shift_inclusive"]:
    for v in VIEWS:
        errs = [r["err_mm"] for r in results[protocol][v].values()]
        print(f"[{protocol}] [{v}] n={len(errs)} mean={np.mean(errs):.1f}mm median={np.median(errs):.1f}mm max={np.max(errs):.1f}mm")
