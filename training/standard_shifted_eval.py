"""Standard evaluation protocol (NEW default, replacing the old fixed-position test):
for each test case and each of N_REPEATS passes, draw a random point subsample AND a
random rigid-body shift (magnitude ~ Uniform(0, 30mm), random direction) applied
identically to the point cloud and the target label - physically equivalent to "the
same patient positioned slightly differently on the table" each time. This makes
positional robustness part of the HEADLINE metric rather than a separate ablation,
since a real deployed robot never gets the idealized fixed-position capture this
project's old benchmark implicitly assumed.

For each repeat, the model's prediction (made on the SHIFTED point cloud) is
un-shifted (shift subtracted back out) before averaging across repeats - since
raw_error = ||pred - shifted_gt|| = ||(pred - shift) - original_gt||, this is
mathematically identical to evaluating in the shifted frame, but lets every repeat's
estimate be averaged in the SAME (original) frame, exactly mirroring the existing
10-repeat point-subsampling-averaging convention used throughout this project.

Works for both the single-view PointNetRegressor (PLAX-only, A4C-only) and the
multi-head PointNetMultiHead (PLAX+A4C) models - set MODEL_KIND below.
"""
import json
import os
import sys

import numpy as np
import torch

MODEL_KIND = sys.argv[1] if len(sys.argv) > 1 else "multihead"  # "multihead" | "plax_only" | "a4c_only"
DATA_DIR = "<DATA_ROOT>/plax_curated157"
N_REPEATS = 10
SHIFT_MAX_MM = 30.0
SEED = 0

if MODEL_KIND == "multihead":
    import train_plax_a4c_pointnet_curated157_bigjitter as T
    CKPT = os.path.join(DATA_DIR, "checkpoints_plax_a4c_bigjitter", "plax_a4c_pointnet_best.pt")
    OUT = os.path.join(DATA_DIR, "checkpoints_plax_a4c_bigjitter", "test_predictions_standard_shifted_eval.json")
    model = T.PointNetMultiHead(n_views=len(T.VIEWS)).to(T.DEVICE)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    train_cases, val_cases, test_cases = T.load_split()
    labels = T.load_labels()
    test_ds = T.PlaxA4cDataset(test_cases, labels, augment=False)
    VIEWS = T.VIEWS
elif MODEL_KIND == "plax_only":
    import train_plax_pointnet_curated157_bigjitter as T
    CKPT = os.path.join(DATA_DIR, "checkpoints_plax_only_bigjitter", "plax_pointnet_best.pt")
    OUT = os.path.join(DATA_DIR, "checkpoints_plax_only_bigjitter", "test_predictions_standard_shifted_eval.json")
    model = T.PointNetRegressor().to(T.DEVICE)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    train_cases, val_cases, test_cases = T.load_split()
    labels = T.load_labels()
    test_ds = T.PlaxDataset(test_cases, labels, augment=False)
    VIEWS = ["PLAX"]
elif MODEL_KIND == "a4c_only":
    import train_a4c_pointnet_curated157_bigjitter as T
    CKPT = os.path.join(DATA_DIR, "checkpoints_a4c_bigjitter", "a4c_pointnet_best.pt")
    OUT = os.path.join(DATA_DIR, "checkpoints_a4c_bigjitter", "test_predictions_standard_shifted_eval.json")
    model = T.PointNetRegressor().to(T.DEVICE)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    train_cases, val_cases, test_cases = T.load_split()
    labels = T.load_labels()
    test_ds = T.A4cDataset(test_cases, labels, augment=False)
    VIEWS = ["A4C"]
else:
    raise ValueError(MODEL_KIND)

print(f"MODEL_KIND={MODEL_KIND}  CKPT={CKPT}  n_test={len(test_ds)}  views={VIEWS}")

rng = np.random.default_rng(SEED)
SCALE = 300.0

# case_id -> view -> list of un-shifted position estimates (mm), to average across repeats
by_case_estimates = {v: {} for v in VIEWS}
gt_by_case = {v: {} for v in VIEWS}

with torch.no_grad():
    for rep in range(N_REPEATS):
        for i in range(len(test_ds)):
            item = test_ds[i]
            if MODEL_KIND == "multihead":
                pts, target, case_id = item  # pts:(N,3) normalized, target:(6,) normalized
            else:
                pts, target, case_id = item  # pts:(N,3) normalized, target:(3,) normalized

            pts_mm = pts * SCALE  # back to mm for shifting
            mag = rng.uniform(0.0, SHIFT_MAX_MM)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            shift = (direction * mag).astype(np.float32)

            pts_shifted_mm = pts_mm + shift
            pts_t = torch.from_numpy((pts_shifted_mm / SCALE)[None].astype(np.float32)).to(T.DEVICE)
            pred = model(pts_t).cpu().numpy()[0] * SCALE  # multihead:(6,) or single:(3,)

            if MODEL_KIND == "multihead":
                for vi, v in enumerate(VIEWS):
                    pred_v = pred[vi * 3:(vi + 1) * 3]
                    unshifted_est = pred_v - shift  # bring estimate back to the ORIGINAL frame
                    by_case_estimates[v].setdefault(case_id, []).append(unshifted_est)
                    gt_by_case[v][case_id] = target[vi * 3:(vi + 1) * 3] * SCALE  # original (unshifted) GT
            else:
                v = VIEWS[0]
                unshifted_est = pred - shift
                by_case_estimates[v].setdefault(case_id, []).append(unshifted_est)
                gt_by_case[v][case_id] = target * SCALE

results = {v: {} for v in VIEWS}
for v in VIEWS:
    for case_id, estimates in by_case_estimates[v].items():
        avg_pred = np.mean(estimates, axis=0)
        gt = gt_by_case[v][case_id]
        err = float(np.linalg.norm(avg_pred - gt))
        results[v][case_id] = {
            "pred_local_mm": avg_pred.tolist(),
            "gt_local_mm": gt.tolist(),
            "err_mm": err,
        }

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {OUT}")
for v in VIEWS:
    errs = [r["err_mm"] for r in results[v].values()]
    print(f"[{v}] n={len(errs)} mean={np.mean(errs):.1f}mm median={np.median(errs):.1f}mm max={np.max(errs):.1f}mm")
