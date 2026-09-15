"""Free experiment: ensemble the 5 already-trained multihead bigjitter checkpoints
(seeds 0-4 from run_multiseed.py) by averaging their per-case predictions, under both
the no-perturbation and standard shift-inclusive protocols. Zero additional training
cost - just re-uses existing checkpoints.
"""
import json
import os

import numpy as np
import torch

DATA_DIR = "<DATA_ROOT>/plax_curated157"
SEEDS = [0, 1, 2, 3, 4]
N_REPEATS = 10
SHIFT_MAX_MM = 30.0
SEED_EVAL = 12345

import train_plax_a4c_pointnet_curated157_bigjitter as T

train_cases, val_cases, test_cases = T.load_split()
labels = T.load_labels()
test_ds = T.PlaxA4cDataset(test_cases, labels, augment=False)
VIEWS = T.VIEWS
SCALE = 300.0

models = []
for seed in SEEDS:
    m = T.PointNetMultiHead(n_views=len(VIEWS)).to(T.DEVICE)
    ckpt = os.path.join(DATA_DIR, f"checkpoints_plax_a4c_bigjitter_seed{seed}", "plax_a4c_pointnet_best.pt")
    m.load_state_dict(torch.load(ckpt))
    m.eval()
    models.append(m)
print(f"loaded {len(models)} models")


def ensemble_predict(pts_t):
    preds = [m(pts_t).cpu().numpy()[0] for m in models]
    return np.mean(preds, axis=0) * SCALE


# no-perturbation
no_pert = {v: {} for v in VIEWS}
with torch.no_grad():
    for i in range(len(test_ds)):
        pts, target, case_id = test_ds[i]
        pts_t = torch.from_numpy(pts[None].astype(np.float32)).to(T.DEVICE)
        pred = ensemble_predict(pts_t)
        for vi, v in enumerate(VIEWS):
            pred_v = pred[vi*3:(vi+1)*3]
            gt_v = target[vi*3:(vi+1)*3] * SCALE
            no_pert[v][case_id] = {"pred_local_mm": pred_v.tolist(), "gt_local_mm": gt_v.tolist(),
                                     "err_mm": float(np.linalg.norm(pred_v - gt_v))}

# shift-inclusive
rng = np.random.default_rng(SEED_EVAL)
by_case_estimates = {v: {} for v in VIEWS}
gt_by_case = {v: {} for v in VIEWS}
with torch.no_grad():
    for rep in range(N_REPEATS):
        for i in range(len(test_ds)):
            pts, target, case_id = test_ds[i]
            pts_mm = pts * SCALE
            mag = rng.uniform(0.0, SHIFT_MAX_MM)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            shift = (direction * mag).astype(np.float32)
            pts_shifted_mm = pts_mm + shift
            pts_t = torch.from_numpy((pts_shifted_mm / SCALE)[None].astype(np.float32)).to(T.DEVICE)
            pred = ensemble_predict(pts_t)
            for vi, v in enumerate(VIEWS):
                pred_v = pred[vi*3:(vi+1)*3]
                unshifted = pred_v - shift
                by_case_estimates[v].setdefault(case_id, []).append(unshifted)
                gt_by_case[v][case_id] = target[vi*3:(vi+1)*3] * SCALE

shift_incl = {v: {} for v in VIEWS}
for v in VIEWS:
    for case_id, estimates in by_case_estimates[v].items():
        avg_pred = np.mean(estimates, axis=0)
        gt = gt_by_case[v][case_id]
        shift_incl[v][case_id] = {"pred_local_mm": avg_pred.tolist(), "gt_local_mm": gt.tolist(),
                                    "err_mm": float(np.linalg.norm(avg_pred - gt))}

out = {"no_perturbation": no_pert, "shift_inclusive": shift_incl}
OUT = os.path.join(DATA_DIR, "ensemble_5seed_multihead_results.json")
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved {OUT}")

for protocol_name, protocol in [("no_perturbation", no_pert), ("shift_inclusive", shift_incl)]:
    for v in VIEWS:
        errs = [r["err_mm"] for r in protocol[v].values()]
        print(f"[{protocol_name}] [{v}] n={len(errs)} mean={np.mean(errs):.1f}mm median={np.median(errs):.1f}mm max={np.max(errs):.1f}mm")
