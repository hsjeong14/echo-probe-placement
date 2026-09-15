"""v3 experiment (cheap-changes bundle, PointNet architecture unchanged):
1) Residual/offset regression: predict (target - train_mean_per_view) instead of the
   absolute position. Network outputting 0 exactly reproduces the frame-anchored
   mean-position floor, so training can't do WORSE than that baseline by construction
   (up to optimization noise), and focuses capacity on the case-specific residual.
2) Huber loss instead of MSE (less prone to being dominated by/averaging toward
   outliers).
3) N_POINTS 2048->4096, sampled from a voxel-downsampled (3mm) point cloud instead of
   the raw dense capture, so training sees a more spatially-uniform sample of the
   thorax rather than one biased toward densely-sampled flat regions.
Everything else (encoder, trunk, heads, 25mm translation jitter, yaw jitter, LR
schedule, epochs, multi-seed protocol) is identical to run_multiseed.py, so results
are directly comparable.
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "<DATA_ROOT>/plax_curated157"
PC_DIR = "<DATA_ROOT>/plax_full187/pointclouds_local_frame_full187_voxel3mm"
CKPT_DIR_BASE = os.path.join(DATA_DIR, "checkpoints_v3_residual")
os.makedirs(CKPT_DIR_BASE, exist_ok=True)

VIEWS = ["PLAX", "A4C"]
N_POINTS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JITTER_STD_MM = 25.0
SCALE = 300.0
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split():
    with open(DATA_DIR + "/plax_curated157_split.json") as f:
        s = json.load(f)
    return s["train_cases"], s["val_cases"], s["test_cases"]


def load_labels():
    with open(DATA_DIR + "/local_frame_labels_curated157.json") as f:
        d = json.load(f)
    return d["cases"]


class PlaxA4cDatasetResidual(Dataset):
    def __init__(self, case_ids, labels, mean_offset, augment=False):
        self.items = []
        for c in case_ids:
            views = labels[c]["views"]
            if not all(v in views for v in VIEWS):
                continue
            pc_path = os.path.join(PC_DIR, f"{c}_pointcloud_local_mm.npy")
            if not os.path.exists(pc_path):
                continue
            targets_mm = {v: np.array(views[v]["position_m"], dtype=np.float32) * 1000.0 for v in VIEWS}
            self.items.append((c, pc_path, targets_mm))
        self.augment = augment
        self.mean_offset = mean_offset  # dict view -> mm vector (train-set mean position)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        case_id, pc_path, targets_mm = self.items[idx]
        pts = np.load(pc_path).astype(np.float32)
        if len(pts) >= N_POINTS:
            sel = np.random.choice(len(pts), N_POINTS, replace=False)
        else:
            sel = np.random.choice(len(pts), N_POINTS, replace=True)
        pts = pts[sel].copy()
        targets = {v: t.copy() for v, t in targets_mm.items()}

        if self.augment:
            theta = np.random.uniform(-0.15, 0.15)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ R.T
            shift = np.random.normal(0, JITTER_STD_MM, size=3).astype(np.float32)
            pts += np.random.normal(0, 1.5, size=pts.shape).astype(np.float32)
            pts += shift
            for v in VIEWS:
                targets[v] = targets[v] @ R.T + shift

        # residual target: subtract the (also rotated+shifted, for augmented cases) mean
        residual = {}
        for v in VIEWS:
            m = self.mean_offset[v]
            if self.augment:
                m = m @ R.T + shift
            residual[v] = (targets[v] - m) / SCALE

        target_vec = np.concatenate([residual[v] for v in VIEWS])
        return pts / SCALE, target_vec, case_id


class PointNetMultiHead(nn.Module):
    def __init__(self, n_views=2):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)) for _ in range(n_views)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = torch.max(x, dim=2)[0]
        feat = self.trunk(x)
        return torch.cat([head(feat) for head in self.heads], dim=1)


def eval_protocols(model, test_ds, mean_offset, n_repeats=10, shift_max_mm=30.0, seed=0):
    model.eval()
    # no-perturbation
    no_pert = {v: {} for v in VIEWS}
    with torch.no_grad():
        for i in range(len(test_ds)):
            pts, target_res, case_id = test_ds[i]
            pts_t = torch.from_numpy(pts[None].astype(np.float32)).to(DEVICE)
            pred_res = model(pts_t).cpu().numpy()[0] * SCALE
            for vi, v in enumerate(VIEWS):
                pred_abs = pred_res[vi*3:(vi+1)*3] + mean_offset[v]
                gt_abs = target_res[vi*3:(vi+1)*3] * SCALE + mean_offset[v]
                no_pert[v][case_id] = {"pred_local_mm": pred_abs.tolist(), "gt_local_mm": gt_abs.tolist(),
                                        "err_mm": float(np.linalg.norm(pred_abs - gt_abs))}

    # shift-inclusive
    rng = np.random.default_rng(seed + 5000)
    by_case_estimates = {v: {} for v in VIEWS}
    gt_by_case = {v: {} for v in VIEWS}
    with torch.no_grad():
        for rep in range(n_repeats):
            for i in range(len(test_ds)):
                pts, target_res, case_id = test_ds[i]
                pts_mm = pts * SCALE
                mag = rng.uniform(0.0, shift_max_mm)
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                shift = (direction * mag).astype(np.float32)
                pts_shifted_mm = pts_mm + shift
                pts_t = torch.from_numpy((pts_shifted_mm / SCALE)[None].astype(np.float32)).to(DEVICE)
                pred_res = model(pts_t).cpu().numpy()[0] * SCALE
                for vi, v in enumerate(VIEWS):
                    pred_abs = pred_res[vi*3:(vi+1)*3] + mean_offset[v] + shift  # mean_offset frame moved with shift too
                    unshifted = pred_abs - shift
                    by_case_estimates[v].setdefault(case_id, []).append(unshifted)
                    gt_by_case[v][case_id] = target_res[vi*3:(vi+1)*3] * SCALE + mean_offset[v]
    shift_incl = {v: {} for v in VIEWS}
    for v in VIEWS:
        for case_id, estimates in by_case_estimates[v].items():
            avg_pred = np.mean(estimates, axis=0)
            gt = gt_by_case[v][case_id]
            shift_incl[v][case_id] = {"pred_local_mm": avg_pred.tolist(), "gt_local_mm": gt.tolist(),
                                        "err_mm": float(np.linalg.norm(avg_pred - gt))}
    return no_pert, shift_incl


def run_one(seed):
    set_seed(seed)
    train_cases, val_cases, test_cases = load_split()
    labels = load_labels()

    # compute train-set mean position per view (in mm) - the "frame-anchored baseline"
    mean_offset = {}
    for v in VIEWS:
        positions = []
        for c in train_cases:
            views = labels[c]["views"]
            if v in views:
                positions.append(np.array(views[v]["position_m"], dtype=np.float32) * 1000.0)
        mean_offset[v] = np.mean(positions, axis=0)

    train_ds = PlaxA4cDatasetResidual(train_cases, labels, mean_offset, augment=True)
    val_ds = PlaxA4cDatasetResidual(val_cases, labels, mean_offset, augment=False)
    test_ds = PlaxA4cDatasetResidual(test_cases, labels, mean_offset, augment=False)
    print(f"[seed={seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = PointNetMultiHead(n_views=len(VIEWS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = nn.HuberLoss(delta=1.0)  # operating on /SCALE-normalized targets

    ckpt_dir = os.path.join(CKPT_DIR_BASE, f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_best.pt")

    best_val_mm = float("inf")
    for epoch in range(300):
        model.train()
        for pts, target, _ in train_loader:
            pts, target = pts.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            pred = model(pts)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_errs = {v: [] for v in VIEWS}
        with torch.no_grad():
            for pts, target, _ in val_loader:
                pts, target = pts.to(DEVICE), target.to(DEVICE)
                pred = model(pts)
                for vi, v in enumerate(VIEWS):
                    p = pred[:, vi*3:(vi+1)*3]
                    t = target[:, vi*3:(vi+1)*3]
                    val_errs[v].extend(torch.norm((p - t) * SCALE, dim=1).cpu().numpy().tolist())
        val_mm = float(np.mean([np.mean(val_errs[v]) for v in VIEWS]))
        if val_mm < best_val_mm:
            best_val_mm = val_mm
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    no_pert, shift_incl = eval_protocols(model, test_ds, mean_offset, seed=seed)

    out = {"seed": seed, "best_val_mm": best_val_mm, "no_perturbation": no_pert, "shift_inclusive": shift_incl}
    with open(os.path.join(ckpt_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    summary_str = f"[v3_residual seed={seed}] best_val={best_val_mm:.1f}  "
    for v in VIEWS:
        np_errs = [r["err_mm"] for r in no_pert[v].values()]
        sh_errs = [r["err_mm"] for r in shift_incl[v].values()]
        summary_str += f"{v}: no_pert={np.mean(np_errs):.1f}/{np.median(np_errs):.1f} shift={np.mean(sh_errs):.1f}/{np.median(sh_errs):.1f}  "
    print(summary_str)


if __name__ == "__main__":
    for seed in SEEDS:
        run_one(seed)
    print("ALL DONE v3")
