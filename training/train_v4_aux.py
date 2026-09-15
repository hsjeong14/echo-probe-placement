"""v4 = v3 (residual regression + Huber loss + voxel-downsampled 4096pt sampling)
PLUS two free auxiliary supervision targets extracted from TotalSegmentator masks
(no extra annotation cost): heart centroid and sternum lower tip, in the same
body-aligned local frame. Two extra output heads predict these; their loss is added
to the main PLAX/A4C loss at a reduced weight (0.3x) so it regularizes the shared
trunk toward "read internal anatomy from the surface" without dominating the primary
task. Auxiliary targets undergo the IDENTICAL yaw/shift augmentation as the point
cloud and PLAX/A4C targets (same local frame), and use the same residual-from-
train-mean convention for consistent loss scale.
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "<DATA_ROOT>/plax_curated157"
PC_DIR = "<DATA_ROOT>/plax_full187/pointclouds_local_frame_full187_voxel3mm"
CKPT_DIR_BASE = os.path.join(DATA_DIR, "checkpoints_v4_aux")
os.makedirs(CKPT_DIR_BASE, exist_ok=True)

MAIN_VIEWS = ["PLAX", "A4C"]
AUX_KEYS = ["heart_center", "sternum_tip"]
ALL_VIEWS = MAIN_VIEWS + AUX_KEYS
N_POINTS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JITTER_STD_MM = 25.0
SCALE = 300.0
AUX_LOSS_WEIGHT = 0.3
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


def load_aux():
    with open(DATA_DIR + "/anatomical_aux_targets_local_mm.json") as f:
        return json.load(f)


class PlaxA4cAuxDataset(Dataset):
    def __init__(self, case_ids, labels, aux, mean_offset, augment=False):
        self.items = []
        for c in case_ids:
            views = labels[c]["views"]
            if not all(v in views for v in MAIN_VIEWS):
                continue
            if c not in aux:
                continue
            pc_path = os.path.join(PC_DIR, f"{c}_pointcloud_local_mm.npy")
            if not os.path.exists(pc_path):
                continue
            targets_mm = {v: np.array(views[v]["position_m"], dtype=np.float32) * 1000.0 for v in MAIN_VIEWS}
            targets_mm["heart_center"] = np.array(aux[c]["heart_center_local_mm"], dtype=np.float32)
            targets_mm["sternum_tip"] = np.array(aux[c]["sternum_lower_tip_local_mm"], dtype=np.float32)
            self.items.append((c, pc_path, targets_mm))
        self.augment = augment
        self.mean_offset = mean_offset

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
            for v in ALL_VIEWS:
                targets[v] = targets[v] @ R.T + shift

        residual = {}
        for v in ALL_VIEWS:
            m = self.mean_offset[v]
            if self.augment:
                m = m @ R.T + shift
            residual[v] = (targets[v] - m) / SCALE

        target_vec = np.concatenate([residual[v] for v in ALL_VIEWS])  # (12,)
        return pts / SCALE, target_vec, case_id


class PointNetMultiHeadAux(nn.Module):
    def __init__(self, n_views):
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
    no_pert = {v: {} for v in MAIN_VIEWS}
    with torch.no_grad():
        for i in range(len(test_ds)):
            pts, target_res, case_id = test_ds[i]
            pts_t = torch.from_numpy(pts[None].astype(np.float32)).to(DEVICE)
            pred_res = model(pts_t).cpu().numpy()[0] * SCALE
            for vi, v in enumerate(MAIN_VIEWS):
                pred_abs = pred_res[vi*3:(vi+1)*3] + mean_offset[v]
                gt_abs = target_res[vi*3:(vi+1)*3] * SCALE + mean_offset[v]
                no_pert[v][case_id] = {"pred_local_mm": pred_abs.tolist(), "gt_local_mm": gt_abs.tolist(),
                                        "err_mm": float(np.linalg.norm(pred_abs - gt_abs))}

    rng = np.random.default_rng(seed + 6000)
    by_case_estimates = {v: {} for v in MAIN_VIEWS}
    gt_by_case = {v: {} for v in MAIN_VIEWS}
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
                for vi, v in enumerate(MAIN_VIEWS):
                    pred_abs = pred_res[vi*3:(vi+1)*3] + mean_offset[v] + shift
                    unshifted = pred_abs - shift
                    by_case_estimates[v].setdefault(case_id, []).append(unshifted)
                    gt_by_case[v][case_id] = target_res[vi*3:(vi+1)*3] * SCALE + mean_offset[v]
    shift_incl = {v: {} for v in MAIN_VIEWS}
    for v in MAIN_VIEWS:
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
    aux = load_aux()

    mean_offset = {}
    for v in MAIN_VIEWS:
        positions = []
        for c in train_cases:
            views = labels[c]["views"]
            if v in views:
                positions.append(np.array(views[v]["position_m"], dtype=np.float32) * 1000.0)
        mean_offset[v] = np.mean(positions, axis=0)
    for v in AUX_KEYS:
        key = "heart_center_local_mm" if v == "heart_center" else "sternum_lower_tip_local_mm"
        positions = [np.array(aux[c][key], dtype=np.float32) for c in train_cases if c in aux]
        mean_offset[v] = np.mean(positions, axis=0)

    train_ds = PlaxA4cAuxDataset(train_cases, labels, aux, mean_offset, augment=True)
    val_ds = PlaxA4cAuxDataset(val_cases, labels, aux, mean_offset, augment=False)
    test_ds = PlaxA4cAuxDataset(test_cases, labels, aux, mean_offset, augment=False)
    print(f"[seed={seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = PointNetMultiHeadAux(n_views=len(ALL_VIEWS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = nn.HuberLoss(delta=1.0, reduction="none")

    ckpt_dir = os.path.join(CKPT_DIR_BASE, f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_best.pt")

    def compute_loss(pred, target):
        # pred/target: (B, 12) = [PLAX(3), A4C(3), heart(3), sternum(3)]
        per_elem = loss_fn(pred, target)  # (B,12)
        main_loss = per_elem[:, 0:6].mean()
        aux_loss = per_elem[:, 6:12].mean()
        return main_loss + AUX_LOSS_WEIGHT * aux_loss

    best_val_mm = float("inf")
    for epoch in range(300):
        model.train()
        for pts, target, _ in train_loader:
            pts, target = pts.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            pred = model(pts)
            loss = compute_loss(pred, target)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_errs = {v: [] for v in MAIN_VIEWS}
        with torch.no_grad():
            for pts, target, _ in val_loader:
                pts, target = pts.to(DEVICE), target.to(DEVICE)
                pred = model(pts)
                for vi, v in enumerate(MAIN_VIEWS):
                    p = pred[:, vi*3:(vi+1)*3]
                    t = target[:, vi*3:(vi+1)*3]
                    val_errs[v].extend(torch.norm((p - t) * SCALE, dim=1).cpu().numpy().tolist())
        val_mm = float(np.mean([np.mean(val_errs[v]) for v in MAIN_VIEWS]))
        if val_mm < best_val_mm:
            best_val_mm = val_mm
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    no_pert, shift_incl = eval_protocols(model, test_ds, mean_offset, seed=seed)

    out = {"seed": seed, "best_val_mm": best_val_mm, "no_perturbation": no_pert, "shift_inclusive": shift_incl}
    with open(os.path.join(ckpt_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    summary_str = f"[v4_aux seed={seed}] best_val={best_val_mm:.1f}  "
    for v in MAIN_VIEWS:
        np_errs = [r["err_mm"] for r in no_pert[v].values()]
        sh_errs = [r["err_mm"] for r in shift_incl[v].values()]
        summary_str += f"{v}: no_pert={np.mean(np_errs):.1f}/{np.median(np_errs):.1f} shift={np.mean(sh_errs):.1f}/{np.median(sh_errs):.1f}  "
    print(summary_str)


if __name__ == "__main__":
    for seed in SEEDS:
        run_one(seed)
    print("ALL DONE v4")
