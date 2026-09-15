"""Run all three bigjitter models (multihead, plax_only, a4c_only) across 5 seeds each,
to get mean+-SD test-set error - addresses the ICRA reviewer's demand for multi-seed /
cross-validation evidence (single-seed n=32 results could be within seed noise).

For each (model_kind, seed): set seed, train, save checkpoint under a seed-suffixed dir,
then evaluate under BOTH the plain (no-perturbation) protocol and the standard
shift-inclusive protocol (§7.5.1), saving per-case results for later aggregation.
"""
import json
import os
import random
import sys

import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = "<DATA_ROOT>/plax_curated157"
RESULTS_DIR = os.path.join(DATA_DIR, "multiseed_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_no_perturbation(model, test_ds, VIEWS, model_kind, device, scale=300.0):
    model.eval()
    results = {v: {} for v in VIEWS}
    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            pts, target, case_id = item
            pts_t = torch.from_numpy(pts[None].astype(np.float32)).to(device)
            pred = model(pts_t).cpu().numpy()[0] * scale
            if model_kind == "multihead":
                for vi, v in enumerate(VIEWS):
                    pred_v = pred[vi*3:(vi+1)*3]
                    gt_v = target[vi*3:(vi+1)*3] * scale
                    results[v][case_id] = {"pred_local_mm": pred_v.tolist(), "gt_local_mm": gt_v.tolist(),
                                            "err_mm": float(np.linalg.norm(pred_v - gt_v))}
            else:
                v = VIEWS[0]
                gt_v = target * scale
                results[v][case_id] = {"pred_local_mm": pred.tolist(), "gt_local_mm": gt_v.tolist(),
                                        "err_mm": float(np.linalg.norm(pred - gt_v))}
    return results


def eval_shift_inclusive(model, test_ds, VIEWS, model_kind, device, scale=300.0,
                          n_repeats=10, shift_max_mm=30.0, seed=0):
    model.eval()
    rng = np.random.default_rng(seed + 1000)  # offset from train seed
    by_case_estimates = {v: {} for v in VIEWS}
    gt_by_case = {v: {} for v in VIEWS}
    with torch.no_grad():
        for rep in range(n_repeats):
            for i in range(len(test_ds)):
                item = test_ds[i]
                pts, target, case_id = item
                pts_mm = pts * scale
                mag = rng.uniform(0.0, shift_max_mm)
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                shift = (direction * mag).astype(np.float32)
                pts_shifted_mm = pts_mm + shift
                pts_t = torch.from_numpy((pts_shifted_mm / scale)[None].astype(np.float32)).to(device)
                pred = model(pts_t).cpu().numpy()[0] * scale
                if model_kind == "multihead":
                    for vi, v in enumerate(VIEWS):
                        pred_v = pred[vi*3:(vi+1)*3]
                        unshifted = pred_v - shift
                        by_case_estimates[v].setdefault(case_id, []).append(unshifted)
                        gt_by_case[v][case_id] = target[vi*3:(vi+1)*3] * scale
                else:
                    v = VIEWS[0]
                    unshifted = pred - shift
                    by_case_estimates[v].setdefault(case_id, []).append(unshifted)
                    gt_by_case[v][case_id] = target * scale
    results = {v: {} for v in VIEWS}
    for v in VIEWS:
        for case_id, estimates in by_case_estimates[v].items():
            avg_pred = np.mean(estimates, axis=0)
            gt = gt_by_case[v][case_id]
            results[v][case_id] = {"pred_local_mm": avg_pred.tolist(), "gt_local_mm": gt.tolist(),
                                    "err_mm": float(np.linalg.norm(avg_pred - gt))}
    return results


def run_one(model_kind, seed):
    set_seed(seed)
    if model_kind == "multihead":
        import importlib
        import train_plax_a4c_pointnet_curated157_bigjitter as T
        importlib.reload(T)
        T.CKPT_DIR = os.path.join(DATA_DIR, f"checkpoints_plax_a4c_bigjitter_seed{seed}")
        os.makedirs(T.CKPT_DIR, exist_ok=True)
        VIEWS = T.VIEWS
        train_cases, val_cases, test_cases = T.load_split()
        labels = T.load_labels()
        train_ds = T.PlaxA4cDataset(train_cases, labels, augment=True)
        val_ds = T.PlaxA4cDataset(val_cases, labels, augment=False)
        test_ds = T.PlaxA4cDataset(test_cases, labels, augment=False)
        model = T.PointNetMultiHead(n_views=len(VIEWS)).to(T.DEVICE)
        ckpt_name = "plax_a4c_pointnet_best.pt"
    elif model_kind == "plax_only":
        import importlib
        import train_plax_pointnet_curated157_bigjitter as T
        importlib.reload(T)
        T.CKPT_DIR = os.path.join(DATA_DIR, f"checkpoints_plax_only_bigjitter_seed{seed}")
        os.makedirs(T.CKPT_DIR, exist_ok=True)
        VIEWS = ["PLAX"]
        train_cases, val_cases, test_cases = T.load_split()
        labels = T.load_labels()
        train_ds = T.PlaxDataset(train_cases, labels, augment=True)
        val_ds = T.PlaxDataset(val_cases, labels, augment=False)
        test_ds = T.PlaxDataset(test_cases, labels, augment=False)
        model = T.PointNetRegressor().to(T.DEVICE)
        ckpt_name = "plax_pointnet_best.pt"
    elif model_kind == "a4c_only":
        import importlib
        import train_a4c_pointnet_curated157_bigjitter as T
        importlib.reload(T)
        T.CKPT_DIR = os.path.join(DATA_DIR, f"checkpoints_a4c_bigjitter_seed{seed}")
        os.makedirs(T.CKPT_DIR, exist_ok=True)
        VIEWS = ["A4C"]
        train_cases, val_cases, test_cases = T.load_split()
        labels = T.load_labels()
        train_ds = T.A4cDataset(train_cases, labels, augment=True)
        val_ds = T.A4cDataset(val_cases, labels, augment=False)
        test_ds = T.A4cDataset(test_cases, labels, augment=False)
        model = T.PointNetRegressor().to(T.DEVICE)
        ckpt_name = "a4c_pointnet_best.pt"
    else:
        raise ValueError(model_kind)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = torch.nn.MSELoss()

    best_val_mm = float("inf")
    for epoch in range(300):
        model.train()
        for batch in train_loader:
            pts, target = batch[0].to(T.DEVICE), batch[1].to(T.DEVICE)
            opt.zero_grad()
            pred = model(pts)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_errs = []
        with torch.no_grad():
            for batch in val_loader:
                pts, target = batch[0].to(T.DEVICE), batch[1].to(T.DEVICE)
                pred = model(pts)
                if model_kind == "multihead":
                    for vi in range(len(VIEWS)):
                        p = pred[:, vi*3:(vi+1)*3]
                        t = target[:, vi*3:(vi+1)*3]
                        val_errs.extend(torch.norm((p - t) * 300.0, dim=1).cpu().numpy().tolist())
                else:
                    val_errs.extend(torch.norm((pred - target) * 300.0, dim=1).cpu().numpy().tolist())
        val_mm = float(np.mean(val_errs))
        if val_mm < best_val_mm:
            best_val_mm = val_mm
            torch.save(model.state_dict(), os.path.join(T.CKPT_DIR, ckpt_name))

    model.load_state_dict(torch.load(os.path.join(T.CKPT_DIR, ckpt_name)))
    model.eval()

    no_pert = eval_no_perturbation(model, test_ds, VIEWS, model_kind, T.DEVICE)
    shifted = eval_shift_inclusive(model, test_ds, VIEWS, model_kind, T.DEVICE, seed=seed)

    out = {"model_kind": model_kind, "seed": seed, "best_val_mm": best_val_mm,
           "no_perturbation": no_pert, "shift_inclusive": shifted}
    out_path = os.path.join(RESULTS_DIR, f"{model_kind}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    summary = {}
    for v in VIEWS:
        no_pert_errs = [r["err_mm"] for r in no_pert[v].values()]
        shift_errs = [r["err_mm"] for r in shifted[v].values()]
        summary[v] = {
            "no_pert_mean": float(np.mean(no_pert_errs)), "no_pert_median": float(np.median(no_pert_errs)),
            "shift_mean": float(np.mean(shift_errs)), "shift_median": float(np.median(shift_errs)),
        }
    print(f"[{model_kind} seed={seed}] best_val={best_val_mm:.1f}  " +
          "  ".join(f"{v}: no_pert={summary[v]['no_pert_mean']:.1f}/{summary[v]['no_pert_median']:.1f} "
                     f"shift={summary[v]['shift_mean']:.1f}/{summary[v]['shift_median']:.1f}" for v in VIEWS))
    return out_path


if __name__ == "__main__":
    model_kinds = ["multihead", "plax_only", "a4c_only"]
    for mk in model_kinds:
        for seed in SEEDS:
            run_one(mk, seed)
    print("ALL DONE")
