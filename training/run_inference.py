"""Run the trained v4 model on a single point cloud and produce a prediction in the
exact JSON schema `isaac_sim_scripts/fr5_prediction_check_window.py` expects
(`{"PLAX": {case_id: {"pred_local_mm": [...], "gt_local_mm": [...], "err_mm": ...}},
"A4C": {...}}`), so the two connect directly: predict here, then visualize/drive the
robot to the result inside Isaac Sim.

v4 predicts a RESIDUAL (offset from the training-set mean position), not an absolute
coordinate (see train_v4_aux.py) - MEAN_OFFSET_LOCAL_MM below is that population
average (two 3-vectors, mm, in the body-aligned local frame), needed to convert the
model's output back to an absolute position. This is aggregate training-set
statistics, not per-patient annotation data, so it's safe to ship even though the
individual expert labels are excluded from this snapshot.

Usage:
    python run_inference.py \
        --checkpoint <CHECKPOINT_PATH>/model_best.pt \
        --pointcloud <path/to/case_pointcloud_local_mm.npy> \
        --case-id my_case \
        --out predictions.json \
        [--gt-json <path/to/labels.json>]   # optional: if you have ground truth for
                                             # this case (e.g. re-checking a held-out
                                             # test case), fills in "gt_local_mm" too,
                                             # so fr5_prediction_check_window.py can
                                             # show both markers. Omit for a real new
                                             # (unlabeled) capture - "gt_local_mm" is
                                             # then just set equal to the prediction,
                                             # and the "true position" marker in the
                                             # viewer becomes meaningless (expected:
                                             # there IS no ground truth for a new scan).

The input point cloud must already be in the same body-aligned local mm frame used
throughout this project (bbox-centered on the subject's own skin mesh; see the
paper's coordinate-system section) and should ideally be voxel-downsampled first
(voxel_downsample_pointclouds.py) for consistency with how the model was trained.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

VIEWS = ["PLAX", "A4C"]
N_POINTS = 4096
SCALE = 300.0

# Population average PLAX/A4C position (body-aligned local mm frame) over the
# training set - NOT per-patient data, see module docstring.
MEAN_OFFSET_LOCAL_MM = {
    "PLAX": [30.65569568520908, 176.3750738547189, 108.79671592248818],
    "A4C": [107.99197578207709, 75.23204707675913, 101.54076663653679],
}


class PointNetMultiHeadAux(nn.Module):
    """Must match train_v4_aux.py's architecture exactly for checkpoint loading."""

    def __init__(self, n_views=4):  # PLAX, A4C, heart_center, sternum_tip
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


def run(checkpoint_path, pointcloud_path, case_id, out_path, gt_json_path=None, n_repeats=10, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNetMultiHeadAux(n_views=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    pts_full = np.load(pointcloud_path).astype(np.float32)
    print(f"loaded point cloud: {len(pts_full)} points")

    rng = np.random.default_rng(seed)
    preds = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if len(pts_full) >= N_POINTS:
                sel = rng.choice(len(pts_full), N_POINTS, replace=False)
            else:
                sel = rng.choice(len(pts_full), N_POINTS, replace=True)
            pts = pts_full[sel] / SCALE
            pts_t = torch.from_numpy(pts[None]).to(device)
            pred_res = model(pts_t).cpu().numpy()[0] * SCALE  # (12,) = 4 heads x 3
            preds.append(pred_res)
    pred_res_avg = np.mean(preds, axis=0)  # average over N_REPEATS point-subsampling draws

    gt_by_view = None
    if gt_json_path:
        with open(gt_json_path) as f:
            gt_by_view = json.load(f)

    out = {v: {} for v in VIEWS}
    for vi, v in enumerate(VIEWS):
        pred_abs = pred_res_avg[vi * 3:(vi + 1) * 3] + np.array(MEAN_OFFSET_LOCAL_MM[v])
        if gt_by_view and v in gt_by_view and case_id in gt_by_view[v]:
            gt_abs = np.array(gt_by_view[v][case_id])
        else:
            gt_abs = pred_abs  # no ground truth available for a real new capture
        out[v][case_id] = {
            "pred_local_mm": pred_abs.tolist(),
            "gt_local_mm": gt_abs.tolist(),
            "err_mm": float(np.linalg.norm(pred_abs - gt_abs)),
        }
        print(f"[{v}] predicted local-frame position (mm): {pred_abs.round(1).tolist()}")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")
    print("Point isaac_sim_scripts/fr5_prediction_check_window.py's "
          "test_predictions_plax_a4c_local_mm.json path at this file (and set its "
          "CASE_ID) to visualize the result and optionally drive the robot there.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pointcloud", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--gt-json", default=None)
    args = p.parse_args()
    run(args.checkpoint, args.pointcloud, args.case_id, args.out, args.gt_json)
