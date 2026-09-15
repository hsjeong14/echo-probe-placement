"""v4 qualitative results grid: rows = {Point Cloud, PLAX GT, PLAX Predicted, A4C GT,
A4C Predicted}, columns = 6 representative test cases (same cases as the v1 figure,
for direct before/after comparison). Predictions from v4 (residual regression + Huber
loss + voxel-downsampled 4096pt sampling + heart/sternum auxiliary supervision),
seed 0 (v4's 5-seed SD is tiny - 0.3mm PLAX / 0.5mm A4C - so any single seed is
representative).
"""
import importlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, "<REPO_ROOT>/assets_ext/i4h-sensor-simulation/ultrasound-simulator/examples")

AXIS_PERM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0])
PHANTOM_TRANSLATE_M = np.array([0.5520152197845651, -0.0037616623180198594, 1.0158508657337284])
BASE = "<CT_DATASET_ROOT>"
MESH_DIR = "<SCRATCH_DIR>/phantom_meshes_obj_fixed2"
OUT_PATH = "<REPO_ROOT>/figures/qualitative_v4_5x6.png"

with open("<REPO_ROOT>/data/d455_world_pose_at_sweep.json") as f:
    _campose = json.load(f)
CAM_FX, CAM_FY = _campose["intrinsics"]["fx"], _campose["intrinsics"]["fy"]
CAM_CX, CAM_CY = _campose["intrinsics"]["cx"], _campose["intrinsics"]["cy"]
CAM_TO_WORLD = np.array(_campose["cam_to_world_4x4_row_vector"])
WORLD_TO_CAM = np.linalg.inv(CAM_TO_WORLD)

with open("<REPO_ROOT>/results/v4_aux/seed0_results.json") as f:
    v4_out = json.load(f)
preds = v4_out["no_perturbation"]  # {"PLAX": {case_id: {pred_local_mm, gt_local_mm, err_mm}}, "A4C": {...}}

with open("<REPO_ROOT>/data/case_local_centers_mm.json") as f:
    centers_mm = {k: np.array(v) for k, v in json.load(f).items()}
with open("<SCRATCH_DIR>/corrected_labels_curated157.json") as f:
    ras_labels = json.load(f)["cases"]
with open("<REPO_ROOT>/data/predicted_normal_orientation_plax_a4c.json") as f:
    normal_orient = json.load(f)

POINTCLOUD_DIR = "<REPO_ROOT>/data/pointclouds_local_frame_full187"

VIEWS = ["PLAX", "A4C"]
CASE_IDS = ["s0334", "s0884", "s0612", "s1046", "s0794", "s1085"]  # same cases as the v1 figure, for before/after comparison
print(f"cases: {CASE_IDS}")


def rot_matrix_to_rxryrz(R):
    R = np.array(R)
    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return [rx, ry, rz]


def local_to_ras_mm(pos_local_mm, case_id):
    return AXIS_PERM.T @ (np.array(pos_local_mm) + centers_mm[case_id])


PC_W, PC_H = 320, 180
PC_SS = 3


def local_mm_to_px(local_mm):
    local_mm = np.atleast_2d(local_mm)
    world_pts = (local_mm / 1000.0) @ ROT_Z_180 + PHANTOM_TRANSLATE_M[None, :]
    homog = np.concatenate([world_pts, np.ones((len(world_pts), 1))], axis=1)
    cam_pts = homog @ WORLD_TO_CAM
    x_cam, y_cam, z_cam = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    d = -z_cam
    u = CAM_CX + x_cam * CAM_FX / d
    v = CAM_CY - y_cam * CAM_FY / d
    u_s = u / 1280.0 * PC_W
    v_s = v / 720.0 * PC_H
    return u_s, v_s


def render_pointcloud_panel(case_id):
    pc = np.load(os.path.join(POINTCLOUD_DIR, f"{case_id}_pointcloud_local_mm.npy"))

    canvas = np.zeros((PC_H * PC_SS, PC_W * PC_SS, 3), dtype=np.uint8)
    u, v = local_mm_to_px(pc)
    ui = np.round(u * PC_SS).astype(np.int64)
    vi = np.round(v * PC_SS).astype(np.int64)
    ok = (ui >= 0) & (ui < PC_W * PC_SS) & (vi >= 0) & (vi < PC_H * PC_SS)
    canvas[vi[ok], ui[ok]] = (60, 220, 90)

    def splat(local_mm, rgb, radius_px=5):
        r = radius_px * PC_SS
        uc, vc = local_mm_to_px(np.array(local_mm))
        uc, vc = int(round(float(uc[0]) * PC_SS)), int(round(float(vc[0]) * PC_SS))
        rr, cc = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        mask = rr * rr + cc * cc <= r * r
        vv = vc + rr[mask]
        uu = uc + cc[mask]
        keep = (vv >= 0) & (vv < PC_H * PC_SS) & (uu >= 0) & (uu < PC_W * PC_SS)
        canvas[vv[keep], uu[keep]] = rgb

    splat(preds["PLAX"][case_id]["gt_local_mm"], (255, 230, 40))
    splat(preds["A4C"][case_id]["gt_local_mm"], (170, 70, 230))
    splat(preds["PLAX"][case_id]["pred_local_mm"], (255, 40, 40))
    splat(preds["A4C"][case_id]["pred_local_mm"], (255, 160, 40))

    img = canvas.reshape(PC_H, PC_SS, PC_W, PC_SS, 3).mean(axis=(1, 3)).astype(np.uint8)
    return img


images = {}
pc_images = {}
tangential_err = {}  # (case_id, view) -> mm
for case_id in CASE_IDS:
    pc_images[case_id] = render_pointcloud_panel(case_id)
    print(f"point cloud panel: {case_id}")
for case_id in CASE_IDS:
    mesh = trimesh.load(f"{MESH_DIR}/{case_id}_skin.obj", process=False)
    verts = np.asarray(mesh.vertices)
    tree = cKDTree(verts)

    os.environ["CT_NIFTI_PATH"] = f"{BASE}/{case_id}/ct.nii.gz"
    if "srv" in sys.modules:
        srv = importlib.reload(sys.modules["srv"])
    else:
        import fr5_ct_slice_server as srv
        sys.modules["srv"] = srv

    for view in VIEWS:
        gt_view = ras_labels[case_id]["views"][view]
        gt_pos_mm = np.array(gt_view["position_mm"])
        gt_rot = rot_matrix_to_rxryrz(gt_view["rotation_matrix"])

        r = preds[view][case_id]
        pred_local = np.array(r["pred_local_mm"])
        dist_to_skin, idx = tree.query(pred_local)
        snapped_local = verts[idx]
        snapped_ras_mm = local_to_ras_mm(snapped_local, case_id)
        pred_rot_rad = normal_orient[view][case_id]["rotation_rad_ras"]

        gt_local = np.array(r["gt_local_mm"])
        tangential_err[(case_id, view)] = float(np.linalg.norm(snapped_local - gt_local))

        images[(case_id, view, "gt")] = srv._render(gt_pos_mm, gt_rot)
        images[(case_id, view, "pred")] = srv._render(snapped_ras_mm, pred_rot_rad)
    print(f"rendered {case_id}")

ROW_SPECS = [
    ("PLAX", "gt", "PLAX\ntarget"),
    ("PLAX", "pred", "PLAX\npredicted"),
    ("A4C", "gt", "A4C\ntarget"),
    ("A4C", "pred", "A4C\npredicted"),
]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(5, 6, figsize=(13.5, 11.3), facecolor="white")
fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.02, wspace=0.04, hspace=0.08)

for col, case_id in enumerate(CASE_IDS):
    tang_p = tangential_err[(case_id, "PLAX")]
    tang_a = tangential_err[(case_id, "A4C")]
    axes[0, col].set_title(f"{case_id}\nPLAX {tang_p:.1f}mm / A4C {tang_a:.1f}mm", fontsize=9.5, fontweight="bold")

    ax = axes[0, col]
    ax.imshow(pc_images[case_id])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for row, (view, kind, _) in enumerate(ROW_SPECS, start=1):
        ax = axes[row, col]
        ax.imshow(images[(case_id, view, kind)])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

axes[0, 0].set_ylabel("Point\nCloud", fontsize=10.5,
                       fontweight="bold", rotation=0, ha="right", va="center", labelpad=10)
for row, (_, _, label) in enumerate(ROW_SPECS, start=1):
    axes[row, 0].set_ylabel(label, fontsize=10.5, fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=10)

fig.suptitle("v4 qualitative results (residual + Huber + voxel-sampling + anatomical aux target, seed 0)",
              fontsize=11, y=0.985)
fig.savefig(OUT_PATH, dpi=200, facecolor="white")
print(f"saved {OUT_PATH}")

with open("<REPO_ROOT>/results/qualitative_v4_tangential_err.json", "w") as f:
    json.dump({f"{c}_{v}": tangential_err[(c, v)] for c in CASE_IDS for v in VIEWS}, f, indent=2)
