"""Simulator negative control (ICRA reviewer request): does rendered cardiac-chamber
visibility actually degrade as the probe position moves away from the expert-labeled
GT, or is the simulator "too permissive" (chambers stay visible even far off-target,
as the reviewer suspected from Case A's 43mm-error example)?

For each test case (PLAX view), render at GT and at points offset by 10/20/30/40mm
(2 random tangential directions per magnitude, snapped to the nearest skin-mesh
vertex - same snapping convention used for tangential error elsewhere in this
project) with the GT's own orientation held fixed (isolates POSITION error from
orientation error). Quantify chamber visibility as the fraction of rendered
scan-line samples where the blood-pool mask (heart chambers + great vessels,
already computed internally by fr5_ct_slice_server.py for echo-intensity gating)
exceeds 0.5 - i.e. "what fraction of this image is actually inside a heart chamber."

Perturbation/snapping happens in the LOCAL mesh frame (where MESH_DIR's .obj vertices
live), then converted back to RAS-mm for rendering, matching the frame convention used
by compute_render_similarity_all_test.py.
"""
import importlib
import json
import os
import sys

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, "<REPO_ROOT>/assets_ext/i4h-sensor-simulation/ultrasound-simulator/examples")

AXIS_PERM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
BASE = "<CT_DATASET_ROOT>"
MESH_DIR = "<SCRATCH_DIR>/phantom_meshes_obj_fixed2"
OUT_DIR = "<SCRATCH_DIR>/negative_control"
os.makedirs(OUT_DIR, exist_ok=True)

with open("<SCRATCH_DIR>/corrected_labels_curated157.json") as f:
    ras_labels = json.load(f)["cases"]
with open("<REPO_ROOT>/data/case_local_centers_mm.json") as f:
    centers_mm = {k: np.array(v) for k, v in json.load(f).items()}

VIEW = "PLAX"
TEST_CASE_IDS = ["s0028", "s0029", "s0086", "s0327", "s0334", "s0369", "s0408", "s0467",
                  "s0472", "s0519", "s0543", "s0550", "s0553", "s0612", "s0733", "s0764",
                  "s0794", "s0836", "s0842", "s0869", "s0884", "s0945", "s0950", "s0991",
                  "s1031", "s1046", "s1085", "s1111", "s1145", "s1210", "s1364", "s1382"]
CASE_IDS = sorted(set(ras_labels.keys()) & set(TEST_CASE_IDS))
MAGNITUDES_MM = [0.0, 10.0, 20.0, 30.0, 40.0]
N_DIRECTIONS = 2
SEED = 0
rng = np.random.default_rng(SEED)


def ras_to_local(pos_ras_mm, case_id):
    return AXIS_PERM @ np.array(pos_ras_mm) - centers_mm[case_id]


def local_to_ras(pos_local_mm, case_id):
    return AXIS_PERM.T @ (np.array(pos_local_mm) + centers_mm[case_id])


def rot_matrix_to_rxryrz(R):
    R = np.array(R)
    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return [rx, ry, rz]


def chamber_visible_fraction(srv, position_mm, rotation_rad):
    R = srv._rotation_matrix(*rotation_rad)
    position_mm = np.asarray(position_mm, dtype=np.float64)
    local_pts = np.stack([srv._polar_local_x, np.zeros_like(srv._polar_local_x), srv._polar_local_z], axis=-1)
    world_pts = position_mm[None, None, :] + local_pts @ R.T
    flat_world = world_pts.reshape(-1, 3)
    homog = np.concatenate([flat_world, np.ones((flat_world.shape[0], 1))], axis=1)
    voxel_pts = (srv._inv_affine @ homog.T).T[:, :3]
    blood = map_coordinates(srv._blood_mask, voxel_pts.T, order=1, mode="constant", cval=0.0)
    blood = np.clip(blood, 0.0, 1.0)
    return float(np.mean(blood > 0.5))


results = {}
for i, case_id in enumerate(CASE_IDS):
    if VIEW not in ras_labels[case_id]["views"] or case_id not in centers_mm:
        continue
    mesh_path = f"{MESH_DIR}/{case_id}_skin.obj"
    if not os.path.exists(mesh_path):
        continue
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices)  # LOCAL frame
    tree = cKDTree(verts)

    os.environ["CT_NIFTI_PATH"] = f"{BASE}/{case_id}/ct.nii.gz"
    if "srv" in sys.modules:
        srv = importlib.reload(sys.modules["srv"])
    else:
        import fr5_ct_slice_server as srv
        sys.modules["srv"] = srv

    gt_view = ras_labels[case_id]["views"][VIEW]
    gt_pos_ras_mm = np.array(gt_view["position_mm"])
    gt_rot_rad = rot_matrix_to_rxryrz(gt_view["rotation_matrix"])
    gt_pos_local_mm = ras_to_local(gt_pos_ras_mm, case_id)

    case_result = {"0mm": chamber_visible_fraction(srv, gt_pos_ras_mm, gt_rot_rad)}

    for mag in MAGNITUDES_MM[1:]:
        fracs = []
        for d in range(N_DIRECTIONS):
            rand_dir = rng.normal(size=3)
            rand_dir /= np.linalg.norm(rand_dir)
            candidate_local = gt_pos_local_mm + rand_dir * mag
            _, idx = tree.query(candidate_local)
            snapped_local = verts[idx]
            snapped_ras = local_to_ras(snapped_local, case_id)
            fracs.append(chamber_visible_fraction(srv, snapped_ras, gt_rot_rad))
        case_result[f"{int(mag)}mm"] = float(np.mean(fracs))

    results[case_id] = case_result
    print(f"[{i+1}] {case_id}: " + "  ".join(f"{k}={v:.3f}" for k, v in case_result.items()))

with open(os.path.join(OUT_DIR, "chamber_visibility_vs_offset.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Aggregate (mean chamber-visible fraction across cases) ===")
for k in ["0mm", "10mm", "20mm", "30mm", "40mm"]:
    vals = [r[k] for r in results.values() if k in r]
    print(f"{k}: mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  n={len(vals)}")
