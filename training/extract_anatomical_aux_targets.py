"""Extract two free auxiliary supervision targets per case from TotalSegmentator masks
we already have locally (no extra annotation cost): heart centroid and sternum lower
tip (xiphoid-adjacent point). Converted to the same body-aligned local frame (mm) used
for the PLAX/A4C position labels, via the project's established AXIS_PERM + per-case
bbox-center convention (PIPELINE_TECHNICAL_REPORT.md Sec 4.2/4.3).
"""
import json
import os

import nibabel as nib
import numpy as np

AXIS_PERM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
BASE = "<CT_DATASET_ROOT>"

with open("<REPO_ROOT>/data/case_local_centers_mm.json") as f:
    centers_mm = {k: np.array(v) for k, v in json.load(f).items()}
with open("<SCRATCH_DIR>/corrected_labels_curated157.json") as f:
    ras_labels = json.load(f)["cases"]

CASE_IDS = sorted(set(ras_labels.keys()) & set(centers_mm.keys()))
print(f"{len(CASE_IDS)} cases")


def ras_to_local(pos_ras_mm, case_id):
    return AXIS_PERM @ np.array(pos_ras_mm) - centers_mm[case_id]


def mask_centroid_ras_mm(seg_path, affine):
    seg = nib.load(seg_path).get_fdata()
    idx = np.argwhere(seg > 0.5)
    if len(idx) == 0:
        return None
    centroid_voxel = idx.mean(axis=0)
    homog = np.array([centroid_voxel[0], centroid_voxel[1], centroid_voxel[2], 1.0])
    ras = (affine @ homog)[:3]
    return ras


def mask_inferior_tip_ras_mm(seg_path, affine):
    """Most-inferior point of the mask, converted to RAS - for sternum lower tip."""
    seg = nib.load(seg_path).get_fdata()
    idx = np.argwhere(seg > 0.5)
    if len(idx) == 0:
        return None
    homog = np.concatenate([idx, np.ones((len(idx), 1))], axis=1)
    ras_all = (affine @ homog.T).T[:, :3]
    inferior_idx = np.argmin(ras_all[:, 2])  # RAS z-axis = Superior; min = most inferior
    return ras_all[inferior_idx]


results = {}
for i, case_id in enumerate(CASE_IDS):
    ct_path = f"{BASE}/{case_id}/ct.nii.gz"
    seg_dir = f"{BASE}/{case_id}/segmentations"
    heart_path = f"{seg_dir}/heart.nii.gz"
    sternum_path = f"{seg_dir}/sternum.nii.gz"
    if not (os.path.exists(heart_path) and os.path.exists(sternum_path)):
        print(f"[{i+1}/{len(CASE_IDS)}] {case_id}: missing seg files, skip")
        continue

    affine = nib.load(ct_path).affine

    heart_ras = mask_centroid_ras_mm(heart_path, affine)
    sternum_ras = mask_inferior_tip_ras_mm(sternum_path, affine)
    if heart_ras is None or sternum_ras is None:
        print(f"[{i+1}/{len(CASE_IDS)}] {case_id}: empty mask, skip")
        continue

    heart_local = ras_to_local(heart_ras, case_id)
    sternum_local = ras_to_local(sternum_ras, case_id)

    results[case_id] = {
        "heart_center_local_mm": heart_local.tolist(),
        "sternum_lower_tip_local_mm": sternum_local.tolist(),
    }
    print(f"[{i+1}/{len(CASE_IDS)}] {case_id}: heart={heart_local.round(1)}  sternum_tip={sternum_local.round(1)}")

OUT = "<SCRATCH_DIR>/anatomical_aux_targets_local_mm.json"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {OUT}  (n={len(results)})")
