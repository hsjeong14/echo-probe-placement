"""Voxel-downsample every case's raw point cloud (often 100k-480k points from dense
D455 capture) to a spatially-uniform ~8-15k point subset, so subsequent random
training subsampling (2048-4096 pts) doesn't disproportionately miss thin/detailed
thorax regions in favor of oversampled flat areas - addresses the "raw random
subsample loses chest detail" critique. One-time preprocessing, cached to disk.
"""
import json
import os

import numpy as np

DATA_DIR = "<DATA_ROOT>/plax_curated157"
PC_DIR = "<DATA_ROOT>/plax_full187/pointclouds_local_frame_full187"
OUT_DIR = "<DATA_ROOT>/plax_full187/pointclouds_local_frame_full187_voxel3mm"
os.makedirs(OUT_DIR, exist_ok=True)

VOXEL_MM = 3.0


def voxel_downsample(pts, voxel_size):
    keys = np.floor(pts / voxel_size).astype(np.int64)
    # unique voxel keys -> keep the point closest to that voxel's own mean
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_voxels = counts.shape[0]
    sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sums, inverse, pts)
    means = sums / counts[:, None]
    return means.astype(np.float32)


with open(DATA_DIR + "/plax_curated157_split.json") as f:
    split = json.load(f)
all_cases = split["train_cases"] + split["val_cases"] + split["test_cases"]

for i, case_id in enumerate(all_cases):
    src = os.path.join(PC_DIR, f"{case_id}_pointcloud_local_mm.npy")
    if not os.path.exists(src):
        print(f"[{i+1}/{len(all_cases)}] {case_id}: MISSING source, skip")
        continue
    pts = np.load(src).astype(np.float32)
    down = voxel_downsample(pts, VOXEL_MM)
    np.save(os.path.join(OUT_DIR, f"{case_id}_pointcloud_local_mm.npy"), down)
    print(f"[{i+1}/{len(all_cases)}] {case_id}: {len(pts)} -> {len(down)} points")

print("DONE")
