"""Compute tangential error (predicted point snapped to nearest skin-mesh vertex, vs
GT) for v4 seed 0's no-perturbation predictions, across ALL 32 test cases (both
views) - to rank cases for a "true best 6" qualitative figure (not carried over from
the old v1 figure's case selection).
"""
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree

MESH_DIR = "<SCRATCH_DIR>/phantom_meshes_obj_fixed2"

with open("<REPO_ROOT>/results/v4_aux/seed0_results.json") as f:
    v4_out = json.load(f)
preds = v4_out["no_perturbation"]

VIEWS = ["PLAX", "A4C"]
CASE_IDS = sorted(preds["PLAX"].keys())

results = {}
for case_id in CASE_IDS:
    mesh = trimesh.load(f"{MESH_DIR}/{case_id}_skin.obj", process=False)
    verts = np.asarray(mesh.vertices)
    tree = cKDTree(verts)

    results[case_id] = {}
    for view in VIEWS:
        r = preds[view][case_id]
        pred_local = np.array(r["pred_local_mm"])
        gt_local = np.array(r["gt_local_mm"])
        dist_to_skin, idx = tree.query(pred_local)
        snapped_local = verts[idx]
        tang_err = float(np.linalg.norm(snapped_local - gt_local))
        results[case_id][view] = {"tangential_err_mm": tang_err, "dist_to_skin_mm": float(dist_to_skin),
                                    "raw_err_mm": r["err_mm"]}

    combined = results[case_id]["PLAX"]["tangential_err_mm"] + results[case_id]["A4C"]["tangential_err_mm"]
    results[case_id]["combined_tangential_mm"] = combined
    print(f"{case_id}: PLAX={results[case_id]['PLAX']['tangential_err_mm']:.1f}  A4C={results[case_id]['A4C']['tangential_err_mm']:.1f}  combined={combined:.1f}")

OUT = "<REPO_ROOT>/results/v4_tangential_all32.json"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {OUT}")

ranked = sorted(CASE_IDS, key=lambda c: results[c]["combined_tangential_mm"])
print("\n=== best 6 by combined PLAX+A4C tangential error ===")
for c in ranked[:6]:
    print(f"{c}: combined={results[c]['combined_tangential_mm']:.1f}  (PLAX={results[c]['PLAX']['tangential_err_mm']:.1f}, A4C={results[c]['A4C']['tangential_err_mm']:.1f})")
