# Paste into Isaac Sim's Script Editor (Window > Script Editor) and run while
# fr5_ultrasound_scene_cadaver_ct.usd is Playing. Unlike wasd_probe_teleop.py (which only
# nudges the TCP a few mm at a time and doesn't control orientation), this solves full
# 6-DOF IK - position AND orientation - for an arbitrary target probe pose and smoothly
# drives all 6 joints there on its own; you don't touch individual joints at all.
#
# Usage (run these as separate Script-Editor snippets after this one has run once):
#   go_to_pose_world([0.55, 0.05, 1.25], some_3x3_rotation_matrix)   # raw Isaac world m
#   go_to_ct_pose(position_mm, rotation_rad)                         # CT world-mm, same
#                                                                     # convention as
#                                                                     # fr5_ct_slice_server.py's
#                                                                     # POST /set_pose
#   go_to_view_PLAX_s0024()                                          # auto-discovered from
#                                                                     # every saved view for
#                                                                     # the active CT_CASE_ID
#                                                                     # (see the bottom of
#                                                                     # this file) - printed
#                                                                     # on run
#
# FK chain: reuses wasd_probe_teleop.py's validated base_link->wrist3_link link offsets
# (unaffected by the camera/probe bracket rework), but with TCP_LOCAL recomputed for the
# CURRENT path (wrist3_link/LumifyHolder/ProbeBracket/Probe/TCP) - LumifyHolder,
# ProbeBracket and Probe are all pure translates (identity rotation) in the current scene,
# so the new offset is just their sum, read live from fr5_ultrasound_scene_cadaver_ct.usd.
#
# IK is a 6-unknown/6-target least-squares solve (3 position + 9 rotation-matrix-element
# residuals, weighted) via scipy, tried from the arm's own current live joint angles PLUS
# a batch of random restarts within the joint limits - a single local solve from "wherever
# the arm currently is" can land in a joint-limited/bad branch even when a good one exists
# a few restarts away (a 6-DOF arm generically has multiple valid configurations - elbow
# up/down, wrist-flip - for one target pose), so the lowest-error result across all tries
# is kept.
#
# Assumes /World/Robot's own mounting xformOp:rotateZ is 0 (fixed in this session - see
# the joint1-range/base-mounting discussion) with a pure (0,0,0.84) translate and no
# rotation, so the scratch FK chain (which computes TCP pose in the robot's own local
# frame, ignoring /World/Robot's placement) only needs that one fixed offset added back
# to get true Isaac world coordinates - no rotation correction needed any more.

import glob
import json
import math
import os
import re
import time

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import least_squares

try:
    from isaacsim.core.prims import Articulation
except ImportError:
    from omni.isaac.core.articulations import Articulation  # older Isaac Sim fallback

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
JOINT_LIMITS_DEG = {
    "j1": (-174.998, 174.998), "j2": (-264.999, 84.998), "j3": (-161.998, 161.998),
    "j4": (-264.999, 84.998), "j5": (-174.998, 174.998), "j6": (-174.998, 174.998),
}
LINKS = [  # (name, translate, orient wxyz) - fixed mechanical offsets from FR5.usd
    ("shoulder_link", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ("upperarm_link", (0.0, 0.0, 0.152), (0.7071054577827454, 0.7071080803871155, 0.0, 0.0)),
    ("forearm_link", (-0.425, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ("wrist1_link", (-0.39501, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ("wrist2_link", (0.0, 0.0, 0.1021), (0.7071054577827454, 0.7071080803871155, 0.0, 0.0)),
    ("wrist3_link", (0.0, 0.0, 0.102), (-0.7071054577827454, 0.7071080803871155, 0.0, 0.0)),
]
# wrist3_link -> LumifyHolder(0,0.0475,0) -> ProbeBracket(0,0,0) -> Probe(0,0,0) ->
# TCP(-0.000267,-0.049005,0.298309), all identity rotation, so this is just their sum -
# read live from the current scene file, not hand-derived. Recompute if the bracket is
# ever repositioned again.
TCP_LOCAL = (-0.00026718797987952334, -0.00150480040175554, 0.29830869043822317)

# /World/Robot's own placement in the scene - pure translate, zero rotation (fixed this
# session; used to be rotateZ=180, which is what put joint1's range boundary right on the
# phantom - see the base-mounting fix).
ROBOT_WORLD_TRANSLATE_M = np.array([0.0, 0.0, 0.84])

_scratch = Usd.Stage.CreateInMemory()
_path = ""
_scratch_prims = []
for _name, _t, _q in LINKS:
    _path += "/" + _name
    _xf = UsdGeom.Xform.Define(_scratch, _path)
    _xf.AddTranslateOp().Set(Gf.Vec3d(*_t))
    _xf.AddOrientOp().Set(Gf.Quatf(*_q))
    _xf.AddRotateZOp().Set(0.0)
    _scratch_prims.append(_scratch.GetPrimAtPath(_path))
UsdGeom.Xform.Define(_scratch, _path + "/TCP").AddTranslateOp().Set(Gf.Vec3d(*TCP_LOCAL))
_tcp_scratch_prim = _scratch.GetPrimAtPath(_path + "/TCP")
_xcache = UsdGeom.XformCache()


def _quat_to_matrix(quat: Gf.Quatd) -> np.ndarray:
    w = quat.GetReal()
    x, y, z = quat.GetImaginary()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _fk_tcp_full(angles_deg):
    """Returns (position_m in the robot's own local frame, 3x3 world-orientation matrix)."""
    for prim, theta in zip(_scratch_prims, angles_deg):
        prim.GetAttribute("xformOp:rotateZ").Set(float(theta))
    _xcache.Clear()
    m = _xcache.GetLocalToWorldTransform(_tcp_scratch_prim)
    pos = np.array(m.ExtractTranslation())
    R = _quat_to_matrix(m.ExtractRotationQuat())
    return pos, R


_ROT_RESIDUAL_SCALE = 0.3  # meters-equivalent weight so position (m) and rotation
# (dimensionless matrix-element) residuals pull the solver comparably hard - 0.05 let the
# solver satisfy position first and then get stuck in an orientation-only local minimum
# (verified: 73 deg rotation error on an easily-reachable target); 0.3 fixed that.

_N_RANDOM_RESTARTS = 25  # a 6-DOF arm generically has multiple valid IK branches (elbow
# up/down, wrist-flip, etc.) for one target pose - a single local solve from "wherever
# the arm currently is" can land in a joint-limited/bad branch even when a good branch
# exists a few restarts away (verified: single-start solve got wrist joints pinned at
# their limits with 73 deg orientation error; best-of-25 random-start solve for the exact
# same target found an exact solution with 59+ deg of margin on every joint).


def _residuals(x, target_local_pos, target_R):
    pos, R = _fk_tcp_full(x)
    pos_res = pos - target_local_pos
    rot_res = (R - target_R).flatten() * _ROT_RESIDUAL_SCALE
    return np.concatenate([pos_res, rot_res])


def _solve_ik_pose(target_local_pos, target_R, x0_deg):
    lo = np.array([JOINT_LIMITS_DEG[j][0] for j in JOINT_NAMES])
    hi = np.array([JOINT_LIMITS_DEG[j][1] for j in JOINT_NAMES])

    def _solve_from(x0):
        x0c = np.clip(x0, lo, hi)
        # diff_step is explicit because scipy's default relative finite-difference step
        # is too small once the parameters are in degrees (~O(10-100)) - without it the
        # estimated Jacobian is noisy enough that the solver stalls well short of
        # convergence (verified: ~30mm/8deg residual error instead of ~microns).
        return least_squares(
            _residuals, x0c, args=(target_local_pos, target_R),
            bounds=(lo, hi), xtol=1e-13, ftol=1e-15, gtol=1e-15, max_nfev=4000, diff_step=1e-6,
        )

    rng = np.random.default_rng()
    candidates = [np.asarray(x0_deg, dtype=np.float64)] + [rng.uniform(lo, hi) for _ in range(_N_RANDOM_RESTARTS)]
    best_score, best_x, best_pos_err, best_rot_err = None, None, None, None
    for x0 in candidates:
        res = _solve_from(x0)
        pos_out, R_out = _fk_tcp_full(res.x)
        pos_err_mm = float(np.linalg.norm(pos_out - target_local_pos) * 1000.0)
        rot_err_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_out.T @ target_R) - 1.0) / 2.0, -1.0, 1.0))))
        score = pos_err_mm + rot_err_deg
        if best_score is None or score < best_score:
            best_score, best_x, best_pos_err, best_rot_err = score, res.x, pos_err_mm, rot_err_deg

    print(f"[go_to_pose] IK solved (best of {len(candidates)}): "
          f"pos error={best_pos_err:.2f} mm, rot error={best_rot_err:.2f} deg")
    if best_pos_err > 5.0 or best_rot_err > 5.0:
        print("[go_to_pose] WARNING: large residual error - target may be outside the arm's reachable workspace")
    return best_x


_stage = omni.usd.get_context().get_stage()
_JOINT_BASE = "/World/Robot/Physics/"
_robot = Articulation("/World/Robot")
_robot.initialize()


def _read_current_angles_deg():
    pos_rad = _robot.get_joint_positions()
    if pos_rad.ndim == 2:
        pos_rad = pos_rad[0]
    names = _robot.dof_names
    return np.array([math.degrees(float(pos_rad[names.index(j)])) for j in JOINT_NAMES])


def _apply_angles(angles_deg):
    for jn, deg in zip(JOINT_NAMES, angles_deg):
        prim = _stage.GetPrimAtPath(_JOINT_BASE + jn)
        UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(float(deg))


_motion = {"active": False}


def _on_tick(_e):
    if not _motion["active"]:
        return
    t = (time.time() - _motion["start_time"]) / _motion["duration"]
    if t >= 1.0:
        t = 1.0
        _motion["active"] = False
    t_ease = t * t * (3.0 - 2.0 * t)  # smoothstep - gentle start/stop, no PD-controller jerk
    angles = _motion["start_deg"] + (_motion["target_deg"] - _motion["start_deg"]) * t_ease
    _apply_angles(angles)


# Unique variable name - Isaac Sim's Script Editor shares one global namespace across
# separate Run calls (see live_ultrasound_window.py's comment on why this matters).
_go_to_pose_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_tick, name="fr5_go_to_pose_stream"
)


def go_to_pose_world(target_pos_m, target_R, duration_s=2.5):
    """target_pos_m: (3,) Isaac world meters. target_R: 3x3 world-orientation matrix."""
    target_local_pos = np.asarray(target_pos_m, dtype=np.float64) - ROBOT_WORLD_TRANSLATE_M
    x0 = _read_current_angles_deg()
    target_deg = _solve_ik_pose(target_local_pos, np.asarray(target_R, dtype=np.float64), x0)
    _motion["start_deg"] = x0
    _motion["target_deg"] = target_deg
    _motion["start_time"] = time.time()
    _motion["duration"] = max(float(duration_s), 0.1)
    _motion["active"] = True
    print(f"[go_to_pose] moving over {duration_s:.1f}s -> " + str(dict(zip(JOINT_NAMES, np.round(target_deg, 2)))))
    return target_deg


# ---- CT world-mm convenience wrapper - same position/rotation convention as
# fr5_ct_slice_server.py's POST /set_pose (position in mm, rotation in rad, R=Rz*Ry*Rx) ----
PHANTOM_POS_M = np.array([0.53169432, -0.20765703, 1.15666311])  # CT_Skin centered on the table's true cover surface
CALIB_OFFSET_MM = np.array([-33.4, 330.1, -254.3])
_AXIS_PERMUTATION_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def go_to_ct_pose(position_mm, rotation_rad, duration_s=2.5):
    position_mm = np.asarray(position_mm, dtype=np.float64)
    dx = (position_mm[0] - CALIB_OFFSET_MM[0]) / 1000.0
    dz = (position_mm[1] - CALIB_OFFSET_MM[1]) / 1000.0
    dy = -(position_mm[2] - CALIB_OFFSET_MM[2]) / 1000.0
    target_pos_m = PHANTOM_POS_M + np.array([dx, dy, dz])
    R_ct = _rotation_matrix(*rotation_rad)
    R_isaac = _AXIS_PERMUTATION_M.T @ R_ct  # inverse of the position/rotation calibration
    return go_to_pose_world(target_pos_m, R_isaac, duration_s=duration_s)


# ---- Auto-discover EVERY saved target view for a case and register a
# go_to_view_<NAME>_<case>() function for each, instead of hand-copying one view (PLAX)
# at a time - the reference tool can save up to 9 named views per case
# (PLAX, PSAX AV/MV/PM/APEX, A4C/A3C/A2C, SC; see view_names in simulation_ct_ver3_dino.py)
# but only actually has some of them saved for any given case. Re-running this script
# after saving more views in the reference tool picks the new ones up automatically.
CT_CASE_ID = "s0024"
# Note: /media/<user>/.../cadaver/sim_echo_rl/ct_target/s0024 only ever had PLAX saved -
# .../cadaver/Project/ct_target/s0024 (a different directory, saved later - May 29) has
# all 9 views for this same case, so that's the one to scan.
CT_TARGET_DIR = f"/media/<user>/DATA/Research/Project/cadaver/Project/ct_target/{CT_CASE_ID}"

# vtkNIFTIImageReader keeps each case's own volume Origin at (0,0,0) (confirmed by
# loading ct.nii.gz directly with vtk and reading GetOrigin()), so the reference tool's
# saved matrix translation is voxel_index*spacing "pseudo-mm", not the NIfTI's true sform
# world-mm - this per-case sform translation (read once from that case's own ct.nii.gz
# header with nibabel, outside Isaac Sim since nibabel isn't in Isaac Sim's own python
# env) converts pseudo-mm back to true world-mm. Add an entry here for any other case you
# want go_to_view_*() functions for (open the case's ct.nii.gz with nibabel, print
# img.affine[:3, 3]).
_SFORM_ORIGIN_MM_BY_CASE = {
    "s0024": np.array([-227.580078125, -40.080078125, -821.9000244140625]),
    "s0011": np.array([-236.54492188, -45.54492188, -61.0]),
}


def _load_ct_view_json(path):
    with open(path) as f:
        data = json.load(f)
    m = np.array(data["matrix"], dtype=np.float64).reshape(4, 4)
    R = m[:3, :3]
    position_mm = m[:3, 3] + _SFORM_ORIGIN_MM_BY_CASE[CT_CASE_ID]
    # The reference tool's own probe-forward axis is the matrix's -Z column (not +Z),
    # lateral is +X (see simulation_ct_ver3_dino.py ~line 3168: z_axis = -mat4x4[:3, 2]) -
    # flip Y/Z columns (a proper rotation, 180 deg about local X) to match the local
    # +Z=forward / +X=lateral convention used throughout this project's own scripts and
    # fr5_ct_slice_server.py. Independently verified for PLAX by rendering through that
    # server and seeing the probe sit on the chest wall aiming at the heart.
    R_use = R @ np.diag([1.0, -1.0, -1.0])
    ry = np.arcsin(np.clip(-R_use[2, 0], -1.0, 1.0))
    rx = np.arctan2(R_use[2, 1], R_use[2, 2])
    rz = np.arctan2(R_use[1, 0], R_use[0, 0])
    return position_mm, [rx, ry, rz]


def _make_go_to_view(position_mm, rotation_rad, view_name):
    def _fn(duration_s=3.0):
        print(f"[go_to_pose] going to '{view_name}' ({CT_CASE_ID})")
        return go_to_ct_pose(position_mm, rotation_rad, duration_s=duration_s)

    return _fn


_discovered_views = {}
if CT_CASE_ID in _SFORM_ORIGIN_MM_BY_CASE:
    for _json_path in sorted(glob.glob(os.path.join(CT_TARGET_DIR, "ct_*_target_view_params.json"))):
        _view_name = os.path.basename(_json_path)[len("ct_"): -len("_target_view_params.json")]
        try:
            _pos_mm, _rot_rad = _load_ct_view_json(_json_path)
        except Exception as _e:  # noqa: BLE001 - one bad file must not block the rest
            print(f"[go_to_pose] failed to load {_json_path}: {_e}")
            continue
        _func_name = "go_to_view_" + re.sub(r"[^0-9A-Za-z]+", "_", _view_name).strip("_") + f"_{CT_CASE_ID}"
        globals()[_func_name] = _make_go_to_view(_pos_mm, _rot_rad, _view_name)
        _discovered_views[_func_name] = _view_name
else:
    print(f"[go_to_pose] no sform origin registered for case {CT_CASE_ID} in _SFORM_ORIGIN_MM_BY_CASE - "
          "add one (see the comment above) to enable go_to_view_*() discovery for it")

print(
    "[go_to_pose] ready. Call go_to_pose_world(pos_m, R3x3) or go_to_ct_pose(position_mm, rotation_rad) "
    "as a separate Script-Editor run.\n"
    f"[go_to_pose] discovered {len(_discovered_views)} saved view(s) for {CT_CASE_ID}: "
    + (", ".join(f"{fn}()" for fn in _discovered_views) if _discovered_views else "(none)")
)
print(
    "Run `_go_to_pose_update_sub = None` to stop any in-progress motion updates."
)
