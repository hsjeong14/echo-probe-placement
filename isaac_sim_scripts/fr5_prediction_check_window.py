# Paste into Isaac Sim's Script Editor and run while the FR5 ultrasound scene is open,
# Play is active, AND fr5_ct_slice_server.py is running on localhost:8002 with
# CT_NIFTI_PATH pointing at THIS SAME CASE_ID's ct.nii.gz (see below - the two must
# match or the B-mode panel will show a different phantom than the one in view):
#   CT_NIFTI_PATH=<CT_DATASET_ROOT>/s0794/ct.nii.gz \
#     python fr5_ct_slice_server.py
#
# PREREQUISITE: run go_to_target_pose.py first - not for its go_to_pose_world() (this
# script solves its own IK locally, see _move_robot_local_ik below), but because its
# _on_tick()/update-subscription is what actually APPLIES joint-angle changes each frame;
# this script just writes into the same shared `_motion` dict that subscription reads
# (Script Editor shares one global namespace across separate Run calls).
#
# One window, three panels: D455 Depth | Point Cloud (top-down scatter with the
# shared-encoder multi-head model's predicted position, for whichever VIEW is selected
# below, in RED and the expert-labeled true position in BLUE) | Ultrasound B-mode. Two
# buttons let you instantly PREVIEW what the B-mode image looks like at the predicted
# vs. true position (a direct query to fr5_ct_slice_server.py, no robot motion needed),
# and two more actually drive the robot's probe there.
#
# Orientation: the trained model only predicts POSITION, not a full probe pose, so the
# predicted-position orientation is assumed perpendicular to the skin surface at that
# point, with a per-view marker/roll clock convention (PLAX 10:30, A4C 2:30) - see
# predicted_normal_orientation_plax_a4c.json (compute_predicted_normal_orientation_plax_a4c.py),
# built from a PCA-estimated local surface normal at each case's predicted point (NOT
# trimesh's own vertex_normals - those turned out to have inconsistent/inverted winding
# on several of these marching-cubes meshes, caught by checking the anterior-chest
# normal's local-z sign against physical expectation), oriented outward via a local
# cross-sectional centroid so it doesn't depend on the mesh's face winding being
# consistent. The true/expert position always uses the real recorded probe orientation
# instead (no assumption needed there). NOTE: only the roll is view-specific - the
# perpendicular-to-skin FORWARD axis assumption is applied identically to both views and
# is known to be a worse approximation for A4C (real A4C angles toward the cardiac base).

import io
import json
import os
import urllib.request

import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Gf, UsdGeom

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError("Pillow not available in this Kit Python") from e

_REQUIRED_GLOBALS = [
    "_fk_tcp_full", "_read_current_angles_deg", "_residuals", "_motion", "_apply_angles",
    "JOINT_NAMES", "JOINT_LIMITS_DEG", "ROBOT_WORLD_TRANSLATE_M", "time", "_quat_to_matrix",
]
if any(name not in globals() for name in _REQUIRED_GLOBALS):
    raise RuntimeError(
        f"{[n for n in _REQUIRED_GLOBALS if n not in globals()]} not found - "
        "run go_to_target_pose.py first (in this same Script Editor session)."
    )

from scipy.optimize import least_squares  # noqa: E402


def _move_robot_local_ik(target_pos_m, target_R_world, label, duration_s=3.0):
    """Seeded ONLY from the robot's current joint angles (unlike go_to_pose_world's own
    best-of-25-random-restart solve) - go_to_pose_world's random restarts can converge to
    a mathematically valid but physically awkward branch (elbow/wrist flipped relative to
    wherever the arm currently is), which both swings through a weird path getting there
    AND can leave the arm in a self-intersecting final pose (observed: the camera bracket
    ending up poking through the forearm). Solving locally from the current pose instead
    stays in the same IK branch the arm is already in, so both the path and the final
    pose stay close to (and as reasonable as) the current, known-good configuration."""
    target_local_pos = np.asarray(target_pos_m, dtype=np.float64) - ROBOT_WORLD_TRANSLATE_M
    x0 = _read_current_angles_deg()
    lo = np.array([JOINT_LIMITS_DEG[j][0] for j in JOINT_NAMES])
    hi = np.array([JOINT_LIMITS_DEG[j][1] for j in JOINT_NAMES])
    res = least_squares(
        _residuals, x0, args=(target_local_pos, target_R_world),
        bounds=(lo, hi), xtol=1e-13, ftol=1e-15, gtol=1e-15, max_nfev=4000, diff_step=1e-6,
    )
    pos_out, R_out = _fk_tcp_full(res.x)
    pos_err_mm = float(np.linalg.norm(pos_out - target_local_pos) * 1000.0)
    rot_err_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_out.T @ target_R_world) - 1.0) / 2.0, -1.0, 1.0))))
    print(f"[predcheck] local IK to {label}: pos_err={pos_err_mm:.2f}mm rot_err={rot_err_deg:.2f}deg "
          f"(seeded from current pose only, no random restarts)")
    if pos_err_mm > 5.0 or rot_err_deg > 5.0:
        print("[predcheck] WARNING: large residual - target may not be reachable from the current arm "
              "configuration without a bigger (branch-changing) move; consider go_to_pose_world() instead.")

    _motion["start_deg"] = x0
    _motion["target_deg"] = res.x
    _motion["start_time"] = time.time()
    _motion["duration"] = max(float(duration_s), 0.1)
    _motion["active"] = True

# Change these to any of the 32 held-out curated157 test cases / either view and
# re-run (also update CT_NIFTI_PATH on fr5_ct_slice_server.py to match CASE_ID, then
# restart that server).
CASE_ID = "s0334"
VIEW = "A4C"  # "PLAX" or "A4C" - predictions come from the shared-encoder multi-head model

PROJECT_DIR = "<REPO_ROOT>"
DATA_DIR = os.path.join(PROJECT_DIR, "fr5_ultrasound_rl/data")
MESH_DIR = os.path.join(PROJECT_DIR, "assets/plax_50case_meshes")
US_SERVER_BASE = "http://localhost:8002"

with open(os.path.join(DATA_DIR, "test_predictions_plax_a4c_local_mm.json")) as f:
    _preds_by_view = json.load(f)
with open(os.path.join(DATA_DIR, "case_local_centers_mm.json")) as f:
    _centers_all = json.load(f)
with open(os.path.join(DATA_DIR, "case_local_bbox_mm.json")) as f:
    _bboxes_all = json.load(f)
with open(os.path.join(DATA_DIR, "corrected_labels_curated157.json")) as f:
    _ras_labels_all = json.load(f)["cases"]
with open(os.path.join(DATA_DIR, "predicted_normal_orientation_plax_a4c.json")) as f:
    _normal_orient_by_view = json.load(f)

if VIEW not in _preds_by_view or CASE_ID not in _preds_by_view[VIEW]:
    raise RuntimeError(f"{VIEW}/{CASE_ID} not in test_predictions_plax_a4c_local_mm.json - "
                        f"pick one of: {sorted(_preds_by_view.get(VIEW, {}))}")
_preds_all = _preds_by_view[VIEW]
_normal_orient_all = _normal_orient_by_view[VIEW]

_PRED_LOCAL_MM = np.array(_preds_all[CASE_ID]["pred_local_mm"])
_GT_LOCAL_MM = np.array(_preds_all[CASE_ID]["gt_local_mm"])

# Point-cloud panel always shows BOTH views' target (GT) positions for this case;
# predicted positions are loaded too but only splatted when the "Show Predicted"
# button is toggled on (see _show_pred_state / the pointcloud-panel button below).
_PLAX_GT_LOCAL_MM = np.array(_preds_by_view["PLAX"][CASE_ID]["gt_local_mm"])
_PLAX_PRED_LOCAL_MM = np.array(_preds_by_view["PLAX"][CASE_ID]["pred_local_mm"])
_A4C_GT_LOCAL_MM = np.array(_preds_by_view["A4C"][CASE_ID]["gt_local_mm"])
_A4C_PRED_LOCAL_MM = np.array(_preds_by_view["A4C"][CASE_ID]["pred_local_mm"])

_CENTER_MM = np.array(_centers_all[CASE_ID])
_BBOX_MIN_MM = np.array(_bboxes_all[CASE_ID]["bbox_min_mm"])
_BBOX_MAX_MM = np.array(_bboxes_all[CASE_ID]["bbox_max_mm"])

# ICP-verified RAS-mm -> local-mesh-frame permutation (convert_case_to_usd.py), and its
# use alongside the phantom's Isaac placement (translate + 180deg about Z) - same chain
# validated (via skin-surface-distance checks) in build_local_frame_labels_50case.py /
# batch_render_test_predictions.py.
_AXIS_PERM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
_ROT_Z_180 = np.diag([-1.0, -1.0, 1.0])
_PHANTOM_TRANSLATE_M = Gf.Vec3d(0.5520152197845651, -0.0037616623180198594, 1.0158508657337284)
_PHANTOM_TRANSLATE_NP = np.array([_PHANTOM_TRANSLATE_M[0], _PHANTOM_TRANSLATE_M[1], _PHANTOM_TRANSLATE_M[2]])
_PHANTOM_ROTATE_Z_DEG = 180.0

_gt_view = _ras_labels_all[CASE_ID]["views"][VIEW]
_GT_POS_RAS_MM = np.array(_gt_view["position_mm"])
_GT_ROT_MATRIX_RAS = np.array(_gt_view["rotation_matrix"])


def _rot_matrix_to_rxryrz(R):
    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return [rx, ry, rz]


_GT_ROTATION_RAD = _rot_matrix_to_rxryrz(_GT_ROT_MATRIX_RAS)  # true position always uses the real recorded orientation

# Predicted position: no recorded orientation exists (model predicts position only), so
# assume the probe sits perpendicular to the skin surface there - see module docstring /
# compute_predicted_normal_orientation.py. R_local's columns are [lateral, elevation,
# forward] in the mesh's own local frame (same convention as fr5_ct_slice_server.py's
# R = Rz*Ry*Rx: column 0 = in-plane lateral, column 2 = beam/forward direction).
_pred_orient = _normal_orient_all[CASE_ID]
_PRED_R_LOCAL = np.array(_pred_orient["R_local"])
_PRED_ROTATION_RAD_RAS = _pred_orient["rotation_rad_ras"]

# World-frame rotations for the "Move Robot" buttons: the phantom's placement rotates
# LOCAL-frame quantities by 180deg about Z (same as position, see _local_mm_to_world_m),
# and rotation matrices transform the same way as any other direction vector under a
# pure linear map - R_world = ROT_Z_180 @ R_local. For the true/expert orientation,
# R_local_gt = AXIS_PERM @ R_ras_gt (see build_local_frame_labels_50case.py's corrected
# derivation - NOT a similarity transform, since AXIS_PERM is itself the ras->local
# change-of-basis and applies identically to any direction vector, rotation-matrix
# columns included).
_GT_R_LOCAL = _AXIS_PERM @ _GT_ROT_MATRIX_RAS
_PRED_R_WORLD = _ROT_Z_180 @ _PRED_R_LOCAL
_GT_R_WORLD = _ROT_Z_180 @ _GT_R_LOCAL
# NOTE: this world-frame chain has not been visually confirmed in Isaac Sim yet (the
# position-only chain used elsewhere in this file HAS been validated, via the
# skin-surface-distance checks in build_local_frame_labels_50case.py) - if "Move Robot"
# makes the probe point somewhere obviously wrong (away from the phantom, sideways,
# etc.), that's this chain, tell me the direction it's actually pointing and I'll fix
# the sign/permutation empirically the same way the mesh orientation bug was fixed.

# Same world-frame position+orientation, computed for BOTH views (not just whichever
# VIEW is selected above) - so the "Move Robot" buttons can offer PLAX and A4C
# separately instead of only whichever single view VIEW happens to be set to.
_MOVE_TARGETS = {}
for _v in ("PLAX", "A4C"):
    _v_gt_view = _ras_labels_all[CASE_ID]["views"][_v]
    _v_gt_r_local = _AXIS_PERM @ np.array(_v_gt_view["rotation_matrix"])
    _v_pred_r_local = np.array(_normal_orient_by_view[_v][CASE_ID]["R_local"])
    _MOVE_TARGETS[_v] = {
        "pred_local_mm": np.array(_preds_by_view[_v][CASE_ID]["pred_local_mm"]),
        "pred_r_world": _ROT_Z_180 @ _v_pred_r_local,
        "gt_local_mm": np.array(_preds_by_view[_v][CASE_ID]["gt_local_mm"]),
        "gt_r_world": _ROT_Z_180 @ _v_gt_r_local,
    }


def _local_mm_to_ras_mm(pos_local_mm):
    return _AXIS_PERM.T @ (np.array(pos_local_mm) + _CENTER_MM)


def _world_m_to_local_mm(pos_world_m):
    return 1000.0 * (_ROT_Z_180 @ (np.array(pos_world_m) - _PHANTOM_TRANSLATE_NP))


def _world_m_to_ras_mm(pos_world_m):
    return _local_mm_to_ras_mm(_world_m_to_local_mm(pos_world_m))


def _local_mm_to_world_m(pos_local_mm):
    return (np.array(pos_local_mm) / 1000.0) @ _ROT_Z_180 + _PHANTOM_TRANSLATE_NP


def _world_R_to_rot_rad_ras(R_world):
    # Inverse of R_world = _ROT_Z_180 @ R_local (see _PRED_R_WORLD/_GT_R_WORLD below) and
    # R_local = _AXIS_PERM @ R_ras (build_local_frame_labels_full187.py) composed together:
    # since _ROT_Z_180 is its own inverse, R_ras = _AXIS_PERM.T @ _ROT_Z_180 @ R_world.
    R_local = _ROT_Z_180 @ R_world
    R_ras = _AXIS_PERM.T @ R_local
    return _rot_matrix_to_rxryrz(R_ras)


# ---- scene setup: hide the old CT_Skin, load this case's phantom ----
WRIST3_LUMIFY_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upperarm_link/forearm_link/"
    "wrist1_link/wrist2_link/wrist3_link/LumifyHolder"
)
SENSOR_PRIM_PATH = WRIST3_LUMIFY_PATH + "/CameraBracket/D455DepthSensor"
DEPTH_CAMERA_PRIM_PATH = SENSOR_PRIM_PATH + "/RSD455/Camera_Pseudo_Depth"
TCP_PRIM_PATH = WRIST3_LUMIFY_PATH + "/ProbeBracket/Probe/TCP"
CAM_RESOLUTION = (720, 1280)

_stage = omni.usd.get_context().get_stage()
_xcache = UsdGeom.XformCache()

_tcp_prim = _stage.GetPrimAtPath(TCP_PRIM_PATH)
if not _tcp_prim.IsValid():
    raise RuntimeError(f"TCP prim not found at {TCP_PRIM_PATH}")

_ct_skin_prim = _stage.GetPrimAtPath("/World/CT_Skin")
if _ct_skin_prim.IsValid():
    UsdGeom.Imageable(_ct_skin_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    for _op in UsdGeom.Xformable(_ct_skin_prim).GetOrderedXformOps():
        if _op.GetOpName() == "xformOp:translate":
            _t = _op.Get()
            _op.Set(Gf.Vec3d(_t[0], _t[1], -50.0))

_case_prim = _stage.DefinePrim("/World/CasePhantom", "Xform")
_case_prim.GetReferences().ClearReferences()
_case_prim.GetReferences().AddReference(os.path.join(MESH_DIR, f"{CASE_ID}_skin.usd"))
_case_xformable = UsdGeom.Xformable(_case_prim)
_case_xformable.ClearXformOpOrder()
_case_xformable.AddTranslateOp().Set(_PHANTOM_TRANSLATE_M)
_case_xformable.AddRotateYOp().Set(0.0)
_case_xformable.AddRotateXOp().Set(0.0)
_case_xformable.AddRotateZOp().Set(_PHANTOM_ROTATE_Z_DEG)
_case_xformable.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
print(f"[predcheck] loaded {CASE_ID} phantom, view={VIEW}, model error = {_preds_all[CASE_ID]['err_mm']:.1f}mm")

if not _stage.GetPrimAtPath(DEPTH_CAMERA_PRIM_PATH).IsValid():
    raise RuntimeError(f"{DEPTH_CAMERA_PRIM_PATH} not found - is the FR5 ultrasound scene open?")

from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor  # noqa: E402

_predcheck_depth_sensor = SingleViewDepthCameraSensor(
    RtxCamera(DEPTH_CAMERA_PRIM_PATH), resolution=CAM_RESOLUTION, annotators=["distance_to_image_plane"]
)
_predcheck_depth_sensor.set_enabled_post_processing(True)
_predcheck_cam_prim = _stage.GetPrimAtPath(DEPTH_CAMERA_PRIM_PATH)
_predcheck_cam = UsdGeom.Camera(_predcheck_cam_prim)


def _depth_to_rgba(depth_m, near=0.05, far=2.0):
    depth_m = np.nan_to_num(depth_m, nan=far, posinf=far, neginf=near)
    norm = np.clip((depth_m - near) / (far - near), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)


# ---- point-cloud panel: projected through the SAME D455 camera view as the Depth
# panel above it (not a top-down local-XY scatter). Built at the camera's NATIVE
# resolution (matching exactly how the Depth panel feeds set_bytes_data - see
# _tick()'s `_providers["depth"].set_bytes_data(..., [dw, dh])`) and displayed in the
# same 260x150 UI box, so both panels go through IDENTICAL widget-level downscaling -
# pre-scaling only the point-cloud side (the earlier approach) risked a size/zoom
# mismatch if ui.ImageWithProvider's own scaling isn't a naive stretch-to-fit.
_PC_MARKER_RADIUS_PX = 15  # native-resolution equivalent of a ~3px marker at 260-wide display


def _project_world_to_pc_panel(world_pts_m, fx, fy, cx, cy, w, h, world_to_cam):
    """World-frame points -> this camera's own NATIVE pixel plane (0..w, 0..h) -
    mirrors the forward projection in _tick() (u,v -> 3D) run in reverse (3D -> u,v)."""
    homog = np.concatenate([world_pts_m, np.ones((len(world_pts_m), 1))], axis=1)
    cam_pts = homog @ world_to_cam
    x_cam, y_cam, z_cam = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    d = -z_cam
    u_raw = np.full_like(d, np.nan)
    v_raw = np.full_like(d, np.nan)
    in_front = d > 1e-6
    u_raw[in_front] = cx + x_cam[in_front] * fx / d[in_front]
    v_raw[in_front] = cy - y_cam[in_front] * fy / d[in_front]
    return u_raw, v_raw


# Point-cloud panel marker toggle: targets (GT) for BOTH views are always shown;
# predicted positions only appear once this is switched on via the "Show Predicted"
# button below.
_show_pred_state = {"on": False}


def _toggle_show_pred():
    _show_pred_state["on"] = not _show_pred_state["on"]
    print(f"[predcheck] point-cloud predicted markers -> {'ON' if _show_pred_state['on'] else 'OFF'}")


def _render_pointcloud_panel(world_pts_m, fx, fy, cx, cy, w, h, world_to_cam):
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    canvas[..., 3] = 255
    if len(world_pts_m) > 0:
        us, vs = _project_world_to_pc_panel(world_pts_m, fx, fy, cx, cy, w, h, world_to_cam)
        ok = np.isfinite(us) & np.isfinite(vs) & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        ui_, vi_ = us[ok].astype(np.int32), vs[ok].astype(np.int32)
        # give each point a small native-res radius (not just 1px) so the cloud
        # survives the widget's downscale to the small display box as a solid patch
        # rather than thinning out to near-invisible scattered dots
        r = 3
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if du * du + dv * dv > r * r:
                    continue
                uu = np.clip(ui_ + du, 0, w - 1)
                vv = np.clip(vi_ + dv, 0, h - 1)
                canvas[vv, uu] = (60, 220, 90, 255)  # green points

    def _splat(local_mm, rgb, radius=_PC_MARKER_RADIUS_PX):
        world_pt = _local_mm_to_world_m(local_mm)[None, :]
        u, v = _project_world_to_pc_panel(world_pt, fx, fy, cx, cy, w, h, world_to_cam)
        u, v = float(u[0]), float(v[0])
        if not (np.isfinite(u) and np.isfinite(v)):
            return
        u, v = int(round(u)), int(round(v))
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if du * du + dv * dv <= radius * radius:
                    uu, vv = u + du, v + dv
                    if 0 <= uu < w and 0 <= vv < h:
                        canvas[vv, uu] = (*rgb, 255)

    _splat(_PLAX_GT_LOCAL_MM, (255, 230, 40))  # PLAX target - yellow (blue was hard to see against the green cloud)
    _splat(_A4C_GT_LOCAL_MM, (170, 70, 230))  # A4C target - purple
    if _show_pred_state["on"]:
        _splat(_PLAX_PRED_LOCAL_MM, (255, 40, 40))  # PLAX predicted - red
        _splat(_A4C_PRED_LOCAL_MM, (255, 160, 40))  # A4C predicted - orange
    return canvas


# ---- UI window ----
# The window title includes CASE_ID/VIEW, so switching cases and re-running this
# script would otherwise create a SEPARATE new window each time (Script Editor's
# shared namespace means `_window`/`_predcheck_update_sub` from the PREVIOUS run are
# still globals here, but overwriting them below does NOT close/unsubscribe the old
# ones automatically) - leaving the old window open, frozen on its last frame (its
# tick subscription stops once `_predcheck_update_sub` is reassigned and the old
# subscription object gets GC'd, so it looks like a stale, non-updating phantom).
# Explicitly clean up any previous run's window/subscription first so re-running
# with a new CASE_ID replaces the same window instead of leaving an old one behind.
if "_window" in globals():
    try:
        _window.destroy()
    except Exception:  # noqa: BLE001
        pass
if "_predcheck_update_sub" in globals():
    try:
        _predcheck_update_sub.unsubscribe()
    except Exception:  # noqa: BLE001
        pass

_ROW_IMG_W, _ROW_IMG_H = 260, 150
# All three panels share the SAME display box size - depth and pointcloud both feed
# native-camera-resolution data into it (see _tick()), so both get identical
# widget-level downscaling and their framing/zoom matches exactly.
_PANEL_SIZE = {"depth": (_ROW_IMG_W, _ROW_IMG_H), "pointcloud": (_ROW_IMG_W, _ROW_IMG_H), "us": (_ROW_IMG_W, _ROW_IMG_H)}
_window = ui.Window(
    f"FR5 Prediction Check ({CASE_ID} / {VIEW})",
    width=max(w for w, h in _PANEL_SIZE.values()) + 20,
    # +24 for the "Show Predicted" row, +24 for the extra Move-Robot row (PLAX/A4C are
    # now two separate rows instead of one)
    height=sum(h + 24 for w, h in _PANEL_SIZE.values()) + 24 + 24 + 140,
)
_providers = {name: ui.ByteImageProvider() for name in ["depth", "pointcloud", "us"]}
_labels = {
    "depth": "D455 Depth",
    "pointcloud": "Point Cloud (yellow=PLAX target, purple=A4C target)",
    "us": "Ultrasound B-mode",
}
_status_label = None
_us_mode = {"mode": "live"}  # "live" | "preview_pred" | "preview_gt"


def _set_mode(mode):
    _us_mode["mode"] = mode
    print(f"[predcheck] B-mode panel mode -> {mode}")


def _move_robot_to(target_world_m, target_R_world, label):
    print(f"[predcheck] moving probe to {label} position + orientation...")
    _move_robot_local_ik(target_world_m, target_R_world, label)


with _window.frame:
    with ui.VStack():
        for name in ["depth", "pointcloud", "us"]:
            _pw, _ph = _PANEL_SIZE[name]
            with ui.VStack(height=_ph + 24):
                ui.Label(_labels[name], height=20, alignment=ui.Alignment.CENTER)
                ui.ImageWithProvider(_providers[name], width=_pw, height=_ph)
            if name == "pointcloud":
                with ui.HStack(height=24):
                    ui.Button("Show Predicted (red/orange)", clicked_fn=_toggle_show_pred)
        ui.Spacer(height=6)
        with ui.HStack(height=24):
            ui.Button("Preview: Predicted", clicked_fn=lambda: _set_mode("preview_pred"))
            ui.Button("Preview: True", clicked_fn=lambda: _set_mode("preview_gt"))
        with ui.HStack(height=24):
            ui.Button("Live (follow TCP)", clicked_fn=lambda: _set_mode("live"))
        for _v in ("PLAX", "A4C"):
            with ui.HStack(height=24):
                ui.Button(
                    f"Move Robot -> {_v} Predicted",
                    clicked_fn=lambda v=_v: _move_robot_to(
                        _local_mm_to_world_m(_MOVE_TARGETS[v]["pred_local_mm"]),
                        _MOVE_TARGETS[v]["pred_r_world"],
                        f"{v} predicted (normal-assumed orientation)",
                    ),
                )
                ui.Button(
                    f"Move Robot -> {_v} True",
                    clicked_fn=lambda v=_v: _move_robot_to(
                        _local_mm_to_world_m(_MOVE_TARGETS[v]["gt_local_mm"]),
                        _MOVE_TARGETS[v]["gt_r_world"],
                        f"{v} true (expert-recorded orientation)",
                    ),
                )
        _status_label = ui.Label(
            f"{CASE_ID} [{VIEW}]: model error = {_preds_all[CASE_ID]['err_mm']:.1f}mm  |  mode = live", height=20
        )

_printed_errors = set()


def _err_once(key, e):
    if key not in _printed_errors:
        print(f"[predcheck] {key} failed: {e}")
        _printed_errors.add(key)


_US_THROTTLE_N = 6
_tick_count = 0


def _tick(_e):
    global _tick_count
    _tick_count += 1

    # Camera intrinsics/pose are read unconditionally (not just when depth data is
    # available) since the point-cloud panel's predicted/GT markers need them every
    # tick regardless of whether a fresh depth frame came through this time.
    h, w = CAM_RESOLUTION
    focal_len = _predcheck_cam.GetFocalLengthAttr().Get()
    h_ap = _predcheck_cam.GetHorizontalApertureAttr().Get()
    v_ap = _predcheck_cam.GetVerticalApertureAttr().Get()
    fx, fy = focal_len * w / h_ap, focal_len * h / v_ap
    cx, cy = w / 2.0, h / 2.0
    _xcache.Clear()
    cam_world_m = np.array(_xcache.GetLocalToWorldTransform(_predcheck_cam_prim)).reshape(4, 4)
    world_to_cam = np.linalg.inv(cam_world_m)

    world_pts = np.zeros((0, 3))
    try:
        data, _ = _predcheck_depth_sensor.get_data("distance_to_image_plane")
        if data is not None:
            depth = data.numpy()
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            dh, dw = depth.shape
            _providers["depth"].set_bytes_data(_depth_to_rgba(depth).flatten().tolist(), [dw, dh])

            us_, vs_ = np.meshgrid(np.arange(dw), np.arange(dh))
            valid = np.isfinite(depth) & (depth > 0.05) & (depth < 2.0)
            u_v, v_v, d_v = us_[valid].astype(np.float64), vs_[valid].astype(np.float64), depth[valid].astype(np.float64)
            x_cam = (u_v - cx) * d_v / fx
            y_cam = -(v_v - cy) * d_v / fy
            z_cam = -d_v
            cam_pts_h = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)
            world_pts_h = cam_pts_h @ cam_world_m
            world_pts = world_pts_h[:, :3] / world_pts_h[:, 3:4]
            # Filter to this case's own local-frame skin bbox, same as the offline
            # pipeline (build_pointclouds_local_frame_full187.py) - NOT a fixed
            # world-space radius from the table center, which also captures the bare
            # table/pedestal surface whenever it's in view. Padding is ASYMMETRIC:
            # generous in X/Y (lateral - so the body's actual silhouette edges aren't
            # clipped) but tight on the low-Z (Anterior) side, since the table surface
            # sits at ~bbox_min Z (the phantom's own underside touches it) - a lateral
            # pad there would let a rim of flat table right next to the body through.
            # Trimming a few mm off the bottom clips only the lowest sliver of the
            # body's own posterior contact line, which is an acceptable trade for a
            # live debug view (the offline training-data pipeline, which needs the
            # full body surface, still uses the old symmetric 40mm pad).
            local_mm = 1000.0 * ((world_pts - _PHANTOM_TRANSLATE_NP[None, :]) @ _ROT_Z_180)
            _PC_BBOX_PAD_XY_MM = 40.0
            _PC_BBOX_MIN_Z_MARGIN_MM = 8.0  # strips the flat table plane at the phantom's base
            lo = _BBOX_MIN_MM.copy()
            lo[0] -= _PC_BBOX_PAD_XY_MM
            lo[1] -= _PC_BBOX_PAD_XY_MM
            lo[2] += _PC_BBOX_MIN_Z_MARGIN_MM
            hi = _BBOX_MAX_MM + _PC_BBOX_PAD_XY_MM
            in_bbox = np.all((local_mm >= lo[None, :]) & (local_mm <= hi[None, :]), axis=1)
            world_pts = world_pts[in_bbox]
            if len(world_pts) > 8000:
                sel = np.random.default_rng(0).choice(len(world_pts), 8000, replace=False)
                world_pts = world_pts[sel]
    except Exception as e:  # noqa: BLE001
        _err_once("depth/pointcloud", e)

    try:
        pc_img = _render_pointcloud_panel(world_pts, fx, fy, cx, cy, w, h, world_to_cam)
        _providers["pointcloud"].set_bytes_data(pc_img.flatten().tolist(), [w, h])  # native res, same as depth
    except Exception as e:  # noqa: BLE001
        _err_once("pointcloud_panel", e)

    if _tick_count % _US_THROTTLE_N != 0:
        return

    try:
        mode = _us_mode["mode"]
        if mode == "preview_pred":
            pos_mm = _local_mm_to_ras_mm(_PRED_LOCAL_MM).tolist()
            rot_rad = _PRED_ROTATION_RAD_RAS  # assumed perpendicular to the skin surface, see module docstring
        elif mode == "preview_gt":
            pos_mm = _GT_POS_RAS_MM.tolist()
            rot_rad = _GT_ROTATION_RAD  # real expert-recorded orientation
        else:
            # Live mode reflects wherever the robot ACTUALLY is/points right now - reads
            # both position and orientation off the real TCP transform every tick, instead
            # of assuming GT orientation regardless of the arm's real pose (that assumption
            # was wrong: after "Move Robot -> Predicted" the arm physically points along
            # the predicted/normal-assumed orientation, not GT, so the old code showed a
            # render inconsistent with where the probe actually was).
            _xcache.Clear()
            tcp_world_m4 = _xcache.GetLocalToWorldTransform(_tcp_prim)
            tcp_world_m = np.array(tcp_world_m4.ExtractTranslation())
            tcp_R_world = _quat_to_matrix(tcp_world_m4.ExtractRotationQuat())
            pos_mm = _world_m_to_ras_mm(tcp_world_m).tolist()
            rot_rad = _world_R_to_rot_rad_ras(tcp_R_world)

        body = json.dumps({"position": pos_mm, "rotation": rot_rad}).encode()
        req = urllib.request.Request(
            US_SERVER_BASE + "/set_pose", data=body, headers={"Content-Type": "application/json"}
        )
        png_bytes = urllib.request.urlopen(req, timeout=0.5).read()
        rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        _providers["us"].set_bytes_data(list(rgba.tobytes()), [rgba.width, rgba.height])
        if _status_label is not None:
            _status_label.text = f"{CASE_ID} [{VIEW}]: model error = {_preds_all[CASE_ID]['err_mm']:.1f}mm  |  mode = {mode}"
    except Exception as e:  # noqa: BLE001
        _err_once("us", e)


_predcheck_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _tick, name="fr5_prediction_check_window_stream"
)
print(
    f"[predcheck] window opened for {CASE_ID}. Make sure fr5_ct_slice_server.py is running with "
    f"CT_NIFTI_PATH for {CASE_ID}. Run `_predcheck_update_sub = None` to stop streaming."
)
