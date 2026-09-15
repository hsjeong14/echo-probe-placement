# Paste into Isaac Sim's Script Editor and run while the FR5 ultrasound scene is open,
# Play is active, and the robot is in its "initial pose" (run go_to_initial_pose.py
# first if unsure). Cycles through all 187 labeled CT cases' skin meshes (see
# assets/plax_50case_meshes/ - same directory as the earlier 50-case sweep, now holding
# all 187), referencing each one in turn at EXACTLY the same transform /World/CT_Skin
# uses, capturing a real D455 depth image for each, and saving PNG + a raw depth .npy to
# disk. The robot itself never moves - only the phantom mesh is swapped, one case at a
# time, with a few frames' pause after each swap for the render to catch up before
# capturing.
#
# RESUMABLE: any case that already has a _depth_m.npy in OUT_DIR is skipped, so if this
# gets interrupted partway through (187 cases takes a while), just re-run the same
# script and it picks up where it left off instead of re-capturing everything.

import os

import numpy as np
import omni.kit.app
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

_ALL_CASE_IDS = [
    "s0024", "s0032", "s0040", "s0050", "s0059", "s0065", "s0070", "s0071", "s0091", "s0123",
    "s0124", "s0141", "s0146", "s0163", "s0171", "s0197", "s0223", "s0224", "s0239", "s0253",
    "s0303", "s0315", "s0332", "s0344", "s0350", "s0358", "s0370", "s0371", "s0372", "s0402",
    "s0436", "s0440", "s0446", "s0447", "s0456", "s0476", "s0484", "s0502", "s0507", "s0516",
    "s0546", "s0549", "s0551", "s0561", "s0571", "s0574", "s0578", "s0591", "s0593", "s0613",
    "s0617", "s0619", "s0621", "s0623", "s0625", "s0628", "s0629", "s0635", "s0637", "s0644",
    "s0648", "s0650", "s0661", "s0662", "s0663", "s0667", "s0669", "s0680", "s0686", "s0703",
    "s0708", "s0720", "s0723", "s0726", "s0731", "s0739", "s0778", "s0796", "s0797", "s0801",
    "s0830", "s0863", "s0864", "s0878", "s0880", "s0885", "s0899", "s0904", "s0913", "s0915",
    "s0918", "s0923", "s0924", "s0957", "s0959", "s0961", "s0963", "s0983", "s0992", "s0994",
    "s1006", "s1008", "s1012", "s1038", "s1044", "s1061", "s1063", "s1069", "s1086", "s1090",
    "s1099", "s1120", "s1127", "s1135", "s1143", "s1209", "s1238", "s1247", "s1273", "s1287",
    "s1297", "s1336", "s1340", "s1348", "s1349", "s1361", "s1369", "s1371", "s1372", "s1379",
    "s1380", "s1388", "s1397", "s1400", "s0231", "s0375", "s0429", "s0461", "s0477", "s0639",
    "s0835", "s0985", "s1070", "s1174", "s1176", "s1224", "s1267", "s1283", "s1363", "s0028",
    "s0029", "s0037", "s0080", "s0086", "s0102", "s0327", "s0334", "s0365", "s0369", "s0392",
    "s0408", "s0467", "s0472", "s0519", "s0543", "s0550", "s0553", "s0612", "s0733", "s0764",
    "s0794", "s0836", "s0842", "s0869", "s0884", "s0945", "s0950", "s0991", "s1031", "s1046",
    "s1085", "s1111", "s1145", "s1210", "s1364", "s1382", "s1384",
]
MESH_DIR = "<REPO_ROOT>/assets/plax_50case_meshes"
OUT_DIR = "<REPO_ROOT>/isaac_sim_scripts/plax_full187_depth_captures"
os.makedirs(OUT_DIR, exist_ok=True)

CASE_IDS = [c for c in _ALL_CASE_IDS if not os.path.exists(os.path.join(OUT_DIR, f"{c}_depth_m.npy"))]
print(f"[sweep] {len(_ALL_CASE_IDS)} total cases, {len(_ALL_CASE_IDS) - len(CASE_IDS)} already captured, "
      f"{len(CASE_IDS)} remaining this run")

CASE_PRIM_PATH = "/World/CasePhantom"
PHANTOM_TRANSLATE_M = Gf.Vec3d(0.5520152197845651, -0.0037616623180198594, 1.0158508657337284)
PHANTOM_ROTATE_Z_DEG = 180.0

WRIST3_LUMIFY_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upperarm_link/forearm_link/"
    "wrist1_link/wrist2_link/wrist3_link/LumifyHolder"
)
SENSOR_PRIM_PATH = WRIST3_LUMIFY_PATH + "/CameraBracket/D455DepthSensor"
DEPTH_CAMERA_PRIM_PATH = SENSOR_PRIM_PATH + "/RSD455/Camera_Pseudo_Depth"
CAM_RESOLUTION = (720, 1280)

_stage = omni.usd.get_context().get_stage()

_ct_skin_prim = _stage.GetPrimAtPath("/World/CT_Skin")
if _ct_skin_prim.IsValid():
    _vis_attr = UsdGeom.Imageable(_ct_skin_prim).CreateVisibilityAttr()
    _vis_attr.Set(UsdGeom.Tokens.invisible)
    _ct_skin_xformable = UsdGeom.Xformable(_ct_skin_prim)
    _ct_skin_orig_ops = _ct_skin_xformable.GetOrderedXformOps()
    _ct_skin_orig_translate = None
    for _op in _ct_skin_orig_ops:
        if _op.GetOpName() == "xformOp:translate":
            _ct_skin_orig_translate = _op.Get()
            _op.Set(Gf.Vec3d(_ct_skin_orig_translate[0], _ct_skin_orig_translate[1], -50.0))
    print(f"[sweep] /World/CT_Skin hidden (was {_ct_skin_orig_translate})")
else:
    print("[sweep] WARNING: /World/CT_Skin not found - nothing to hide")

if not _stage.GetPrimAtPath(DEPTH_CAMERA_PRIM_PATH).IsValid():
    raise RuntimeError(f"{DEPTH_CAMERA_PRIM_PATH} not found - is the FR5 ultrasound scene open?")

from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor  # noqa: E402

_sweep187_depth_sensor = SingleViewDepthCameraSensor(
    RtxCamera(DEPTH_CAMERA_PRIM_PATH), resolution=CAM_RESOLUTION, annotators=["distance_to_image_plane"]
)
_sweep187_depth_sensor.set_enabled_post_processing(True)

_case_prim = _stage.DefinePrim(CASE_PRIM_PATH, "Xform")
_xformable = UsdGeom.Xformable(_case_prim)


def _load_case(case_id):
    usd_path = os.path.join(MESH_DIR, f"{case_id}_skin.usd")
    _case_prim.GetReferences().ClearReferences()
    _case_prim.GetReferences().AddReference(usd_path)
    _xformable.ClearXformOpOrder()
    _xformable.AddTranslateOp().Set(PHANTOM_TRANSLATE_M)
    _xformable.AddRotateYOp().Set(0.0)
    _xformable.AddRotateXOp().Set(0.0)
    _xformable.AddRotateZOp().Set(PHANTOM_ROTATE_Z_DEG)
    _xformable.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _save_capture(case_id):
    data, _ = _sweep187_depth_sensor.get_data("distance_to_image_plane")
    if data is None:
        print(f"[sweep] {case_id}: no depth data (skipped)")
        return False
    depth = data.numpy()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    np.save(os.path.join(OUT_DIR, f"{case_id}_depth_m.npy"), depth)

    valid = np.isfinite(depth) & (depth > 0.01) & (depth < 3.0)
    norm = np.zeros_like(depth)
    if valid.any():
        dmin, dmax = depth[valid].min(), depth[valid].max()
        norm[valid] = 1.0 - (depth[valid] - dmin) / max(dmax - dmin, 1e-6)
    png = (norm * 255).astype(np.uint8)
    Image.fromarray(png).save(os.path.join(OUT_DIR, f"{case_id}_depth.png"))
    print(f"[sweep] {case_id}: saved ({valid.sum()} valid px)")
    return True


_sweep_state = {"idx": -1, "wait_frames": 0, "done": False}
_PAUSE_FRAMES = 15


def _on_tick(_e):
    if _sweep_state["done"]:
        return
    if _sweep_state["wait_frames"] > 0:
        _sweep_state["wait_frames"] -= 1
        return

    if _sweep_state["idx"] >= 0:
        _save_capture(CASE_IDS[_sweep_state["idx"]])

    _sweep_state["idx"] += 1
    if _sweep_state["idx"] >= len(CASE_IDS):
        _sweep_state["done"] = True
        if _ct_skin_prim.IsValid():
            UsdGeom.Imageable(_ct_skin_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            for _op in _ct_skin_xformable.GetOrderedXformOps():
                if _op.GetOpName() == "xformOp:translate" and _ct_skin_orig_translate is not None:
                    _op.Set(_ct_skin_orig_translate)
        print(f"[sweep] ALL DONE - {len(CASE_IDS)} cases captured to {OUT_DIR}")
        return

    next_case = CASE_IDS[_sweep_state["idx"]]
    _load_case(next_case)
    _sweep_state["wait_frames"] = _PAUSE_FRAMES
    print(f"[sweep] loading {next_case} ({_sweep_state['idx']+1}/{len(CASE_IDS)})...")


_sweep187_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_tick, name="fr5_plax187_sweep_stream"
)
print(f"[sweep] started - {len(CASE_IDS)} cases queued. Run `_sweep187_update_sub = None` to abort.")
