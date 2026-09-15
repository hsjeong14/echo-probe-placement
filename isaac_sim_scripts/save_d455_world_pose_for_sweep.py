# Paste into Isaac Sim's Script Editor and run while the FR5 ultrasound scene is open,
# Play is active, and the robot is in the SAME "initial pose" used for the
# capture_depth_all_50_cases.py sweep (run go_to_initial_pose.py first if unsure - the
# robot never moves during that sweep, so this pose is constant across all 50 captures
# and only needs to be saved once).
#
# Saves the D455 depth camera's world transform (4x4, row-vector convention, same as
# capture_depth_pointcloud_ct_frame.py) + intrinsics to JSON, so the depth images
# captured by the sweep can be unprojected into world space and then into each case's
# mesh-local frame (see local_frame_labels_50case.json for that frame's convention)
# entirely offline, without needing Isaac Sim again.

import json

import numpy as np
import omni.usd
from pxr import UsdGeom

WRIST3_LUMIFY_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upperarm_link/forearm_link/"
    "wrist1_link/wrist2_link/wrist3_link/LumifyHolder"
)
SENSOR_PRIM_PATH = WRIST3_LUMIFY_PATH + "/CameraBracket/D455DepthSensor"
DEPTH_CAMERA_PRIM_PATH = SENSOR_PRIM_PATH + "/RSD455/Camera_Pseudo_Depth"
CAM_RESOLUTION = (720, 1280)  # (height, width), matches capture_depth_all_50_cases.py
OUT_PATH = "<REPO_ROOT>/data/d455_world_pose_at_sweep.json"

_stage = omni.usd.get_context().get_stage()
_cam_prim = _stage.GetPrimAtPath(DEPTH_CAMERA_PRIM_PATH)
if not _cam_prim.IsValid():
    raise RuntimeError(f"{DEPTH_CAMERA_PRIM_PATH} not found - is the FR5 ultrasound scene open?")

H, W = CAM_RESOLUTION
_cam = UsdGeom.Camera(_cam_prim)
focal_len = _cam.GetFocalLengthAttr().Get()
h_aperture = _cam.GetHorizontalApertureAttr().Get()
v_aperture = _cam.GetVerticalApertureAttr().Get()
fx = focal_len * W / h_aperture
fy = focal_len * H / v_aperture
cx, cy = W / 2.0, H / 2.0

_xcache = UsdGeom.XformCache()
cam_world_m = _xcache.GetLocalToWorldTransform(_cam_prim)
cam_world_np = np.array(cam_world_m).reshape(4, 4)  # row-vector convention: p_world = p_cam @ M

out = {
    "camera_prim_path": DEPTH_CAMERA_PRIM_PATH,
    "resolution_hw": [H, W],
    "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    "cam_to_world_4x4_row_vector": cam_world_np.tolist(),
    "note": "p_world_h = [x_cam,y_cam,z_cam,1] @ cam_to_world_4x4_row_vector ; "
            "camera looks down local -Z, +X right, +Y up (USD camera convention), "
            "matching capture_depth_pointcloud_ct_frame.py's unprojection.",
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"[save_pose] saved D455 world pose + intrinsics to {OUT_PATH}")
print(f"[save_pose] intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
print(f"[save_pose] cam_to_world =\n{cam_world_np}")
