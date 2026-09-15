# Paste into Isaac Sim's Script Editor (Window > Script Editor) and run while
# the fr5_ultrasound_scene_cadaver_ct.usd stage is open. Streams the probe TCP's
# live world pose to fr5_ct_slice_server.py (port 8002 - raw CT HU-volume slicer)
# so its two browser pages show live frames as the FR5 arm moves: B-mode at
# http://localhost:8002/ and the CT position map at http://localhost:8002/localizer.
# (fr5_cadaver_server.py on 8001 is the older mesh-based ray tracer, not used here;
# fr5_live_server.py on 8000 is for the separate generic-ABDPhantom scene.)
#
# COORDINATE CALIBRATION - position axis mapping verified (see below); rotation still
# a first guess. Position was reverse-engineered by fitting CT_Skin.usd's own mesh
# points against skin.nii.gz's actual world-mm bounding box - sampling HU at 57
# CT_Skin surface points run through this formula now lands 86% of them in the
# -300..50 HU skin/fat range (was 88% landing in air with the old, wrong axis mapping).
# Rotation calibration (CALIB_ROT_OFFSET_RAD) has NOT been independently re-verified -
# if the image orientation looks flipped/rotated even when position is right, that's
# the next thing to fix.
#
# TCP_PRIM_PATH changed: the probe now lives under LumifyHolder/ProbeBracket
# (Lumify holder bracket rework), not directly under wrist3_link/Probe.

import json
import urllib.request

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf

# 8002 = CT-slice server (fr5_ct_slice_server.py). Open http://localhost:8002/ (B-mode)
# and http://localhost:8002/localizer (CT position map) in two browser tabs - both poll
# the server's own latest-frame endpoints, so this script just needs to keep POSTing
# pose updates; no Isaac-Sim-native window needed on this path.
SERVER_URL = "http://localhost:8002/set_pose"
TCP_PRIM_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upperarm_link/forearm_link/"
    "wrist1_link/wrist2_link/wrist3_link/LumifyHolder/ProbeBracket/Probe/TCP"
)

PHANTOM_POS_M = Gf.Vec3d(0.53169432, -0.20765703, 1.15666311)  # CT_Skin centered on the table's true cover surface
CALIB_OFFSET_MM = Gf.Vec3d(-33.4, 330.1, -254.3)  # fr5_cadaver_server.py's own initial_pose position
CALIB_ROT_OFFSET_RAD = Gf.Vec3d(1.952517925040523, 6.683790859541802, 0.0)  # its initial_pose rotation

_stage = omni.usd.get_context().get_stage()
_xcache = UsdGeom.XformCache()
_tcp_prim = _stage.GetPrimAtPath(TCP_PRIM_PATH)
if not _tcp_prim.IsValid():
    raise RuntimeError(f"TCP prim not found at {TCP_PRIM_PATH} - is fr5_ultrasound_scene.usd open?")


# Same axis permutation derived for position - applied directly to the rotation matrix
# instead of the old (mathematically unsound) decompose-Euler-then-add-offset approach.
_AXIS_PERMUTATION_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


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


def _matrix_to_rz_ry_rx_euler(R: np.ndarray):
    # Inverse of R = Rz(rz) @ Ry(ry) @ Rx(rx), matching fr5_ct_slice_server.py's own
    # _rotation_matrix() convention (== ultrasound-simulator's C++ Pose convention).
    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return rx, ry, rz


def _send_pose():
    _xcache.Clear()
    world_m = _xcache.GetLocalToWorldTransform(_tcp_prim)
    pos_m = world_m.ExtractTranslation()
    quat = world_m.ExtractRotationQuat()

    # Reverse-engineered from how CT_Skin.usd's own mesh was authored (fit against
    # skin.nii.gz's actual world-mm bbox) - see live_ultrasound_window.py's comment for
    # the derivation. Isaac Sim world Y/Z map to swapped (and one flipped) CT mm axes,
    # not a simple rotateZ(180) on a 1:1 axis mapping like the old formula assumed.
    dx = pos_m[0] - PHANTOM_POS_M[0]
    dy = pos_m[1] - PHANTOM_POS_M[1]
    dz = pos_m[2] - PHANTOM_POS_M[2]
    position_mm = [
        CALIB_OFFSET_MM[0] + dx * 1000.0,
        CALIB_OFFSET_MM[1] + dz * 1000.0,
        CALIB_OFFSET_MM[2] - dy * 1000.0,
    ]
    R_isaac = _quat_to_matrix(quat)
    R_ct = _AXIS_PERMUTATION_M @ R_isaac
    rx, ry, rz = _matrix_to_rz_ry_rx_euler(R_ct)
    rotation_rad = [rx, ry, rz]

    body = json.dumps({"position": position_mm, "rotation": rotation_rad}).encode()
    req = urllib.request.Request(SERVER_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=0.5).read()
    except Exception as e:  # noqa: BLE001 - streaming loop must not die on one dropped frame
        print(f"[ultrasound stream] send failed: {e}")


import omni.kit.app  # noqa: E402

# Unique variable name - see live_ultrasound_window.py's comment on why.
_probe_stream_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    lambda e: _send_pose(), name="fr5_ultrasound_probe_stream"
)
print("[ultrasound stream] started - open http://localhost:8002/ and http://localhost:8002/localizer .")
print("Run `_probe_stream_update_sub = None` to stop.")
