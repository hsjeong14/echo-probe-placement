# Paste into Isaac Sim's Script Editor (Window > Script Editor) and run while
# fr5_ultrasound_scene_cadaver_ct.usd is open and fr5_cadaver_server.py (in the
# ultrasound-simulator repo) is running on localhost:8001 - the CT-derived
# cadaver phantom matching this scene's CT_Skin (use fr5_live_server.py on
# 8000 instead for the generic ABDPhantom fr5_ultrasound_scene.usd).
#
# Opens a floating "FR5 Ultrasound" window INSIDE Isaac Sim showing the live
# B-mode frame, instead of a separate browser tab.
# TCP_PRIM_PATH changed: the probe now lives under LumifyHolder/ProbeBracket
# (Lumify holder bracket rework), not directly under wrist3_link/Probe.
#
# COORDINATE CALIBRATION - position axis mapping verified: reverse-engineered by
# fitting CT_Skin.usd's own mesh points against skin.nii.gz's actual world-mm
# bounding box. Sampling HU at 57 CT_Skin surface points run through this formula
# lands 86% of them in the -300..50 HU skin/fat range (was 88% landing in AIR with
# the old, wrong axis mapping - Y/Z were swapped and X had the wrong sign). Rotation
# calibration (CALIB_ROT_OFFSET_RAD) has NOT been independently re-verified the same
# way - if position looks right but the image orientation is flipped/rotated, that's
# the next thing to fix.

import io
import json
import urllib.request

import numpy as np
import omni.ui as ui
import omni.usd
from pxr import Gf, Usd, UsdGeom

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError("Pillow not available in this Kit Python - pip install pillow into the isaacsim6 env") from e

# 8001 = mesh-based ray tracer (fr5_cadaver_server.py, requires segmented organ meshes).
# 8002 = raw CT HU-volume slicer (fr5_ct_slice_server.py, no segmentation needed, air
# outside the body naturally renders black - no distance-gating hack required).
SERVER_URL = "http://localhost:8002/set_pose"
TCP_PRIM_PATH = (
    "/World/Robot/Geometry/base_link/shoulder_link/upperarm_link/forearm_link/"
    "wrist1_link/wrist2_link/wrist3_link/LumifyHolder/ProbeBracket/Probe/TCP"
)

PHANTOM_POS_M = Gf.Vec3d(0.53169432, -0.20765703, 1.15666311)  # CT_Skin centered on the table's true cover surface
CALIB_OFFSET_MM = Gf.Vec3d(-33.4, 330.1, -254.3)
CALIB_ROT_OFFSET_RAD = Gf.Vec3d(1.952517925040523, 6.683790859541802, 0.0)

_stage = omni.usd.get_context().get_stage()
_xcache = UsdGeom.XformCache()
_tcp_prim = _stage.GetPrimAtPath(TCP_PRIM_PATH)
if not _tcp_prim.IsValid():
    raise RuntimeError(f"TCP prim not found at {TCP_PRIM_PATH} - is fr5_ultrasound_scene.usd open?")


# Same axis permutation derived for position (dx->mm.x, dz->mm.y, -dy->mm.z) - applied
# directly to the rotation matrix instead of the old (mathematically unsound) approach of
# decomposing to Euler angles and adding a fixed offset. Since M was fit against the
# WHOLE CT_Skin mesh (not just one point), it's a complete linear relationship between
# the two frames - no separate rotational calibration offset should be needed on top.
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


def _current_pose_payload() -> bytes:
    _xcache.Clear()
    world_m = _xcache.GetLocalToWorldTransform(_tcp_prim)
    pos_m = world_m.ExtractTranslation()
    quat = world_m.ExtractRotationQuat()

    # Reverse-engineered directly from how CT_Skin.usd's own mesh was authored (fit
    # against skin.nii.gz's actual world-mm bbox): its LOCAL axes are a permutation of
    # the CT's RAS world-mm axes, not a simple rotateZ(180) on a 1:1 axis mapping like
    # the old formula assumed - Isaac Sim world Y and Z correspond to swapped CT mm axes.
    # world_mm.x = 1000*(wx-tx) - 33.4
    # world_mm.y = 1000*(wz-tz) + 330.1   (Isaac world Z -> CT mm Y)
    # world_mm.z = 1000*(ty-wy) - 254.3   (Isaac world Y -> CT mm Z, flipped)
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
    return json.dumps({"position": position_mm, "rotation": rotation_rad}).encode(), position_mm, rotation_rad


_window = ui.Window("FR5 Ultrasound", width=600, height=690)
_provider = ui.ByteImageProvider()
with _window.frame:
    with ui.VStack():
        _image_widget = ui.ImageWithProvider(_provider, width=580, height=580)
        _hud_label = ui.Label("position_mm: -- | rotation_rad: --", height=30)


def _tick(_e):
    try:
        body, position_mm, rotation_rad = _current_pose_payload()
        req = urllib.request.Request(SERVER_URL, data=body, headers={"Content-Type": "application/json"})
        png_bytes = urllib.request.urlopen(req, timeout=0.5).read()
        rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        _provider.set_bytes_data(list(rgba.tobytes()), [rgba.width, rgba.height])
        _hud_label.text = (
            f"position_mm: [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}] | "
            f"rotation_rad: [{rotation_rad[0]:.2f}, {rotation_rad[1]:.2f}, {rotation_rad[2]:.2f}]"
        )
    except Exception as e:  # noqa: BLE001 - one dropped frame must not kill the subscription
        print(f"[ultrasound window] frame failed: {e}")


import omni.kit.app  # noqa: E402

# Unique variable name - Isaac Sim's Script Editor shares one global namespace across
# separate Run calls, so a generic `_update_sub` here would get overwritten (and its
# subscription silently dropped/GC'd) by any other script using the same name, e.g.
# ct_localizer_window.py - that was why opening the localizer killed this window.
_bmode_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _tick, name="fr5_ultrasound_window_stream"
)
print("[ultrasound window] window opened. Run `_bmode_update_sub = None` to stop streaming.")
