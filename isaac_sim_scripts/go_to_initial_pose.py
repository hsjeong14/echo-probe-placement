# Paste into Isaac Sim's Script Editor and run WHILE PLAYING to send the arm, from
# wherever it currently is, to the "initial" pose: D455 camera directly above the
# TABLE's center, aimed straight down. (Was framed around the phantom's center before,
# but the phantom is itself centered on the table, so anchoring to the table instead
# makes this pose stay correct even if the phantom mesh/case changes later - the table
# doesn't move.) This is now the single canonical "camera overhead" pose script -
# go_to_phantom_overview_pose.py covered the same goal and has been removed.
#
# How the target was derived: the user jogged the arm live (while Playing) until the
# D455 RGB feed showed the camera parallel to and centered over the phantom, then ran
# save_current_pose.py to capture those exact live joint angles. Running that saved pose
# through this project's own camera FK chain (LumifyHolder -> CameraBracket ->
# D455DepthSensor -> base_link->wrist3_link joint chain) confirmed empirically that the
# D455DepthSensor's LOCAL +X axis is the camera's true optical axis (it landed only ~19
# deg off vertical in the user's own hand-tuned pose - much closer than local +Y, the
# best guess from an earlier, less-informed attempt).
#
# Position/tilt (camera directly above the table's true cover-surface center, looking
# exactly straight down) converge exactly regardless of roll, but ROLL - rotation around
# that straight-down viewing axis, i.e. which way is "up" in the RGB frame - is a
# separate, independent parameter that centering alone does not fix; leaving it at
# whatever roll the user's hand-tuned reference pose happened to have (arbitrary, since
# they were only judging "is the phantom centered and big enough", not "is the table edge
# parallel to the image border") made the table itself look rotated/diagonal in frame.
# Fixed by explicitly aligning roll to the TABLE's own known world orientation (its
# xformOp:orient in the scene) instead: chose the camera's other body axis to line up
# with the table's own local X direction in world space (there are 4 ways to pair "which
# remaining axis" with "which sign" against the table's 2 in-plane directions - this is
# the one of those 4 that both converges exactly AND lands in a comfortable arm
# configuration; the others left the arm nearly fully extended or unreachable at this
# height). Target position: directly above the table's true cover-surface center (fetched
# and measured directly from the table asset's mesh - see the table-centering work in
# this project's history) in world X,Y - since the phantom is centered on that same
# point, this is identical to centering on the phantom - height 1.60m. Converges to an
# exact solve (0.0000mm / 0.0000deg residual) with generous margin on every joint (>87
# deg to the nearest limit), only ~38 deg of total joint movement away from the user's
# own already-good hand-tuned pose.
#
# NOT verified against the live D455 feed - this is a principled, deterministic choice
# (table's own real-world orientation), not a guess, but the mapping from "which body
# axis" to "which way it appears on screen" still couldn't be checked offline. If the
# table now looks rotated by some other fixed amount (most likely 90 deg one way or the
# other, or upside down), tell me which and it's a one-line fix (swap which table
# direction the axis is aligned to, or negate it).
#
# This reads the arm's LIVE current joint angles first (same Articulation approach as
# bake_current_pose_as_ready.py) purely to log how far it's moving - the actual target is
# the same fixed joint configuration regardless of where the arm starts, since it's an
# IK solution for a pose, not a path. PhysX's PD drive smoothly carries it there from
# wherever it currently is.
#
# Same caveat as always: verified against known scene geometry (table/phantom bbox,
# camera mounting offsets), NOT against the live D455 RGB feed (which this offline
# session can't see) - check the actual panel and nudge if the framing looks off.

import math

import omni.usd
from pxr import UsdPhysics

try:
    from isaacsim.core.prims import Articulation
except ImportError:
    from omni.isaac.core.articulations import Articulation  # older Isaac Sim fallback

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
INITIAL_POSE_DEG = {
    "j1": 13.2601,
    "j2": -93.5627,
    "j3": -40.4329,
    "j4": -123.7635,
    "j5": 88.0125,
    "j6": 12.0364,
}
PHYSICS_JOINT_BASE = "/World/Robot/Physics/"

_robot = Articulation("/World/Robot")
_robot.initialize()
_current_rad = _robot.get_joint_positions()
if _current_rad.ndim == 2:
    _current_rad = _current_rad[0]
_dof_names = _robot.dof_names
_current_deg = {jn: math.degrees(float(_current_rad[_dof_names.index(jn)])) for jn in JOINT_NAMES}

_stage = omni.usd.get_context().get_stage()
for jn in JOINT_NAMES:
    joint_prim = _stage.GetPrimAtPath(PHYSICS_JOINT_BASE + jn)
    if not joint_prim.IsValid():
        print(f"WARNING: joint prim not found for {jn}")
        continue
    deg = INITIAL_POSE_DEG[jn]
    UsdPhysics.DriveAPI(joint_prim, "angular").GetTargetPositionAttr().Set(deg)
    print(f"[go to initial pose] {jn}: {_current_deg[jn]:7.2f} deg -> {deg:7.2f} deg")

print("Drive targets set - the arm will move to the camera-overhead initial pose over the next few frames.")
