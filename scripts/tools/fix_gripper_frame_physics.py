"""Fix OpenArm USD physics layers (enactic/openarm_isaac_lab):
  1. `*_hand` bodies: author URDF mass/CoM/inertia (0.127 kg, parallel-link gripper `inertials.yaml`) — currently no
     `physics:mass` is authored, so PhysX falls back to 1.0 kg.
  2. `*_ee_tcp` bodies (= URDF `hand_tcp`, a massless TCP frame): author a negligible mass (1e-3 kg) — kept as a rigid
     body so existing `prim_path` / `body_names` references keep working.
  3. Joints attaching them (`*_hand`, `*_ee_tcp_joint`): exported as PhysicsRevoluteJoint with limits [0, 0] — convert to
     PhysicsFixedJoint (as in the URDF), removing the spurious DOFs.

    ./isaaclab.sh -p scripts/tools/fix_gripper_frame_physics.py <physics_layer.usd> [...] [--dry-run] --headless

This script was used once to produce the committed USD changes; it is kept for reproducibility.
"""
import argparse, sys
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("layers", nargs="+"); ap.add_argument("--dry-run", action="store_true")
AppLauncher.add_app_launcher_args(ap); a = ap.parse_args(); app = AppLauncher(a).app
from pxr import Usd, UsdPhysics, Sdf, Gf

HAND = dict(mass=0.127, com=Gf.Vec3f(0.000001, 0.0, -0.000549), inertia=Gf.Vec3f(1.65558e-4, 1.74660e-5, 1.73524e-4))  # URDF
TCP = dict(mass=1e-3, com=Gf.Vec3f(0, 0, 0), inertia=Gf.Vec3f(1e-7, 1e-7, 1e-7))
REVOLUTE_ATTRS = ["physics:axis", "physics:lowerLimit", "physics:upperLimit", "physics:JointEquivalentInertia",
                  "physxJoint:maxJointVelocity", "drive:angular:physics:damping", "drive:angular:physics:maxForce",
                  "drive:angular:physics:stiffness", "drive:angular:physics:targetPosition", "drive:angular:physics:type",
                  "drive:angular:physics:targetVelocity", "state:angular:physics:position", "state:angular:physics:velocity"]

for path in a.layers:
    stage = Usd.Stage.Open(path); layer = stage.GetRootLayer(); changed = []
    for prim in list(stage.Traverse()):
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (name.endswith("_hand") or name.endswith("_ee_tcp")):
            spec = HAND if name.endswith("_hand") else TCP
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr().Set(spec["mass"]); m.CreateCenterOfMassAttr().Set(spec["com"]); m.CreateDiagonalInertiaAttr().Set(spec["inertia"])
            changed.append(f"{prim.GetPath()}: mass={spec['mass']} com={tuple(spec['com'])} I={tuple(spec['inertia'])}")
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            b1 = [t.name for t in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()]
            if b1 and (b1[0].endswith("_hand") or b1[0].endswith("_ee_tcp")):
                primspec = layer.GetPrimAtPath(prim.GetPath()); primspec.typeName = "PhysicsFixedJoint"
                for an in REVOLUTE_ATTRS:
                    if prim.HasAttribute(an): prim.RemoveProperty(an)
                # 회전 관절용 API 스키마 제거 (DriveAPI:angular, JointStateAPI:angular, PhysxJointAPI 는 무해하나 정리)
                for api in ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]:
                    prim.RemoveAppliedSchema(api)
                changed.append(f"{prim.GetPath()}: PhysicsRevoluteJoint -> PhysicsFixedJoint (body1={b1[0]})")
    print(f"\n== {path}"); [print("  ", c) for c in changed]
    if not a.dry_run: layer.Save(); print("  saved.")
    else: print("  (dry-run, not saved)")
import os, threading; threading.Timer(5, lambda: os._exit(0)).start(); app.close()
