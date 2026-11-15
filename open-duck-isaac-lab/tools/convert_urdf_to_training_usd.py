from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _set_config(import_config, name: str, value):
    setter_name = f"set_{name}"
    if hasattr(import_config, setter_name):
        getattr(import_config, setter_name)(value)
    elif hasattr(import_config, name):
        setattr(import_config, name, value)
    else:
        raise RuntimeError(
            f"Required URDF ImportConfig field {name!r} is unavailable; use the documented Isaac Sim version"
        )


def _has_api(prim, api_name: str) -> bool:
    return api_name in [str(name) for name in prim.GetAppliedSchemas()]


def _find_root_link(stage):
    candidates = (
        "trunk_assembly",
        "trunk",
        "base",
        "body",
        "torso",
    )
    for keyword in candidates:
        for prim in stage.Traverse():
            if keyword in prim.GetName().lower() and _has_api(
                prim, "PhysicsRigidBodyAPI"
            ):
                return prim
    for prim in stage.Traverse():
        if _has_api(prim, "PhysicsRigidBodyAPI"):
            return prim
    return None


def _postprocess_training_usd(stage, output_usd: Path):
    from pxr import UsdPhysics

    root_link = _find_root_link(stage)
    if root_link is None:
        raise RuntimeError("No rigid body link found after URDF import.")

    for prim in stage.Traverse():
        if (
            _has_api(prim, "PhysicsArticulationRootAPI")
            and prim.GetPath() != root_link.GetPath()
        ):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"[INFO] Removed ArticulationRootAPI from {prim.GetPath()}")

    if not _has_api(root_link, "PhysicsArticulationRootAPI"):
        UsdPhysics.ArticulationRootAPI.Apply(root_link)
        print(f"[INFO] Applied ArticulationRootAPI to {root_link.GetPath()}")

    for prim in stage.Traverse():
        if prim.GetName().lower() == "root_joint":
            prim.SetActive(False)
            print(f"[INFO] Deactivated unexpected root_joint at {prim.GetPath()}")

    stage.Export(str(output_usd))
    print(f"[OK] Wrote training USD: {output_usd}")
    print(f"[OK] Floating root link: {root_link.GetPath()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-fixed-joints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--convex-decomp", action="store_true", default=False)
    parser.add_argument(
        "--self-collision", action=argparse.BooleanOptionalAction, default=True
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    simulation_app.update()

    project_root = Path(__file__).resolve().parents[1]
    robots_dir = project_root / "Robots"
    urdf_path = robots_dir / "open_duck_mini.urdf"
    output_usd = robots_dir / "open_duck_mini_training.usd"

    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)

    omni.usd.get_context().new_stage()
    simulation_app.update()
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config.")

    # Critical for legged locomotion: this must be False, otherwise the body is fixed to world.
    _set_config(import_config, "fix_base", False)
    _set_config(import_config, "make_default_prim", True)
    _set_config(import_config, "create_physics_scene", False)
    _set_config(import_config, "merge_fixed_joints", args.merge_fixed_joints)
    _set_config(import_config, "self_collision", args.self_collision)
    _set_config(import_config, "collision_from_visuals", False)
    _set_config(import_config, "convex_decomp", args.convex_decomp)
    _set_config(import_config, "import_inertia_tensor", True)
    _set_config(import_config, "distance_scale", 1.0)
    _set_config(import_config, "density", 0.0)
    _set_config(import_config, "default_drive_strength", 0.0)
    _set_config(import_config, "default_position_drive_damping", 0.0)

    status, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError("URDF import failed.")
    print(f"[INFO] Importer returned articulation path: {imported_path}")

    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    _postprocess_training_usd(stage, output_usd)
    simulation_app.close()


if __name__ == "__main__":
    main()
