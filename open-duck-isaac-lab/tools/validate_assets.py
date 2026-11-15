#!/usr/bin/env python3
"""Validate tracked URDF data and optional official-MJCF coordinate equivalence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_duck_v2.profile import JOINT_NAMES, URDF_TO_CANONICAL, HOME, LIMITS


def transform(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return t


def urdf_fk(root, canonical):
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = {link.get("name") for link in root.findall("link")} - children
    if roots != {"trunk_assembly"}:
        raise ValueError(f"Unexpected URDF roots {roots}")
    result = {"trunk_assembly": np.eye(4)}
    q = {n: v - o for n, v, o in zip(JOINT_NAMES, canonical, URDF_TO_CANONICAL)}
    todo = list(root.findall("joint"))
    while todo:
        progressed = False
        for j in todo[:]:
            parent = j.find("parent").get("link")
            if parent not in result:
                continue
            t = transform(j.find("origin"))
            if j.get("type") != "fixed":
                a = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(a * q[j.get("name")]).as_matrix()
                t = t @ rot
            result[j.find("child").get("link")] = result[parent] @ t
            todo.remove(j)
            progressed = True
        if not progressed:
            raise ValueError("Invalid URDF graph")
    return result


def home_collision_min_z(root):
    """Find the actual STL collision surface at HOME, in the trunk frame."""
    import trimesh

    fk = urdf_fk(root, HOME)
    lowest = float("inf")
    lowest_link = None
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            path = ROOT / "Robots" / mesh.get("filename").removeprefix("package://")
            vertices = trimesh.load_mesh(path, process=False).vertices.copy()
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            vertices *= scale
            matrix = fk[link.get("name")] @ transform(collision.find("origin"))
            z = (matrix @ np.c_[vertices, np.ones(len(vertices))].T)[2].min()
            if z < lowest:
                lowest, lowest_link = float(z), link.get("name")
    return lowest, lowest_link


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-mjcf", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    path = ROOT / "Robots/open_duck_mini.urdf"
    root = ET.parse(path).getroot()
    massive = []
    mass = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Missing inertia {link.get('name')}")
        m = float(inertial.find("mass").get("value"))
        mass += m
        if m < 1e-6:
            continue  # fixed frame markers merge at import
        v = inertial.find("inertia").attrib
        inertia = np.array(
            [
                [float(v["ixx"]), float(v["ixy"]), float(v["ixz"])],
                [float(v["ixy"]), float(v["iyy"]), float(v["iyz"])],
                [float(v["ixz"]), float(v["iyz"]), float(v["izz"])],
            ]
        )
        eigen = np.linalg.eigvalsh(inertia)
        if eigen[0] <= 0 or eigen[-1] > eigen[:2].sum() + 1e-12:
            raise ValueError(f"Invalid inertia {link.get('name')}")
        massive.append(link.get("name"))
    if abs(mass - 2.107140705) > 1e-6:
        raise ValueError(f"Unexpected mass {mass}")
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    if set(movable) != set(JOINT_NAMES):
        raise ValueError("Joint set differs from the contract")
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        i = JOINT_NAMES.index(j.get("name"))
        bound = j.find("limit")
        actual = (
            np.array([float(bound.get("lower")), float(bound.get("upper"))])
            + URDF_TO_CANONICAL[i]
        )
        np.testing.assert_allclose(actual, LIMITS[i], atol=1e-6)
    meshes = set()
    for mesh in root.findall(".//mesh"):
        filename = path.parent / mesh.get("filename").removeprefix("package://").lstrip(
            "/"
        )
        if not filename.is_file():
            raise FileNotFoundError(filename)
        meshes.add(str(filename.relative_to(ROOT)))
    result = dict(
        urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mass_kg=mass,
        physical_links=len(massive),
        moving_joints=len(movable),
        mesh_files=len(meshes),
        urdf_inertia_limits_meshes="passed",
        isaac_usd_readback="not_run",
    )
    from open_duck_v2.profile import SPAWN_HEIGHT

    min_z, lowest_link = home_collision_min_z(root)
    if min_z + SPAWN_HEIGHT < 0.002:
        raise ValueError(
            f"Initial collision geometry penetrates or nearly touches ground: {min_z + SPAWN_HEIGHT}"
        )
    result.update(
        home_collision_min_z_relative_trunk_m=min_z,
        lowest_collision_link=lowest_link,
        spawn_height_m=SPAWN_HEIGHT,
        initial_ground_clearance_m=min_z + SPAWN_HEIGHT,
    )
    if a.official_mjcf:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(a.official_mjcf))
        data = mujoco.MjData(model)
        bounds = np.array(LIMITS)
        rng = np.random.default_rng(42)
        max_pos = max_rot = 0.0
        for k in range(32):
            pose = np.clip(
                np.array(HOME) + (rng.uniform(-0.2, 0.2, 14) if k else 0),
                bounds[:, 0] + 0.02,
                bounds[:, 1] - 0.02,
            )
            for name, q in zip(JOINT_NAMES, pose):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
            mujoco.mj_forward(model, data)
            base = model.body("trunk_assembly").id
            rb = data.xmat[base].reshape(3, 3)
            pb = data.xpos[base]
            fk = urdf_fk(root, pose)
            for name in massive:
                b = model.body(name).id
                pos = rb.T @ (data.xpos[b] - pb)
                rot = rb.T @ data.xmat[b].reshape(3, 3)
                max_pos = max(max_pos, float(np.max(np.abs(pos - fk[name][:3, 3]))))
                max_rot = max(
                    max_rot,
                    float(Rotation.from_matrix(rot.T @ fk[name][:3, :3]).magnitude()),
                )
        if max_pos > 2e-5 or max_rot > 5e-5:
            raise ValueError(f"Canonical FK does not match: {max_pos},{max_rot}")
        result.update(
            official_mjcf_fk_poses=32,
            max_position_error_m=max_pos,
            max_rotation_error_rad=max_rot,
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
