#!/usr/bin/env python3
"""Generate bounded squat targets from a recorded standing pose and the v2 URDF.

This computes reference joint angles only. The simulator always integrates the
floating base from actuator forces during playback.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_assets import urdf_fk, transform
from open_duck_v2.profile import LIMITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standing-rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--low-height", type=float, default=0.13)
    p.add_argument("--com-shift-x", type=float, default=-0.020)
    p.add_argument("--knots", type=int, default=20)
    args = p.parse_args()
    urdf = ET.parse(ROOT / "Robots/open_duck_mini.urdf").getroot()
    a = np.load(args.standing_rollout)
    ids = np.flatnonzero(a["labels"] == "stand")
    v = a["values"][-round(2 / float(a["dt"])) :, ids]
    cols = a["columns"].tolist()
    if len(ids) == 0 or not np.isfinite(v).all():
        raise ValueError("A finite standing rollout is required")
    q0 = np.mean(
        v[:, :, [cols.index("joint_" + str(i)) for i in range(14)]], axis=(0, 1)
    )
    quat = np.mean(
        v[:, :, [cols.index(k) for k in ("qw", "qx", "qy", "qz")]], axis=(0, 1)
    )
    quat /= np.linalg.norm(quat)
    rpy0 = Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")
    rpy0[2] = 0
    height0 = float(np.mean(v[:, :, cols.index("height")]))
    R0 = Rotation.from_euler("xyz", rpy0).as_matrix()
    P0 = np.array([0.0, 0.0, height0])
    feet = ["foot_assembly", "foot_assembly_2"]
    fk0 = urdf_fk(urdf, q0)
    foot_targets = [(R0 @ fk0[n][:3, :3], R0 @ fk0[n][:3, 3] + P0) for n in feet]
    links = []
    for link in urdf.findall("link"):
        i = link.find("inertial")
        m = float(i.find("mass").get("value"))
        if m > 1e-6:
            links.append((link.get("name"), m, transform(i.find("origin"))[:3, 3]))

    def com(fk, R, P):
        return sum(
            m * (R @ (fk[n][:3, :3] @ offset + fk[n][:3, 3]) + P)
            for n, m, offset in links
        ) / sum(m for _, m, _ in links)

    c0 = com(fk0, R0, P0)
    bounds = np.array(LIMITS)
    low = np.r_[bounds[:, 0] + 0.022, -0.12, -0.45, -0.15]
    high = np.r_[bounds[:, 1] - 0.022, 0.12, 0.6, 0.15]
    initial = np.r_[q0, 0, rpy0[1], rpy0[0]]
    initial = np.clip(initial, low + 1e-5, high - 1e-5)
    # Allow the CoM to shift backward inside the footprint while lowering.
    results = []
    for height in np.linspace(height0, args.low_height, args.knots):
        u = (height0 - height) / (height0 - args.low_height)
        target_com = c0.copy()
        target_com[0] += args.com_shift_x * u

        def residual(x):
            fk = urdf_fk(urdf, x[:14])
            R = Rotation.from_euler("xyz", [x[16], x[15], 0]).as_matrix()
            P = np.array([x[14], 0, height])
            r = []
            for name, (Rt, Pt) in zip(feet, foot_targets):
                r.extend(200 * (R @ fk[name][:3, 3] + P - Pt))
                r.extend(
                    10 * Rotation.from_matrix(Rt.T @ R @ fk[name][:3, :3]).as_rotvec()
                )
            r.extend(180 * (com(fk, R, P)[:2] - target_com[:2]))
            reg = np.ones(14) * 0.05
            reg[5:9] = [0.06, 0.06, 0.5, 0.5]
            r.extend(reg * (x[:14] - q0))
            r.extend(0.06 * (x[:14] - initial[:14]))
            r.extend([0.6 * (x[15] - rpy0[1]), 0.3 * (x[16] - rpy0[0])])
            return r

        sol = least_squares(
            residual,
            initial,
            bounds=(low, high),
            max_nfev=180,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        x = sol.x
        fk = urdf_fk(urdf, x[:14])
        R = Rotation.from_euler("xyz", [x[16], x[15], 0]).as_matrix()
        P = np.array([x[14], 0, height])
        cp = com(fk, R, P)
        row = dict(
            height=float(height),
            root_x=float(x[14]),
            pitch=float(x[15]),
            roll=float(x[16]),
            q=x[:14].tolist(),
            com=cp.tolist(),
            foot_pos_error_m=[
                float(np.linalg.norm(R @ fk[n][:3, 3] + P - Pt))
                for n, (_, Pt) in zip(feet, foot_targets)
            ],
            cost=float(sol.cost),
        )
        results.append(row)
        initial = x
        print("shifted", json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        version="IK-reference-v1",
        initial_height=height0,
        initial_q=q0.tolist(),
        initial_rpy=rpy0.tolist(),
        initial_com=c0.tolist(),
        solutions=results,
        provenance=dict(
            standing_rollout=str(args.standing_rollout.resolve()),
            standing_rollout_sha256=hashlib.sha256(
                args.standing_rollout.read_bytes()
            ).hexdigest(),
            low_height=args.low_height,
            com_shift_x=args.com_shift_x,
            joint_stop_margin_rad=0.022,
            feet_position_weight=200,
            feet_orientation_weight=10,
            com_weight=180,
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
