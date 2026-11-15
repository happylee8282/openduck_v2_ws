import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch
from open_duck_v2.profile import (
    HOME,
    JOINT_NAMES,
    MOTOR_IDS,
    LIMITS,
    OBS_SLICES,
    MOTOR_REFERENCE,
    MAX_TARGET_SPEED,
    CONTRACT_VERSION,
    contract,
    contract_hash,
)
from open_duck_v2.control import (
    V2Control,
    EpisodeDelay,
    pack_actor_observation,
    servo_effort,
    quat_rotate_inverse,
)
from open_duck_v2.randomization import RobotRandomizer
from open_duck_v2.runtime import Actor, load_rsl_rl_actor, export_policy
from open_duck_v2.motion import load_motion

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_shape_layout_and_proprioception(self):
        c = V2Control(2, "cpu")
        obs = pack_actor_observation(
            torch.zeros(2, 3),
            torch.tensor([[0.0, 0.0, 9.81]]).repeat(2, 1),
            torch.zeros(2, 7),
            c.home,
            torch.zeros(2, 14),
            c.history,
            c.motor_targets,
            torch.ones(2, 2),
            torch.tensor([[1.0, 0.0]]).repeat(2, 1),
        )
        self.assertEqual(obs.shape, (2, 101))
        torch.testing.assert_close(
            obs[:, OBS_SLICES["joint_position_minus_home_rad"]], torch.zeros(2, 14)
        )
        self.assertEqual(OBS_SLICES["commands"], slice(6, 13))
        self.assertEqual(len(set(MOTOR_IDS)), 14)
        self.assertFalse(MOTOR_REFERENCE["hardware_calibrated"])

    def test_native_roundtrip_and_stops(self):
        c = V2Control(10, "cpu", backlash=0)
        native = c.home - c.offsets
        torch.testing.assert_close(native + c.offsets, c.home)
        for _ in range(40):
            old = c.motor_targets.clone()
            native = c.process(torch.empty(10, 14).uniform_(-2, 2))
            self.assertTrue(
                (c.motor_targets >= c.lower).all()
                and (c.motor_targets <= c.upper).all()
            )
            self.assertLessEqual(
                float((c.motor_targets - old).abs().max()),
                MAX_TARGET_SPEED * 0.02 + 1e-6,
            )
            torch.testing.assert_close(native + c.offsets, c.motor_targets)

    def test_home_collision_meshes_start_clear_of_the_ground(self):
        import xml.etree.ElementTree as ET
        from open_duck_v2.profile import SPAWN_HEIGHT
        from tools.validate_assets import home_collision_min_z

        root = ET.parse(ROOT / "Robots/open_duck_mini.urdf").getroot()
        min_z, link = home_collision_min_z(root)
        self.assertIn(link, ("foot_assembly", "foot_assembly_2"))
        self.assertGreaterEqual(SPAWN_HEIGHT + min_z, 0.002)
        self.assertLess(0.15 + min_z, -0.01)

    def test_collision_filters_preserve_cross_limb_and_remote_body_pairs(self):
        from open_duck_v2.collision import mechanical_neighbor_pairs

        pairs = mechanical_neighbor_pairs(ROOT / "Robots/open_duck_mini.urdf", 2)
        pairs = {frozenset(pair) for pair in pairs}
        self.assertIn(frozenset(("foot_assembly", "knee_and_ankle_assembly")), pairs)
        self.assertNotIn(frozenset(("foot_assembly", "foot_assembly_2")), pairs)
        self.assertNotIn(frozenset(("head_assembly", "trunk_assembly")), pairs)
        self.assertNotIn(frozenset(("hip_roll_assembly", "hip_roll_assembly_2")), pairs)

    def test_command_head_ignores_policy_and_limits_target_rate(self):
        c = V2Control(2, "cpu", head_control_mode="command", backlash=0)
        c.process(torch.ones(2, 14), torch.zeros(2, 4))
        torch.testing.assert_close(c.motor_targets[:, 5:9], torch.zeros(2, 4))
        torch.testing.assert_close(c.history[:, :, 5:9], torch.zeros(2, 3, 4))
        c.process(torch.ones(2, 14), torch.ones(2, 4) * 0.2)
        self.assertLessEqual(float(c.motor_targets[:, 5:9].abs().max()), 0.020001)
        c.reset(torch.tensor([0]))
        torch.testing.assert_close(c.motor_targets[0], c.home[0])

    def test_target_filter_reduces_first_step_without_changing_stops(self):
        c = V2Control(1, "cpu", target_filter_tau_s=0.04, backlash=0)
        before = c.motor_targets.clone()
        c.process(torch.ones(1, 14))
        self.assertLess(float((c.motor_targets - before).abs().max()), 0.1048)
        self.assertTrue(
            (c.motor_targets >= c.lower).all() and (c.motor_targets <= c.upper).all()
        )

    def test_nan_action_fails(self):
        with self.assertRaises(ValueError):
            V2Control(1, "cpu").process(torch.full((1, 14), float("nan")))

    def test_episode_delay_and_subset_reset(self):
        d = EpisodeDelay(2, 1, 2, "cpu")
        ids = torch.tensor([0, 1])
        d.reset(ids, torch.tensor([[10.0], [20.0]]), 2, 2)
        torch.testing.assert_close(
            d.push(torch.tensor([[1.0], [2.0]])), torch.tensor([[10.0], [20.0]])
        )
        d.reset(torch.tensor([0]), torch.tensor([[99.0]]), 0, 0)
        y = d.push(torch.tensor([[3.0], [4.0]]))
        self.assertEqual(y[0].item(), 3.0)
        self.assertEqual(y[1].item(), 20.0)

    def test_backlash_is_bounded_and_reset_has_no_leak(self):
        c = V2Control(1, "cpu", backlash=0.01)
        c.process(torch.ones(1, 14) * 0.01)
        torch.testing.assert_close(c.actuator_targets, c.home)
        c.reset(torch.tensor([0]))
        torch.testing.assert_close(c.history, torch.zeros_like(c.history))
        torch.testing.assert_close(c.motor_targets, c.home)

    def test_servo_effort_saturation_and_friction(self):
        e = servo_effort(
            torch.tensor([100.0, -100.0]),
            torch.zeros(2),
            torch.zeros(2),
            17.8,
            0.6,
            0.052,
            torch.full((2,), 3.35),
        )
        torch.testing.assert_close(e, torch.tensor([3.35, -3.35]))
        e = servo_effort(
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([1.0, -1.0]),
            0.0,
            0.0,
            0.052,
            torch.ones(2) * 3.35,
        )
        torch.testing.assert_close(e, torch.tensor([-0.052, 0.052]))

    def test_quaternion_and_specific_force_convention(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        g = torch.tensor([[0.0, 0.0, -9.81]])
        torch.testing.assert_close(
            quat_rotate_inverse(q, -g), torch.tensor([[0.0, 0.0, 9.81]])
        )
        torch.testing.assert_close(quat_rotate_inverse(q, g - g), torch.zeros(1, 3))


class StraightRewardTests(unittest.TestCase):
    def test_reverse_is_not_positive_progress_and_side_motion_is_costly(self):
        from open_duck_v2.straight_rewards import tracking_terms

        command = torch.tensor([[0.1, 0, 0]]).repeat(3, 1)
        v = torch.tensor([[0.1, 0, 0], [-0.1, 0, 0], [0.1, 0.13, 0]])
        t = tracking_terms(command, v, torch.zeros(3, 3))
        self.assertEqual(float(t["progress"][0]), 1)
        self.assertEqual(float(t["progress"][1]), -1)
        self.assertGreater(float(t["lateral_cost"][2]), 6)
        self.assertGreater(float(t["track_x"][0]), float(t["track_x"][1]))

    def test_stop_and_yaw_costs_do_not_saturate_to_no_penalty(self):
        from open_duck_v2.straight_rewards import tracking_terms

        cmd = torch.zeros(2, 7)
        v = torch.tensor([[0.0, 0, 0], [0.1, -0.13, 0]])
        w = torch.tensor([[0.0, 0, 0], [0, 0, 1.4]])
        t = tracking_terms(cmd, v, w)
        self.assertEqual(float(t["still"][0]), 1)
        self.assertEqual(float(t["idle_cost"][0]), 0)
        self.assertGreater(float(t["idle_cost"][1]), 10)
        self.assertGreater(float(t["yaw_cost"][1]), 10)

    def test_bounded_leg_output_matches_export_actor(self):
        from open_duck_v2.training_policy import BoundedLegOutput

        actor = Actor(output_activation="tanh_legs")
        x = torch.randn(4, 101)
        expected = BoundedLegOutput()(actor.net(x))
        torch.testing.assert_close(actor(x), expected)
        torch.testing.assert_close(actor(x)[:, 5:9], torch.zeros(4, 4))


class TerrainTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse(
            (ROOT / "open_duck_complex_terrain/open_duck_env.py").read_text()
        )
        node = next(n for n in source.body if isinstance(n, ast.ClassDef))
        names = {
            "_ground_at",
            "_base_height",
            "_feet_clearance",
            "_get_dones",
            "_phase",
        }
        cls = ast.ClassDef(
            name="Env",
            bases=[],
            keywords=[],
            body=[
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            decorator_list=[],
        )
        m = ast.Module(body=[cls], type_ignores=[])
        ast.fix_missing_locations(m)
        ns = {"torch": torch, "math": math, "quat_rotate_inverse": quat_rotate_inverse}
        exec(compile(m, "<original environment methods>", "exec"), ns)
        self.e = ns["Env"]()
        e = self.e
        e.num_envs = 1
        e.device = "cpu"
        e.step_dt = 0.02
        e.max_episode_length = 1000
        e.cfg = NS(termination_height=0.08, max_root_angle=0.75, gait_cycle_time_s=0.8)
        state = torch.tensor([[0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]])
        e.robot = NS(
            data=NS(
                root_state_w=state,
                body_pos_w=torch.tensor([[[0.0, 0.0, 0.05], [0.02, 0.0, 0.06]]]),
            )
        )
        e.foot_body_ids = torch.tensor([0, 1])
        e.episode_length_buf = torch.tensor([17])
        e.height_scanner = NS(
            data=NS(ray_hits_w=torch.tensor([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]]))
        )

    def test_vertical_translation_leaves_height_clearance_and_death_invariant(self):
        e = self.e
        before = (
            e._base_height().clone(),
            e._feet_clearance().clone(),
            e._get_dones()[0].clone(),
        )
        e.robot.data.root_state_w[:, 2] += 0.5
        e.robot.data.body_pos_w[:, :, 2] += 0.5
        e.height_scanner.data.ray_hits_w[:, :, 2] += 0.5
        torch.testing.assert_close(e._base_height(), before[0])
        torch.testing.assert_close(e._feet_clearance(), before[1])
        torch.testing.assert_close(e._get_dones()[0], before[2])

    def test_local_fall_detected_on_high_ground(self):
        self.e.robot.data.root_state_w[:, 2] = 0.56
        self.e.height_scanner.data.ray_hits_w[:, :, 2] = 0.5
        self.assertTrue(self.e._get_dones()[0].item())

    def test_missing_ground_is_not_silently_accepted(self):
        self.e.height_scanner.data.ray_hits_w[:] = float("inf")
        with self.assertRaises(RuntimeError):
            self.e._base_height()

    def test_phase_observation_changes_with_gait_clock(self):
        a = self.e._phase()
        self.e.episode_length_buf += 10
        self.assertFalse(torch.equal(a, self.e._phase()))


class FakeView:
    def __init__(self):
        self.values = {
            "masses": torch.ones(3, 2),
            "inertias": torch.eye(3).reshape(1, 1, 9).repeat(3, 2, 1),
            "coms": torch.zeros(3, 2, 7),
            "material_properties": torch.ones(3, 2, 3),
            "dof_max_forces": torch.full((3, 14), 3.35),
            "dof_armatures": torch.full((3, 14), 0.028),
        }

    def __getattr__(self, name):
        if name.startswith("get_"):
            return lambda: self.values[name[4:]].clone()
        if name.startswith("set_"):

            def setter(value, ids):
                self.values[name[4:]][ids] = value[ids].clone()

            return setter
        raise AttributeError(name)


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        v = FakeView()
        a = NS(
            joint_indices=slice(None),
            effort_limit=torch.full((3, 14), 3.35),
            effort_limit_sim=torch.full((3, 14), 3.35),
            stiffness=torch.full((3, 14), 17.8),
            damping=torch.full((3, 14), 0.6),
            armature=torch.full((3, 14), 0.028),
            coulomb_friction=torch.full((3, 14), 0.052),
        )
        self.robot = NS(root_physx_view=v, actuators={"servos": a}, num_joints=14)

        def writer(value, joint_ids, env_ids):
            v.values["dof_max_forces"][env_ids] = value.cpu()

        def armature(value, joint_ids, env_ids):
            v.values["dof_armatures"][env_ids] = value.cpu()

        self.robot.write_joint_effort_limit_to_sim = writer
        self.robot.write_joint_armature_to_sim = armature
        self.cfg = NS(
            randomize_domain=True,
            verify_randomization=True,
            body_mass_scale_range=(1.1, 1.1),
            base_com_offset_range=(0.002, 0.002),
            ground_static_friction_range=(0.8, 0.8),
            ground_dynamic_friction_range=(0.7, 0.7),
            ground_restitution_range=(0.0, 0.0),
            motor_effort_scale_range=(0.5, 0.5),
            joint_gain_scale_range=(1.0, 1.0),
            joint_friction_range=(0.052, 0.052),
            armature_scale_range=(1.0, 1.0),
        )

    def test_effort_written_inertia_scaled_and_other_environments_untouched(self):
        r = RobotRandomizer(self.robot, self.cfg, 0)
        r.reset(torch.tensor([1]))
        v = self.robot.root_physx_view.values
        torch.testing.assert_close(v["dof_max_forces"][1], torch.full((14,), 1.675))
        torch.testing.assert_close(v["dof_max_forces"][0], torch.full((14,), 3.35))
        torch.testing.assert_close(
            v["inertias"][1], torch.eye(3).reshape(1, 9).repeat(2, 1) * 1.1
        )
        r.reset(torch.tensor([1]))
        torch.testing.assert_close(v["masses"][1], torch.full((2,), 1.1))

    def test_material_pool_is_reused_across_partial_resets(self):
        self.cfg.material_buckets = 4
        self.cfg.ground_static_friction_range = (0.75, 1.45)
        self.cfg.ground_dynamic_friction_range = (0.65, 1.2)
        self.cfg.ground_restitution_range = (0.0, 0.04)
        r = RobotRandomizer(self.robot, self.cfg, 0)
        untouched = self.robot.root_physx_view.values["material_properties"][0].clone()
        seen = []
        for _ in range(40):
            r.reset(torch.tensor([1, 2]))
            seen.append(
                self.robot.root_physx_view.values["material_properties"][1:].clone()
            )
        values = torch.cat(seen).reshape(-1, 3)
        self.assertLessEqual(len(torch.unique(values, dim=0)), 4)
        self.assertTrue((values[:, 1] <= values[:, 0]).all())
        torch.testing.assert_close(
            self.robot.root_physx_view.values["material_properties"][0], untouched
        )

    def test_broken_writer_is_reported(self):
        self.robot.write_joint_effort_limit_to_sim = lambda *a, **kw: None
        r = RobotRandomizer(self.robot, self.cfg, 0)
        with self.assertRaisesRegex(RuntimeError, "readback"):
            r.reset(torch.tensor([1]))


class RuntimeTests(unittest.TestCase):
    def checkpoint(self, path, normalize=False):
        actor = Actor()
        info = {
            "open_duck_contract": contract(),
            "open_duck_contract_hash": contract_hash(),
            "empirical_normalization": normalize,
            "phase_period_s": 0.8,
        }
        data = {
            "model_state_dict": {
                "actor." + k: v for k, v in actor.net.state_dict().items()
            },
            "infos": info,
        }
        if normalize:
            data["obs_norm_state_dict"] = {
                "_mean": torch.ones(1, 101) * 2.0,
                "_std": torch.ones(1, 101) * 3.0,
            }
        torch.save(data, path)
        return actor

    def test_old_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "old.pt"
            torch.save({"model_state_dict": {}}, p)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_rsl_rl_actor(p)

    def test_normalization_and_torchscript_parity(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "model.pt"
            reference = self.checkpoint(p, True)
            actor = load_rsl_rl_actor(p)
            x = torch.randn(12, 101)
            torch.testing.assert_close(
                actor(x), reference.net((x - 2) / 3.01).clamp(-1, 1)
            )
            output = Path(t) / "policy.pt"
            export_policy(p, output)
            torch.testing.assert_close(actor(x), torch.jit.load(str(output))(x))
            self.assertTrue(output.with_suffix(".pt.contract.json").exists())
            metadata = json.loads(output.with_suffix(".pt.contract.json").read_text())
            self.assertTrue(metadata["empirical_normalization"])
            self.assertTrue(metadata["normalization_baked_into_model"])


class MotionTests(unittest.TestCase):
    def make_file(self, path, fps=100.0):
        order = list(reversed(JOINT_NAMES))
        pos = np.tile(np.array(HOME)[::-1], (101, 1))
        pos[:, order.index("head_yaw")] += 0.01 * np.sin(
            np.arange(101) / fps * 2 * np.pi
        )
        np.savez(
            path,
            joint_names=np.array(order),
            joint_pos=pos,
            fps=fps,
            root_height=np.ones(101) * 0.15,
            root_lin_vel=np.zeros((101, 3)),
            joint_frame=np.array("canonical-v2"),
            root_lin_vel_frame=np.array("body"),
        )

    def test_named_reorder_and_noninteger_fps_interpolation(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "ref.npz"
            self.make_file(p)
            m = load_motion(p, 50.0)
            self.assertEqual(len(m["joint_pos"]), 51)
            np.testing.assert_allclose(m["joint_pos"][0], HOME, atol=1e-6)
            self.assertEqual(m["fps"], 50.0)

    def test_missing_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_motion(ROOT / "data/does-not-exist.npz")

    def test_invalid_joint_limit_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "ref.npz"
            self.make_file(p)
            with np.load(p) as source:
                data = {k: source[k] for k in source.files}
            data["joint_pos"][:, 0] = 100.0
            np.savez(p, **data)
            with self.assertRaisesRegex(ValueError, "stops"):
                load_motion(p)

    def test_retarget_matches_real_knee_and_respects_canonical_stops(self):
        spec = importlib.util.spec_from_file_location(
            "retarget_v2", ROOT / "tools/retarget_bvh_to_open_duck.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        names = [
            "Hips",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
        ]
        channels = {n: ["Xrotation", "Yrotation", "Zrotation"] for n in names}
        slices = {n: slice(i * 3, i * 3 + 3) for i, n in enumerate(names)}
        frames = np.zeros((101, 21), dtype=np.float32)
        frames[:, slices["LeftLeg"].start] = np.sin(np.arange(101) / 10) * 20
        data = module.BvhData(channels, slices, 0.01, frames)
        self.assertEqual(module.find_joint(data, "left", "leg"), "LeftLeg")
        m = module.retarget(data, 50.0)
        self.assertEqual(float(m["fps"]), 50.0)
        self.assertTrue((m["joint_pos"] >= np.array(LIMITS)[:, 0]).all())
        self.assertTrue((m["joint_pos"] <= np.array(LIMITS)[:, 1]).all())
        self.assertLessEqual(abs(m["joint_vel"]).max(), 5.24)


class HardwareRecordTests(unittest.TestCase):
    def test_unmeasured_template_is_not_a_pass(self):
        from open_duck_v2.calibration import calibration_issues

        issues = calibration_issues(
            json.loads((ROOT / "config/calibration.template.json").read_text())
        )
        self.assertGreater(len(issues), 10)

    def test_replay_uses_contract_order_clock_and_joint_limits(self):
        spec = importlib.util.spec_from_file_location(
            "replay_v2", ROOT / "tools/replay_hardware_log.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actor = Actor()
        actor.phase_period_s = 0.8
        rows = []
        for i in range(10):
            rows.append(
                dict(
                    timestamp_s=i * 0.02,
                    joint_frame="canonical-v2",
                    contract_version=CONTRACT_VERSION,
                    joint_names=list(JOINT_NAMES),
                    gyro=[0.0] * 3,
                    accelerometer=[0.0, 0.0, 9.81],
                    commands=[0.0] * 7,
                    joint_pos=list(HOME),
                    joint_vel=[0.0] * 14,
                    contacts=[1.0, 1.0],
                )
            )
        result = list(module.replay(rows, actor))
        self.assertEqual(len(result), 10)
        self.assertEqual(len(result[-1]["observation"]), 101)
        self.assertEqual(result[-1]["motor_ids"], list(MOTOR_IDS))
        rows[1]["timestamp_s"] = 0.06
        with self.assertRaisesRegex(ValueError, "clock"):
            list(module.replay(rows, actor))

    def test_all_three_tasks_register_without_simulator_imports(self):
        import gymnasium as gym
        import open_duck_complex_terrain  # noqa: F401
        import open_duck_imitation  # noqa: F401

        for name in (
            "Isaac-Open-Duck-Direct-v0",
            "Isaac-Open-Duck-Rough-Direct-v0",
            "Isaac-Open-Duck-Imitation-Direct-v0",
        ):
            self.assertIn(name, gym.registry)


class HeadingRewardTests(unittest.TestCase):
    def test_world_heading_is_not_body_gyro_z(self):
        from open_duck_v2.straight_rewards import heading_from_quaternion

        # Fixed ZYX roll/pitch while yaw changes: body gyro z differs from yaw dot.
        roll, pitch = 0.3, -0.2
        yaw = torch.tensor([0.2, 0.201])
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q = torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            -1,
        )
        torch.testing.assert_close(heading_from_quaternion(q), yaw)
        self.assertNotAlmostEqual(np.cos(roll) * np.cos(pitch), 1.0, places=3)

    def test_heading_wrap_is_continuous_across_pi(self):
        from open_duck_v2.straight_rewards import wrap_angle

        difference = torch.tensor([-2 * np.pi + 0.01, 2 * np.pi - 0.01])
        torch.testing.assert_close(
            wrap_angle(difference), torch.tensor([0.01, -0.01]), atol=1e-6, rtol=1e-5
        )


class HeadingHoldTests(unittest.TestCase):
    def test_correction_sign_limit_and_stand_bypass(self):
        from open_duck_v2.heading import HeadingHold

        h = HeadingHold(2, kp=2, max_rate=0.35, filter_tau=0)
        h.reset(torch.tensor([0, 1]), torch.zeros(2))
        h.observe(torch.tensor([0.1, -1.0]))
        commands = torch.zeros(2, 7)
        commands[:, 0] = 0.1
        out = h.commands(commands)
        torch.testing.assert_close(out[:, 2], torch.tensor([-0.2, 0.35]))
        commands.zero_()
        torch.testing.assert_close(h.commands(commands), commands)
        self.assertTrue(torch.equal(h.reference, torch.zeros(2)))

    def test_forward_calibration_preserves_stop(self):
        from open_duck_v2.heading import HeadingHold

        h = HeadingHold(2, forward_scale=0.85)
        commands = torch.zeros(2, 7)
        commands[0, 0] = 0.1
        result = h.commands(commands)
        self.assertAlmostEqual(float(result[0, 0]), 0.085, places=6)
        torch.testing.assert_close(result[1], commands[1])

    def test_wrap_and_partial_reset(self):
        from open_duck_v2.heading import HeadingHold

        h = HeadingHold(2, filter_tau=0)
        h.reset(torch.tensor([0, 1]), torch.tensor([np.pi - 0.01, 0.0]))
        h.observe(torch.tensor([-np.pi + 0.01, 0.1]))
        torch.testing.assert_close(
            h.error, torch.tensor([-0.02, -0.1]), atol=1e-6, rtol=1e-5
        )
        h.reset(torch.tensor([0]), torch.tensor([0.5]))
        torch.testing.assert_close(h.error, torch.tensor([0.0, -0.1]))

    def test_attitude_requires_unit_wxyz(self):
        from open_duck_v2.heading import validated_imu_heading

        yaw = 0.2
        q = torch.tensor([[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]])
        torch.testing.assert_close(
            validated_imu_heading(q), torch.tensor([yaw], dtype=q.dtype)
        )
        for bad in (torch.zeros(1, 4), torch.full((1, 4), float("nan"))):
            with self.assertRaises(ValueError):
                validated_imu_heading(bad)

    def test_replay_applies_same_heading_command_preprocessor(self):
        spec = importlib.util.spec_from_file_location(
            "replay_heading", ROOT / "tools/replay_hardware_log.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actor = Actor()
        actor.phase_period_s = 0.54
        actor.control_profile = {"heading_hold_kp": 2.0, "heading_hold_filter_tau": 0.0}
        row = dict(
            timestamp_s=0.0,
            joint_frame="canonical-v2",
            contract_version=CONTRACT_VERSION,
            joint_names=list(JOINT_NAMES),
            gyro=[0.0] * 3,
            accelerometer=[0, 0, 9.81],
            commands=[0.1, 0, 0, 0, 0, 0, 0],
            joint_pos=list(HOME),
            joint_vel=[0.0] * 14,
            contacts=[1, 1],
        )
        with self.assertRaisesRegex(KeyError, "imu_quaternion_wxyz"):
            list(module.replay([row], actor))
        row["imu_quaternion_wxyz"] = [1, 0, 0, 0]
        next_row = dict(
            row,
            timestamp_s=0.02,
            imu_quaternion_wxyz=[np.cos(0.05), 0, 0, np.sin(0.05)],
        )
        result = list(module.replay([row, next_row], actor))
        self.assertAlmostEqual(result[0]["observation"][8], 0.0, places=6)
        self.assertAlmostEqual(result[1]["observation"][8], -0.2, places=6)


class GaitRewardTimingTests(unittest.TestCase):
    def test_contact_reward_matches_official_swing_apices(self):
        from open_duck_v2.straight_rewards import desired_contacts

        # Measured expert swing peaks: left at cycle .52, right near cycle 0.
        contact = desired_contacts(torch.tensor([0.0, 0.5]), 0.58, 0.25)
        self.assertEqual(contact.tolist(), [[True, False], [False, True]])
        # Legacy timing incorrectly rewarded stance at each of those swing peaks.
        legacy = desired_contacts(torch.tensor([0.0, 0.5]), 0.58, 0)
        self.assertTrue(bool(legacy[0, 1]))
        self.assertTrue(bool(legacy[1, 0]))


class StraightAssessmentTests(unittest.TestCase):
    def test_world_drift_cannot_pass_even_when_body_velocity_matches(self):
        spec = importlib.util.spec_from_file_location(
            "assess_straight", ROOT / "tools/assess_straight_walk.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fields = (
            [
                "vx",
                "vy",
                "yaw_rate",
                "x",
                "y",
                "z",
                "height",
                "tilt_rad",
                "contact_left",
                "contact_right",
                "torque_saturation",
                "action_clipping",
                "alive",
                "stable_contact_left",
                "stable_contact_right",
                "foot_clearance_left",
                "foot_clearance_right",
            ]
            + ["joint_" + str(i) for i in range(14)]
            + ["qw", "qx", "qy", "qz"]
        )
        t = (np.arange(1000) + 1) * 0.02
        a = np.zeros((1000, 1, len(fields)), dtype=np.float32)
        a[:, 0, 0] = 0.1
        a[:, 0, 3] = 0.1 * t
        a[:, 0, 6] = 0.17
        a[:, 0, 12] = 1
        a[:, 0, 31] = 1
        for i in range(2):
            phase = (np.arange(1000) / 27 + i * 0.5) % 1
            contact = phase < 0.5
            a[:, 0, 8 + i] = contact
            a[:, 0, 13 + i] = contact
            a[:, 0, 15 + i] = 0.04 + 0.02 * np.maximum(0, -np.sin(2 * np.pi * phase))
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            report = dict(
                checkpoint="synthetic",
                randomized=False,
                seed=43,
                duration_s=20,
                trials=[
                    dict(
                        command="forward",
                        trial=0,
                        survived=True,
                        stable_foot_landings=[36, 36],
                    )
                ],
            )
            (folder / "evaluation.json").write_text(json.dumps(report))

            def result():
                np.savez(
                    folder / "rollout.npz",
                    values=a,
                    columns=np.array(fields),
                    commands=np.array([[0.1, 0, 0]]),
                    dt=0.02,
                )
                return module.assess(folder / "evaluation.json")

            self.assertTrue(result()["accepted"])
            yaw = 0.025 * t
            a[:, 0, 31] = np.cos(yaw / 2)
            a[:, 0, 34] = np.sin(yaw / 2)
            r = result()
            self.assertFalse(r["accepted"])
            self.assertFalse(r["trials"][0]["checks"]["final_heading_at_most_0_12_rad"])
            a[:, 0, 31] = 1
            a[:, 0, 34] = 0
            a[:, 0, 4] = 0.02 * t
            r = result()
            self.assertFalse(r["trials"][0]["checks"]["net_lateral_at_most_0_15_m"])

    def test_cycle_mean_removes_sway_but_retains_drift(self):
        spec = importlib.util.spec_from_file_location(
            "assess_straight", ROOT / "tools/assess_straight_walk.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        signal = 0.2 * np.sin(np.arange(270) * 2 * np.pi / 27) + 0.03
        np.testing.assert_allclose(
            module.cycle_average(signal[:, None], 27), 0.03, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
