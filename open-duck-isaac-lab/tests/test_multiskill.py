import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from open_duck_v2.control import V2Control
from open_duck_v2.multiskill import HybridControl
from open_duck_v2.posture import (
    PostureReference,
    PostureControl,
    posture_command,
    posture_contract,
    posture_phase,
)
from open_duck_v2.posture_runtime import load_posture_actor
from open_duck_v2.profile import LIMITS
from open_duck_v2.runtime import Actor, load_rsl_rl_actor
from open_duck_v2.straight_rewards import tracking_terms

ROOT = Path(__file__).resolve().parents[1]


class SkillControlTests(unittest.TestCase):
    def setUp(self):
        self.reference = PostureReference(ROOT / "config/posture_reference_v1.json")

    def test_forward_controller_exactly_preserved(self):
        old = V2Control(3, "cpu", max_delay=0, head_control_mode="stabilized")
        new = HybridControl(
            3, "cpu", self.reference, max_delay=0, head_control_mode="stabilized"
        )
        generator = torch.Generator().manual_seed(8)
        for _ in range(30):
            action = torch.rand(3, 14, generator=generator) * 2 - 1
            torch.testing.assert_close(
                new.process(action), old.process(action), atol=0, rtol=0
            )
            torch.testing.assert_close(new.history, old.history, atol=0, rtol=0)

    def test_mixed_posture_does_not_change_other_robots(self):
        old = V2Control(3, "cpu", max_delay=0, head_control_mode="stabilized")
        new = HybridControl(
            3, "cpu", self.reference, max_delay=0, head_control_mode="stabilized"
        )
        new.set_posture_mask(torch.tensor([False, True, False]))
        new.reference_target[1] = self.reference.joints[0]
        for _ in range(20):
            action = torch.zeros(3, 14)
            expected = old.process(action)
            actual = new.process(action)
            torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]], atol=0, rtol=0)
        self.assertGreater(
            float((new.motor_targets[1] - old.motor_targets[1]).abs().max()), 0.3
        )

    def test_handover_remaps_history_without_target_jump(self):
        control = HybridControl(
            2, "cpu", self.reference, max_delay=0, head_control_mode="stabilized"
        )
        control.motor_targets[:] = self.reference.stand_joints
        targets = control.motor_targets.clone()
        control.set_posture_mask(torch.tensor([True, False]))
        self.assertTrue(torch.equal(control.history[0], torch.zeros(3, 14)))
        torch.testing.assert_close(control.motor_targets, targets)
        control.set_posture_mask(torch.tensor([False, False]))
        expected = ((targets[0] - control.home[0]) / 0.25).clamp(-1, 1)
        torch.testing.assert_close(control.history[0], expected.expand(3, -1))
        torch.testing.assert_close(control.motor_targets, targets)

    def test_reference_bounds_and_rate_limits(self):
        control = PostureControl(2, "cpu", self.reference)
        bounds = torch.tensor(LIMITS)
        q, _, _, _ = self.reference.sample(torch.tensor([0.13, 0.165]))
        control.reference_target = q
        for _ in range(50):
            before = control.motor_targets.clone()
            control.process(torch.ones(2, 14))
            delta = (control.motor_targets - before).abs()
            self.assertLessEqual(float(delta[:, 5:9].max()), 0.020001)
            self.assertLessEqual(float(delta.max()), 5.24 * 0.02 + 1e-6)
            self.assertTrue(bool((control.motor_targets >= bounds[:, 0] + 0.019).all()))
            self.assertTrue(bool((control.motor_targets <= bounds[:, 1] - 0.019).all()))

    def test_smooth_squat_command_and_static_phase(self):
        h, v = posture_command(
            torch.tensor([0.0, 2.0, 6.0, 9.0, 13.0, 16.0]), 0.166, 0.13
        )
        torch.testing.assert_close(
            h, torch.tensor([0.166, 0.166, 0.13, 0.13, 0.166, 0.166])
        )
        torch.testing.assert_close(v, torch.zeros_like(v))
        phase = posture_phase(torch.tensor([0.0, 3.0, 9.0]))
        torch.testing.assert_close(phase, torch.tensor([[1.0, 0.0]]).expand(3, -1))

    def test_signed_progress_rewards_reverse_direction(self):
        cmd = torch.tensor([[-0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        v = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        terms = tracking_terms(cmd, v, torch.zeros(2, 3))
        torch.testing.assert_close(terms["progress"], torch.tensor([1.0, -1.0]))
        self.assertLess(float(terms["no_progress"][0]), float(terms["no_progress"][1]))


class PostureContractTests(unittest.TestCase):
    def test_distinct_dimension_reference_contract_and_actor_parity(self):
        reference = ROOT / "config/posture_reference_v1.json"
        sha = hashlib.sha256(reference.read_bytes()).hexdigest()
        spec = posture_contract(sha)
        actor = Actor(obs_dim=103, hidden_dims=(16, 12), output_activation="tanh")
        checkpoint = {
            "model_state_dict": {
                "actor." + k: v for k, v in actor.net.state_dict().items()
            },
            "infos": {
                "open_duck_contract": spec,
                "open_duck_contract_hash": hashlib.sha256(
                    json.dumps(spec, sort_keys=True).encode()
                ).hexdigest(),
                "actor_output_activation": "tanh",
                "control_profile": {"posture_residual_scale": 0.12},
            },
            "obs_norm_state_dict": {
                "_mean": torch.zeros(1, 103),
                "_std": torch.ones(1, 103) - 0.01,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "posture.pt"
            torch.save(checkpoint, path)
            loaded = load_posture_actor(path, reference)
            obs = torch.randn(8, 103)
            torch.testing.assert_close(loaded(obs), actor(obs))
            with self.assertRaises(ValueError):
                load_rsl_rl_actor(path)
            checkpoint["infos"]["open_duck_contract"]["reference_sha256"] = "changed"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_posture_actor(path, reference)


if __name__ == "__main__":
    unittest.main()
