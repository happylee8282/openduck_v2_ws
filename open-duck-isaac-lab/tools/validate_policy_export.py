#!/usr/bin/env python3
"""Validate the actor against RSL-RL, real rollout inputs, TorchScript and ONNX."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from torch import nn
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization
    import onnx
    from onnx.reference import ReferenceEvaluator
    from open_duck_v2.runtime import load_rsl_rl_actor, export_policy
    from open_duck_v2.training_policy import BoundedFullOutput, BoundedLegOutput

    torch.set_num_threads(2)
    data = np.load(args.rollout)
    raw = data["observations"].reshape(-1, 101)
    recorded = data["actions"].reshape(-1, 14)
    indices = np.linspace(0, len(raw) - 1, min(2048, len(raw))).astype(int)
    x = torch.from_numpy(raw[indices])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = checkpoint["model_state_dict"]
    info = checkpoint["infos"]

    def dims(prefix):
        keys = sorted(
            [
                k
                for k in weights
                if k.startswith(prefix + ".") and k.endswith(".weight")
            ],
            key=lambda k: int(k.split(".")[1]),
        )
        return [weights[k].shape[0] for k in keys[:-1]]

    actual = ActorCritic(
        101,
        222,
        14,
        actor_hidden_dims=dims("actor"),
        critic_hidden_dims=dims("critic"),
        activation="elu",
    )
    if info.get("actor_hidden_activation") == "silu":
        for i, layer in enumerate(actual.actor):
            if isinstance(layer, nn.ELU):
                actual.actor[i] = nn.SiLU()
    activation = info.get("actor_output_activation", "clamp")
    if activation in ("tanh", "tanh_legs"):
        actual.actor.add_module(
            "bounded_output",
            BoundedLegOutput() if activation == "tanh_legs" else BoundedFullOutput(),
        )
    actual.load_state_dict(weights, strict=True)
    actual.eval()
    normalizer = EmpiricalNormalization(101, eps=0.01)
    normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
    normalizer.eval()
    actor = load_rsl_rl_actor(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = actor(x)
        rsl = actual.act_inference(normalizer(x))
        if activation == "clamp":
            rsl = rsl.clamp(-1, 1)
        rsl_error = float((expected - rsl).abs().max())
        trace_error = float(np.abs(expected.numpy() - recorded[indices]).max())
    export_policy(args.checkpoint, args.output / "flat_straight_actor.ts.pt")
    export_policy(args.checkpoint, args.output / "flat_straight_actor.onnx")
    script = torch.jit.load(str(args.output / "flat_straight_actor.ts.pt")).eval()
    with torch.inference_mode():
        script_error = float((script(x) - expected).abs().max())
    model = onnx.load(args.output / "flat_straight_actor.onnx")
    onnx.checker.check_model(model)
    ref = ReferenceEvaluator(model)
    small = x[:64]
    onnx_value = ref.run(None, {"obs": small.numpy()})[0]
    onnx_error = float(np.abs(onnx_value - expected[:64].numpy()).max())
    errors = dict(
        rsl_rl_max_abs_error=rsl_error,
        recorded_sim_action_max_abs_error=trace_error,
        torchscript_max_abs_error=script_error,
        onnx_reference_max_abs_error=onnx_error,
    )
    if max(errors.values()) > 1e-5:
        raise RuntimeError(f"Export parity failed: {errors}")
    report = dict(
        status="passed",
        checkpoint=str(args.checkpoint.resolve()),
        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_rollout=str(args.rollout.resolve()),
        actual_simulator_input_samples=len(indices),
        onnx_samples=len(small),
        errors=errors,
        scope="Actor output parity on actual simulator inputs. Heading command preprocessor and motor target processing remain required outside standalone actor exports.",
        control_profile=actor.control_profile,
    )
    (args.output / "export_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
