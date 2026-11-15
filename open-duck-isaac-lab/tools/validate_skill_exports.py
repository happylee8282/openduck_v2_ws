#!/usr/bin/env python3
"""Check three skill actors against RSL-RL and recorded Isaac inputs and exports."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "config/multiskill_policy.json"
    )
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from torch import nn
    import onnx
    from onnx.reference import ReferenceEvaluator
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization
    from open_duck_v2.multiskill import load_manifest
    from open_duck_v2.runtime import load_rsl_rl_actor
    from open_duck_v2.posture_runtime import load_posture_actor
    from open_duck_v2.training_policy import BoundedFullOutput, BoundedLegOutput

    torch.set_num_threads(2)
    manifest = load_manifest(args.bundle)
    data = np.load(args.rollout)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for skill in ("forward", "backward", "posture"):
        file = Path(manifest["skills"][skill]["resolved_checkpoint"])
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        info, weights = checkpoint["infos"], checkpoint["model_state_dict"]
        pose = skill == "posture"
        actor = (
            load_posture_actor(file, manifest["reference_path"])
            if pose
            else load_rsl_rl_actor(file)
        )
        dim = 103 if pose else 101
        # Select isolated cases: transition frames can intentionally blend two actors.
        columns = np.flatnonzero(data["labels"] == skill)
        raw = data["posture_observations" if pose else "observations"][:, columns]
        recorded = data["actions"][:, columns]
        mask = data["active_skill"][:, columns] == (
            2 if pose else 1 if skill == "backward" else 0
        )
        raw, recorded = raw[mask], recorded[mask]
        if not len(raw):
            raise ValueError("No actual policy samples for " + skill)
        indices = np.linspace(0, len(raw) - 1, min(2048, len(raw))).astype(int)
        x = torch.from_numpy(raw[indices])

        def hidden(prefix):
            keys = sorted(
                (
                    k
                    for k in weights
                    if k.startswith(prefix + ".") and k.endswith(".weight")
                ),
                key=lambda k: int(k.split(".")[1]),
            )
            return [weights[k].shape[0] for k in keys[:-1]]

        actual = ActorCritic(
            dim,
            weights["critic.0.weight"].shape[1],
            14,
            actor_hidden_dims=hidden("actor"),
            critic_hidden_dims=hidden("critic"),
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
                BoundedLegOutput()
                if activation == "tanh_legs"
                else BoundedFullOutput(),
            )
        actual.load_state_dict(weights, strict=True)
        actual.eval()
        norm = EmpiricalNormalization(dim, eps=0.01)
        norm.load_state_dict(checkpoint["obs_norm_state_dict"])
        norm.eval()
        with torch.inference_mode():
            y = actor(x)
            rsl = actual.act_inference(norm(x))
            if activation == "clamp":
                rsl = rsl.clamp(-1, 1)
            script = torch.jit.trace(actor, x[:1])
            script.save(str(args.output / (skill + ".ts.pt")))
            script_error = float((script(x) - y).abs().max())
            rsl_error = float((rsl - y).abs().max())
            trace_error = float(np.abs(y.numpy() - recorded[indices]).max())
        target = args.output / (skill + ".onnx")
        torch.onnx.export(
            actor,
            x[:1],
            str(target),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        model = onnx.load(str(target))
        onnx.checker.check_model(model)
        onnx_value = ReferenceEvaluator(model).run(None, {"obs": x[:64].numpy()})[0]
        errors = dict(
            rsl_rl_max_abs_error=rsl_error,
            recorded_sim_action_max_abs_error=trace_error,
            torchscript_max_abs_error=script_error,
            onnx_reference_max_abs_error=float(
                np.abs(onnx_value - y[:64].numpy()).max()
            ),
        )
        if max(errors.values()) > 1e-5:
            raise RuntimeError(str((skill, errors)))
        reports[skill] = dict(
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            obs_dim=dim,
            actual_simulator_samples=len(indices),
            onnx_samples=min(64, len(x)),
            errors=errors,
        )
    report = dict(
        status="passed",
        rollout=str(args.rollout.resolve()),
        skills=reports,
        scope="Actor and normalizer output parity. The multiskill scheduler, heading preprocessing, IK reference and motor target processing remain required.",
    )
    (args.output / "export_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
