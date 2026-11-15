#!/usr/bin/env python3
"""Import the known official 101/14 SiLU ONNX actor without executing Python pickles."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_duck_v2.runtime import Actor
from open_duck_v2.profile import contract, contract_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model = onnx.load(str(args.onnx), load_external_data=False)
    onnx.checker.check_model(model)
    expected = [
        "Sub",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Split",
        "Tanh",
    ]
    if [n.op_type for n in model.graph.node] != expected:
        raise ValueError(
            "Unsupported ONNX graph; this importer only supports the official v2 actor"
        )
    arrays = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    mean = arrays[model.graph.node[0].input[1]]
    reciprocal = arrays[model.graph.node[1].input[1]]
    gems = [n for n in model.graph.node if n.op_type == "Gemm"]
    dims = [arrays[n.input[1]].shape[1] for n in gems[:-1]]
    actor = Actor(hidden_dims=dims, hidden_activation="silu", output_activation="tanh")
    weights = {}
    for i, node in enumerate(gems):
        w = arrays[node.input[1]].T.copy()
        b = arrays[node.input[2]].copy()
        if i == 3:
            w, b = w[:14], b[:14]
        weights[f"{2 * i}.weight"] = torch.tensor(w)
        weights[f"{2 * i}.bias"] = torch.tensor(b)
    actor.net.load_state_dict(weights)
    actor.mean.copy_(torch.tensor(mean[None]))
    actor.scale.copy_(torch.tensor(1 / reciprocal[None]))
    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(42)
    inputs = (mean + rng.normal(size=(32, 101)).astype(np.float32) / reciprocal).astype(
        np.float32
    )
    ref = np.concatenate([evaluator.run(None, {"obs": x[None]})[0] for x in inputs])
    with torch.no_grad():
        got = actor(torch.from_numpy(inputs)).numpy()
    error = float(np.abs(ref - got).max())
    if error > 1e-5:
        raise ValueError(f"ONNX import output mismatch: {error}")
    std = (1 / reciprocal) - 0.01
    if not (std > 0).all():
        raise ValueError("Cannot represent ONNX normalization with RSL eps=.01")
    info = dict(
        open_duck_contract=contract(),
        open_duck_contract_hash=contract_hash(),
        empirical_normalization=True,
        actor_output_activation="tanh",
        actor_hidden_activation="silu",
        actor_hidden_dims=dims,
        phase_period_s=0.54,
        control_profile={
            "head_control_mode": "policy",
            "target_filter_tau_s": 0.0,
            "head_target_speed": 1.0,
            "reward_profile": "legacy",
        },
        initialization={
            "type": "official_onnx_import",
            "source": str(args.onnx.resolve()),
            "sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
            "repository": "apirrone/Open_Duck_Mini",
            "commit": "b23317a485b3cec7d8417f352478778b3475173c",
        },
    )
    checkpoint = dict(
        model_state_dict={"actor." + k: v for k, v in weights.items()},
        infos=info,
        iter=0,
        obs_norm_state_dict={
            "_mean": torch.tensor(mean[None]),
            "_std": torch.tensor(std[None]),
            "_var": torch.tensor(std[None] ** 2),
            "count": torch.tensor(100000000),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = dict(
        status="imported_not_retrained",
        source=info["initialization"],
        max_output_error=error,
        hidden_dims=dims,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
