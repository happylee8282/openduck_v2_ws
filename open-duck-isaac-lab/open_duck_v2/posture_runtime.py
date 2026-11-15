"""103차원 residual 자세 정책을 참조 파일과 함께 검증해 로드한다."""

import hashlib
import json
import torch
from .runtime import Actor
from .posture import PostureReference, posture_contract, POSTURE_OBS_DIM


def load_posture_actor(checkpoint_path, reference_path, device="cpu"):
    """자세 체크포인트·normalizer·IK 참조가 한 계약인지 확인한다.

    reference SHA-256과 residual scale도 계약에 들어가므로, 같은 network라도
    다른 자세표와 임의로 조합할 수 없다.
    """

    ref = PostureReference(reference_path, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    info = checkpoint.get("infos") or {}
    profile = info.get("control_profile", {})
    scale = profile.get("posture_residual_scale", 0.12)
    expected = posture_contract(ref.sha256, scale)
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if (
        info.get("open_duck_contract") != expected
        or info.get("open_duck_contract_hash") != digest
    ):
        raise ValueError("Posture checkpoint/reference contract mismatch")
    # RSL-RL state dict에서 actor 부분만 추출해 독립 추론 module을 재구성한다.
    weights = {
        k.removeprefix("actor."): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("actor.")
    }
    keys = sorted(
        (k for k in weights if k.endswith(".weight")),
        key=lambda k: int(k.split(".")[0]),
    )
    dims = [weights[k].shape[0] for k in keys[:-1]]
    actor = Actor(
        obs_dim=POSTURE_OBS_DIM,
        hidden_dims=dims,
        hidden_activation=info.get("actor_hidden_activation", "elu"),
        output_activation=info.get("actor_output_activation", "tanh"),
    )
    actor.net.load_state_dict(weights, strict=True)
    # 자세 정책은 empirical normalization을 전제로 학습됐으므로 누락을
    # 허용하지 않는다.
    norm = checkpoint.get("obs_norm_state_dict")
    if not norm or norm["_mean"].shape != (1, POSTURE_OBS_DIM):
        raise ValueError("Posture normalizer missing or wrong dimension")
    actor.mean.copy_(norm["_mean"])
    actor.scale.copy_(norm["_std"] + 0.01)
    if (
        not all(torch.isfinite(v).all() for v in actor.state_dict().values())
        or (actor.scale <= 0).any()
    ):
        raise ValueError("Invalid posture weights")
    actor.control_profile = profile
    actor.reference = ref
    actor.phase_period_s = 16.0
    return actor.eval().to(device)
