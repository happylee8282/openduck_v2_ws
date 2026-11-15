"""버전 계약과 학습 normalizer를 보존하는 actor 로드·export 도구.

RSL-RL 전체 runner 없이 독립 추론 가능한 actor를 재구성한다. 체크포인트의
관측 의미, network 구조, activation과 normalization이 현재 코드와 정확히
일치하지 않으면 로드를 거부한다.
"""

from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from .profile import OBS_DIM, contract, contract_hash


class Actor(nn.Module):
    """RSL-RL actor와 같은 MLP 및 입력 normalizer를 가진 추론 module."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=14,
        output_activation="clamp",
        hidden_dims=(256, 256, 128),
        hidden_activation="elu",
    ):
        """checkpoint metadata로 복원할 수 있는 MLP 구조를 생성한다."""

        super().__init__()
        if hidden_activation not in ("elu", "silu"):
            raise ValueError("Unsupported hidden activation")
        activation = nn.ELU if hidden_activation == "elu" else nn.SiLU
        layers = []
        last = obs_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(last, width), activation()])
            last = width
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        # parameter가 아닌 buffer로 두어 TorchScript/ONNX export에도 포함한다.
        self.register_buffer("mean", torch.zeros(1, obs_dim))
        self.register_buffer("scale", torch.ones(1, obs_dim))

    def forward(self, obs):
        """관측을 정규화하고 계약에 지정된 출력 제한을 적용한다."""

        output = self.net((obs - self.mean) / self.scale)
        if self.output_activation in ("tanh", "tanh_legs"):
            output = torch.tanh(output)
            if self.output_activation == "tanh_legs":
                # command-held head 학습에서는 머리 4관절 action을 항상 0으로 둔다.
                output = torch.cat(
                    (output[:, :5], torch.zeros_like(output[:, 5:9]), output[:, 9:]), -1
                )
            return output
        return output.clamp(-1, 1)


def load_rsl_rl_actor(checkpoint_path, device="cpu", obs_dim=OBS_DIM, action_dim=14):
    """v2 101/14 계약의 RSL-RL 체크포인트를 독립 actor로 복원한다.

    계약 전체와 hash를 모두 비교하고, actor topology와 empirical normalizer를
    체크포인트에서 읽는다. 구형 정책은 차원만 맞춰 재사용하지 않는다.
    """

    if obs_dim != OBS_DIM or action_dim != 14:
        raise ValueError("Expected the new v2 101/14 contract")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    info = checkpoint.get("infos") or {}
    if (
        info.get("open_duck_contract_hash") != contract_hash()
        or info.get("open_duck_contract") != contract()
    ):
        raise ValueError(
            "Checkpoint has no matching v2 contract. Retrain using tools/launch.py; legacy 158/66 weights cannot be relabelled."
        )
    # critic과 optimizer를 제외하고 actor prefix의 weight만 추출한다.
    weights = {
        k.removeprefix("actor."): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("actor.")
    }
    activation = info.get("actor_output_activation", "clamp")
    if activation not in ("clamp", "tanh", "tanh_legs"):
        raise ValueError("Unsupported actor activation")
    # Sequential index를 수치로 정렬해 실제 hidden layer 폭을 복원한다.
    linear_keys = sorted(
        (k for k in weights if k.endswith(".weight")),
        key=lambda k: int(k.split(".")[0]),
    )
    hidden_dims = [weights[k].shape[0] for k in linear_keys[:-1]]
    actor = Actor(
        output_activation=activation,
        hidden_dims=hidden_dims,
        hidden_activation=info.get("actor_hidden_activation", "elu"),
    )
    actor.net.load_state_dict(weights, strict=True)
    norm = checkpoint.get("obs_norm_state_dict")
    if bool(norm) != bool(info.get("empirical_normalization", False)):
        raise ValueError("Checkpoint normalization metadata does not match its state")
    if norm:
        if norm["_mean"].shape != (1, OBS_DIM) or norm["_std"].shape != (1, OBS_DIM):
            raise ValueError("Normalizer dimension mismatch")
        actor.mean.copy_(norm["_mean"])
        # RSL-RL 2.3.3 EmpiricalNormalization이 추론 시 사용하는 eps까지
        # 동일하게 포함해야 학습 runner와 출력이 일치한다.
        actor.scale.copy_(norm["_std"] + 0.01)
    if (
        not all(torch.isfinite(p).all() for p in actor.state_dict().values())
        or (actor.scale <= 0).any()
    ):
        raise ValueError("Non-finite weights or invalid normalization scale")
    actor.empirical_normalization = bool(norm)
    actor.phase_period_s = float(info.get("phase_period_s", 0.8))
    actor.control_profile = info.get("control_profile", {})
    return actor.eval().to(device)


def export_policy(checkpoint_path, output_path, device="cpu"):
    """검증된 actor를 ONNX 또는 TorchScript와 계약 sidecar로 내보낸다."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actor = load_rsl_rl_actor(checkpoint_path, device)
    example = torch.zeros(1, OBS_DIM, device=device)
    if output_path.suffix == ".onnx":
        torch.onnx.export(
            actor,
            example,
            str(output_path),
            opset_version=17,
            dynamo=False,
            input_names=["obs"],
            output_names=["continuous_actions"],
            dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        )
    else:
        torch.jit.trace(actor, example).save(str(output_path))
    # 신경망 파일만으로는 heading 전처리·주기·normalization 여부를 알 수
    # 없으므로 같은 이름의 계약 JSON을 항상 함께 기록한다.
    output_path.with_suffix(output_path.suffix + ".contract.json").write_text(
        json.dumps(
            dict(
                contract(),
                phase_period_s=actor.phase_period_s,
                empirical_normalization=actor.empirical_normalization,
                normalization_baked_into_model=True,
                actor_output_activation=actor.output_activation,
                control_profile=actor.control_profile,
                command_preprocessor="IMU attitude heading hold at 50Hz"
                if actor.control_profile.get("heading_hold_kp", 0)
                else "direct commands",
                additional_sensor_requirements=["canonical trunk IMU attitude wxyz"]
                if actor.control_profile.get("heading_hold_kp", 0)
                else [],
            ),
            indent=2,
        )
        + "\n"
    )
    return actor


def install_checkpoint_contract_hook():
    """RSL-RL save에 v2 계약과 실행 profile을 기록하는 hook을 설치한다.

    학습 진입점이 process당 한 번 호출하는 전역 monkeypatch다. 저장된 정보는
    이후 strict loader가 정책과 환경 설정을 재구성하는 근거가 된다.
    """

    from rsl_rl.runners import OnPolicyRunner

    original = OnPolicyRunner.save

    def save(self, path, infos=None):
        """원래 save 전에 현재 환경의 actor 계약과 제어 metadata를 추가한다."""

        info = dict(infos or {})
        specification = contract()
        digest = contract_hash()
        if getattr(self.env.unwrapped.cfg, "motion_skill", "forward") == "posture":
            # 자세 정책은 101차원 보행 계약 대신 reference hash가 포함된 별도
            # 103차원 계약을 저장한다.
            import hashlib
            from .posture import posture_contract

            specification = posture_contract(
                self.env.unwrapped.reference.sha256,
                self.env.unwrapped.cfg.posture_residual_scale,
            )
            digest = hashlib.sha256(
                json.dumps(specification, sort_keys=True).encode()
            ).hexdigest()
        info.update(
            open_duck_contract=specification,
            open_duck_contract_hash=digest,
            empirical_normalization=bool(self.empirical_normalization),
            phase_period_s=float(self.env.unwrapped.cfg.gait_cycle_time_s),
            actor_output_activation=getattr(self, "actor_output_activation", "clamp"),
            actor_hidden_activation=getattr(self, "actor_hidden_activation", "elu"),
            initialization=getattr(self, "policy_initialization", None),
            warm_start=getattr(self, "warm_start_metadata", None),
            fixed_action_noise_std=getattr(self, "fixed_action_noise_std", None),
            critic_context=self.env.unwrapped.cfg.critic_context,
            control_profile={
                key: getattr(self.env.unwrapped.cfg, key)
                for key in (
                    "head_control_mode",
                    "target_filter_tau_s",
                    "head_target_speed",
                    "reward_profile",
                    "motion_skill",
                    "heading_hold_kp",
                    "heading_hold_max_rate",
                    "heading_hold_filter_tau",
                    "lateral_command_trim",
                    "forward_command_scale",
                )
            },
        )
        if getattr(self.env.unwrapped.cfg, "motion_skill", "forward") == "posture":
            info["control_profile"].update(
                posture_residual_scale=self.env.unwrapped.cfg.posture_residual_scale,
                posture_reference_sha256=self.env.unwrapped.reference.sha256,
            )
        original(self, path, infos=info)
        # checkpoint를 열지 않고도 계약을 검토할 수 있는 사람이 읽는 sidecar다.
        Path(path).parent.joinpath("policy_contract.json").write_text(
            json.dumps(
                dict(specification, phase_period_s=info["phase_period_s"]), indent=2
            )
            + "\n"
        )

    OnPolicyRunner.save = save


def environment_cfg_for_actor(actor):
    """actor의 저장 metadata에서 재생에 맞는 환경 config를 복원한다.

    reward profile은 해당 세대의 base config를 선택하고, checkpoint에 기록된
    제어 profile만 덮어쓴다. gait phase 주기도 actor와 동일하게 맞춘다.
    """

    from open_duck_complex_terrain.open_duck_env_cfg import (
        OpenDuckEnvCfg,
        OpenDuckStraightEnvCfg,
        OpenDuckStraightV3EnvCfg,
        OpenDuckStraightV4EnvCfg,
        OpenDuckStraightV5EnvCfg,
        OpenDuckStraightV6EnvCfg,
        OpenDuckBackwardEnvCfg,
    )

    profile = getattr(actor, "control_profile", {})
    cfg = (
        OpenDuckStraightV3EnvCfg()
        if profile.get("reward_profile") == "straight_v3"
        else (
            OpenDuckStraightEnvCfg()
            if profile.get("reward_profile") == "straight_v2"
            else OpenDuckEnvCfg()
        )
    )
    if profile.get("reward_profile") == "straight_v4":
        cfg = OpenDuckStraightV4EnvCfg()
    if profile.get("reward_profile") == "straight_v5":
        cfg = OpenDuckStraightV5EnvCfg()
    if profile.get("reward_profile") == "straight_v6":
        cfg = OpenDuckStraightV6EnvCfg()
    if profile.get("motion_skill") == "backward":
        cfg = OpenDuckBackwardEnvCfg()
    # 학습 시 실제 사용한 명령 전처리와 머리 제어값을 재생 환경에 복원한다.
    for key in (
        "head_control_mode",
        "target_filter_tau_s",
        "head_target_speed",
        "reward_profile",
        "heading_hold_kp",
        "heading_hold_max_rate",
        "heading_hold_filter_tau",
        "lateral_command_trim",
        "forward_command_scale",
    ):
        if key in profile:
            setattr(cfg, key, profile[key])
    cfg.gait_cycle_time_s = actor.phase_period_s
    return cfg
