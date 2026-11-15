"""직진 PPO의 bounded actor 출력, warm-start와 고정 exploration 설정.

RSL-RL runner를 학습 진입 시 한 번 확장하여 추론 actor와 같은 tanh 출력을
사용하고, 검증 정책에서 actor만 이어받을 때 critic이 먼저 적응하도록 한다.
"""

import torch
from torch import nn


class BoundedLegOutput(nn.Module):
    """전체 출력을 tanh로 제한하고 command 제어 머리 action은 0으로 만든다."""

    def forward(self, value):
        """다리 10관절만 ``[-1, 1]`` action으로 반환한다."""

        value = torch.tanh(value)
        return torch.cat(
            (value[:, :5], torch.zeros_like(value[:, 5:9]), value[:, 9:]), dim=-1
        )


class BoundedFullOutput(nn.Module):
    """14관절 actor 평균 전체를 tanh 범위로 제한한다."""

    def forward(self, value):
        """모든 action을 ``[-1, 1]``로 매핑한다."""

        return torch.tanh(value)


def install_straight_policy_hook(fixed_noise_std=0.35, warm_start=None):
    """RSL-RL runner에 출력 제한과 재현 가능한 warm-start 절차를 설치한다.

    exploration 표준편차는 학습 parameter가 아니라 고정값이며 머리 관절은
    0.01로 낮춘다. warm-start 시 actor/normalizer만 복원하고 새 critic은
    40 update 동안 actor를 고정한 채 먼저 학습한다.
    """

    from rsl_rl.runners import OnPolicyRunner

    original_init = OnPolicyRunner.__init__
    original_load = OnPolicyRunner.load

    def configure(runner, std):
        """actor 끝에 bounded output을 붙이고 고정 action noise를 설정한다."""

        policy = runner.alg.policy
        head_held = runner.env.unwrapped.cfg.head_control_mode == "command"
        policy.actor.add_module(
            "bounded_output", BoundedLegOutput() if head_held else BoundedFullOutput()
        )
        with torch.no_grad():
            policy.std.fill_(std)
            policy.std[5:9] = 0.01
        # PPO가 std를 최적화하지 못하게 해 실험별 exploration을 고정한다.
        policy.std.requires_grad_(False)
        runner.fixed_action_noise_std = std
        runner.actor_output_activation = "tanh_legs" if head_held else "tanh"

    def init(self, *args, **kwargs):
        """원래 runner 초기화 뒤 actor warm-start와 자세 초기화를 수행한다."""

        original_init(self, *args, **kwargs)
        if warm_start:
            ckpt = torch.load(warm_start, map_location="cpu", weights_only=True)
            hidden = ckpt["infos"].get("actor_hidden_activation", "elu")
            if hidden == "silu":
                for i, layer in enumerate(self.alg.policy.actor):
                    if isinstance(layer, nn.ELU):
                        self.alg.policy.actor[i] = nn.SiLU()
            weights = {
                k.removeprefix("actor."): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("actor.")
            }
            # critic/optimizer는 가져오지 않고 actor와 관측 normalizer만 옮긴다.
            self.alg.policy.actor.load_state_dict(weights, strict=True)
            self.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
            self.obs_normalizer.until = 0
            self.actor_hidden_activation = hidden
            self.policy_initialization = ckpt["infos"].get("initialization")
            import hashlib
            from pathlib import Path

            self.warm_start_metadata = {
                "checkpoint": str(Path(warm_start).resolve()),
                "sha256": hashlib.sha256(Path(warm_start).read_bytes()).hexdigest(),
                "parent_iteration": ckpt.get("iter"),
                "parent_warm_start": ckpt["infos"].get("warm_start"),
                "actor_frozen_initial_updates": 40,
            }
        configure(self, fixed_noise_std)
        if (
            getattr(self.env.unwrapped.cfg, "motion_skill", "forward") == "posture"
            and not warm_start
        ):
            # 새 자세 actor의 마지막 층을 0으로 만들면 초기 출력이 residual=0,
            # 즉 검증된 IK reference 자체가 된다.
            for layer in reversed(list(self.alg.policy.actor)):
                if isinstance(layer, nn.Linear):
                    with torch.no_grad():
                        layer.weight.zero_()
                        layer.bias.zero_()
                    break
            self.policy_initialization = {
                "type": "zero residual over checked IK references",
                "reference_sha256": self.env.unwrapped.reference.sha256,
            }
        if warm_start:
            update = self.alg.update
            completed = [0]

            def critic_warmup_update():
                """초기 40 update에는 actor gradient를 끄고 critic만 준비한다."""

                train_actor = completed[0] >= 40
                for parameter in self.alg.policy.actor.parameters():
                    parameter.requires_grad_(train_actor)
                result = update()
                completed[0] += 1
                return result

            self.alg.update = critic_warmup_update

    def load(self, *args, **kwargs):
        """resume 시 저장 activation을 복원하고 bounded output을 다시 설치한다."""

        ckpt = torch.load(args[0], map_location="cpu", weights_only=True)
        info = ckpt.get("infos") or {}
        hidden = info.get("actor_hidden_activation", "elu")
        if hidden == "silu":
            for i, layer in enumerate(self.alg.policy.actor):
                if isinstance(layer, nn.ELU):
                    self.alg.policy.actor[i] = nn.SiLU()
        self.actor_hidden_activation = hidden
        self.policy_initialization = info.get("initialization")
        result = original_load(self, *args, **kwargs)
        if self.policy_initialization:
            self.obs_normalizer.until = 0
        configure(self, fixed_noise_std)
        return result

    OnPolicyRunner.__init__ = init
    OnPolicyRunner.load = load
