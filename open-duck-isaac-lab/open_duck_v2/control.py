"""관측 생성과 action→관절 목표 변환을 담당하는 공통 tensor 제어 코드.

Isaac Lab 없이도 CPU 회귀검사를 실행할 수 있도록 순수 PyTorch 연산으로
구성한다. 실물 런타임도 같은 순서와 수식을 재현해야 한다.
"""

from __future__ import annotations

import torch

from .profile import (
    ACTION_SCALE,
    HOME,
    LIMITS,
    MAX_TARGET_SPEED,
    OBS_DIM,
    URDF_TO_CANONICAL,
)


def quat_rotate_inverse(q, v):
    """정규화된 ``wxyz`` quaternion의 역회전으로 world 벡터를 body로 옮긴다."""

    w, xyz = q[..., :1], q[..., 1:]
    return (
        v * (2 * w * w - 1)
        - 2 * w * torch.cross(xyz, v, dim=-1)
        + 2 * xyz * (xyz * v).sum(-1, keepdim=True)
    )


def relative_height(world_z, ground_z):
    """월드 z가 아니라 현재 지면을 기준으로 한 높이를 계산한다."""

    return world_z - ground_z


def pack_actor_observation(
    gyro,
    accelerometer,
    commands,
    joint_pos,
    joint_vel,
    history,
    targets,
    contacts,
    phase,
):
    """센서와 제어 상태를 계약 순서의 101차원 actor 입력으로 결합한다.

    ``joint_pos``와 ``targets``는 canonical 좌표이고, ``history``는 최신
    action부터 과거 방향으로 3개를 저장한다. 마지막에는 차원과 유한값을
    검사해 잘못된 센서 입력이 정책까지 전파되지 않게 한다.
    """

    home = joint_pos.new_tensor(HOME)
    obs = torch.cat(
        (
            gyro,
            accelerometer,
            commands,
            joint_pos - home,
            joint_vel * 0.05,
            history[:, 0],
            history[:, 1],
            history[:, 2],
            targets,
            contacts.to(joint_pos.dtype),
            phase,
        ),
        -1,
    )
    if obs.shape[-1] != OBS_DIM or not torch.isfinite(obs).all():
        raise ValueError(
            f"Invalid v2 policy observation: shape={obs.shape}; expected finite {OBS_DIM} values"
        )
    return obs


def servo_effort(
    target,
    position,
    velocity,
    kp,
    damping,
    friction,
    limit,
    max_speed=MAX_TARGET_SPEED,
    torque_speed_curve=False,
    target_velocity=0.0,
    feedforward=0.0,
):
    """PD, Coulomb 마찰과 토크 제한을 적용한 servo effort를 계산한다.

    기본 경로는 측정되지 않은 토크-속도 곡선을 가정하지 않고 대칭 토크
    제한만 적용한다. ``torque_speed_curve``는 실제 계측 fit이 있을 때만
    사용하기 위한 선택 기능이다.
    """

    effort = (
        kp * (target - position)
        + damping * (target_velocity - velocity)
        - friction * torch.tanh(velocity / 0.02)
        + feedforward
    )
    if torque_speed_curve:
        # 속도에 따라 양/음 방향 포화 경계를 좁히는 선택적 선형 근사다.
        top = torch.clamp(limit * (1 - velocity / max_speed), min=-limit, max=limit)
        bottom = torch.clamp(limit * (-1 - velocity / max_speed), min=-limit, max=limit)
        return torch.clamp(effort, min=bottom, max=top)
    return torch.clamp(effort, min=-limit, max=limit)


class EpisodeDelay:
    """환경별 episode 고정 지연을 구현하는 원형 버퍼.

    병렬 환경마다 하나의 lag를 뽑고 episode 동안 유지한다. 일부 환경만
    reset할 때도 다른 환경의 지연 이력이 손상되지 않는다.
    """

    def __init__(self, num_envs, width, max_delay, device):
        """``max_delay + 1`` 길이의 원형 버퍼와 환경별 lag를 만든다."""

        if max_delay < 0:
            raise ValueError("Negative delay")
        self.buffer = torch.zeros(max_delay + 1, num_envs, width, device=device)
        self.lags = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = 0

    def reset(self, env_ids, value, minimum=0, maximum=None):
        """선택 환경의 전체 지연 이력을 안전한 초기값으로 채우고 lag를 뽑는다."""

        maximum = self.buffer.shape[0] - 1 if maximum is None else maximum
        if not 0 <= minimum <= maximum < self.buffer.shape[0]:
            raise ValueError("Invalid delay bounds")
        self.buffer[:, env_ids] = value
        self.lags[env_ids] = torch.randint(
            minimum, maximum + 1, (len(env_ids),), device=self.buffer.device
        )

    def push(self, value):
        """최신 값을 기록하고 각 환경의 고정 lag 위치 값을 반환한다."""

        self.buffer[self.cursor] = value
        rows = (self.cursor - self.lags) % len(self.buffer)
        out = self.buffer[
            rows, torch.arange(value.shape[0], device=value.device)
        ].clone()
        self.cursor = (self.cursor + 1) % len(self.buffer)
        return out


class V2Control:
    """14차원 정책 action을 제한된 URDF 관절 목표로 변환하는 상태 제어기.

    처리 순서는 action history 갱신 → episode delay → canonical 목표각 →
    머리 모드 → 저역통과 필터 → 속도 제한 → 백래시 → URDF 좌표 변환이다.
    ``motor_targets``는 명령 측, ``actuator_targets``는 백래시 뒤 물리 측
    위치를 뜻한다.
    """

    def __init__(
        self,
        num_envs,
        device,
        dt=0.02,
        max_delay=3,
        backlash=0.0087266463,
        head_control_mode="policy",
        target_filter_tau_s=0.0,
        head_target_speed=1.0,
    ):
        """병렬 환경 수와 50Hz 제어 profile에 맞는 상태 tensor를 초기화한다."""

        if (
            head_control_mode not in ("policy", "command", "stabilized")
            or target_filter_tau_s < 0
        ):
            raise ValueError("Invalid controller profile")
        self.head_control_mode = head_control_mode
        self.target_filter_tau_s = target_filter_tau_s
        self.head_target_speed = head_target_speed
        self.dt = dt
        self.home = (
            torch.tensor(HOME, device=device, dtype=torch.float32)
            .expand(num_envs, -1)
            .clone()
        )
        self.offsets = torch.tensor(URDF_TO_CANONICAL, device=device)
        bounds = torch.tensor(LIMITS, device=device)
        # 물리 hard stop을 직접 치지 않도록 양쪽에 0.02rad를 남긴다.
        self.lower, self.upper = bounds[:, 0] + 0.02, bounds[:, 1] - 0.02
        self.actions = torch.zeros_like(self.home)
        self.history = torch.zeros(num_envs, 3, 14, device=device)
        self.motor_targets = self.home.clone()
        self.actuator_targets = self.home.clone()
        self.backlash = backlash
        self.delay = EpisodeDelay(num_envs, 14, max_delay, device)
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids, minimum_delay=0, maximum_delay=0):
        """선택 환경의 action 이력, 목표각과 지연 버퍼를 HOME으로 되돌린다."""

        self.actions[env_ids] = 0
        self.history[env_ids] = 0
        self.motor_targets[env_ids] = self.home[env_ids]
        self.actuator_targets[env_ids] = self.home[env_ids]
        self.delay.reset(
            env_ids, torch.zeros_like(self.home[env_ids]), minimum_delay, maximum_delay
        )

    def process(self, actions, head_commands=None):
        """정책 action을 한 제어 tick 진행하고 URDF 좌표 목표를 반환한다."""

        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Non-finite or incorrectly shaped v2 action")
        self.history[:, 2] = self.history[:, 1].clone()
        self.history[:, 1] = self.history[:, 0].clone()
        self.actions = actions.clamp(-1, 1)
        if self.head_control_mode == "command":
            # command 모드에서는 actor의 4개 머리 출력을 관측 이력에서도 제거한다.
            self.actions[:, 5:9] = 0.0
        self.history[:, 0] = self.actions
        delayed = self.delay.push(self.actions)
        target = torch.clamp(
            self.home + ACTION_SCALE * delayed, min=self.lower, max=self.upper
        )
        max_speed = torch.full_like(self.home, MAX_TARGET_SPEED)
        if self.head_control_mode == "command":
            head_commands = (
                self.home[:, 5:9] if head_commands is None else head_commands
            )
            if (
                head_commands.shape != self.home[:, 5:9].shape
                or not torch.isfinite(head_commands).all()
            ):
                raise ValueError("Invalid head commands")
            target[:, 5:9] = head_commands.clamp(self.lower[5:9], self.upper[5:9])
            max_speed[:, 5:9] = self.head_target_speed
        if self.head_control_mode == "stabilized":
            if head_commands is not None:
                target[:, 5:9] += head_commands
                target[:, 5:9] = target[:, 5:9].clamp(self.lower[5:9], self.upper[5:9])
            max_speed[:, 5:9] = self.head_target_speed
        if self.target_filter_tau_s:
            import math

            alpha = 1 - math.exp(-self.dt / self.target_filter_tau_s)
            target = self.motor_targets + alpha * (target - self.motor_targets)
        max_change = max_speed * self.dt
        self.motor_targets = self.motor_targets + torch.clamp(
            target - self.motor_targets, -max_change, max_change
        )
        # Play operator는 가상 관절이 아니라 전달계의 유격/데드밴드를 모델링한다.
        # motor target이 밴드 밖으로 나갈 때만 물리 actuator target이 따라간다.
        self.actuator_targets = torch.clamp(
            self.actuator_targets,
            min=self.motor_targets - self.backlash,
            max=self.motor_targets + self.backlash,
        )
        self.actuator_targets = torch.clamp(
            self.actuator_targets, min=self.lower, max=self.upper
        )
        return self.actuator_targets - self.offsets


def readback_close(actual, expected, label, atol=1e-5):
    """PhysX에 쓴 값과 재조회 값이 다르면 randomization 실패로 처리한다."""

    if not torch.allclose(actual.to(expected.device), expected, atol=atol, rtol=1e-4):
        raise RuntimeError(
            f"{label} simulator readback does not match requested values"
        )
