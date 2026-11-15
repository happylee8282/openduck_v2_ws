"""전신 질량을 반영한 자세 참조와 제한된 residual 위치 제어.

IK로 만든 높이별 기준 관절각을 기본 목표로 사용하고, actor는 기준 위에
작은 residual만 더한다. 따라서 자세 정책은 보행 정책과 다른 103차원 계약을
갖는다.
"""

from pathlib import Path
import hashlib
import json

import torch
from .control import V2Control
from .profile import LIMITS, MAX_TARGET_SPEED


POSTURE_OBS_DIM = 103
POSTURE_CRITIC_DIM = 126
POSTURE_VERSION = "open-duck-mini-v2-posture-103-v1"


def posture_contract(reference_sha256, residual_scale=0.12):
    """자세 actor와 IK reference를 함께 묶는 버전 계약을 반환한다."""

    return dict(
        version=POSTURE_VERSION,
        observation_dim=POSTURE_OBS_DIM,
        critic_dim=POSTURE_CRITIC_DIM,
        action_dim=14,
        policy_hz=50,
        reference_sha256=reference_sha256,
        residual_scale=residual_scale,
        actor_fields="base101 with residual action history and fixed phase [1,0], desired_height_m, desired_height_rate_m_s",
        action="canonical reference joint angles plus bounded residual, then rate and physical limits",
        additional_sensor_requirements="canonical trunk gyro and specific-force acceleration; joint positions/velocities; both foot contacts",
    )


class PostureReference:
    """높이별 IK 해를 읽고 GPU에서 선형 보간하는 immutable 참조표."""

    def __init__(self, path, device="cpu"):
        """JSON을 읽어 높이순으로 정렬하고 hash·관절 한계를 검증한다."""

        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        # 생성 파일의 행 순서에 의존하지 않도록 높이 오름차순으로 정규화한다.
        rows = sorted(data["solutions"], key=lambda x: x["height"])
        self.heights = torch.tensor(
            [r["height"] for r in rows], device=device, dtype=torch.float32
        )
        self.joints = torch.tensor(
            [r["q"] for r in rows], device=device, dtype=torch.float32
        )
        self.rpy = torch.tensor(
            [[r["roll"], r["pitch"], 0.0] for r in rows],
            device=device,
            dtype=torch.float32,
        )
        self.root_x = torch.tensor(
            [r["root_x"] for r in rows], device=device, dtype=torch.float32
        )
        self.com = torch.tensor(
            [r["com"] for r in rows], device=device, dtype=torch.float32
        )
        limits = self.joints.new_tensor(LIMITS)
        if (
            not torch.isfinite(self.joints).all()
            or (self.joints < limits[:, 0] + 0.019).any()
            or (self.joints > limits[:, 1] - 0.019).any()
        ):
            raise ValueError("Posture reference exceeds physical joint stops")
        if (self.heights[1:] <= self.heights[:-1]).any():
            raise ValueError("Posture heights must be strictly ordered")
        self.stand_height = float(self.heights[-1])
        self.low_height = float(self.heights[0])
        self.stand_joints = self.joints[-1].clone()

    def sample(self, height):
        """요청 높이를 참조 범위로 제한하고 인접 IK 해 사이를 보간한다."""

        height = height.clamp(self.heights[0], self.heights[-1])
        hi = torch.searchsorted(self.heights, height.contiguous()).clamp(
            1, len(self.heights) - 1
        )
        lo = hi - 1
        u = (height - self.heights[lo]) / (self.heights[hi] - self.heights[lo])

        def interp(table):
            """1차원 scalar 표와 2차원 vector 표에 같은 비율을 적용한다."""

            ratio = u if table.ndim == 1 else u[:, None]
            return table[lo] + ratio * (table[hi] - table[lo])

        return (
            interp(self.joints),
            interp(self.rpy),
            interp(self.root_x),
            interp(self.com),
        )


def posture_command(time, stand_height, low_height, period=16.0):
    """16초 주기의 낮추기·유지·기립 높이와 높이 속도를 생성한다.

    5차 smoothstep을 사용해 전환의 속도와 가속도가 끝점에서 부드럽게 0이
    되도록 한다.
    """

    t = torch.remainder(time, period)
    down = ((t - 2.0) / 4.0).clamp(0, 1)
    up = ((t - 9.0) / 4.0).clamp(0, 1)

    def smooth(u):
        """끝점의 1·2차 미분이 0인 5차 smoothstep."""

        return 6 * u**5 - 15 * u**4 + 10 * u**3

    def rate(u):
        """4초 구간 시간에 대한 ``smooth``의 미분."""

        return 30 * u**2 * (1 - u) ** 2 / 4

    blend = smooth(down) - smooth(up)
    derivative = rate(down) - rate(up)
    extent = stand_height - low_height
    return stand_height - extent * blend, -extent * derivative


class PostureControl(V2Control):
    """IK 관절각에 actor residual을 더하는 자세 전용 제어기."""

    def __init__(
        self,
        num_envs,
        device,
        reference,
        residual_scale=0.12,
        dt=0.02,
        backlash=0.0087266463,
    ):
        """참조 standing pose를 HOME으로 사용하는 무지연 제어기를 만든다."""

        super().__init__(
            num_envs,
            device,
            dt=dt,
            max_delay=0,
            backlash=backlash,
            head_control_mode="stabilized",
            head_target_speed=1.0,
        )
        self.home = reference.stand_joints.expand(num_envs, -1).clone()
        self.reference_target = self.home.clone()
        self.residual_scale = residual_scale
        self.reset(torch.arange(num_envs, device=device))

    def process(self, actions, head_commands=None):
        """residual action을 기준 자세에 더하고 속도·관절·백래시 제한을 적용한다."""

        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Invalid posture action")
        self.history[:, 2] = self.history[:, 1].clone()
        self.history[:, 1] = self.history[:, 0].clone()
        self.actions = actions.clamp(-1, 1)
        self.history[:, 0] = self.actions
        # action=0이면 순수 IK reference를 추종하고 ±1은 최대 residual_scale
        # rad만큼 수정한다.
        target = (self.reference_target + self.residual_scale * self.actions).clamp(
            self.lower, self.upper
        )
        speed = torch.full_like(target, MAX_TARGET_SPEED)
        speed[:, 5:9] = 1.0
        self.motor_targets += torch.clamp(
            target - self.motor_targets, -speed * self.dt, speed * self.dt
        )
        self.actuator_targets = torch.clamp(
            self.actuator_targets,
            min=self.motor_targets - self.backlash,
            max=self.motor_targets + self.backlash,
        )
        self.actuator_targets = self.actuator_targets.clamp(self.lower, self.upper)
        return self.actuator_targets - self.offsets


def posture_phase(value):
    """보행 clock 대신 자세 actor에 고정 phase ``[1, 0]``를 제공한다."""

    return torch.stack((torch.ones_like(value), torch.zeros_like(value)), -1)
