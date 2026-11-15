"""trunk 정렬 IMU 자세로 직진 명령의 heading을 유지한다.

월드 위치나 root pose를 강제로 고정하지 않는다. 101차원 actor가 받기 전에
yaw-rate 명령만 보정하는 50Hz 상태 전처리기이며, 실물 런타임은 canonical
축의 ``wxyz`` IMU quaternion을 제공해야 한다.
"""

import math

import torch
from .straight_rewards import heading_from_quaternion, wrap_angle


def validated_imu_heading(quaternion):
    """IMU quaternion의 shape·유한값·단위 norm을 확인하고 world yaw를 구한다."""

    if quaternion.shape[-1] != 4 or not torch.isfinite(quaternion).all():
        raise ValueError("IMU attitude must be finite canonical wxyz")
    norm = quaternion.norm(dim=-1)
    if ((norm - 1).abs() > 0.01).any():
        raise ValueError("IMU quaternion must have unit norm")
    return heading_from_quaternion(quaternion / norm[..., None])


class HeadingHold:
    """직진 중 초기 heading을 유지하도록 명령을 보정하는 상태 제어기.

    사용자가 회전을 명령하거나 정지 중일 때는 yaw 명령을 덮어쓰지 않는다.
    ``forward_scale``과 ``lateral_trim``은 학습 가중치가 아니라 정책별 명령
    calibration 값이다.
    """

    def __init__(
        self,
        num_envs,
        device="cpu",
        kp=2.0,
        max_rate=0.35,
        filter_tau=0.25,
        dt=0.02,
        lateral_trim=0.0,
        forward_scale=1.0,
    ):
        """비례 gain, yaw-rate 제한과 1차 error filter를 초기화한다."""

        if (
            kp < 0
            or max_rate <= 0
            or filter_tau < 0
            or dt <= 0
            or not 0 < forward_scale <= 2
        ):
            raise ValueError("Invalid heading hold gains")
        self.reference = torch.zeros(num_envs, device=device)
        self.error = torch.zeros_like(self.reference)
        self.kp, self.max_rate = kp, max_rate
        self.alpha = 1 - math.exp(-dt / filter_tau) if filter_tau else 1.0
        self.lateral_trim = lateral_trim
        self.forward_scale = forward_scale

    def reset(self, ids, measured_heading):
        """선택 환경의 현재 heading을 새 직진 기준으로 저장한다."""

        self.reference[ids] = measured_heading
        self.error[ids] = 0

    def observe(self, measured_heading):
        """±π 경계를 연속적으로 처리하며 필터된 heading error를 갱신한다."""

        error = wrap_angle(self.reference - measured_heading)
        self.error += self.alpha * wrap_angle(error - self.error)

    def commands(self, external):
        """직진 중인 행에만 yaw 보정·전진 배율·측면 trim을 적용한다."""

        result = external.clone()
        moving = external[:, :2].norm(dim=-1) > 0.02
        straight = external[:, 2].abs() < 0.05
        active = moving & straight
        # reference - measured가 양수면 양의 yaw-rate로 기준 방향으로 돌아간다.
        correction = (self.kp * self.error).clamp(-self.max_rate, self.max_rate)
        result[:, 2] = torch.where(active, correction, external[:, 2])
        result[:, 0] = torch.where(
            active, external[:, 0] * self.forward_scale, external[:, 0]
        )
        result[:, 1] += active * self.lateral_trim
        return result
