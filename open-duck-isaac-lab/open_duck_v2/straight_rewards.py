"""직진·정지 명령에 반응하는 공통 보상 항.

별도 표기가 없으면 속도는 body 축, 단위는 SI다. 환경 코드는 이 함수가
반환한 무차원 항에 profile별 weight를 곱해 최종 reward를 만든다.
"""

import torch


def tracking_terms(commands, velocity, angular):
    """명령 추종, 진행, 측면/회전 오차와 정지 품질 항을 한 번에 계산한다.

    전진 명령에서 실제 후진은 ``progress=-1``이 되며, 단순 속력만으로
    진행했다고 오판하지 않는다. 큰 cost는 gradient 폭주를 막기 위해 16에서
    포화하지만 0으로 사라지지는 않는다.
    """

    speed = commands[:, :2].norm(dim=-1)
    moving = speed > 0.02
    idle = (~moving) & (commands[:, 2].abs() < 0.05)
    direction = commands[:, :2] / speed.clamp_min(0.02)[:, None]
    along = (velocity[:, :2] * direction).sum(-1)
    return {
        "moving": moving,
        "idle": idle,
        "along": along,
        "track_x": torch.exp(-((velocity[:, 0] - commands[:, 0]) / 0.04).square()),
        "track_y": torch.exp(-((velocity[:, 1] - commands[:, 1]) / 0.025).square()),
        "track_yaw": torch.exp(-((angular[:, 2] - commands[:, 2]) / 0.15).square()),
        "progress": (along / speed.clamp_min(0.04)).clamp(-1, 1) * moving,
        "lateral_cost": ((velocity[:, 1] - commands[:, 1]) / 0.05)
        .square()
        .clamp_max(16),
        "yaw_cost": ((angular[:, 2] - commands[:, 2]) / 0.25).square().clamp_max(16),
        "idle_cost": (
            (velocity[:, :2] / 0.03).square().sum(-1) + (angular[:, 2] / 0.15).square()
        ).clamp_max(16)
        * idle,
        "still": torch.exp(
            -(velocity[:, :2] / 0.02).square().sum(-1) - (angular[:, 2] / 0.10).square()
        )
        * idle,
        "no_progress": torch.exp(-along.clamp_min(0) / 0.03) * moving,
    }


def heading_from_quaternion(q):
    """``wxyz`` body quaternion에서 Z-up world heading을 구한다.

    body gyro z는 회전 속도이고 이 값은 누적 자세 방향이므로 서로 대체할 수
    없다.
    """

    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def wrap_angle(angle):
    """임의 각도를 ``[-π, π]``로 감싸 경계에서 연속적으로 비교한다."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def desired_contacts(phase, duty_factor, phase_offset):
    """주기 phase에서 좌·우 발의 목표 접촉 상태를 계산한다.

    ``phase`` 단위는 cycle이고 오른발은 왼발에서 반 주기 떨어진다. offset은
    접촉 reward의 정렬만 바꾸며 actor가 보는 phase clock 자체는 바꾸지 않는다.
    """

    left = torch.remainder(phase + phase_offset, 1.0)
    right = torch.remainder(left + 0.5, 1.0)
    return torch.stack((left < duty_factor, right < duty_factor), dim=-1)
