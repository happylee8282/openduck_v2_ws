"""v2 기준 servo 수식을 Isaac Lab actuator 인터페이스에 연결한다.

여기서 ``coulomb_friction``은 PhysX의 무차원 마찰계수가 아니라 관절축에
작용하는 N·m 단위의 마찰 토크다.
"""

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
import torch

from .control import servo_effort


class V2ServoActuator(IdealPDActuator):
    """Isaac의 position target을 명시적 effort로 바꾸는 v2 servo actuator."""

    def __init__(self, cfg, *args, **kwargs):
        """환경별 effort tensor와 같은 shape로 Coulomb 마찰값을 준비한다."""

        super().__init__(cfg, *args, **kwargs)
        self.coulomb_friction = torch.full_like(self.effort_limit, cfg.coulomb_friction)

    def compute(self, control_action, joint_pos, joint_vel):
        """PD+마찰+포화를 계산하고 simulator에 effort 명령만 전달한다."""

        velocity = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * (control_action.joint_positions - joint_pos)
            + self.damping * velocity
            + control_action.joint_efforts
        )
        self.applied_effort = servo_effort(
            control_action.joint_positions,
            joint_pos,
            joint_vel,
            self.stiffness,
            self.damping,
            self.coulomb_friction,
            self.effort_limit,
            self.cfg.target_speed_limit,
            self.cfg.torque_speed_curve,
            control_action.joint_velocities,
            control_action.joint_efforts,
        )
        # 위치/속도 target을 그대로 넘기면 PhysX drive가 PD를 한 번 더 적용한다.
        # 따라서 계산된 effort만 남겨 제어 수식이 중복되지 않게 한다.
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class V2ServoActuatorCfg(IdealPDActuatorCfg):
    """v2 servo의 추가 마찰·속도 profile을 담는 Isaac Lab config."""

    class_type = V2ServoActuator
    coulomb_friction: float = 0.052
    target_speed_limit: float = 5.24
    torque_speed_curve: bool = False
