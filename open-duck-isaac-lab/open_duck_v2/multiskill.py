"""고정 전진 정책과 별도 학습한 후진·자세 정책의 전환 런타임.

전환 과정에서도 Isaac이나 실물 servo에는 관절 목표만 보낸다. body pose,
질량, 충돌 형상이나 관절 한계를 덮어쓰지 않으며, action history를 재배치해
목표각 불연속을 줄인다.
"""

from pathlib import Path
import hashlib
import json

import torch
from .control import V2Control
from .profile import ACTION_SCALE, MAX_TARGET_SPEED, OBS_SLICES
from .posture import posture_phase


class HybridControl(V2Control):
    """보행 action과 IK+residual 자세 action을 한 제어 상태에서 처리한다."""

    def __init__(self, num_envs, device, reference, residual_scale=0.12, **kwargs):
        """기본 보행 제어기 위에 환경별 자세 mask와 참조 목표를 추가한다."""

        super().__init__(num_envs, device, **kwargs)
        self.pose_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reference_target = reference.stand_joints.expand(num_envs, -1).clone()
        self.residual_scale = residual_scale

    def set_posture_mask(self, mask):
        """자세 진입/이탈 환경의 action history를 연속적인 상태로 바꾼다."""

        entering = mask & ~self.pose_mask
        leaving = self.pose_mask & ~mask
        if entering.any():
            # 자세 actor의 residual=0이 IK reference이므로 과거 보행 action을
            # 그대로 넘기지 않는다.
            self.history[entering] = 0
            self.actions[entering] = 0
            self.delay.buffer[:, entering] = 0
        if leaving.any():
            # 현재 motor target을 보행 action 좌표로 역변환해, 전진 actor가
            # 다음 tick을 시작할 때 history가 갑자기 HOME으로 뛰지 않게 한다.
            full = (
                (self.motor_targets[leaving] - self.home[leaving]) / ACTION_SCALE
            ).clamp(-1, 1)
            self.history[leaving] = full[:, None]
            self.actions[leaving] = full
            self.delay.buffer[:, leaving] = full[None]
        self.pose_mask.copy_(mask)

    def process(self, actions, head_commands=None):
        """보행 제어를 진행한 뒤 자세 mask 행만 IK residual 결과로 교체한다."""

        if not self.pose_mask.any():
            return super().process(actions, head_commands)
        # super().process가 전체 batch의 history/delay를 동일하게 갱신하게 한 뒤
        # 자세 환경의 목표각만 reference 기반 계산으로 교체한다.
        before = self.motor_targets.clone()
        act_before = self.actuator_targets.clone()
        super().process(actions, head_commands)
        mask = self.pose_mask
        rows = (self.delay.cursor - 1 - self.delay.lags) % len(self.delay.buffer)
        delayed = self.delay.buffer[
            rows, torch.arange(len(actions), device=actions.device)
        ]
        target = (self.reference_target + self.residual_scale * delayed).clamp(
            self.lower, self.upper
        )
        speed = torch.full_like(target, MAX_TARGET_SPEED)
        speed[:, 5:9] = 1.0
        commanded = before + torch.clamp(
            target - before, -speed * self.dt, speed * self.dt
        )
        physical = act_before.clamp(
            min=commanded - self.backlash, max=commanded + self.backlash
        ).clamp(self.lower, self.upper)
        self.motor_targets[mask] = commanded[mask]
        self.actuator_targets[mask] = physical[mask]
        return self.actuator_targets - self.offsets


def load_manifest(path):
    """멀티스킬 manifest의 정책·자세 참조 경로와 SHA-256을 검증한다."""

    path = Path(path)
    data = json.loads(path.read_text())
    # 공식 manifest는 <repository>/config 아래에 있으므로 부모의 부모를
    # repository root로 사용한다.
    root = path.resolve().parents[1]
    for name, record in data["skills"].items():
        file = root / record["checkpoint"]
        if (
            not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"{name} checkpoint missing or changed")
        record["resolved_checkpoint"] = str(file)
    reference = root / data["posture_reference"]["path"]
    if (
        hashlib.sha256(reference.read_bytes()).hexdigest()
        != data["posture_reference"]["sha256"]
    ):
        raise ValueError("Posture reference changed")
    data["reference_path"] = str(reference)
    return data


class MultiSkillRuntime:
    """전진·후진·자세 actor를 선택하고 전환 조건을 관리하는 상태기계.

    active_skill 값은 0=전진/정지, 1=후진, 2=자세다. 전진 actor는 정지
    controller도 담당하며, 후진 정지에서 원래 정지 controller로 1초 동안
    crossfade한다.
    """

    def __init__(self, env, forward, backward, posture, reference):
        """환경 제어기를 HybridControl로 교체하고 전환 상태 tensor를 만든다."""

        self.env = env
        self.forward = forward
        self.backward = backward
        self.posture = posture
        self.reference = reference
        n = env.num_envs
        device = env.device
        self.control = HybridControl(
            n,
            device,
            reference,
            posture.control_profile.get("posture_residual_scale", 0.12),
            dt=env.step_dt,
            max_delay=env.cfg.action_delay_max,
            backlash=env.cfg.backlash_half_width,
            head_control_mode=env.cfg.head_control_mode,
            head_target_speed=env.cfg.head_target_speed,
            target_filter_tau_s=env.cfg.target_filter_tau_s,
        )
        # 이미 reset된 환경의 제어 상태를 보존한 채 hybrid 제어기로 교체한다.
        self.control.motor_targets.copy_(env.control.motor_targets)
        self.control.actuator_targets.copy_(env.control.actuator_targets)
        self.control.history.copy_(env.control.history)
        env.control = self.control
        env.actions = self.control.actions
        self.height = torch.full((n,), reference.stand_height, device=device)
        self.height_rate = torch.zeros_like(self.height)
        self.start_height = self.height.clone()
        self.goal = self.height.clone()
        self.elapsed = torch.full_like(self.height, 4.0)
        self.pose_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.quiet = torch.zeros_like(self.height)
        self.last_direction = torch.zeros(n, device=device)
        self.time_still = torch.full_like(self.height, 2.0)
        self.active_skill = torch.zeros(n, dtype=torch.long, device=device)
        self.applied_commands = torch.zeros(n, 7, device=device)
        self.backward_stop_weight = torch.zeros_like(self.height)
        self.last_policy_observation = None
        self.last_posture_observation = None

    def step(self, external, low_requested):
        """외부 속도·자세 요청을 한 tick 처리하고 Isaac 환경을 진행한다.

        낮은 자세는 1초 이상 정지한 뒤 진입한다. 자세 중 보행 요청은 먼저
        4초 기립과 1초 안정화를 끝낸 뒤 허용하며, 전진↔후진 즉시 반전도
        1초 정지를 삽입한다.
        """

        env = self.env
        dt = env.step_dt
        external = external.clone()
        low_requested = low_requested.to(env.device).bool()
        desired = torch.where(
            low_requested,
            self.height.new_full(self.height.shape, self.reference.low_height),
            self.height.new_full(self.height.shape, self.reference.stand_height),
        )
        # 정지 상태에서만 자세 모드로 들어간다. 낮은 자세에서 걷기 요청이
        # 들어오면 desired가 stand가 되어 우선 기립 궤적을 수행한다.
        entering = (~self.pose_mask) & low_requested & (self.time_still >= 1.0)
        self.pose_mask |= entering
        change = self.pose_mask & ((desired - self.goal).abs() > 1e-6)
        self.start_height[change] = self.height[change]
        self.goal[change] = desired[change]
        self.elapsed[change] = 0
        self.elapsed = (self.elapsed + dt).clamp_max(4.0)
        # 4초 5차 smoothstep 높이 궤적과 그 시간 미분이다.
        u = self.elapsed / 4.0
        blend = 6 * u**5 - 15 * u**4 + 10 * u**3
        rate = 30 * u**2 * (1 - u) ** 2 / 4.0
        self.height = self.start_height + (self.goal - self.start_height) * blend
        self.height_rate = (self.goal - self.start_height) * rate
        at_stand = (
            self.pose_mask
            & ~low_requested
            & (self.elapsed >= 4.0)
            & ((self.height - self.reference.stand_height).abs() < 1e-5)
        )
        contacts = env._get_foot_contacts().all(-1)
        joint_speed = (
            env.robot.data.joint_vel[:, env.controlled_joint_ids].abs().max(-1).values
        )
        # 기립 완료 후에도 양발 접촉·낮은 관절 속도를 1초 확인하고 이탈한다.
        quiet = at_stand & contacts & (joint_speed < 0.3)
        self.quiet = torch.where(quiet, self.quiet + dt, torch.zeros_like(self.quiet))
        self.pose_mask &= ~(self.quiet >= 1.0)
        direction = torch.sign(external[:, 0])
        # 진행 방향이 즉시 반전되면 정책을 바로 바꾸지 않고 1초간 정지한다.
        reversal = (
            (direction != 0)
            & (self.last_direction != 0)
            & (direction != self.last_direction)
            & (self.time_still < 1.0)
        )
        braking = self.pose_mask | low_requested | reversal
        external[braking, :3] = 0
        moving = external[:, :2].norm(dim=-1) > 0.02
        self.time_still = torch.where(
            moving, torch.zeros_like(self.time_still), self.time_still + dt
        )
        self.last_direction = torch.where(
            moving, torch.sign(external[:, 0]), self.last_direction
        )
        self.control.set_posture_mask(self.pose_mask)
        q, _, _, _ = self.reference.sample(self.height)
        self.control.reference_target = q
        env.commands.copy_(external)
        env._obs_stamp = -1
        # 환경이 만든 동일한 raw 관측을 세 actor의 공통 입력 기반으로 사용한다.
        obs = env._get_observations()["policy"].clone()
        effective = env._effective_commands().clone()
        reverse = (external[:, 0] < -0.02) & ~self.pose_mask
        p = self.backward.control_profile
        # 후진 actor는 전진 actor와 다른 명령 calibration metadata를 사용한다.
        effective[reverse, 0] = external[reverse, 0] * p.get(
            "forward_command_scale", 1.0
        )
        effective[reverse, 1] = external[reverse, 1] + p.get(
            "lateral_command_trim", 0.0
        )
        effective[reverse, 2] = (
            env.heading_hold.error[reverse] * p.get("heading_hold_kp", 2.0)
        ).clamp(
            -p.get("heading_hold_max_rate", 0.35), p.get("heading_hold_max_rate", 0.35)
        )
        obs[:, OBS_SLICES["commands"]] = effective
        actions = torch.zeros(env.num_envs, 14, device=env.device)
        normal = ~self.pose_mask & ~reverse
        if normal.any():
            actions[normal] = self.forward(obs[normal])
        if reverse.any():
            actions[reverse] = self.backward(obs[reverse])
        # 별도 학습한 후진 자세를 원래 정지 controller로 1초 동안 넘긴다.
        # 순수 전진/정지 경로는 이 crossfade의 영향을 받지 않는다.
        handover = (
            normal & (self.last_direction < 0) & (~moving) & (self.time_still < 1.0)
        )
        self.backward_stop_weight = torch.where(
            handover, (1.0 - self.time_still).clamp(0, 1), 0.0
        )
        if handover.any():
            weight = self.backward_stop_weight[handover, None]
            actions[handover] = (1 - weight) * actions[
                handover
            ] + weight * self.backward(obs[handover])

        # 자세 actor는 base101에 목표 높이·높이 속도를 붙인 103차원을 받는다.
        pose_observation = torch.zeros(env.num_envs, 103, device=env.device)
        if self.pose_mask.any():
            raw = obs[self.pose_mask].clone()
            raw[:, OBS_SLICES["commands"]] = 0
            raw[:, OBS_SLICES["phase_cos_sin"]] = posture_phase(
                self.height[self.pose_mask]
            )
            pose = torch.cat(
                (
                    raw,
                    self.height[self.pose_mask, None],
                    self.height_rate[self.pose_mask, None],
                ),
                -1,
            )
            pose_observation[self.pose_mask] = pose
            actions[self.pose_mask] = self.posture(pose)
        self.last_policy_observation = obs
        self.last_posture_observation = pose_observation
        self.active_skill = torch.where(self.pose_mask, 2, torch.where(reverse, 1, 0))
        self.applied_commands = external
        return env.step(actions)
