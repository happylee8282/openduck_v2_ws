"""물성·접촉·actuator domain randomization과 simulator readback 검증."""

import torch
from .control import readback_close


def uniform(bounds, shape, device):
    """``bounds=(low, high)`` 범위의 균등분포 tensor를 만든다."""

    lo, hi = bounds
    if not lo <= hi:
        raise ValueError(f"Invalid randomization range {bounds}")
    return torch.empty(shape, device=device).uniform_(lo, hi)


class RobotRandomizer:
    """병렬 환경 reset 시 v2 로봇 물성과 actuator 값을 무작위화한다.

    초기 simulator 값을 immutable 기준으로 저장하고 매 reset마다 그 기준에
    새 scale을 곱한다. 직전 random 값에 다시 scale을 곱하지 않으므로 값이
    episode마다 누적 표류하지 않는다.
    """

    def __init__(self, robot, cfg, root_body_id):
        """nominal tensor와 재사용할 접촉 재질 bucket을 준비한다."""

        self.robot, self.cfg, self.root_body_id = robot, cfg, root_body_id
        view = robot.root_physx_view
        self.defaults = {
            name: getattr(view, f"get_{name}")().cpu().clone()
            for name in ("masses", "inertias", "coms", "material_properties")
        }
        self.actuator_defaults = {
            name: {
                attr: getattr(a, attr).clone()
                for attr in (
                    "effort_limit",
                    "stiffness",
                    "damping",
                    "armature",
                    "coulomb_friction",
                )
            }
            for name, a in robot.actuators.items()
        }
        # PhysX의 scene 전체 고유 재질 한도(64K)를 넘지 않도록 reset마다
        # 재질을 생성하지 않고 유한한 pool을 모든 shape에서 재사용한다.
        bucket_count = getattr(cfg, "material_buckets", 32)
        if not 1 <= bucket_count <= 1024:
            raise ValueError("material_buckets must be between 1 and 1024")
        static = uniform(cfg.ground_static_friction_range, (bucket_count,), "cpu")
        dynamic = torch.minimum(
            uniform(cfg.ground_dynamic_friction_range, (bucket_count,), "cpu"), static
        )
        restitution = uniform(cfg.ground_restitution_range, (bucket_count,), "cpu")
        self.material_buckets = torch.stack((static, dynamic, restitution), dim=-1)
        self.applied_resets = 0

    def reset(self, env_ids):
        """선택 환경에 새 물성과 actuator 표본을 적용하고 필요 시 재조회한다."""

        if not self.cfg.randomize_domain or not len(env_ids):
            return
        view = self.robot.root_physx_view
        ids = env_ids.cpu().long()
        count = len(ids)
        # PhysX tensor API가 CPU 값을 요구하는 항목은 CPU에서 준비하고,
        # actuator 내부 tensor는 각 actuator가 위치한 장치에서 갱신한다.
        masses = view.get_masses().cpu().clone()
        scales = uniform(
            self.cfg.body_mass_scale_range, self.defaults["masses"][ids].shape, "cpu"
        )
        masses[ids] = self.defaults["masses"][ids] * scales
        inertias = view.get_inertias().cpu().clone()
        # 질량과 관성에 같은 body별 scale을 적용해 둘의 기준 비율을 보존한다.
        inertias[ids] = self.defaults["inertias"][ids] * scales[..., None]
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)
        coms = view.get_coms().cpu().clone()
        coms[ids] = self.defaults["coms"][ids]
        coms[ids, self.root_body_id, :3] += uniform(
            self.cfg.base_com_offset_range, (count, 3), "cpu"
        )
        view.set_coms(coms, ids)
        materials = view.get_material_properties().cpu().clone()
        shape = materials[ids, :, 0].shape
        bucket_ids = torch.randint(len(self.material_buckets), shape, device="cpu")
        materials[ids] = self.material_buckets[bucket_ids]
        view.set_material_properties(materials, ids)
        # 각 actuator group의 effort/gain/friction/armature를 함께 갱신한다.
        for name, a in self.robot.actuators.items():
            d = self.actuator_defaults[name]
            aid = env_ids.to(a.effort_limit.device)
            effort = d["effort_limit"][aid] * uniform(
                self.cfg.motor_effort_scale_range, (count, 1), a.effort_limit.device
            )
            a.effort_limit[aid] = effort
            a.effort_limit_sim[aid] = effort
            self.robot.write_joint_effort_limit_to_sim(
                effort, joint_ids=a.joint_indices, env_ids=env_ids
            )
            for attr in ("stiffness", "damping"):
                getattr(a, attr)[aid] = d[attr][aid] * uniform(
                    self.cfg.joint_gain_scale_range, (count, 1), a.effort_limit.device
                )
            a.coulomb_friction[aid] = uniform(
                self.cfg.joint_friction_range,
                d["coulomb_friction"][aid].shape,
                a.effort_limit.device,
            )
            a.armature[aid] = d["armature"][aid] * uniform(
                self.cfg.armature_scale_range, (count, 1), a.effort_limit.device
            )
            self.robot.write_joint_armature_to_sim(
                a.armature[aid], joint_ids=a.joint_indices, env_ids=env_ids
            )
            if self.cfg.verify_randomization:
                # writer가 실패했는데도 학습이 진행되지 않도록 실제 PhysX 값을
                # 즉시 다시 읽어 요청값과 비교한다.
                indices = torch.arange(self.robot.num_joints)[a.joint_indices]
                got = view.get_dof_max_forces().cpu()[ids][:, indices]
                readback_close(got, effort.cpu(), "effort limits")
                readback_close(
                    view.get_dof_armatures().cpu()[ids][:, indices],
                    a.armature[aid].cpu(),
                    "armature",
                )
        if self.cfg.verify_randomization:
            # body 물성 및 접촉 재질도 선택된 환경만 변경됐는지 확인한다.
            for label, expected in (
                ("masses", masses),
                ("inertias", inertias),
                ("coms", coms),
                ("material_properties", materials),
            ):
                readback_close(
                    getattr(view, f"get_{label}")().cpu()[ids], expected[ids], label
                )
        self.applied_resets += count
