"""Open Duck Mini v2 정책이 공유하는 기준 좌표와 버전 계약.

모든 물리량은 SI 단위를 사용한다. 이 파일의 모터 값은 모델 기준값이며
실제 로봇을 계측해 얻은 fit으로 취급하면 안 된다.
"""

from __future__ import annotations

import hashlib
import json
import math

# 관측 의미가 바뀌면 차원 수가 같더라도 반드시 버전을 올려야 한다.
CONTRACT_VERSION = "open-duck-mini-v2-101-v1"

# 정책, 모터 I/O, 로그가 공통으로 사용하는 canonical 관절 순서다.
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOTOR_IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14)
# canonical 좌표는 좌우 다리와 머리를 같은 의미로 다루기 위한 정책 좌표다.
# q_canonical = q_urdf + offset. 추적 중인 URDF와 공식 Playground MJCF의
# 상대 관절 프레임에서 구했고 FK 회귀검사로 확인했다.
URDF_TO_CANONICAL = (
    0.0,
    0.0,
    -math.pi / 4,
    math.pi / 2,
    -math.pi / 4,
    math.pi / 4,
    -math.pi / 4,
    0.0,
    0.0,
    0.0,
    0.0,
    math.pi / 4,
    math.pi / 2,
    -math.pi / 4,
)
# actor action=0이 가리키는 canonical 기준 자세다.
HOME = (
    0.002,
    0.053,
    -0.630,
    1.368,
    -0.784,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.003,
    -0.065,
    0.635,
    1.379,
    -0.796,
)
# URDF 원본 한계다. 아래 LIMITS에서 canonical 좌표로 옮기고, 실제 제어기는
# 여기에 추가로 0.02rad 안전 여유를 둔다.
URDF_LIMITS = (
    (-0.523599, 0.523599),
    (-0.436332, 0.436332),
    (-0.436332, 1.309),
    (-3.14159, 0.0),
    (-0.785398, 2.35619),
    (-1.13446, 0.349066),
    (0.0, 1.5708),
    (-2.79253, 2.79253),
    (-0.523599, 0.523599),
    (-0.523599, 0.523599),
    (-0.436332, 0.436332),
    (-1.309, 0.436332),
    (-3.14159, 0.0),
    (-0.785398, 2.35619),
)
LIMITS = tuple((lo + o, hi + o) for (lo, hi), o in zip(URDF_LIMITS, URDF_TO_CANONICAL))
# HOME에서 전체 STL 충돌 형상은 trunk 기준 z=-0.16683m까지 내려온다.
# 기존 0.15m는 지면과 겹치므로 약 5mm 간격이 생기도록 0.172m에서 시작한다.
SPAWN_HEIGHT = 0.172
STANCE_HEIGHT = 0.17

# PhysX는 500Hz, 정책은 decimation 10을 적용해 50Hz로 실행한다.
POLICY_HZ = 50
PHYSICS_DT = 0.002
DECIMATION = 10
ACTION_SCALE = 0.25
MAX_TARGET_SPEED = 5.24

# 측정 전 초기 시뮬레이션에 사용하는 actuator 기준값이다.
MOTOR_REFERENCE = dict(
    effort_nm=3.35,
    kp=17.8,
    damping_nm_s=0.60,
    coulomb_friction_nm=0.052,
    armature_kg_m2=0.028,
    nominal_voltage_v=7.4,
    backlash_half_width_rad=math.radians(0.5),
    provenance="Open_Duck_Playground b9be205, BAM-derived reference",
    hardware_calibrated=False,
)
# 순서 자체가 체크포인트 계약이다. 필드를 삽입하거나 재배열하면 기존 정책과
# 호환되지 않으므로 CONTRACT_VERSION도 함께 변경해야 한다.
OBS_FIELDS = (
    ("gyro_rad_s", 3),
    ("accelerometer_m_s2", 3),
    ("commands", 7),
    ("joint_position_minus_home_rad", 14),
    ("joint_velocity_scaled_0_05", 14),
    ("last_action", 14),
    ("previous_action", 14),
    ("older_action", 14),
    ("motor_targets_rad", 14),
    ("feet_contacts_left_right", 2),
    ("phase_cos_sin", 2),
)
OBS_DIM = sum(n for _, n in OBS_FIELDS)
CRITIC_DIM = OBS_DIM + 3 + 1 + 117
OBS_SLICES = {}
_start = 0
for _name, _length in OBS_FIELDS:
    OBS_SLICES[_name] = slice(_start, _start + _length)
    _start += _length


def contract():
    """현재 보행 actor 계약을 JSON 직렬화 가능한 사전으로 반환한다.

    체크포인트 저장·로드와 export sidecar가 모두 이 값을 비교하므로 단순한
    문서가 아니라 실행 시 호환성 검사의 기준이다.
    """

    return dict(
        version=CONTRACT_VERSION,
        observation_dim=OBS_DIM,
        critic_dim=CRITIC_DIM,
        action_dim=14,
        joint_names=list(JOINT_NAMES),
        motor_ids=list(MOTOR_IDS),
        urdf_to_canonical=list(URDF_TO_CANONICAL),
        home_rad=list(HOME),
        joint_limits_rad=[list(x) for x in LIMITS],
        policy_hz=POLICY_HZ,
        action_scale=ACTION_SCALE,
        max_target_speed_rad_s=MAX_TARGET_SPEED,
        observation_fields=[dict(name=n, length=k) for n, k in OBS_FIELDS],
        imu_frame="canonical trunk-aligned axes; specific force includes gravity",
        commands="vx, vy, yaw_rate, neck_pitch, head_pitch, head_yaw, head_roll",
        phase="cos/sin of externally supplied cycle; default period 0.8 seconds",
        empirical_normalization=False,
        motor_reference=MOTOR_REFERENCE,
    )


def contract_hash():
    """키 순서를 정규화한 정책 계약의 SHA-256을 반환한다."""

    return hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()


def native_home():
    """canonical HOME을 Isaac/URDF 관절 좌표의 이름별 사전으로 변환한다."""

    return {n: q - o for n, q, o in zip(JOINT_NAMES, HOME, URDF_TO_CANONICAL)}
