"""v2 실물 계측 기록의 완전성과 학습 범위 적합성을 검사한다.

시뮬레이션 기준값이 채워져 있다는 이유로 실측 완료로 판정하지 않는다.
각 항목은 실제 하드웨어 측정값과 그 근거 로그를 요구한다.
"""

import math
from .profile import JOINT_NAMES


def calibration_issues(record):
    """계측 JSON 사전을 검사하고 발견한 문제 메시지 목록을 반환한다.

    빈 목록은 기록 형식과 필수값이 유효하다는 뜻일 뿐, 실제 보행 성능이나
    로봇 안전성이 입증됐다는 뜻은 아니다.
    """

    issues = []
    if record.get("source") != "measured_hardware":
        issues.append("source must be measured_hardware")
    for key in (
        "measured_mass_kg",
        "measured_supply_v",
        "command_latency_p95_s",
        "imu_latency_p95_s",
    ):
        value = record.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            issues.append(f"{key} requires a finite positive measurement")
    # 학습 mass randomization 범위를 벗어나면 측정값이 틀렸다고 보지 않고,
    # simulator 범위를 갱신하고 재학습해야 할 조건으로 보고한다.
    mass = record.get("measured_mass_kg")
    if (
        isinstance(mass, (int, float))
        and not 2.107140705 * 0.94 <= mass <= 2.107140705 * 1.08
    ):
        issues.append(
            "measured mass falls outside the configured training randomization"
        )
    # 3 policy tick = 60ms가 현재 학습한 최대 지연이다.
    for key in ("command_latency_p95_s", "imu_latency_p95_s"):
        value = record.get(key)
        if isinstance(value, (int, float)) and value > 0.06:
            issues.append(f"{key} exceeds the trained delay range; revise and retrain")
    # 위치 배열만으로는 순서 오류를 잡을 수 없으므로 14개 관절 이름을 모두
    # 키로 요구한다.
    offsets = record.get("joint_zero_offsets_rad", {})
    if set(offsets) != set(JOINT_NAMES) or any(
        not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in offsets.values()
    ):
        issues.append("all 14 named joint zero offsets must be measured")
    for key in (
        "motor_order_verified",
        "joint_signs_verified",
        "imu_axes_verified",
        "feet_left_right_verified",
    ):
        if record.get(key) is not True:
            issues.append(f"{key} must be verified on hardware")
    fits = record.get("servo_fit", {})
    for key in (
        "effort_nm",
        "kp",
        "damping_nm_s",
        "coulomb_friction_nm",
        "armature_kg_m2",
        "max_speed_rad_s",
    ):
        value = fits.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            issues.append(f"servo_fit.{key} requires a measured positive value")
    if not record.get("measurement_log_sha256"):
        issues.append("measurement log SHA-256 is required")
    return issues
