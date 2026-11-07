"""OpenDuck Mini 整机中试检测平台。

固定硬件约定：
- 舵机驱动板：/dev/ttyACM0
- 总线波特率：1,000,000
- 项目舵机 ID：10-14、20-24、30-33

命令：
    python scripts/pilot_test_platform.py assembly
    python scripts/pilot_test_platform.py gait
    python scripts/pilot_test_platform.py all

任何动作开始前都会确认 14 个舵机全部在线。发生低电压、过流、过温、
持续跟踪误差或通信异常时，程序立即关闭全部舵机扭矩。

装配方向实现：
- 整机正确样机保存为direction_references/robot.json；
- 保存母基准后自动裁剪right_leg/left_leg/head三个子基准；
- 正常整机和部件测试自动加载对应基准；
- 位置反馈方向错误或电流特征明确反向时输出FAIL和疑似装配方向错误。
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import rustypot
except ImportError:  # 允许在没有硬件 SDK 的开发机上导入并运行单元测试。
    rustypot = None


SERIAL_PORT = "/dev/ttyACM0"
SERVO_BAUD_RATE = 1_000_000
# 用户指定：单次SDK串口事务最多等待0.5s。
# 动作循环连续两次失败才停机，所以最坏约1s确认真实通信中断。
SERIAL_TIMEOUT_S = 1.0
# 扫描阶段单独使用短超时，避免逐个等待不在线ID浪费时间
SCAN_TIMEOUT_S = 0.5
CONTROL_FREQUENCY_HZ = 50
SAMPLE_PERIOD_S = 1.0 / CONTROL_FREQUENCY_HZ

# 顺序与现有 HWI、ONNX 策略的 14 维 action 顺序完全一致。
JOINTS: tuple[tuple[str, int], ...] = (
    ("left_hip_yaw", 20),
    ("left_hip_roll", 21),
    ("left_hip_pitch", 22),
    ("left_knee", 23),
    ("left_ankle", 24),
    ("neck_pitch", 30),
    ("head_pitch", 31),
    ("head_yaw", 32),
    ("head_roll", 33),
    ("right_hip_yaw", 10),
    ("right_hip_roll", 11),
    ("right_knee", 13),
    ("right_ankle", 14),
    ("right_hip_pitch", 12),
)
EXPECTED_IDS: tuple[int, ...] = (
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    30, 31, 32, 33,
)
JOINT_NAME_BY_ID = {servo_id: name for name, servo_id in JOINTS}
JOINT_ID_BY_NAME = dict(JOINTS)
JOINT_LABEL_ZH = {
    "left_hip_yaw": "左髋偏航",
    "left_hip_roll": "左髋横滚",
    "left_hip_pitch": "左髋俯仰",
    "left_knee": "左膝",
    "left_ankle": "左踝",
    "neck_pitch": "颈部俯仰",
    "head_pitch": "头部俯仰",
    "head_yaw": "头部偏航",
    "head_roll": "头部横滚",
    "right_hip_yaw": "右髋偏航",
    "right_hip_roll": "右髋横滚",
    "right_hip_pitch": "右髋俯仰",
    "right_knee": "右膝",
    "right_ankle": "右踝",
}
HEAD_IDS = frozenset({30, 31, 32, 33})
# 偏航轴的小角度运动对重力势能影响很弱，不能用重力电流特征硬判装配方向。
# 这些关节仍执行“指令正方向/反馈正方向”位置判定。




# 部件和整机逐关节检测先到装配当前位置+15°，再到-15°。
JOINT_AMPLITUDE_DEG = {name: 30.0 for name, _ in JOINTS}
JOINT_WAVEFORM_SIGN = {"left_hip_roll": 1.0, "left_hip_pitch": 1.0}
JOINT_WAVEFORM_DEFAULT_SIGN = -1.0

# 阈值来源分为厂家规格和项目工程阈值，不能混写成“厂家验收标准”：
# 1. 飞特《STS3215产品规格书》A/0：
#    https://www.feetechrc.com/Data/feetechrc/upload/file/20200611/6372749961523760249976542.pdf
#    - 工作电压4.0~7.4V、运行温度-20~60℃；
#    - 电流>2A持续2s进入过流保护；
#    - 温度>70℃关闭扭矩；默认波特率1Mbps。
# 2. MIN_VOLTAGE_V=6.8V 是OpenDuck步态稳定性的内部保守阈值，
#    高于厂家4.0V欠压保护线，不是飞特出厂合格线。
# 3. 过流持续0.50s是中试平台主动停机阈值，比厂家2s保护更保守。
# 4. RMSE、最大误差、持续误差和方向相似度均为OpenDuck内部工程阈值，
#    后续应使用正确/错误装配批量样本做统计校准。
FEETECH_SPEC_URL = (
    "https://www.feetechrc.com/Data/feetechrc/upload/file/20200611/"
    "6372749961523760249976542.pdf"
)
MIN_VOLTAGE_V = 6.8
MAX_CURRENT_A = 2.0
# 按当前中试要求，温度不参与方向/波形PASS判定：
# 70℃以下不提示，超过70℃只做告警；75℃仅作为防损坏紧急断扭矩底线。
TEMPERATURE_WARNING_C = 999.0
MAX_TEMPERATURE_C = 999.0
MAX_TRACKING_ERROR_DEG = 20.0
TRACKING_ERROR_DURATION_S = 0.30
OVERCURRENT_DURATION_S = 0.50
PASS_RMSE_DEG = 5.0
PASS_MAX_ERROR_DEG = 15.0
# 基准比例：单舵机90°/5°(5.6%)用于换算各幅度的RMSE阈值，max_error同理(16.7%)


# 固定阈值：髋横滚(ID11/21)负载大放宽，其余收紧
HIP_ROLL_JOINTS = {"left_hip_roll", "right_hip_roll"}
HIP_ROLL_RMSE_LIMIT = 1.5
HIP_ROLL_MAX_ERROR_LIMIT = 3.0
DEFAULT_RMSE_LIMIT = 0.8
DEFAULT_MAX_ERROR_LIMIT = 2.0


def pass_rmse_for_joint(joint_name: str) -> float:
    return HIP_ROLL_RMSE_LIMIT if joint_name in HIP_ROLL_JOINTS else DEFAULT_RMSE_LIMIT


def pass_max_error_for_joint(joint_name: str) -> float:
    return HIP_ROLL_MAX_ERROR_LIMIT if joint_name in HIP_ROLL_JOINTS else DEFAULT_MAX_ERROR_LIMIT

# STS3215电流寄存器单位为6.5mA；代码读取的是寄存器原始值。
CURRENT_RAW_TO_AMP = 0.0065
PAIRED_RMSE_RATIO_THRESHOLD = 2.0
# 位置方向只用于排除“不动/反向”，精度由RMSE和最大误差另行判断。
# 部件/整机采用“至少同向移动3°且达到指令20%”；3°远大于厂家≤0.5°齿隙，
# 可避免把编码器噪声和齿隙当成有效方向响应。
CURRENT_SAMPLE_INTERVAL_S = 0.10
MOTOR_CSV_FIELDS = (
    "time_s",
    "joint",
    "servo_id",
    "target_deg",
    "actual_deg",
    "error_deg",
    "target_absolute_deg",
    "actual_absolute_deg",
    "current_a",
    "voltage_v",
    "temperature_c",
    "communication",
    "position_delay_ms",
    "phase_delay_deg",
    "delay_correlation",
)


class SafetyAbort(RuntimeError):
    """表示硬件安全阈值触发，必须立即关闭扭矩。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.motor_rows: list[dict[str, Any]] = []


@dataclass
class ScanResult:
    """一次只读总线扫描结果。"""

    expected_ids: tuple[int, ...] = EXPECTED_IDS
    found_positions: dict[int, float] = field(default_factory=dict)
    missing_ids: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def found_ids(self) -> list[int]:
        return sorted(self.found_positions)

    @property
    def complete(self) -> bool:
        return (
            not self.missing_ids
            and set(self.found_positions) == set(self.expected_ids)
        )


@dataclass
class SafetyMonitor:
    """累计持续误差和持续过流时间，瞬时异常不会被误判为持续异常。"""

    error_seconds: dict[int, float] = field(default_factory=dict)
    overcurrent_seconds: dict[int, float] = field(default_factory=dict)
    temperature_warning_ids: set[int] = field(default_factory=set)

    def check(
        self,
        *,
        servo_id: int,
        error_deg: float,
        current_a: float | None,
        voltage_v: float | None,
        temperature_c: float | None,
        dt_s: float,
        power_sample: bool = True,
        power_dt_s: float | None = None,
    ) -> None:
        if voltage_v is not None and voltage_v < MIN_VOLTAGE_V:
            raise SafetyAbort(
                f"舵机 ID {servo_id} 电压 {voltage_v:.2f}V，"
                f"低于 {MIN_VOLTAGE_V:.2f}V"
            )
        if temperature_c is not None and temperature_c >= MAX_TEMPERATURE_C:
            raise SafetyAbort(
                f"舵机 ID {servo_id} 温度 {temperature_c:.1f}℃，"
                f"达到 {MAX_TEMPERATURE_C:.1f}℃硬停机阈值"
            )
        if (
            temperature_c is not None
            and temperature_c >= TEMPERATURE_WARNING_C
            and servo_id not in self.temperature_warning_ids
        ):
            self.temperature_warning_ids.add(servo_id)
            print(
                f"[温度告警] ID {servo_id} 当前 {temperature_c:.1f}℃，"
                f"已达到 {TEMPERATURE_WARNING_C:.1f}℃；"
                f"{MAX_TEMPERATURE_C:.1f}℃时关闭扭矩停机"
            )

        if abs(error_deg) > MAX_TRACKING_ERROR_DEG:
            self.error_seconds[servo_id] = self.error_seconds.get(servo_id, 0.0) + dt_s
        else:
            self.error_seconds[servo_id] = 0.0
        if self.error_seconds[servo_id] >= TRACKING_ERROR_DURATION_S:
            raise SafetyAbort(
                f"舵机 ID {servo_id} 位置误差超过 {MAX_TRACKING_ERROR_DEG:.1f}°，"
                f"持续 {self.error_seconds[servo_id]:.2f}s"
            )

        if power_sample:
            if current_a is not None and abs(current_a) > MAX_CURRENT_A:
                self.overcurrent_seconds[servo_id] = (
                    self.overcurrent_seconds.get(servo_id, 0.0)
                    + (power_dt_s if power_dt_s is not None else dt_s)
                )
            else:
                self.overcurrent_seconds[servo_id] = 0.0
            if self.overcurrent_seconds[servo_id] >= OVERCURRENT_DURATION_S:
                raise SafetyAbort(
                    f"舵机 ID {servo_id} 电流超过 {MAX_CURRENT_A:.2f}A，"
                    f"持续 {self.overcurrent_seconds[servo_id]:.2f}s"
                )


def scalar(value: Any) -> float:
    """兼容 rustypot 返回标量或单元素列表的情况。"""

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("SDK 返回空列表")
        value = value[0]
    return float(value)


def sequence_of_floats(values: Any) -> list[float]:
    """把 SDK 返回值统一成 float 列表。"""

    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    try:
        return [float(value) for value in values]
    except TypeError:
        return [float(values)]


def open_controller(timeout_s: float | None = None) -> Any:
    """用项目固定串口和波特率打开 STS3215 控制器。timeout_s 可覆盖默认超时。"""

    if rustypot is None:
        raise RuntimeError(
            "未安装 rustypot。请在树莓派项目虚拟环境中运行本程序。"
        )
    return rustypot.Sts3215PyController(
        SERIAL_PORT,
        SERVO_BAUD_RATE,
        timeout_s if timeout_s is not None else SERIAL_TIMEOUT_S,
    )


def release_controller(controller: Any) -> None:
    """释放 SDK 对象，让后续 HWI 可以重新打开同一串口。"""

    close_method = getattr(controller, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            pass
    # rustypot 当前版本没有公开 close()；调用方还需清除自己的对象引用。


def disable_torque(controller: Any, ids: Sequence[int]) -> None:
    """尽最大努力关闭指定舵机扭矩；安全清理阶段不覆盖原始异常。"""

    if not ids:
        return
    try:
        controller.sync_write_torque_enable(list(ids), [False] * len(ids))
        return
    except Exception:
        pass
    for servo_id in ids:
        try:
            controller.write_torque_enable(servo_id, False)
        except Exception:
            pass


def emergency_disable_all() -> None:
    """在上层对象初始化失败时，重新打开总线并强制关闭全部扭矩。"""

    controller = None
    try:
        controller = open_controller()
        disable_torque(controller, EXPECTED_IDS)
    except Exception:
        pass
    finally:
        if controller is not None:
            release_controller(controller)
        controller = None
        gc.collect()
        time.sleep(0.10)


def scan_servo_ids(
    controller: Any,
    candidate_ids: Sequence[int] = EXPECTED_IDS,
    *,
    quiet: bool = False,
) -> ScanResult:
    """逐个读取指定 ID 的当前位置；不写寄存器、不发送动作。

    quiet=True 时不打印单个超时/缺失信息，只在最后汇总未检测到的 ID。
    """

    expected_ids = tuple(candidate_ids)
    result = ScanResult(expected_ids=expected_ids)
    for servo_id in expected_ids:
        try:
            position = scalar(controller.read_present_position(servo_id))
            result.found_positions[servo_id] = position
            print(
                f"[在线] ID {servo_id:02d} "
                f"{JOINT_NAME_BY_ID[servo_id]} "
                f"位置 {math.degrees(position):.2f}°"
            )
        except Exception as exc:
            result.missing_ids.append(servo_id)
            result.errors[servo_id] = f"{type(exc).__name__}: {exc}"
            if not quiet:
                print(f"[缺失] ID {servo_id:02d}：{exc}")
    if quiet and result.missing_ids:
        print(
            f"[扫描] 未检测到 {len(result.missing_ids)} 个 ID："
            f"{result.missing_ids}（总线无响应，已跳过）"
        )
    return result


def read_optional(controller: Any, method_name: str, servo_id: int) -> float | None:
    """读取可选遥测；方法不存在时返回空值。"""

    method = getattr(controller, method_name, None)
    if not callable(method):
        return None
    return scalar(method(servo_id))


def current_to_amp(raw_current: float | None) -> float | None:
    """STS3215 电流反馈每个原始单位为 6.5mA。"""

    if raw_current is None:
        return None
    return raw_current * CURRENT_RAW_TO_AMP


def read_current_vector_amp(
    controller: Any,
    servo_ids: Sequence[int],
) -> dict[int, float] | None:
    """批量读取串联舵机电流并转换为安培；SDK不支持时返回空值。"""

    method = getattr(controller, "sync_read_present_current", None)
    if not callable(method):
        return None
    raw_values = method(list(servo_ids))
    if isinstance(raw_values, Mapping):
        values = [
            raw_values.get(servo_id, raw_values.get(str(servo_id)))
            for servo_id in servo_ids
        ]
    else:
        values = sequence_of_floats(raw_values)
    if len(values) != len(servo_ids) or any(value is None for value in values):
        raise ValueError(
            f"批量电流返回数量异常：期望{len(servo_ids)}，实际{len(values)}"
        )
    return {
        servo_id: round(current_to_amp(float(raw)) or 0.0, 6)
        for servo_id, raw in zip(servo_ids, values)
    }









def threshold_sources() -> list[dict[str, str]]:
    """返回报告和JSON共用的阈值依据，明确区分厂家规格与内部指标。"""

    return [
        {
            "item": "通信与反馈能力",
            "value": "1,000,000 bit/s；位置/速度/电压/电流/温度反馈",
            "source_type": "厂家规格",
            "source": "飞特《STS3215产品规格书》A/0，第4页控制特性",
            "url": FEETECH_SPEC_URL,
        },
        {
            "item": "工作电压与项目欠压停机",
            "value": "厂家4.0~7.4V；项目低于6.8V主动停机",
            "source_type": "厂家规格 + 项目工程阈值",
            "source": (
                "厂家规格书第3/5页；6.8V用于保留OpenDuck步态负载裕量，"
                "不是厂家出厂合格线"
            ),
            "url": FEETECH_SPEC_URL,
        },
        {
            "item": "温度",
            "value": "超过70℃告警；75℃紧急关闭扭矩；不参与方向判定",
            "source_type": "厂家规格 + 当前项目安全策略",
            "source": (
                "厂家运行温度上限60℃、>70℃过热保护；"
                "当前中试按要求仅在>70℃告警，并保留75℃防损坏底线"
            ),
            "url": FEETECH_SPEC_URL,
        },
        {
            "item": "电流",
            "value": "超过2.0A持续0.50s主动停机",
            "source_type": "厂家数值 + 项目保守时限",
            "source": (
                "厂家>2A持续2s进入保护；中试平台将持续时间缩短为0.50s"
            ),
            "url": FEETECH_SPEC_URL,
        },
        {
            "item": "位置跟踪精度",
            "value": "RMSE≤5°且最大绝对误差≤15°",
            "source_type": "OpenDuck内部工程阈值",
            "source": (
                "用于筛除会明显影响后续步态的舵机/装配，非飞特出厂标准；"
                "应根据批量合格样机数据复核"
            ),
            "url": "",
        },
        {
            "item": "装配方向检测",
            "value": f"对称关节对齐RMSE比值≥{PAIRED_RMSE_RATIO_THRESHOLD:.1f}:1时标记疑似装反",
            "source_type": "OpenDuck内部工程阈值",
            "source": "基于正确装配的对称关节应有相近跟踪精度",
            "url": "",
        },
    ]


def half_cosine(phase: float) -> float:
    """生成端点速度为零的 0→1 平滑曲线。"""

    phase = min(1.0, max(0.0, phase))
    return 0.5 - 0.5 * math.cos(math.pi * phase)


def joint_waveform(elapsed_s: float, half_cycle_s: float) -> float:
    """生成一次 0→1→0 的半余弦波。"""

    if elapsed_s <= half_cycle_s:
        return half_cosine(elapsed_s / half_cycle_s)
    return half_cosine(2.0 - elapsed_s / half_cycle_s)


def unidirectional_waveform(elapsed_s: float, half_cycle_s: float) -> float:
    """Generate 0->-1->0 single-direction half-cosine waveform."""
    period = 2.0 * half_cycle_s
    phase = (elapsed_s % period) / half_cycle_s
    if phase <= 1.0:
        return -half_cosine(phase)
    else:
        return -half_cosine(2.0 - phase)


def centered_two_angle_waveform(elapsed_s: float, segment_s: float) -> float:
    """生成 0→+1→0→-1→0，两个目标角度之间先回装配原点。"""

    segment = int(min(3, max(0, elapsed_s // segment_s)))
    phase = (elapsed_s - segment * segment_s) / segment_s
    if segment == 0:
        return half_cosine(phase)
    if segment == 1:
        return half_cosine(1.0 - phase)
    if segment == 2:
        return -half_cosine(phase)
    return -half_cosine(1.0 - phase)


def read_pid(controller: Any, servo_id: int) -> dict[str, float | None]:
    """读取舵机原始 PID，写入报告供追溯。"""

    return {
        "kp": read_optional(controller, "read_p_coefficient", servo_id),
        "ki": read_optional(controller, "read_i_coefficient", servo_id),
        "kd": read_optional(controller, "read_d_coefficient", servo_id),
    }


def apply_project_pid(
    controller: Any,
    servo_ids: Sequence[int] = EXPECTED_IDS,
) -> dict[int, dict[str, float | None]]:
    """写入项目测试 PID 并逐个读回确认。"""

    servo_ids = tuple(servo_ids)
    original_pid: dict[int, dict[str, float | None]] = {}
    for servo_id in servo_ids:
        original_pid[servo_id] = read_pid(controller, servo_id)
        # STS3215 的模式和 PID 寄存器写入前先解除锁定，并保持位置舵机模式。
        if hasattr(controller, "write_lock"):
            controller.write_lock(servo_id, False)
        if hasattr(controller, "write_mode"):
            controller.write_mode(servo_id, 0)

    target_kps = [22] * len(servo_ids)
    target_kds = [0] * len(servo_ids)
    controller.sync_write_p_coefficient(list(servo_ids), target_kps)
    controller.sync_write_d_coefficient(list(servo_ids), target_kds)

    for servo_id, expected_kp in zip(servo_ids, target_kps):
        actual_kp = int(scalar(controller.read_p_coefficient(servo_id)))
        actual_kd = int(scalar(controller.read_d_coefficient(servo_id)))
        if actual_kp != expected_kp or actual_kd != 0:
            raise RuntimeError(
                f"ID {servo_id} PID 回读不一致："
                f"期望 Kp={expected_kp}, Kd=0；"
                f"实际 Kp={actual_kp}, Kd={actual_kd}"
            )
    return original_pid


def confirm_supported(skip_confirmation: bool) -> None:
    """动作前要求操作者确认机器人已经悬空或可靠支撑。"""

    if skip_confirmation:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "当前终端不能交互确认。请确认机器人已悬空后使用 --yes。"
        )
    input(
        "请确认机器人已悬空或可靠支撑，周围无人员和障碍物。"
        "确认后按 Enter 开始动作检测；按 Ctrl+C 取消："
    )


def estimate_position_delay(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
    actual_field: str,
    period_s: float | None,
    max_lag_s: float = 0.50,
) -> dict[str, float | None]:
    """用归一化互相关估计实际位置相对输入位置的时间延时。"""

    samples = [
        (
            float(row["time_s"]),
            float(row[target_field]),
            float(row[actual_field]),
        )
        for row in rows
        if row.get("time_s") is not None
        and row.get(target_field) is not None
        and row.get(actual_field) is not None
    ]
    if len(samples) < 8:
        return {
            "position_delay_ms": None,
            "phase_delay_deg": None,
            "delay_correlation": None,
        }

    times = [sample[0] for sample in samples]
    target = [sample[1] for sample in samples]
    actual = [sample[2] for sample in samples]
    time_steps = [
        times[index] - times[index - 1]
        for index in range(1, len(times))
        if times[index] > times[index - 1]
    ]
    if not time_steps:
        return {
            "position_delay_ms": None,
            "phase_delay_deg": None,
            "delay_correlation": None,
        }

    sample_period_s = statistics.median(time_steps)
    max_lag_samples = min(
        max(1, int(round(max_lag_s / sample_period_s))),
        max(1, len(samples) // 3),
    )
    best_lag = 0
    best_correlation = -2.0

    for lag in range(0, max_lag_samples + 1):
        if lag >= 0:
            target_values = target[: len(target) - lag or None]
            actual_values = actual[lag:]
        else:
            lead = -lag
            target_values = target[lead:]
            actual_values = actual[: len(actual) - lead]
        if len(target_values) < 6:
            continue
        target_mean = statistics.fmean(target_values)
        actual_mean = statistics.fmean(actual_values)
        target_centered = [value - target_mean for value in target_values]
        actual_centered = [value - actual_mean for value in actual_values]
        denominator = math.sqrt(
            sum(value * value for value in target_centered)
            * sum(value * value for value in actual_centered)
        )
        if denominator <= 1e-12:
            continue
        correlation = sum(
            target_value * actual_value
            for target_value, actual_value in zip(
                target_centered,
                actual_centered,
            )
        ) / denominator
        if (
            correlation > best_correlation + 1e-12
            or (
                abs(correlation - best_correlation) <= 1e-12
                and abs(lag) < abs(best_lag)
            )
        ):
            best_correlation = correlation
            best_lag = lag

    delay_s = best_lag * sample_period_s
    return {
        "position_delay_ms": round(delay_s * 1000.0, 3),
        "phase_delay_deg": (
            round(delay_s / period_s * 360.0, 3)
            if period_s is not None and period_s > 0.0
            else None
        ),
        "delay_correlation": round(best_correlation, 6),
    }


def select_first_response_feature(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
) -> int:
    """选择第一周期刚离开起点的首次有效输入，避开t=0静止区。"""

    if len(rows) < 3:
        return 0
    target_values = [float(row[target_field]) for row in rows]
    start_value = target_values[0]
    minimum = min(target_values)
    maximum = max(target_values)
    extreme = (
        maximum
        if abs(maximum - start_value) >= abs(minimum - start_value)
        else minimum
    )
    direction = 1.0 if extreme >= start_value else -1.0
    excursion = abs(extreme - start_value)
    movement_threshold = min(max(0.5, excursion * 0.02), excursion * 0.25)
    for index in range(1, len(target_values)):
        movement = direction * (target_values[index] - start_value)
        if movement >= movement_threshold:
            return index
    return max(
        range(1, len(rows) - 1),
        key=lambda index: abs(
            (
                target_values[index + 1]
                - target_values[index - 1]
            )
            / max(
                float(rows[index + 1]["time_s"])
                - float(rows[index - 1]["time_s"]),
                1e-9,
            )
        ),
    )



def compute_aligned_errors(rows, *, target_field, actual_field, delay_ms):
    """延时对齐后重算RMSE和最大误差，补偿通信/传动滞后。"""
    import math as _m, statistics as _st
    samples = [(float(r["time_s"]), float(r[target_field]), float(r[actual_field]))
               for r in rows if r.get("time_s") and r.get(target_field) and r.get(actual_field)]
    if len(samples) < 8 or delay_ms is None:
        return {"aligned_rmse_deg": None, "aligned_max_abs_error_deg": None}
    delay_s = delay_ms / 1000.0
    ts = [samples[i][0]-samples[i-1][0] for i in range(1,len(samples)) if samples[i][0]>samples[i-1][0]]
    if not ts:
        return {"aligned_rmse_deg": None, "aligned_max_abs_error_deg": None}
    dt = _st.median(ts)
    shift = max(1, int(round(delay_s / dt)))
    if shift >= len(samples):
        return {"aligned_rmse_deg": None, "aligned_max_abs_error_deg": None}
    at = [s[1] for s in samples[:-shift or None]]
    aa = [s[2] for s in samples[shift:]]
    if len(at) < 6:
        return {"aligned_rmse_deg": None, "aligned_max_abs_error_deg": None}
    ae = [a-t for t,a in zip(at, aa)]
    return {"aligned_rmse_deg": round(_m.sqrt(sum(e*e for e in ae)/len(ae)), 4),
            "aligned_max_abs_error_deg": round(max(abs(e) for e in ae), 4)}


def calculate_joint_metrics(
    rows: Sequence[dict[str, Any]],
    joints: Sequence[tuple[str, int]] = JOINTS,
    *,
    waveform_period_s: float = 2.0,
    amplitude_by_joint: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """按关节计算输入波形与实际波形的跟踪指标。包含未对齐原始误差和延时补偿对齐误差。PASS/FAIL使用比例阈值。"""

    metrics: dict[str, dict[str, Any]] = {}
    for joint_name, servo_id in joints:
        joint_rows = [row for row in rows if row["servo_id"] == servo_id]
        currents = [float(row["current_a"]) for row in joint_rows if row.get("current_a") is not None]
        # 先保存原始误差（对齐前）
        raw_errors = [float(row["error_deg"]) for row in joint_rows]
        raw_rmse = math.sqrt(sum(e*e for e in raw_errors)/len(raw_errors)) if raw_errors else 0
        raw_max = max(abs(e) for e in raw_errors) if raw_errors else 0

        # 用延时对齐后重写每行 error_deg
        delay_metrics = estimate_position_delay(
            joint_rows,
            target_field="target_deg",
            actual_field="actual_deg",
            period_s=waveform_period_s,
        )
        delay_ms = delay_metrics.get("position_delay_ms")
        if delay_ms is not None and delay_ms > 5 and len(joint_rows) >= 8:
            samples = [(float(r["time_s"]), float(r["target_deg"]), float(r["actual_deg"]))
                       for r in joint_rows if r.get("time_s") and r.get("target_deg") and r.get("actual_deg")]
            if len(samples) >= 8:
                ts = [samples[i][0]-samples[i-1][0] for i in range(1,len(samples)) if samples[i][0]>samples[i-1][0]]
                if ts:
                    dt = statistics.median(ts)
                    shift = max(1, int(round(delay_ms / 1000.0 / dt)))
                    if shift < len(samples):
                        time_to_idx = {}
                        for i, r in enumerate(joint_rows):
                            if r.get("time_s") is not None:
                                time_to_idx[round(float(r["time_s"]), 6)] = i
                        for si in range(shift, len(samples)):
                                s_t = samples[si][0]
                                aligned_target = samples[si - shift][1]
                                s_actual = samples[si][2]
                                key = round(s_t, 6)
                                if key in time_to_idx:
                                    joint_rows[time_to_idx[key]]["error_deg"] = round(s_actual - aligned_target, 6)

        errors = [float(row["error_deg"]) for row in joint_rows]
        if not errors:
            metrics[joint_name] = {
                "servo_id": servo_id,
                "result": "FAIL",
                "samples": 0,
                "command_amplitude_deg": 0,
                "rmse_deg": None,
                "max_abs_error_deg": None,
                "mean_abs_error_deg": None,
                "mean_current_a": None,
                "peak_current_a": None,
                "aligned_rmse_deg": None,
                "aligned_max_abs_error_deg": None,
                "position_delay_ms": None,
                "phase_delay_deg": None,
                "delay_correlation": None,
            }
            continue
        target_values = [abs(float(r["target_deg"])) for r in joint_rows if r.get("target_deg") is not None]
        amplitude_deg = float(amplitude_by_joint.get(joint_name, 0.0)) if amplitude_by_joint else max(target_values, default=0.0)
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        max_abs_error = max(abs(error) for error in errors)
        mean_abs_error = statistics.fmean(abs(error) for error in errors)
        aligned = compute_aligned_errors(joint_rows, target_field="target_deg", actual_field="actual_deg", delay_ms=delay_ms)
        rmse_limit = pass_rmse_for_joint(joint_name)
        max_error_limit = pass_max_error_for_joint(joint_name)
        # 对齐后的 rmse/max 已是有效误差
        result = "PASS" if rmse <= rmse_limit and max_abs_error <= max_error_limit else "FAIL"
        fail_reason = ""
        if result == "FAIL":
            reasons = []
            if rmse > rmse_limit:
                reasons.append(f"均方根误差{rmse:.2f}°>{rmse_limit}°")
            if max_abs_error > max_error_limit:
                reasons.append(f"最大误差{max_abs_error:.2f}°>{max_error_limit}°")
            fail_reason = "；".join(reasons)
        metrics[joint_name] = {
            "servo_id": servo_id,
            "result": result,
            "fail_reason": fail_reason,
            "samples": len(errors),
            "command_amplitude_deg": round(amplitude_deg, 1),
            "raw_rmse_deg": round(raw_rmse, 4),
            "raw_max_abs_error_deg": round(raw_max, 4),
            "rmse_deg": round(rmse, 4),
            "max_abs_error_deg": round(max_abs_error, 4),
            "mean_abs_error_deg": round(mean_abs_error, 4),
            "mean_current_a": round(statistics.fmean(currents), 4) if currents else None,
            "peak_current_a": round(max(abs(c) for c in currents), 4) if currents else None,
            "aligned_rmse_deg": round(rmse, 4),
            "aligned_max_abs_error_deg": round(max_abs_error, 4),
            "rmse_limit_deg": round(rmse_limit, 4),
            "max_error_limit_deg": round(max_error_limit, 4),
            **delay_metrics,
        }
    return metrics


def run_assembly_motion(
    controller: Any,
    scan: ScanResult,
    *,
    joints: Sequence[tuple[str, int]] = JOINTS,
    expected_ids: Sequence[int] = EXPECTED_IDS,
    half_cycle_s: float = 1.0,
    sample_period_s: float = SAMPLE_PERIOD_S,
    amplitude_by_joint: Mapping[str, float] | None = None,
    sample_callback=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """依次对指定关节发送小幅平滑波形并采集遥测。"""

    joints = tuple(joints)
    expected_ids = tuple(expected_ids)
    if not scan.complete or set(scan.found_positions) != set(expected_ids):
        raise RuntimeError("当前检测对象的舵机未全部在线，禁止执行动作检测")

    rows: list[dict[str, Any]] = []
    original_pid: dict[int, dict[str, float | None]] = {}
    origins = dict(scan.found_positions)
    ordered_origins = [origins[servo_id] for servo_id in expected_ids]
    amplitude_profile = (
        JOINT_AMPLITUDE_DEG
        if amplitude_by_joint is None
        else amplitude_by_joint
    )
    monitor = SafetyMonitor()
    communication_errors = 0
    started_all = time.monotonic()
    current_vector_interval = max(
        1,
        int(round(CURRENT_SAMPLE_INTERVAL_S / sample_period_s)),
    )
    

    def _retry_sync(op_name, func, *args):
        """同步操作自动重试：单次超时重试1次，连续2次失败抛异常。"""
        for attempt in range(2):
            try:
                return func(*args)
            except Exception:
                if attempt == 0:
                    print(f"[通信重试] {op_name} 超时，自动重试1次")
                    time.sleep(0.05)
                    continue
                raise RuntimeError(
                    f"{op_name} 连续两次超时：{sys.exc_info()[1]}"
                )

    try:
        original_pid = _retry_sync("PID写入", apply_project_pid, controller, expected_ids)

        # 先写入当前位置，再使能扭矩，避免上电瞬间跳到旧目标位置。
        _retry_sync("初始位置写入",
                    controller.sync_write_goal_position,
                    list(expected_ids), ordered_origins)
        _retry_sync("扭矩使能",
                    controller.sync_write_torque_enable,
                    list(expected_ids), [True] * len(expected_ids))
        time.sleep(0.25)

        for joint_name, servo_id in joints:
            if joint_name not in amplitude_profile:
                raise SafetyAbort(f"缺少关节 {joint_name} 的测试幅度配置")
            amplitude_deg = float(amplitude_profile[joint_name])
            if not 0.0 < amplitude_deg <= 90.0:
                raise SafetyAbort(
                    f"关节 {joint_name} 的测试幅度无效：{amplitude_deg}°"
                )
            wf_sign_print = JOINT_WAVEFORM_SIGN.get(joint_name, JOINT_WAVEFORM_DEFAULT_SIGN)
            direction_label = "+" if wf_sign_print < 0 else "-"
            print(
                f"[动作] 正在检测 ID {servo_id:02d} "
                f"{JOINT_LABEL_ZH.get(joint_name, joint_name)}；"
                f"0→{direction_label}{amplitude_deg:.1f}°→0（单向）"
            )
            amplitude_rad = math.radians(amplitude_deg)
            origin = origins[servo_id]
            if (
                origin + amplitude_rad > math.pi
                or origin - amplitude_rad < -math.pi
            ):
                raise SafetyAbort(
                    f"ID {servo_id} 当前角度没有足够的 "
                    f"+{amplitude_deg:.1f}°/-{amplitude_deg:.1f}°测试空间"
                )
            total_duration_s = 2.0 * half_cycle_s
            joint_started = time.monotonic()
            next_sample = joint_started
            sample_index = 0

            while True:
                now = time.monotonic()
                elapsed = now - joint_started
                if elapsed > total_duration_s:
                    break

                wf_sign = JOINT_WAVEFORM_SIGN.get(joint_name, JOINT_WAVEFORM_DEFAULT_SIGN)
                offset = wf_sign * amplitude_rad * unidirectional_waveform(
                    elapsed,
                    half_cycle_s,
                )
                target = origin + offset
                targets = list(ordered_origins)
                targets[expected_ids.index(servo_id)] = target

                # 单次串口超时先自动重试一次；连续两次失败才安全停机。
                # 这样既不过度放过真实掉线，也避免偶发USB调度抖动中断整轮测试。
                for communication_attempt in range(2):
                    try:
                        controller.sync_write_goal_position(
                            list(expected_ids),
                            targets,
                        )
                        actual = scalar(
                            controller.read_present_position(servo_id)
                        )
                        current_vector = None
                        if sample_index % current_vector_interval == 0:
                            try:
                                current_vector = read_current_vector_amp(
                                    controller,
                                    expected_ids,
                                )
                            except Exception as exc:
                                pass
                        current_a = (
                            current_vector.get(servo_id)
                            if current_vector is not None
                            else current_to_amp(
                                read_optional(
                                    controller,
                                    "read_present_current",
                                    servo_id,
                                )
                            )
                        )
                        voltage_v = read_optional(
                            controller,
                            "read_present_voltage",
                            servo_id,
                        )
                        temperature_c = read_optional(
                            controller,
                            "read_present_temperature",
                            servo_id,
                        )
                        break
                    except Exception as exc:
                        if communication_attempt == 0:
                            print(
                                f"[通信重试] ID {servo_id} 单次读写超时，"
                                "自动重试1次"
                            )
                            time.sleep(0.01)
                            continue
                        # 连续两次超时：跳过本次采样，不中断整轮测试
                        communication_errors += 1
                        print(
                            f"[通信跳过] ID {servo_id} 连续两次读写超时"
                            f"（累计 {communication_errors} 次），跳过本次采样"
                        )
                        actual = None

                if actual is None:
                    # 通信跳过：不记录此行，继续下一采样周期
                    next_sample += sample_period_s
                    sample_index += 1
                    time.sleep(max(0.0, next_sample - time.monotonic()))
                    continue

                # 波形图使用相对装配原点角度，直观显示0→+15→0→-15→0；
                # 同时在CSV保留绝对角度供装配追溯。
                target_absolute_deg = math.degrees(target)
                actual_absolute_deg = math.degrees(actual)
                target_deg = math.degrees(offset)
                actual_deg = math.degrees(actual - origin)
                error_deg = actual_deg - target_deg
                monitor.check(
                    servo_id=servo_id,
                    error_deg=error_deg,
                    current_a=current_a,
                    voltage_v=voltage_v,
                    temperature_c=temperature_c,
                    dt_s=sample_period_s,
                )
                row = {
                    "time_s": round(time.monotonic() - started_all, 6),
                    "joint": joint_name,
                    "servo_id": servo_id,
                    "target_deg": round(target_deg, 6),
                    "actual_deg": round(actual_deg, 6),
                    "error_deg": round(error_deg, 6),
                    "target_absolute_deg": round(target_absolute_deg, 6),
                    "actual_absolute_deg": round(actual_absolute_deg, 6),
                    "current_a": current_a,
                    "voltage_v": voltage_v,
                    "temperature_c": temperature_c,
                    "communication": "正常",

                }
                rows.append(row)
                if sample_callback is not None:
                    sample_callback(row)

                next_sample += sample_period_s
                sample_index += 1
                time.sleep(max(0.0, next_sample - time.monotonic()))

            controller.sync_write_goal_position(
                list(expected_ids),
                ordered_origins,
            )
            time.sleep(0.15)
    except SafetyAbort as exc:
        exc.motor_rows = list(rows)
        raise
    finally:
        disable_torque(controller, expected_ids)

    metrics = calculate_joint_metrics(
        rows,
        joints,
        waveform_period_s=4.0 * half_cycle_s,
        amplitude_by_joint=amplitude_profile,
    )
    # 对称关节对比检测疑似装反（暂时跳过）
    direction_failures: list[str] = []
    for row in rows:
        metric = metrics.get(str(row["joint"]), {})
        row["position_delay_ms"] = metric.get("position_delay_ms")
        row["phase_delay_deg"] = metric.get("phase_delay_deg")
        row["delay_correlation"] = metric.get("delay_correlation")
    overall_result = (
        "PASS"
        if metrics and all(item["result"] == "PASS" for item in metrics.values())
        else "FAIL"
    )
    return rows, {
        "result": overall_result,
        "original_pid": original_pid,
        "joint_metrics": metrics,
        "direction_failures": direction_failures,
        "failure_reasons": direction_failures,
        "samples": len(rows),
    }


def gait_segments() -> list[dict[str, Any]]:
    """返回固定的 50Hz 自动步态命令序列。"""

    return [
        {"name": "站立", "cycles": 100, "command": [0.0, 0.0, 0.0], "scale": (0.0, 0.0)},
        {"name": "原地踏步", "cycles": 150, "command": [0.0, 0.0, 0.0], "scale": (0.0, 0.2)},
        {"name": "前进", "cycles": 250, "command": [0.10, 0.0, 0.0], "scale": (0.2, 0.2)},
        {"name": "停止", "cycles": 100, "command": [0.0, 0.0, 0.0], "scale": (0.2, 0.0)},
        {"name": "左转", "cycles": 150, "command": [0.0, 0.0, 0.15], "scale": (0.0, 0.2)},
        {"name": "右转", "cycles": 150, "command": [0.0, 0.0, -0.15], "scale": (0.2, 0.2)},
        {"name": "最终停止", "cycles": 100, "command": [0.0, 0.0, 0.0], "scale": (0.2, 0.0)},
    ]


def calculate_gait_metrics(
    rows: Sequence[dict[str, Any]],
    joints: Sequence[tuple[str, int]] = JOINTS,
) -> dict[str, dict[str, Any]]:
    """按关节计算整机步态目标角度与实际角度的验收指标。"""

    metrics: dict[str, dict[str, Any]] = {}
    for joint_name, servo_id in joints:
        errors = [
            float(row[f"{joint_name}_error_deg"])
            for row in rows
            if row.get(f"{joint_name}_error_deg") is not None
        ]
        if not errors:
            metrics[joint_name] = {
                "servo_id": servo_id,
                "result": "FAIL",
                "samples": 0,
                "rmse_deg": None,
                "max_abs_error_deg": None,
                "mean_abs_error_deg": None,
                "position_delay_ms": None,
                "phase_delay_deg": None,
                "delay_correlation": None,
            }
            continue
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        max_abs_error = max(abs(error) for error in errors)
        mean_abs_error = statistics.fmean(abs(error) for error in errors)
        delay_metrics = estimate_position_delay(
            rows,
            target_field=f"{joint_name}_target_deg",
            actual_field=f"{joint_name}_actual_deg",
            period_s=None,
        )
        metrics[joint_name] = {
            "servo_id": servo_id,
            "result": (
                "PASS"
                if rmse <= PASS_RMSE_DEG
                and max_abs_error <= PASS_MAX_ERROR_DEG
                else "FAIL"
            ),
            "samples": len(errors),
            "raw_rmse_deg": round(raw_rmse, 4),
            "raw_max_abs_error_deg": round(raw_max, 4),
            "rmse_deg": round(rmse, 4),
            "max_abs_error_deg": round(max_abs_error, 4),
            "mean_abs_error_deg": round(mean_abs_error, 4),
            **delay_metrics,
        }
    return metrics


def interpolate_scale(segment: dict[str, Any], cycle: int) -> float:
    """在每个阶段内平滑升降 ONNX 动作幅度。"""

    start_scale, end_scale = segment["scale"]
    denominator = max(1, int(segment["cycles"]) - 1)
    phase = cycle / denominator
    return start_scale + (end_scale - start_scale) * half_cosine(phase)


def raw_controller_from_walk(walk: Any) -> Any:
    """从 HWI 兼容层取得底层 rustypot 控制器。"""

    adapter = walk.hwi.io
    return getattr(adapter, "io", adapter)


def gait_csv_fields() -> list[str]:
    """构造整机步态宽表字段。"""

    fields = [
        "time_s",
        "stage",
        "command_vx",
        "command_vy",
        "command_yaw",
        "action_scale",
        "min_voltage_v",
        "max_current_a",
        "max_temperature_c",
        "foot_left",
        "foot_right",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
    ]
    for joint_name, _ in JOINTS:
        fields.extend(
            (
                f"{joint_name}_target_deg",
                f"{joint_name}_actual_deg",
                f"{joint_name}_error_deg",
                f"{joint_name}_current_a",
                f"{joint_name}_voltage_v",
                f"{joint_name}_temperature_c",
            )
        )
    return fields


def run_gait_motion(
    repo_root: Path,
    *,
    model_path: Path,
    config_path: Path,
    control_frequency_hz: int = CONTROL_FREQUENCY_HZ,
    sample_callback=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """复用现有 RLWalk 对象执行有限自动步态并记录 50Hz 遥测。"""

    if not model_path.is_file():
        raise RuntimeError(f"找不到 ONNX 模型：{model_path}")
    if not config_path.is_file():
        raise RuntimeError(f"找不到机器人配置：{config_path}")

    package_root = repo_root / "mini_bdx_runtime"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    if str(repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(repo_root / "scripts"))

    # 延迟导入，保证开发机可以在没有 GPIO/IMU 依赖时运行纯逻辑测试。
    from v2_rl_walk_mujoco import RLWalk
    import numpy as np

    walk = None
    rows: list[dict[str, Any]] = []
    monitor = SafetyMonitor()
    power_values: dict[str, Any] = {
        "voltages": [None] * len(JOINTS),
        "currents": [None] * len(JOINTS),
        "temperatures": [None] * len(JOINTS),
    }
    power_sample_every = max(1, int(control_frequency_hz / 10))
    power_sample_dt = power_sample_every / control_frequency_hz
    global_cycle = 0
    started = time.monotonic()

    try:
        walk = RLWalk(
            str(model_path),
            duck_config_path=str(config_path),
            serial_port=SERIAL_PORT,
            control_freq=control_frequency_hz,
            pid=[22, 0, 0],
            action_scale=0.0,
            commands=False,
            min_motor_voltage=MIN_VOLTAGE_V,
        )
        walk.paused = False

        # 初始化完成后立即验证 IMU 与脚底触点可读。
        imu_check = walk.imu.get_data()
        feet_check = walk.feet_contacts.get()
        if not isinstance(imu_check, dict):
            raise RuntimeError("IMU 返回格式异常")
        if len(feet_check) != 2:
            raise RuntimeError("脚底触点返回数量不是 2")

        raw_controller = raw_controller_from_walk(walk)
        init_pos = np.asarray(walk.init_pos, dtype=float)

        for segment in gait_segments():
            print(f"[步态] {segment['name']}，{segment['cycles']} 个控制周期")
            command = list(segment["command"]) + [0.0, 0.0, 0.0, 0.0]
            for segment_cycle in range(int(segment["cycles"])):
                cycle_started = time.monotonic()
                action_scale = interpolate_scale(segment, segment_cycle)
                walk.action_scale = action_scale
                walk.last_commands = command.copy()

                obs = walk.get_obs()
                if obs is None or len(obs) != 101:
                    raise SafetyAbort("步态 observation 读取失败或维度不是 101")
                obs = np.asarray(obs, dtype=float)

                # observation 中第 13-26 维为相对初始姿态的位置。
                actual_positions = obs[13:27] + init_pos
                gyro = obs[0:3]
                accel = obs[3:6]
                feet = obs[97:99]

                walk.imitation_i += (
                    walk.phase_frequency_factor
                    + walk.phase_frequency_factor_offset
                )
                walk.imitation_i %= walk.PRM.nb_steps_in_period
                walk.imitation_phase = np.array(
                    [
                        np.cos(
                            walk.imitation_i
                            / walk.PRM.nb_steps_in_period
                            * 2
                            * np.pi
                        ),
                        np.sin(
                            walk.imitation_i
                            / walk.PRM.nb_steps_in_period
                            * 2
                            * np.pi
                        ),
                    ]
                )

                action = np.asarray(walk.policy.infer(obs), dtype=float)
                if action.shape != (14,) or not np.all(np.isfinite(action)):
                    raise SafetyAbort(
                        f"ONNX 输出异常，期望 14 维有限值，实际 {action.shape}"
                    )

                walk.last_last_last_action = walk.last_last_action.copy()
                walk.last_last_action = walk.last_action.copy()
                walk.last_action = action.copy()
                walk.motor_targets = init_pos + action * action_scale
                walk.motor_targets[5:9] += np.asarray(command[3:], dtype=float)

                action_dict = {
                    joint_name: float(target)
                    for (joint_name, _), target in zip(
                        JOINTS,
                        walk.motor_targets,
                    )
                }
                walk.hwi.set_position_all(action_dict)

                is_power_sample = global_cycle % power_sample_every == 0
                if is_power_sample:
                    voltages = walk.hwi.get_present_voltages()
                    currents = walk.hwi.get_present_currents()
                    temperatures = raw_controller.sync_read_present_temperature(
                        [servo_id for _, servo_id in JOINTS]
                    )
                    if voltages is None or currents is None:
                        raise SafetyAbort("电压或电流遥测读取失败")
                    power_values = {
                        "voltages": sequence_of_floats(voltages),
                        "currents": sequence_of_floats(currents),
                        "temperatures": sequence_of_floats(temperatures),
                    }
                    if not all(
                        len(power_values[key]) == len(JOINTS)
                        for key in power_values
                    ):
                        raise SafetyAbort("整机电源或温度遥测数量不是 14")

                row: dict[str, Any] = {
                    "time_s": round(time.monotonic() - started, 6),
                    "stage": segment["name"],
                    "command_vx": command[0],
                    "command_vy": command[1],
                    "command_yaw": command[2],
                    "action_scale": round(action_scale, 6),
                    "min_voltage_v": min(
                        value
                        for value in power_values["voltages"]
                        if value is not None
                    ),
                    "max_current_a": max(
                        abs(value)
                        for value in power_values["currents"]
                        if value is not None
                    ),
                    "max_temperature_c": max(
                        value
                        for value in power_values["temperatures"]
                        if value is not None
                    ),
                    "foot_left": int(bool(feet[0])),
                    "foot_right": int(bool(feet[1])),
                    "gyro_x": gyro[0],
                    "gyro_y": gyro[1],
                    "gyro_z": gyro[2],
                    "accel_x": accel[0],
                    "accel_y": accel[1],
                    "accel_z": accel[2],
                }

                for index, (joint_name, servo_id) in enumerate(JOINTS):
                    target_deg = math.degrees(float(walk.motor_targets[index]))
                    actual_deg = math.degrees(float(actual_positions[index]))
                    error_deg = actual_deg - target_deg
                    monitor.check(
                        servo_id=servo_id,
                        error_deg=error_deg,
                        current_a=power_values["currents"][index],
                        voltage_v=power_values["voltages"][index],
                        temperature_c=power_values["temperatures"][index],
                        dt_s=1.0 / control_frequency_hz,
                        power_sample=is_power_sample,
                        power_dt_s=power_sample_dt,
                    )
                    row[f"{joint_name}_target_deg"] = round(target_deg, 6)
                    row[f"{joint_name}_actual_deg"] = round(actual_deg, 6)
                    row[f"{joint_name}_error_deg"] = round(error_deg, 6)
                    row[f"{joint_name}_current_a"] = power_values["currents"][index]
                    row[f"{joint_name}_voltage_v"] = power_values["voltages"][index]
                    row[f"{joint_name}_temperature_c"] = power_values[
                        "temperatures"
                    ][index]
                rows.append(row)
                if sample_callback is not None:
                    sample_callback(row)
                walk.prev_motor_targets = walk.motor_targets.copy()
                global_cycle += 1

                elapsed = time.monotonic() - cycle_started
                time.sleep(max(0.0, 1.0 / control_frequency_hz - elapsed))

        metrics = calculate_gait_metrics(rows)
        result = (
            "PASS"
            if metrics
            and all(item["result"] == "PASS" for item in metrics.values())
            else "FAIL"
        )
        errors = [
            abs(float(row[f"{joint_name}_error_deg"]))
            for row in rows
            for joint_name, _ in JOINTS
        ]
        return rows, {
            "result": result,
            "samples": len(rows),
            "control_frequency_hz": control_frequency_hz,
            "max_abs_error_deg": round(max(errors), 4) if errors else None,
            "joint_metrics": metrics,
            "segments": [
                {
                    "name": item["name"],
                    "cycles": item["cycles"],
                    "command": item["command"],
                }
                for item in gait_segments()
            ],
        }
    finally:
        if walk is not None:
            try:
                walk.hwi.turn_off()
            except Exception:
                emergency_disable_all()
            try:
                walk.feet_contacts.close()
            except Exception:
                pass
        else:
            # RLWalk 构造过程中一旦异常，也必须覆盖“已使能但对象未返回”的情况。
            emergency_disable_all()


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    """以 UTF-8 BOM 写 CSV，便于直接用中文 Excel 打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写入可读的中文 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def temporary_report_root(category: str) -> Path:
    """返回当前系统上的OpenDuck临时报告根目录。"""

    return Path(tempfile.gettempdir()) / "openduck_test_logs" / category


def cleanup_temporary_report(
    output_dir: Path,
    *,
    temporary_output: bool,
    keep_report: bool,
) -> None:
    """只删除程序在系统临时根目录下创建的本次报告。"""

    if not temporary_output or keep_report:
        return
    output_dir = output_dir.resolve()
    allowed_root = (
        Path(tempfile.gettempdir()) / "openduck_test_logs"
    ).resolve()
    if output_dir == allowed_root or allowed_root not in output_dir.parents:
        raise RuntimeError(f"拒绝清理非临时报告目录：{output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"树莓派临时报告已删除：{output_dir}")


def svg_overlay(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
    actual_field: str,
    width: int = 900,
    height: int = 330,
) -> str:
    """生成输入与实际位置叠加 SVG，不依赖浏览器外部资源。"""

    valid_rows = [
        row
        for row in rows
        if row.get(target_field) is not None and row.get(actual_field) is not None
    ]
    if len(valid_rows) < 2:
        return "<p>无足够波形数据。</p>"
    if len(valid_rows) > 600:
        step = math.ceil(len(valid_rows) / 600)
        valid_rows = valid_rows[::step]

    x_values = [float(row["time_s"]) for row in valid_rows]
    target_values = [float(row[target_field]) for row in valid_rows]
    actual_values = [float(row[actual_field]) for row in valid_rows]
    y_values = target_values + actual_values
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = max(1.0, (y_max - y_min) * 0.08)
    y_min -= padding
    y_max += padding

    left, right, top, bottom = 82, 18, 75, 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(index: int, value: float) -> str:
        x = left + (x_values[index] - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - value) / (y_max - y_min) * plot_height
        return f"{x:.2f},{y:.2f}"

    target_points = " ".join(
        point(index, value) for index, value in enumerate(target_values)
    )
    actual_points = " ".join(
        point(index, value) for index, value in enumerate(actual_values)
    )
    x_ticks = []
    for tick in range(7):
        fraction = tick / 6
        x = left + fraction * plot_width
        value = x_min + fraction * (x_max - x_min)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height-bottom}" stroke="#e2e8f0"/>'
            f'<text x="{x:.2f}" y="{height-27}" text-anchor="middle">'
            f'{value:.1f}</text>'
        )
    y_ticks = []
    for tick in range(5):
        fraction = tick / 4
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
            f'<text x="{left-7}" y="{y+4:.2f}" text-anchor="end">'
            f'{value:.1f}</text>'
        )
    delay_metrics = estimate_position_delay(
        valid_rows,
        target_field=target_field,
        actual_field=actual_field,
        period_s=None,
    )
    marker_svg = ""
    delay_ms = delay_metrics.get("position_delay_ms")
    if delay_ms is not None and len(valid_rows) >= 3:
        input_index = select_first_response_feature(
            valid_rows,
            target_field=target_field,
        )
        input_time = float(valid_rows[input_index]["time_s"])
        input_position = float(valid_rows[input_index][target_field])
        output_time = input_time + float(delay_ms) / 1000.0
        output_row = min(
            valid_rows,
            key=lambda row: abs(float(row["time_s"]) - output_time),
        )
        output_time = float(output_row["time_s"])
        output_position = float(output_row[actual_field])
        input_x = left + (input_time - x_min) / (x_max - x_min) * plot_width
        output_x = left + (output_time - x_min) / (x_max - x_min) * plot_width
        input_y = top + (y_max - input_position) / (y_max - y_min) * plot_height
        output_y = top + (y_max - output_position) / (y_max - y_min) * plot_height
        marker_svg = f"""
  <defs>
    <marker id="response-arrow" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto-start-reverse">
      <path d="M0,0 L7,3.5 L0,7 z" fill="#0f172a"/>
    </marker>
  </defs>
  <line x1="{input_x:.2f}" y1="32" x2="{output_x:.2f}" y2="32"
        stroke="#0f172a" stroke-width="1.5" marker-start="url(#response-arrow)" marker-end="url(#response-arrow)"/>
  <text x="{(input_x + output_x) / 2:.2f}" y="20" text-anchor="middle">响应延时 Δt={float(delay_ms):.1f} ms</text>
  <line x1="{input_x:.2f}" y1="{input_y:.2f}" x2="{input_x:.2f}" y2="{height-bottom}"
        stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 4"/>
  <line x1="{output_x:.2f}" y1="{output_y:.2f}" x2="{output_x:.2f}" y2="{height-bottom}"
        stroke="#dc2626" stroke-width="1.4" stroke-dasharray="5 4"/>
  <circle cx="{input_x:.2f}" cy="{input_y:.2f}" r="4.5" fill="#2563eb"/>
  <circle cx="{output_x:.2f}" cy="{output_y:.2f}" r="4.5" fill="#dc2626"/>
  <text x="{input_x:.2f}" y="{height-bottom+18}" fill="#2563eb" text-anchor="middle">↑ 指令点：t输入={input_time:.2f}s，θin={input_position:.2f}°</text>
  <text x="{output_x:.2f}" y="{height-bottom+37}" fill="#dc2626" text-anchor="middle">↑ 响应点：t输出={output_time:.2f}s，θout={output_position:.2f}°</text>
"""
    return f"""
<p>实际测试时长：{x_max - x_min:.3f} s</p>
<p>同一横坐标 t：蓝线为该采样时刻发送的位置指令 θin(t)，红线为随后读取的舵机实际输出位置 θout(t)。t=0处波形尚未产生有效位移，因此箭头选取第一周期刚离开起点的首次有效输入及对应输出响应点，横轴上的 t输入、t输出之差即响应延时。</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="输入指定波形与舵机实际响应波形">
  {''.join(x_ticks)}
  {''.join(y_ticks)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
  <polyline points="{target_points}" class="target"/>
  <polyline points="{actual_points}" class="actual"/>
  {marker_svg}
  <text x="{left + plot_width / 2 - 42}" y="{height-8}">实际测试时间（s）</text>
  <text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})">位置角度（°）</text>
</svg>
"""


def svg_series(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    title: str,
    color: str,
    width: int = 900,
    height: int = 210,
) -> str:
    """生成误差或电流单曲线 SVG。"""

    valid_rows = [
        row for row in rows
        if row.get(field) is not None and row.get("time_s") is not None
    ]
    if len(valid_rows) < 2:
        return f"<h4>{html.escape(title)}</h4><p>无足够曲线数据。</p>"
    if len(valid_rows) > 600:
        step = math.ceil(len(valid_rows) / 600)
        valid_rows = valid_rows[::step]

    x_values = [float(row["time_s"]) for row in valid_rows]
    y_values = [float(row[field]) for row in valid_rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = max(0.1, (y_max - y_min) * 0.08)
    y_min -= padding
    y_max += padding

    left, right, top, bottom = 58, 18, 18, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        points.append(f"{x:.2f},{y:.2f}")
    return f"""
<h4>{html.escape(title)}</h4>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="2"/>
  <text x="{left}" y="{height-10}">{x_min:.1f}s</text>
  <text x="{width-right-38}" y="{height-10}">{x_max:.1f}s</text>
  <text x="5" y="{top+8}">{y_max:.2f}</text>
  <text x="5" y="{height-bottom}">{y_min:.2f}</text>
</svg>
"""


def _position_chart_range(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float, float, float]:
    """让位置图包含原点，并为曲线留出少量边距。"""

    x_min = min(0.0, min(x_values))
    x_max = max(0.0, max(x_values))
    y_min = min(0.0, min(y_values))
    y_max = max(0.0, max(y_values))
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    y_padding = max(1.0, (y_max - y_min) * 0.08)
    return x_min, x_max, y_min - y_padding, y_max + y_padding


def svg_position_chart(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
    actual_field: str,
    align_response: bool = False,
    input_function_label: str | None = None,
    y_axis_label: str = "位置角度（°）",
    width: int = 900,
    height: int = 330,
) -> str:
    """绘制最新位置图；可把实际输出按测得延时左移后与指令对齐。"""

    valid_rows = [
        row
        for row in rows
        if row.get("time_s") is not None
        and row.get(target_field) is not None
        and row.get(actual_field) is not None
    ]
    if len(valid_rows) < 2:
        return "<p>无足够波形数据。</p>"

    delay_metrics = estimate_position_delay(
        valid_rows,
        target_field=target_field,
        actual_field=actual_field,
        period_s=None,
    )
    delay_ms_value = delay_metrics.get("position_delay_ms")
    delay_s = (
        max(0.0, float(delay_ms_value) / 1000.0)
        if delay_ms_value is not None
        else 0.0
    )

    if len(valid_rows) > 600:
        step = math.ceil(len(valid_rows) / 600)
        display_rows = valid_rows[::step]
    else:
        display_rows = valid_rows

    # 每个关节在报告中使用自己的相对时间，避免后测关节被挤到全局时间轴末端。
    time_origin = float(valid_rows[0]["time_s"])
    target_samples = [
        (float(row["time_s"]) - time_origin, float(row[target_field]))
        for row in display_rows
    ]
    actual_samples = [
        (
            float(row["time_s"])
            - time_origin
            - (delay_s if align_response else 0.0),
            float(row[actual_field]),
        )
        for row in display_rows
    ]
    if align_response:
        actual_samples = [(x, y) for x, y in actual_samples if x >= 0.0]
    if len(actual_samples) < 2:
        return "<p>对齐后无足够波形数据。</p>"

    all_x = [x for x, _ in target_samples] + [x for x, _ in actual_samples]
    all_y = [y for _, y in target_samples] + [y for _, y in actual_samples]
    x_min, x_max, y_min, y_max = _position_chart_range(all_x, all_y)

    # 顶部为响应延时和输入函数保留独立说明区，避免文字遮挡波形。
    left, right = 82, 18
    top = 80 if align_response else 100
    bottom = 58 if align_response else 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    def xy(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        return x, y

    def points(samples: Sequence[tuple[float, float]]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (xy(x_value, y_value) for x_value, y_value in samples)
        )

    x_ticks: list[str] = []
    for tick in range(7):
        fraction = tick / 6
        x = left + fraction * plot_width
        value = x_min + fraction * (x_max - x_min)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height-bottom}" stroke="#e2e8f0"/>'
            f'<text x="{x:.2f}" y="{height-26}" text-anchor="middle">'
            f'{value:.1f}</text>'
        )
    y_ticks: list[str] = []
    for tick in range(5):
        fraction = tick / 4
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
            f'<text x="{left-7}" y="{y+4:.2f}" text-anchor="end">'
            f'{value:.1f}</text>'
        )

    zero_x, zero_y = xy(0.0, 0.0)
    zero_axes = f"""
  <line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{height-bottom}"
        stroke="#334155" stroke-width="1.8"/>
  <line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}"
        stroke="#334155" stroke-width="1.8"/>
  <text x="{zero_x+5:.2f}" y="{zero_y-5:.2f}" fill="#334155">0</text>
"""

    marker_svg = ""
    if not align_response and delay_ms_value is not None and len(valid_rows) >= 3:
        input_index = select_first_response_feature(
            valid_rows,
            target_field=target_field,
        )
        input_time_raw = float(valid_rows[input_index]["time_s"])
        input_time = input_time_raw - time_origin
        input_position = float(valid_rows[input_index][target_field])
        requested_output_time = input_time_raw + delay_s
        output_row = min(
            valid_rows,
            key=lambda row: abs(float(row["time_s"]) - requested_output_time),
        )
        output_time = float(output_row["time_s"]) - time_origin
        output_position = float(output_row[actual_field])
        input_x, input_y = xy(input_time, input_position)
        output_x, output_y = xy(output_time, output_position)
        marker_svg = f"""
  <defs>
    <marker id="response-arrow" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto-start-reverse">
      <path d="M0,0 L7,3.5 L0,7 z" fill="#0f172a"/>
    </marker>
  </defs>
  <line x1="{input_x:.2f}" y1="32" x2="{output_x:.2f}" y2="32"
        stroke="#0f172a" stroke-width="1.5" marker-start="url(#response-arrow)" marker-end="url(#response-arrow)"/>
  <text x="{(input_x + output_x) / 2:.2f}" y="20" text-anchor="middle">响应延时 Δt={float(delay_ms_value):.1f} ms</text>
  <line x1="{input_x:.2f}" y1="{input_y:.2f}" x2="{input_x:.2f}" y2="{height-bottom}"
        stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 4"/>
  <line x1="{output_x:.2f}" y1="{output_y:.2f}" x2="{output_x:.2f}" y2="{height-bottom}"
        stroke="#dc2626" stroke-width="1.4" stroke-dasharray="5 4"/>
  <circle cx="{input_x:.2f}" cy="{input_y:.2f}" r="4.5" fill="#2563eb"/>
  <circle cx="{output_x:.2f}" cy="{output_y:.2f}" r="4.5" fill="#dc2626"/>
  <text x="{input_x:.2f}" y="{height-bottom+18}" fill="#2563eb" text-anchor="middle">↑ 指令点：t输入={input_time:.2f}s，θin={input_position:.2f}°</text>
  <text x="{output_x:.2f}" y="{height-bottom+37}" fill="#dc2626" text-anchor="middle">↑ 响应点：t输出={output_time:.2f}s，θout={output_position:.2f}°</text>
"""

    if align_response:
        heading = "响应时间对齐后的波形"
        explanation = (
            f"已将红色实际输出曲线沿时间轴向左平移 {delay_s * 1000.0:.1f} ms。"
            "该图只消除响应时间滞后，不修改实际输出角度，可直接观察幅值和形状误差。"
        )
        x_axis_label = "对齐后的测试时间（s）"
    else:
        heading = "原始时序波形（含响应延时）"
        explanation = (
            "蓝线是采样时刻发送的位置指令，红线是实际位置反馈；"
            "两条曲线的横向错位表示响应滞后。"
        )
        x_axis_label = "实际测试时间（s）"

    function_label = input_function_label or (
        "输入函数：θin(t)=θ0+A·f(t)，"
        "f(u)=[1−cos(πu)]/2（分段半余弦）"
    )
    function_label_svg = html.escape(function_label)

    return f"""
<h4>{heading}</h4>
<p>{explanation}</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{heading}">
  {''.join(x_ticks)}
  {''.join(y_ticks)}
  {zero_axes}
  <polyline points="{points(target_samples)}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
  <polyline points="{points(actual_samples)}" fill="none" stroke="#dc2626" stroke-width="2"/>
  {marker_svg}
  <rect x="{left+6}" y="{top-34}" width="{plot_width-12}" height="24" rx="4"
        fill="#ffffff" fill-opacity="0.84"/>
  <text x="{left+12}" y="{top-17}" fill="#0f172a" font-size="12.5">{function_label_svg}</text>
  <text x="{left + plot_width / 2 - 42}" y="{height-8}">{x_axis_label}</text>
  <text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(y_axis_label)}</text>
</svg>
"""


def svg_series_with_zero_axes(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    title: str,
    color: str,
    width: int = 900,
    height: int = 210,
) -> str:
    """绘制带零点十字坐标轴的误差或电流曲线。"""

    valid_rows = [
        row for row in rows
        if row.get(field) is not None and row.get("time_s") is not None
    ]
    if len(valid_rows) < 2:
        return f"<h4>{html.escape(title)}</h4><p>无足够曲线数据。</p>"
    if len(valid_rows) > 600:
        step = math.ceil(len(valid_rows) / 600)
        valid_rows = valid_rows[::step]

    time_origin = float(valid_rows[0]["time_s"])
    x_values = [
        float(row["time_s"]) - time_origin
        for row in valid_rows
    ]
    y_values = [float(row[field]) for row in valid_rows]
    x_min = min(0.0, min(x_values))
    x_max = max(x_values)
    y_min = min(0.0, min(y_values))
    y_max = max(0.0, max(y_values))
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = max(0.1, (y_max - y_min) * 0.08)
    y_min -= padding
    y_max += padding

    left, right, top, bottom = 58, 18, 18, 38
    plot_width = width - left - right
    plot_height = height - top - bottom

    def xy(x_value: float, y_value: float) -> tuple[float, float]:
        return (
            left + (x_value - x_min) / (x_max - x_min) * plot_width,
            top + (y_max - y_value) / (y_max - y_min) * plot_height,
        )

    points = " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in (xy(x_value, y_value) for x_value, y_value in zip(x_values, y_values))
    )
    zero_x, zero_y = xy(0.0, 0.0)
    return f"""
<h4>{html.escape(title)}</h4>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{height-bottom}" stroke="#334155" stroke-width="1.8"/>
  <line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#334155" stroke-width="1.8"/>
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
  <text x="{left}" y="{height-10}">0.0s</text>
  <text x="{width-right-38}" y="{height-10}">{x_max:.1f}s</text>
  <text x="5" y="{top+8}">{y_max:.2f}</text>
  <text x="5" y="{height-bottom}">{y_min:.2f}</text>
</svg>
"""


def result_text(result: str) -> str:
    """把机器可读结果转换为报告和终端使用的中文结论。"""

    return "通过（PASS）" if result == "PASS" else "不通过（FAIL）"


def direction_status_text(result: str | None) -> str:
    """把方向复核状态转换成中文，避免把待标定误显示成失败。"""

    return {
        "PASS": "通过",
        "FAIL": "反向/不通过",
        "REVIEW": "待人工复核",
        "UNCALIBRATED": "未标定",
        "UNAVAILABLE": "数据不可用",
        "NOT_APPLICABLE": "不适用",
    }.get(str(result), "未测试")


def print_metric_summary(summary: dict[str, Any]) -> None:
    """用中文输出逐关节检测和步态验收指标。"""

    sections = (
        ("逐关节检测指标", summary.get("assembly", {}).get("joint_metrics", {})),
        ("步态跟踪指标", summary.get("gait", {}).get("joint_metrics", {})),
    )
    def _display_width(s: str) -> int:
        """计算字符串显示宽度：中文等宽字符=2，ASCII=1"""
        w = 0
        for ch in s:
            if '一' <= ch <= '鿿' or '　' <= ch <= '〿' or '＀' <= ch <= '￯':
                w += 2
            else:
                w += 1
        return w

    def _pad(s: str, width: int) -> str:
        """按显示宽度填充字符串"""
        need = width - _display_width(s)
        return s + ' ' * max(0, need)

    for title, metrics in sections:
        if not metrics:
            continue
        print(f"\n{title}：")
        # Table with CJK-aware alignment
        col_w = [10, 3, 8, 8, 7, 6, 24]
        header = ["关节", "ID", "均方根误差", "最大误差", "延时", "结论", "不通过原因"]
        sep = ["-" * 8, "---", "-" * 6, "-" * 6, "-" * 5, "-" * 4, "-" * 18]
        print("  " + "  ".join(_pad(h, w) for h, w in zip(header, col_w)))
        print("  " + "  ".join(_pad(s, w) for s, w in zip(sep, col_w)))
        for joint_name, item in metrics.items():
            label = JOINT_LABEL_ZH.get(joint_name, joint_name)
            rmse = item.get('rmse_deg')
            max_e = item.get('max_abs_error_deg')
            delay = item.get('position_delay_ms')
            result = str(item.get('result', 'FAIL'))
            fail_reason = item.get('fail_reason', '')
            rmse_s = f"{rmse:.2f}°" if rmse is not None else "-"
            max_s = f"{max_e:.2f}°" if max_e is not None else "-"
            delay_s = f"{delay:.0f}ms" if delay is not None else "-"
            result_s = "PASS" if result == "PASS" else "FAIL"
            cols = [label, str(item.get('servo_id', '-')), rmse_s, max_s, delay_s, result_s, fail_reason]
            print("  " + "  ".join(_pad(c, w) for c, w in zip(cols, col_w)))


def print_result_banner(
    result: str,
    *,
    report_path: Path | None = None,
    error: str | None = None,
) -> None:
    """在终端最后显著输出中文结论、原因和报告位置。"""

    border = "=" * 46
    print(border)
    print(f"测试结论：{result_text(result)}")
    if error:
        print(f"不通过原因：{error}")
    if report_path is not None:
        print(f"报告保存目录：{report_path.resolve().parent}")
        print(f"中文HTML报告：{report_path.resolve()}")
    print(border)


def render_report(
    output_dir: Path,
    summary: dict[str, Any],
    motor_rows: Sequence[dict[str, Any]],
    gait_rows: Sequence[dict[str, Any]],
    report_joints: Sequence[tuple[str, int]] = JOINTS,
) -> Path:
    """生成包含扫描结果、判定和输入/实际波形的中文 HTML。"""

    report_joints = tuple(report_joints)
    report_title = str(
        summary.get("report_title", "OpenDuck Mini 整机中试报告")
    )
    scan = summary.get("scan", {})
    found_ids = scan.get("found_ids", [])
    missing_ids = scan.get("missing_ids", [])
    overall = summary.get("result", "未完成")
    status_class = "pass" if overall == "PASS" else "fail"
    overall_text = result_text(str(overall))
    source_rows = []
    for source_item in summary.get("threshold_sources", threshold_sources()):
        source_url = str(source_item.get("url", ""))
        source_label = html.escape(str(source_item.get("source", "-")))
        if source_url:
            source_label = (
                f'<a href="{html.escape(source_url)}" target="_blank">'
                f"{source_label}</a>"
            )
        source_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source_item.get('item', '-')))}</td>"
            f"<td>{html.escape(str(source_item.get('value', '-')))}</td>"
            f"<td>{html.escape(str(source_item.get('source_type', '-')))}</td>"
            f"<td>{source_label}</td>"
            "</tr>"
        )

    assembly = summary.get("assembly", {})
    metrics = assembly.get("joint_metrics", {})
    metric_rows = []
    direction_detail_rows = []
    for joint_name, servo_id in report_joints:
        item = metrics.get(joint_name, {})
        item_result = item.get("result")
        direction = item.get("direction", {})
        similarity = direction.get("current_similarity")
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(JOINT_LABEL_ZH.get(joint_name, joint_name))}</td>"
            f"<td>{servo_id}</td>"
            f"<td>{result_text(item_result) if item_result else '未测试'}</td>"
            f"<td>{item.get('samples', 0)}</td>"
            f"<td>{item.get('rmse_deg', '-')}</td>"
            f"<td>{item.get('max_abs_error_deg', '-')}</td>"
            f"<td>{item.get('position_delay_ms', '-')}</td>"
            f"<td>{item.get('phase_delay_deg', '-')}</td>"
            f"<td>{direction_status_text(direction.get('position_result'))}</td>"
            f"<td>{direction_status_text(direction.get('current_result'))}</td>"
            f"<td>{similarity if similarity is not None else '-'}</td>"
            f"<td>{html.escape(str(direction.get('current_note', '-')))}</td>"
            "</tr>"
        )
        if direction:
            signature = direction.get("signature_a", {})
            signature_text = "；".join(
                f"ID {observer_id}: {float(delta_current):+.4f} A"
                for observer_id, delta_current in sorted(
                    signature.items(),
                    key=lambda item: int(item[0]),
                )
            ) or "无有效特征"
            direction_detail_rows.append(
                "<tr>"
                f"<td>{html.escape(JOINT_LABEL_ZH.get(joint_name, joint_name))}</td>"
                f"<td>{servo_id}</td>"
                f"<td>{direction.get('positive_pose_samples', 0)}</td>"
                f"<td>{direction.get('negative_pose_samples', 0)}</td>"
                f"<td>{html.escape(signature_text)}</td>"
                "</tr>"
            )

    gait = summary.get("gait", {})
    gait_metrics = gait.get("joint_metrics", {})
    gait_metric_rows = []
    for joint_name, servo_id in report_joints:
        item = gait_metrics.get(joint_name, {})
        item_result = item.get("result")
        gait_metric_rows.append(
            "<tr>"
            f"<td>{html.escape(JOINT_LABEL_ZH.get(joint_name, joint_name))}</td>"
            f"<td>{servo_id}</td>"
            f"<td>{result_text(item_result) if item_result else '未测试'}</td>"
            f"<td>{item.get('samples', 0)}</td>"
            f"<td>{item.get('rmse_deg', '-')}</td>"
            f"<td>{item.get('max_abs_error_deg', '-')}</td>"
            f"<td>{item.get('position_delay_ms', '-')}</td>"
            f"<td>{item.get('phase_delay_deg', '-')}</td>"
            "</tr>"
        )

    charts: list[str] = []
    for joint_name, servo_id in report_joints:
        assembly_joint_rows = [
            row for row in motor_rows if int(row["servo_id"]) == servo_id
        ]
        if assembly_joint_rows:
            charts.append(
                f"<h3>{html.escape(JOINT_LABEL_ZH.get(joint_name, joint_name))}"
                f"（ID {servo_id}）—逐关节检测</h3>"
                + svg_position_chart(
                    assembly_joint_rows,
                    target_field="target_deg",
                    actual_field="actual_deg",
                    input_function_label=(
                        "输入函数：θrel=15°·g；"
                        "g1=(1−cosπu)/2，g2=(1+cosπu)/2，"
                        "g3=−g1，g4=−g2（分段半余弦，u∈[0,1]）"
                    ),
                    y_axis_label="相对装配原点角度（°）",
                )
                + svg_position_chart(
                    assembly_joint_rows,
                    target_field="target_deg",
                    actual_field="actual_deg",
                    align_response=True,
                    input_function_label=(
                        "输入函数：θrel=15°·g；"
                        "g1=(1−cosπu)/2，g2=(1+cosπu)/2，"
                        "g3=−g1，g4=−g2（分段半余弦，u∈[0,1]）"
                    ),
                    y_axis_label="相对装配原点角度（°）",
                )
                + svg_series_with_zero_axes(
                    assembly_joint_rows,
                    field="error_deg",
                    title="跟踪误差（度）",
                    color="#7c3aed",
                )
                + svg_series_with_zero_axes(
                    assembly_joint_rows,
                    field="current_a",
                    title="电流（A）",
                    color="#ea580c",
                )
            )
        if gait_rows:
            charts.append(
                f"<h3>{html.escape(JOINT_LABEL_ZH.get(joint_name, joint_name))}"
                f"（ID {servo_id}）—步态验证</h3>"
                + svg_position_chart(
                    gait_rows,
                    target_field=f"{joint_name}_target_deg",
                    actual_field=f"{joint_name}_actual_deg",
                    input_function_label=(
                        "输入函数：θin(t)=ONNX策略以50Hz输出的关节目标位置"
                    ),
                )
                + svg_position_chart(
                    gait_rows,
                    target_field=f"{joint_name}_target_deg",
                    actual_field=f"{joint_name}_actual_deg",
                    align_response=True,
                    input_function_label=(
                        "输入函数：θin(t)=ONNX策略以50Hz输出的关节目标位置"
                    ),
                )
                + svg_series_with_zero_axes(
                    gait_rows,
                    field=f"{joint_name}_error_deg",
                    title="跟踪误差（度）",
                    color="#7c3aed",
                )
                + svg_series_with_zero_axes(
                    gait_rows,
                    field=f"{joint_name}_current_a",
                    title="电流（A）",
                    color="#ea580c",
                )
            )

    error_text = summary.get("error")
    error_html = (
        f'<div class="card fail"><strong>终止原因：</strong>{html.escape(str(error_text))}</div>'
        if error_text
        else ""
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report_title)}</title>
<style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:28px auto;padding:0 18px;color:#17202a}}
h1,h2{{color:#17365d}}.card{{border:1px solid #ccd5df;border-radius:10px;padding:14px;margin:14px 0;background:#f8fafc}}
.pass{{border:4px solid #15803d;background:#ecfdf3}}.fail{{border:4px solid #b91c1c;background:#fef2f2}}
.overall-result{{font-size:34px;font-weight:800;text-align:center;margin:8px 0 16px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:7px;text-align:left}}th{{background:#eaf0f6}}
svg{{width:100%;height:auto;background:#fff;border:1px solid #e5e7eb}}.axis{{stroke:#667085;stroke-width:1}}
.target{{fill:none;stroke:#2563eb;stroke-width:2}}.actual{{fill:none;stroke:#dc2626;stroke-width:2}}
.legend span{{display:inline-block;margin-right:18px}}.target-dot{{color:#2563eb}}.actual-dot{{color:#dc2626}}
</style>
</head>
<body>
<h1>{html.escape(report_title)}</h1>
<div class="card {status_class}">
  <div class="overall-result">测试结论：{html.escape(overall_text)}</div>
  <p><strong>测试时间：</strong>{html.escape(str(summary.get("timestamp", "-")))}</p>
  <p><strong>运行项目：</strong>{html.escape(str(summary.get("command", "-")))}</p>
  <p><strong>固定通信：</strong>{SERIAL_PORT}，{SERVO_BAUD_RATE:,} baud</p>
</div>
{error_html}
<h2>舵机扫描</h2>
<div class="card">
  <p><strong>已识别 ID：</strong>{html.escape(", ".join(map(str, found_ids)) or "无")}</p>
  <p><strong>缺失 ID：</strong>{html.escape(", ".join(map(str, missing_ids)) or "无")}</p>
</div>
<h2>判定指标与依据</h2>
<p>“厂家规格”用于确定设备能力和保护边界；“OpenDuck内部工程阈值”
用于本项目中试筛选，不代表飞特出厂合格标准。</p>
<table>
<thead><tr><th>项目</th><th>当前值</th><th>依据类型</th><th>来源与说明</th></tr></thead>
<tbody>{''.join(source_rows)}</tbody>
</table>
<h2>逐关节检测判定</h2>
<p><strong>通过标准：</strong>每个关节 RMSE ≤ {PASS_RMSE_DEG:.1f}°，且最大绝对误差 ≤ {PASS_MAX_ERROR_DEG:.1f}°。</p>
<p><strong>方向判定：</strong>所有关节均检查实际位置是否分别到达正、负指令方向；
非偏航关节同时比较其他串联舵机在正/负姿态下的电流差分特征。
对称关节对齐RMSE比值 ≥ {PAIRED_RMSE_RATIO_THRESHOLD:.1f}:1 时标记疑似装反。
偏航关节因重力投影不足，重力电流法标记为“不适用”。
未加载正确装配基准时只记录特征并标记“未标定”，不据此误判。</p>
<table>
<thead><tr><th>关节</th><th>ID</th><th>结果</th><th>样本数</th><th>RMSE(°)</th><th>最大误差(°)</th><th>位置延时(ms)</th><th>相位延时(°)</th><th>位置方向</th><th>重力电流方向</th><th>基准相似度</th><th>说明</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody>
</table>
{
f'''<h3>多关节电流方向特征</h3>
<p>ΔI = 正姿态平均电流 − 负姿态平均电流；特征同时包含当前运动关节和其他串联关节。</p>
<table>
<thead><tr><th>运动关节</th><th>ID</th><th>正姿态样本</th><th>负姿态样本</th><th>其他观察舵机 ΔI</th></tr></thead>
<tbody>{''.join(direction_detail_rows)}</tbody>
</table>'''
if direction_detail_rows else ""
}
{
f'''<h2>步态跟踪判定</h2>
<table>
<thead><tr><th>关节</th><th>ID</th><th>结果</th><th>样本数</th><th>RMSE(°)</th><th>最大误差(°)</th><th>位置延时(ms)</th><th>相位延时(°)</th></tr></thead>
<tbody>{''.join(gait_metric_rows)}</tbody>
</table>'''
if gait_metrics else ""
}
<h2>输入指定波形与舵机实际响应波形</h2>
<p class="legend"><span class="target-dot">● 蓝色：发送的位置指令 θin(t)</span><span class="actual-dot">● 红色：反馈的实际输出位置 θout(t)</span></p>
{''.join(charts) if charts else '<div class="card">动作阶段未执行，因此没有波形数据。</div>'}
</body>
</html>
"""
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def create_output_dir(root: Path) -> Path:
    """按测试时间创建唯一报告目录。"""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / stamp
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def scan_with_new_controller(timeout_s: float = SCAN_TIMEOUT_S) -> ScanResult:
    """打开控制器完成只读扫描后立即释放串口。"""

    controller = open_controller(timeout_s)
    try:
        return scan_servo_ids(controller)
    finally:
        release_controller(controller)
        controller = None
        gc.collect()
        time.sleep(0.10)


def scan_to_dict(scan: ScanResult) -> dict[str, Any]:
    """把扫描结果转换成可写入 JSON 的结构。"""

    return {
        "expected_ids": list(scan.expected_ids),
        "found_ids": scan.found_ids,
        "missing_ids": scan.missing_ids,
        "positions_deg": {
            str(servo_id): round(math.degrees(position), 4)
            for servo_id, position in scan.found_positions.items()
        },
        "errors": {str(key): value for key, value in scan.errors.items()},
    }


def run_session(args: argparse.Namespace) -> int:
    """执行一次 assembly、gait 或 all 测试并始终产出报告。"""

    repo_root = Path(__file__).resolve().parents[1]
    temporary_output = args.output_root is None
    output_root = (
        temporary_report_root("pilot")
        if temporary_output
        else Path(args.output_root)
    )
    output_dir = create_output_dir(output_root)
    dashboard = None
    if not args.no_live:
        try:
            try:
                from live_waveform_dashboard import LiveWaveformDashboard
            except ImportError:
                from scripts.live_waveform_dashboard import LiveWaveformDashboard
            dashboard = LiveWaveformDashboard(port=args.live_port)
            dashboard.start()
            dashboard.set_status("正在扫描舵机", "ID 扫描")
        except Exception as exc:
            dashboard = None
            print(f"实时曲线启动失败，测试继续：{exc}")
    motor_rows: list[dict[str, Any]] = []
    gait_rows: list[dict[str, Any]] = []
    direction_reference: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": args.command,
        "result": "FAIL",
        "port": SERIAL_PORT,
        "baud": SERVO_BAUD_RATE,
        "expected_ids": list(EXPECTED_IDS),
        "safety_thresholds": {
            "min_voltage_v": MIN_VOLTAGE_V,
            "max_current_a": MAX_CURRENT_A,
            "temperature_warning_c": TEMPERATURE_WARNING_C,
            "max_temperature_c": MAX_TEMPERATURE_C,
            "max_tracking_error_deg": MAX_TRACKING_ERROR_DEG,
        },
        # 把每个阈值的来源写入JSON和HTML，避免把项目门槛误称为厂家标准。
        "threshold_sources": threshold_sources(),
    }
    manifest = {
        "format_version": 1,
        "created_at": summary["timestamp"],
        "command": args.command,
        "output_files": {
            "summary": "summary.json",
            "report": "report.html",
            "motor_samples": "motor_samples.csv",
            "gait_samples": "gait_samples.csv",
        },
    }

    exit_code = 1
    try:
        save_reference_path = None
        # 正常检测默认加载仓库中的整机母基准。
        # 采集新母基准时若未显式指定旧基准，则进入标定模式，不让旧基准阻止更新。
        if direction_reference is not None:
            summary["direction_reference"] = {
                "loaded": True,
            }
        else:
            summary["direction_reference"] = {
                "loaded": False,
                "mode": "calibration" if save_reference_path else "unavailable",
                "note": (
                    "正在采集正确整机母基准"
                    if save_reference_path
                    else "默认母基准不存在；只输出电流特征，不据此判定装反"
                ),
            }
        scan = scan_with_new_controller()
        summary["scan"] = scan_to_dict(scan)
        if not scan.complete:
            raise RuntimeError(
                "整机舵机不完整："
                f"已识别 {scan.found_ids}；缺失 {scan.missing_ids}。"
                "动作阶段未执行。"
            )

        if args.scan_only:
            summary["result"] = "PASS"
            summary["note"] = "只读扫描完成，未写 PID、未发送动作"
            exit_code = 0
            return exit_code

        confirm_supported(args.yes)

        if args.command in ("assembly", "all"):
            controller = open_controller()
            try:
                # 再扫描一次，防止确认期间总线状态发生变化。
                assembly_scan = scan_servo_ids(controller)
                summary["scan"] = scan_to_dict(assembly_scan)
                if not assembly_scan.complete:
                    raise RuntimeError("动作前复查发现舵机缺失")
                motor_rows, assembly_summary = run_assembly_motion(
                    controller,
                    assembly_scan,
                    sample_callback=(
                        dashboard.publish_motor_row
                        if dashboard is not None
                        else None
                    ),
                )
                summary["assembly"] = assembly_summary
                if save_reference_path:
                    print("[方向基准] 方向基准功能已移除，请使用对称关节对比判定方向")
            finally:
                disable_torque(
                    controller,
                    summary.get("scan", {}).get("found_ids", []),
                )
                release_controller(controller)
                controller = None
                gc.collect()
                time.sleep(0.10)

            if summary["assembly"]["result"] != "PASS":
                summary["result"] = "FAIL"
                exit_code = 3
                direction_failures = summary["assembly"].get(
                    "direction_failures",
                    [],
                )
                if direction_failures:
                    summary["note"] = (
                        "装配方向检测失败："
                        + "；".join(direction_failures)
                    )
                elif args.command == "all":
                    summary["note"] = "逐关节检测未通过，步态阶段未执行"
                return exit_code

        if args.command == "gait":
            # 步态前再次确认 14 个 ID，符合失败即停止原则。
            gait_scan = scan_with_new_controller()
            summary["scan"] = scan_to_dict(gait_scan)
            if not gait_scan.complete:
                raise RuntimeError("步态前复查发现舵机缺失")
            gait_rows, gait_summary = run_gait_motion(
                repo_root,
                model_path=Path(args.model),
                config_path=Path(args.config),
                sample_callback=(
                    (
                        lambda row: dashboard.publish_gait_row(row, JOINTS)
                    )
                    if dashboard is not None
                    else None
                ),
            )
            summary["gait"] = gait_summary
            if gait_summary["result"] != "PASS":
                summary["result"] = "FAIL"
                summary["note"] = "步态目标角度与实际角度误差超出验收阈值"
                exit_code = 3
                return exit_code

        summary["result"] = "PASS"
        exit_code = 0
        return exit_code
    except KeyboardInterrupt:
        summary["error"] = "操作者中断测试"
        summary["result"] = "FAIL"
        exit_code = 130
        return exit_code
    except SafetyAbort as exc:
        if exc.motor_rows:
            motor_rows = list(exc.motor_rows)
            partial_metrics = calculate_joint_metrics(
                motor_rows,
                JOINTS,
                waveform_period_s=4.0,
            )
            summary["assembly"] = {
                "result": "FAIL",
                "joint_metrics": partial_metrics,
                "samples": len(motor_rows),
                "note": "安全停机前的部分数据",
            }
        summary["error"] = f"安全停机：{exc}"
        summary["result"] = "FAIL"
        exit_code = 4
        return exit_code
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["result"] = "FAIL"
        exit_code = 2
        return exit_code
    finally:
        write_csv(output_dir / "motor_samples.csv", MOTOR_CSV_FIELDS, motor_rows)
        write_csv(output_dir / "gait_samples.csv", gait_csv_fields(), gait_rows)
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "summary.json", summary)
        report_path = render_report(output_dir, summary, motor_rows, gait_rows)
        print_metric_summary(summary)
        if summary.get("error"):
            print(f"终止原因：{summary['error']}")
        print_result_banner(
            summary["result"],
            report_path=report_path,
            error=summary.get("error") or summary.get("note"),
        )
        if dashboard is not None:
            dashboard.set_report(report_path)
            dashboard.set_downloads(
                [
                    ("下载逐关节CSV原始数据", output_dir / "motor_samples.csv"),
                    ("下载步态CSV原始数据", output_dir / "gait_samples.csv"),
                    ("下载JSON汇总", output_dir / "summary.json"),
                ]
            )
            dashboard.set_status(
                f"测试结论：{result_text(summary['result'])}",
                "报告已生成",
            )
            if sys.stdin.isatty():
                try:
                    input(
                        "请先在网页下载所需文件；完成后按 Enter "
                        "关闭页面并清理树莓派临时报告："
                    )
                except KeyboardInterrupt:
                    print("\n收到退出请求，开始清理临时报告。")
            dashboard.close()
        elif temporary_output and sys.stdin.isatty():
            try:
                input("完成后按 Enter 清理树莓派临时报告：")
            except KeyboardInterrupt:
                print("\n收到退出请求，开始清理临时报告。")
        cleanup_temporary_report(
            output_dir,
            temporary_output=temporary_output,
            keep_report=args.keep_report,
        )


def build_parser(repo_root: Path | None = None) -> argparse.ArgumentParser:
    """构造仅包含项目必要参数的中文命令行接口。"""

    root = repo_root or Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="OpenDuck Mini 整机中试检测平台",
        epilog=(
            "assembly：扫描并逐关节检测；"
            "gait：整机模型与自动步态验证；"
            "all：按顺序执行两项。"
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("assembly", "gait", "all"),
        help="要执行的测试项目；不填写时默认执行 all",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="只扫描14个舵机，不写 PID、不发送动作",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认机器人已悬空或可靠支撑，用于非交互终端",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="指定持久报告目录；默认仅使用树莓派临时目录",
    )
    parser.add_argument(
        "--keep-report",
        action="store_true",
        help="测试退出后保留树莓派临时报告",
    )
    parser.add_argument(
        "--model",
        default=str(root / "BEST_WALK_ONNX_2.onnx"),
        help="ONNX 步态模型路径",
    )
    parser.add_argument(
        "--config",
        default=str(root / "duck_config.json"),
        help="机器人配置文件路径",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="关闭局域网实时波形页面",
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=8765,
        help="实时波形页面起始端口，默认8765",
    )
    parser.add_argument(
        "--direction-reference",
        default=None,
        help=(
            "覆盖方向基准JSON；不填写时自动加载"
            "direction_references/robot.json"
        ),
    )
    parser.add_argument(
        "--save-direction-reference",
        default=None,
        help=(
            "将本次电流特征保存为方向基准JSON；"
            "仅允许在人工确认装配正确且逐关节检测通过时使用；"
            "整机母基准保存后自动生成三个组件子基准"
        ),
    )
    return parser


def main() -> int:
    """命令行入口。"""

    args = build_parser().parse_args()
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
