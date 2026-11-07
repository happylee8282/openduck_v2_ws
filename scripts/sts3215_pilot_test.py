"""OpenDuck Mini安装前STS3215单舵机测试。

代码主流程：
1. 扫描项目ID，要求总线上恰好只有一个舵机；
2. 保存舵机原始ID/PID；
3. 写入并回读项目测试参数：腿部Kp=22、头颈Kp=8、Kd=0；
4. 发送平滑的0→90°→0半余弦位置指令；
5. 同一采样周期记录输入位置、实际位置、电流、电压和温度；
6. 计算误差、响应延时和位置反馈方向；
7. 输出CSV、JSON、中文HTML和实时网页。

限制：单舵机没有其他关节的重力电流响应，只能判断编码器反馈方向，
不能判断装入机构后的机械方向；物理装配方向由部件/整机脚本完成。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    import rustypot
except ImportError:  # 允许开发机在没有硬件 SDK 时导入并运行逻辑测试。
    rustypot = None

try:
    from scripts.pilot_test_platform import (
        SCAN_TIMEOUT_S,
        cleanup_temporary_report,
        compute_aligned_errors,
        current_to_amp,
        estimate_position_delay,
        MAX_TEMPERATURE_C,
        SERIAL_TIMEOUT_S,
        TEMPERATURE_WARNING_C,
        select_first_response_feature,
        svg_position_chart,
        svg_series_with_zero_axes,
        temporary_report_root,
        threshold_sources,
    )
except ImportError:
    from pilot_test_platform import (
        SCAN_TIMEOUT_S,
        cleanup_temporary_report,
        compute_aligned_errors,
        current_to_amp,
        estimate_position_delay,
        MAX_TEMPERATURE_C,
        SERIAL_TIMEOUT_S,
        TEMPERATURE_WARNING_C,
        select_first_response_feature,
        svg_position_chart,
        svg_series_with_zero_axes,
        temporary_report_root,
        threshold_sources,
    )


SERVO_BAUD_RATE = 1_000_000

PROJECT_MOTOR_IDS = [
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    30, 31, 32, 33,
]


def as_scalar(value):
    """Normalize rustypot scalar/list return variants to one float."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("rustypot returned an empty list")
        value = value[0]
    return float(value)


def scan_motor(io):
    """扫描全部项目 ID，并确认当前总线上恰好只有一个舵机。"""
    found = []
    for motor_id in PROJECT_MOTOR_IDS:
        try:
            position = as_scalar(io.read_present_position(motor_id))
            found.append((motor_id, position))
        except Exception:
            continue
    if not found:
        raise RuntimeError("未扫描到 STS3215 项目舵机")
    if len(found) != 1:
        raise RuntimeError(
            "单舵机检测要求总线上恰好连接一个舵机；"
            f"当前识别到 ID {[motor_id for motor_id, _ in found]}"
        )
    return found[0]


def read_optional(io, method_name, motor_id):
    """Return optional telemetry as a float without aborting the test."""
    method = getattr(io, method_name, None)
    if method is None:
        return None
    try:
        return as_scalar(method(motor_id))
    except Exception:
        return None


def half_cosine(start, end, phase):
    """Smooth interpolation with zero velocity at both endpoints."""
    phase = min(1.0, max(0.0, phase))
    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
    return start + (end - start) * blend


def waveform(t, amplitude_rad, half_cycle_s):
    """Generate repeating 0 -> amplitude -> 0 half-cosine cycles."""
    cycle_s = 2.0 * half_cycle_s
    local_t = t % cycle_s
    if local_t <= half_cycle_s:
        return half_cosine(0.0, amplitude_rad, local_t / half_cycle_s)
    return half_cosine(amplitude_rad, 0.0, (local_t - half_cycle_s) / half_cycle_s)


def move_smoothly(io, motor_id, start, end, duration_s, sample_period_s):
    """Move to the waveform start without issuing a position step."""
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        phase = elapsed / duration_s
        io.write_goal_position(motor_id, half_cosine(start, end, phase))
        if phase >= 1.0:
            break
        time.sleep(sample_period_s)


def svg_chart(rows, field, title, color, width=1000, height=260):
    """Create a dependency-free SVG line chart for the HTML report."""
    values = [(r["time_s"], r[field]) for r in rows if r[field] is not None]
    if len(values) < 2:
        return f"<h3>{html.escape(title)}</h3><p>暂无数据。</p>"

    times = [v[0] for v in values]
    ys = [v[1] for v in values]
    x_min, x_max = min(times), max(times)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    left, right, top, bottom = 65, 20, 25, 40
    plot_w = width - left - right
    plot_h = height - top - bottom

    def point(x, y):
        px = left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
        py = top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h
        return f"{px:.1f},{py:.1f}"

    points = " ".join(point(x, y) for x, y in values)
    return f"""
    <h3>{html.escape(title)}</h3>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#ccd3dc"/>
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#667085"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#667085"/>
      <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
      <text x="{left}" y="{height - 10}" font-size="13">{x_min:.1f}s</text>
      <text x="{left + plot_w - 35}" y="{height - 10}" font-size="13">{x_max:.1f}s</text>
      <text x="5" y="{top + 12}" font-size="13">{y_max:.2f}</text>
      <text x="5" y="{top + plot_h}" font-size="13">{y_min:.2f}</text>
    </svg>"""


def overlay_chart(rows, width=1000, height=360):
    """Create an SVG overlay of commanded and measured position."""
    if len(rows) < 2:
        return "<p>No waveform data available.</p>"
    x_min, x_max = rows[0]["time_s"], rows[-1]["time_s"]
    ys = [r["target_deg"] for r in rows] + [r["actual_deg"] for r in rows]
    y_min, y_max = min(ys), max(ys)
    y_min -= 5.0
    y_max += 5.0
    left, right, top, bottom = 85, 20, 75, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    def point(x, y):
        px = left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
        py = top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h
        return f"{px:.1f},{py:.1f}"

    target = " ".join(point(r["time_s"], r["target_deg"]) for r in rows)
    actual = " ".join(point(r["time_s"], r["actual_deg"]) for r in rows)
    x_ticks = []
    for tick in range(7):
        fraction = tick / 6
        x = left + fraction * plot_w
        value = x_min + fraction * (x_max - x_min)
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{top + plot_h}" stroke="#e2e8f0"/>'
            f'<text x="{x:.1f}" y="{height - 28}" font-size="12" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    y_ticks = []
    for tick in range(5):
        fraction = tick / 4
        y = top + fraction * plot_h
        value = y_max - fraction * (y_max - y_min)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="12" '
            f'text-anchor="end">{value:.1f}</text>'
        )
    delay_metrics = estimate_position_delay(
        rows,
        target_field="target_deg",
        actual_field="actual_deg",
        period_s=None,
    )
    marker_svg = ""
    delay_ms = delay_metrics.get("position_delay_ms")
    if delay_ms is not None and len(rows) >= 3:
        input_index = select_first_response_feature(
            rows,
            target_field="target_deg",
        )
        input_row = rows[input_index]
        input_time = float(input_row["time_s"])
        output_time = input_time + float(delay_ms) / 1000.0
        output_row = min(
            rows,
            key=lambda row: abs(float(row["time_s"]) - output_time),
        )
        output_time = float(output_row["time_s"])
        input_position = float(input_row["target_deg"])
        output_position = float(output_row["actual_deg"])
        input_x, input_y = map(
            float,
            point(input_time, input_position).split(","),
        )
        output_x, output_y = map(
            float,
            point(output_time, output_position).split(","),
        )
        arrow_y = 32
        marker_svg = f"""
      <defs>
        <marker id="response-arrow" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto-start-reverse">
          <path d="M0,0 L7,3.5 L0,7 z" fill="#0f172a"/>
        </marker>
      </defs>
      <line x1="{input_x:.1f}" y1="{arrow_y}" x2="{output_x:.1f}" y2="{arrow_y}"
            stroke="#0f172a" stroke-width="1.5" marker-start="url(#response-arrow)" marker-end="url(#response-arrow)"/>
      <text x="{(input_x + output_x) / 2:.1f}" y="20" font-size="13" text-anchor="middle">响应延时 Δt={float(delay_ms):.1f} ms</text>
      <line x1="{input_x:.1f}" y1="{input_y:.1f}" x2="{input_x:.1f}" y2="{top + plot_h}"
            stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 4"/>
      <line x1="{output_x:.1f}" y1="{output_y:.1f}" x2="{output_x:.1f}" y2="{top + plot_h}"
            stroke="#dc2626" stroke-width="1.4" stroke-dasharray="5 4"/>
      <circle cx="{input_x:.1f}" cy="{input_y:.1f}" r="4.5" fill="#2563eb"/>
      <circle cx="{output_x:.1f}" cy="{output_y:.1f}" r="4.5" fill="#dc2626"/>
      <text x="{input_x:.1f}" y="{top + plot_h + 18}" fill="#2563eb" font-size="12" text-anchor="middle">↑ 指令点：t输入={input_time:.2f}s，θin={input_position:.2f}°</text>
      <text x="{output_x:.1f}" y="{top + plot_h + 37}" fill="#dc2626" font-size="12" text-anchor="middle">↑ 响应点：t输出={output_time:.2f}s，θout={output_position:.2f}°</text>
"""
    return f"""
    <h3>输入指定波形与舵机实际响应波形对比</h3>
    <p><span class="target">● 蓝线：发送的位置指令 θin(t)</span> &nbsp; <span class="actual">● 红线：反馈的实际输出位置 θout(t)</span></p>
    <p>同一横坐标 t：蓝线为该采样时刻发送给舵机的位置指令，红线为随后读取的位置反馈。t=0处波形尚未产生有效位移，因此箭头选取第一周期刚离开起点的首次有效输入及对应输出响应点，横轴上的 t输入、t输出之差即响应延时。</p>
    <p>实际测试时长：{x_max - x_min:.3f} s</p>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="输入指定波形与舵机实际响应波形">
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#ccd3dc"/>
      {''.join(x_ticks)}
      {''.join(y_ticks)}
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#667085"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#667085"/>
      <polyline points="{target}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
      <polyline points="{actual}" fill="none" stroke="#dc2626" stroke-width="2"/>
      {marker_svg}
      <text x="{left + plot_w / 2 - 45}" y="{height - 8}" font-size="13">实际测试时间（s）</text>
      <text x="25" y="{top + plot_h / 2}" font-size="13" text-anchor="middle" transform="rotate(-90 25 {top + plot_h / 2})">位置角度（°）</text>
    </svg>"""


def write_outputs(output_dir, metadata, rows, summary):
    """Write machine-readable logs and a browser-viewable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"
    json_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "summary": summary}, handle, ensure_ascii=False, indent=2)

    result_color = "#15803d" if summary["result"] == "PASS" else "#b91c1c"
    result_background = "#ecfdf3" if summary["result"] == "PASS" else "#fef2f2"
    result_zh = "通过（PASS）" if summary["result"] == "PASS" else "不通过（FAIL）"
    source_rows = []
    for source_item in threshold_sources():
        url = str(source_item.get("url", ""))
        source_text = html.escape(str(source_item.get("source", "-")))
        if url:
            source_text = (
                f'<a href="{html.escape(url)}" target="_blank">'
                f"{source_text}</a>"
            )
        source_rows.append(
            "<tr>"
            f"<td>{html.escape(str(source_item.get('item', '-')))}</td>"
            f"<td>{html.escape(str(source_item.get('value', '-')))}</td>"
            f"<td>{html.escape(str(source_item.get('source_type', '-')))}</td>"
            f"<td>{source_text}</td>"
            "</tr>"
        )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>STS3215 单舵机测试报告</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:28px auto;padding:0 18px;color:#17202a}}
.card{{border:1px solid #d0d5dd;border-radius:10px;padding:16px;margin:14px 0;background:#f8fafc}}
.result-card{{border:4px solid {result_color};background:{result_background};text-align:center}}
.result{{font-size:34px;font-weight:800;color:{result_color}}}
.target{{color:#2563eb;font-weight:700}} .actual{{color:#dc2626;font-weight:700}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d5dd;padding:8px;text-align:left}}
svg{{width:100%;height:auto;background:#fff}}
</style></head><body>
<h1>OpenDuck Mini - STS3215 单舵机测试报告</h1>
<div class="card result-card"><div class="result">测试结论：{result_zh}</div>
<p>已识别 ID：{metadata["motor_id"]}</p>
<p>缺失 ID：无（单舵机检测只要求1个在线 ID）</p>
<p>Kp：{metadata["kp_after"]} | Kd：{metadata["kd_after"]}</p></div>
<p><strong>通过标准：</strong>通信异常次数为0，均方根误差 ≤ {metadata["max_rmse_deg"]:.1f}°，最大绝对误差 ≤ {metadata["max_error_deg"]:.1f}°。</p>
<table>
<tr><th>采样点数</th><td>{summary["samples"]}</td><th>通信异常次数</th><td>{summary["communication_errors"]}</td></tr>
<tr><th>实际测试时长</th><td>{summary["actual_duration_s"]:.3f} s</td><th>坐标单位</th><td>横轴：时间（s）；纵轴：位置角度（°）</td></tr>
<tr><th>均方根误差（RMSE）</th><td>{summary["rmse_deg"]:.3f}°</td><th>最大绝对误差</th><td>{summary["max_abs_error_deg"]:.3f}°</td></tr>
<tr><th>平均绝对误差</th><td>{summary["mean_abs_error_deg"]:.3f}°</td><th>峰值电流反馈</th><td>{summary["peak_current"]}</td></tr>
<tr><th>位置响应延时</th><td>{summary["position_delay_ms"]} ms</td><th>等效相位延时</th><td>{summary["phase_delay_deg"]}°</td></tr>
<tr><th>延时估计相关系数</th><td>{summary["delay_correlation"]}</td><th>延时含义</th><td>正值表示实际输出滞后于输入目标</td></tr>
<tr><th>位置反馈方向</th><td>{'通过' if summary["position_direction_passed"] else '不通过'}</td><th>方向达到量</th><td>{summary["signed_actual_peak_deg"]:.3f}°；要求 ≥ {summary["direction_required_deg"]:.3f}°</td></tr>
<tr><th>装配物理方向</th><td>不判定</td><th>原因</th><td>单舵机没有其他关节的重力电流响应，需在部件或整机脚本中复核</td></tr>
</table>
<h2>判定指标与依据</h2>
<p>厂家规格用于设备能力和保护边界；RMSE、最大误差和方向门槛属于OpenDuck内部工程阈值，不是飞特出厂标准。</p>
<table>
<thead><tr><th>项目</th><th>当前值</th><th>依据类型</th><th>来源与说明</th></tr></thead>
<tbody>{''.join(source_rows)}</tbody>
</table>
{svg_position_chart(
    rows,
    target_field="target_deg",
    actual_field="actual_deg",
    input_function_label=(
        "输入函数：θin(t)=s·90°·f(t)，"
        "f(u)=[1−cos(πu)]/2（分段半余弦，s为运动方向）"
    ),
)}
{svg_position_chart(
    rows,
    target_field="target_deg",
    actual_field="actual_deg",
    align_response=True,
    input_function_label=(
        "输入函数：θin(t)=s·90°·f(t)，"
        "f(u)=[1−cos(πu)]/2（分段半余弦，s为运动方向）"
    ),
)}
{svg_series_with_zero_axes(rows, field="error_deg", title="跟踪误差（度）", color="#7c3aed", width=1000)}
{svg_series_with_zero_axes(rows, field="current", title="电流反馈（控制器单位）", color="#ea580c", width=1000)}
</body></html>"""
    html_path.write_text(report, encoding="utf-8")
    return csv_path, json_path, html_path


def main():
    parser = argparse.ArgumentParser(description="STS3215 单舵机中试波形检测")
    parser.add_argument("--port", default="/dev/ttyACM0", help="驱动板串口")
    parser.add_argument(
        "--kp",
        type=int,
        default=None,
        help="覆盖项目测试 Kp；默认腿部22、头颈8",
    )
    parser.add_argument("--kd", type=int, default=0, help="项目测试 Kd")
    parser.add_argument("--amplitude-deg", type=float, default=90.0, help="相对运动幅度（度）")
    parser.add_argument("--cycles", type=int, default=3, help="测试周期数")
    parser.add_argument("--half-cycle", type=float, default=2.0, help="单程波形时间")
    parser.add_argument("--sample-period", type=float, default=0.02, help="采样周期")
    parser.add_argument("--max-rmse-deg", type=float, default=5.0, help="RMSE 复核阈值")
    parser.add_argument("--max-error-deg", type=float, default=15.0, help="最大误差复核阈值")
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
    parser.add_argument("--scan-only", action="store_true", help="只读扫描，不写参数、不运动")
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="关闭局域网实时波形页面",
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=8765,
        help="实时波形页面起始端口，默认 8765",
    )
    args = parser.parse_args()
    temporary_output = args.output_root is None

    if rustypot is None:
        raise RuntimeError("未安装 rustypot，请在树莓派项目虚拟环境中运行")
    # 与部件/整机脚本一致，单次通信事务超时固定为0.5s。
    io = rustypot.Sts3215PyController(
        args.port,
        SERVO_BAUD_RATE,
        SCAN_TIMEOUT_S,
    )
    motor_id, initial_position = scan_motor(io)
    original_kp = read_optional(io, "read_p_coefficient", motor_id)
    original_ki = read_optional(io, "read_i_coefficient", motor_id)
    original_kd = read_optional(io, "read_d_coefficient", motor_id)

    test_kp = args.kp if args.kp is not None else 22
    test_kd = args.kd
    print(
        f"已识别 STS3215：ID={motor_id}，"
        f"当前位置={math.degrees(initial_position):.2f}°"
    )
    print(
        f"测试前舵机存储 PID：Kp={original_kp}，"
        f"Ki={original_ki}，Kd={original_kd}"
    )
    print(
        f"本次项目测试 PID：Kp={test_kp}，Kd={test_kd}"
        f"（{'头颈参数' if 30 <= motor_id <= 33 else '腿部参数'}）"
    )
    if args.scan_only:
        print("只读扫描：未写入 PID，未发送运动指令")
        return

    dashboard = None
    if not args.no_live:
        try:
            try:
                from live_waveform_dashboard import (
                    LiveWaveformDashboard,
                    wait_for_operator,
                )
            except ImportError:
                from scripts.live_waveform_dashboard import (
                    LiveWaveformDashboard,
                    wait_for_operator,
                )
            dashboard = LiveWaveformDashboard(port=args.live_port)
            dashboard.start()
            dashboard.set_status(f"已识别 ID {motor_id}", "等待开始")
            wait_for_operator("请在电脑浏览器打开实时曲线地址，按 Enter 开始测试：")
        except Exception as exc:
            dashboard = None
            print(f"实时曲线启动失败，测试继续：{exc}")

    if not (0.0 < args.amplitude_deg <= 90.0):
        raise ValueError("--amplitude-deg must be in the range (0, 90]")
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")

    # Unlock EEPROM/RAM settings if required, select servo mode, and apply project test gains.
    if hasattr(io, "write_lock"):
        io.write_lock(motor_id, False)
    if hasattr(io, "write_mode"):
        io.write_mode(motor_id, 0)
    io.write_p_coefficient(motor_id, test_kp)
    io.write_d_coefficient(motor_id, test_kd)
    kp_after = int(as_scalar(io.read_p_coefficient(motor_id)))
    kd_after = int(as_scalar(io.read_d_coefficient(motor_id)))
    if (kp_after, kd_after) != (test_kp, test_kd):
        raise RuntimeError(
            f"PID readback mismatch: expected {test_kp}/{test_kd}, "
            f"got {kp_after}/{kd_after}"
        )
    print(f"PID 写入并回读确认：Kp={kp_after}，Kd={kd_after}")

    rows = []
    communication_errors = 0
    temperature_warning_printed = False
    amplitude_rad = math.radians(args.amplitude_deg)
    total_duration = args.cycles * 2.0 * args.half_cycle

    # Treat the detected shaft position as the incoming-test zero. This avoids
    # an unsafe jump to the servo's absolute zero before the waveform starts.
    origin_position = initial_position
    direction = 1.0 if origin_position + amplitude_rad <= math.pi else -1.0

    try:
        started = time.monotonic()
        next_sample = started
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed > total_duration:
                break
            target_offset = direction * waveform(elapsed, amplitude_rad, args.half_cycle)
            target = origin_position + target_offset
            try:
                io.write_goal_position(motor_id, target)
                actual = as_scalar(io.read_present_position(motor_id))
                speed = read_optional(io, "read_present_speed", motor_id)
                current = read_optional(io, "read_present_current", motor_id)
                voltage = read_optional(io, "read_present_voltage", motor_id)
                temperature = read_optional(io, "read_present_temperature", motor_id)
            except Exception as exc:
                communication_errors += 1
                print(f"通信异常（超时、无响应或读写失败）{elapsed:.3f}s：{exc}")
                next_sample += args.sample_period
                time.sleep(max(0.0, next_sample - time.monotonic()))
                continue

            if temperature is not None and temperature >= MAX_TEMPERATURE_C:
                raise RuntimeError(
                    f"舵机 ID {motor_id} 温度 {temperature:.1f}℃，"
                    f"达到 {MAX_TEMPERATURE_C:.1f}℃硬停机阈值"
                )
            if (
                temperature is not None
                and temperature >= TEMPERATURE_WARNING_C
                and not temperature_warning_printed
            ):
                temperature_warning_printed = True
                print(
                    f"[温度告警] ID {motor_id} 当前 {temperature:.1f}℃，"
                    f"已达到 {TEMPERATURE_WARNING_C:.1f}℃；"
                    f"{MAX_TEMPERATURE_C:.1f}℃时关闭扭矩停机"
                )

            # Report relative displacement so the plot directly shows 0 -> 90 -> 0.
            target_deg = math.degrees(target_offset)
            actual_deg = math.degrees(actual - origin_position)
            rows.append(
                {
                    "time_s": round(elapsed, 6),
                    "target_deg": round(target_deg, 6),
                    "actual_deg": round(actual_deg, 6),
                    "error_deg": round(actual_deg - target_deg, 6),
                    "target_absolute_deg": round(math.degrees(target), 6),
                    "actual_absolute_deg": round(math.degrees(actual), 6),
                    "speed_rad_s": speed,
                    "current": current,
                    "voltage": voltage,
                    "temperature_c": temperature,
                }
            )
            if dashboard is not None:
                dashboard.publish(
                    f"ID {motor_id}",
                    time_s=elapsed,
                    target_deg=target_deg,
                    actual_deg=actual_deg,
                    error_deg=actual_deg - target_deg,
                    current_a=current_to_amp(current),
                    stage="单舵机波形检测",
                )
            next_sample += args.sample_period
            time.sleep(max(0.0, next_sample - time.monotonic()))

        # Return to the physical position where the incoming test started.
        io.write_goal_position(motor_id, origin_position)
    except BaseException:
        # Stop applying torque on interruption or an unexpected failure.
        if hasattr(io, "disable_torque"):
            try:
                io.disable_torque(motor_id)
            except Exception:
                pass
        if dashboard is not None:
            dashboard.set_status("测试异常终止", "已关闭舵机扭矩")
            dashboard.close()
        raise

    if not rows:
        raise RuntimeError("No valid telemetry samples were collected")

    # 先算延时
    delay_metrics = estimate_position_delay(
        rows,
        target_field="target_deg",
        actual_field="actual_deg",
        period_s=2.0 * args.half_cycle,
    )
    delay_ms = delay_metrics.get("position_delay_ms")
    for row in rows:
        row.update(delay_metrics)

    # 保存原始误差（对齐前）
    raw_errors = [r["error_deg"] for r in rows]
    raw_rmse = math.sqrt(sum(e*e for e in raw_errors)/len(raw_errors))
    raw_max = max(abs(e) for e in raw_errors)

    # 仅当延时显著（>5ms）时才做对齐
    if delay_ms is not None and delay_ms > 5 and len(rows) >= 8:
        samples = [(float(r["time_s"]), float(r["target_deg"]), float(r["actual_deg"]))
                   for r in rows if r.get("time_s") and r.get("target_deg") and r.get("actual_deg")]
        if len(samples) >= 8:
            ts = [samples[i][0]-samples[i-1][0] for i in range(1,len(samples)) if samples[i][0]>samples[i-1][0]]
            if ts:
                dt = statistics.median(ts)
                shift = max(1, int(round(delay_ms / 1000.0 / dt)))
                if shift < len(samples):
                    time_to_idx = {}
                    for i, r in enumerate(rows):
                        if r.get("time_s") is not None:
                            time_to_idx[round(float(r["time_s"]), 6)] = i
                    for si in range(shift, len(samples)):
                            s_t = samples[si][0]
                            aligned_target = samples[si - shift][1]
                            s_actual = samples[si][2]
                            key = round(s_t, 6)
                            if key in time_to_idx:
                                rows[time_to_idx[key]]["error_deg"] = round(s_actual - aligned_target, 6)

    errors = [r["error_deg"] for r in rows]
    currents = [r["current"] for r in rows if r["current"] is not None]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_abs = max(abs(e) for e in errors)
    mean_abs = statistics.fmean(abs(e) for e in errors)
    communication_passed = communication_errors == 0
    rmse_passed = rmse <= args.max_rmse_deg
    max_error_passed = max_abs <= args.max_error_deg
    signed_actual_peak = max(
        direction * float(row["actual_deg"])
        for row in rows
    )
    direction_required = args.amplitude_deg * 0.60
    position_direction_passed = signed_actual_peak >= direction_required
    result = "PASS" if (
        communication_passed
        and rmse_passed
        and max_error_passed
        and position_direction_passed
    ) else "FAIL"
    failure_reasons = []
    if not communication_passed:
        failure_reasons.append(f"通信异常 {communication_errors} 次")
    if not rmse_passed:
        failure_reasons.append(
            f"均方根误差 {rmse:.3f}° 超过 {args.max_rmse_deg:.1f}°"
        )
    if not max_error_passed:
        failure_reasons.append(
            f"最大绝对误差 {max_abs:.3f}° 超过 {args.max_error_deg:.1f}°"
        )
    if not position_direction_passed:
        failure_reasons.append(
            f"位置反馈方向达到量 {signed_actual_peak:.3f}°，"
            f"小于要求 {direction_required:.3f}°"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        temporary_report_root("motor")
        if temporary_output
        else Path(args.output_root)
    )
    output_dir = output_root / f"motor_{motor_id}_{stamp}"
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "port": args.port,
        "baud": SERVO_BAUD_RATE,
        "motor_id": motor_id,
        "original_kp": original_kp,
        "original_ki": original_ki,
        "original_kd": original_kd,
        "kp_after": kp_after,
        "kd_after": kd_after,
        "amplitude_deg": args.amplitude_deg,
        "cycles": args.cycles,
        "half_cycle_s": args.half_cycle,
        "sample_period_s": args.sample_period,
        "max_rmse_deg": args.max_rmse_deg,
        "max_error_deg": args.max_error_deg,
        "origin_position_deg": math.degrees(origin_position),
        "waveform_direction": int(direction),
    }
    summary = {
        "result": result,
        "samples": len(rows),
        "communication_errors": communication_errors,
        "raw_rmse_deg": round(raw_rmse, 4),
        "raw_max_abs_error_deg": round(raw_max, 4),
        "rmse_deg": rmse,
        "max_abs_error_deg": max_abs,
        "mean_abs_error_deg": mean_abs,
        "aligned_rmse_deg": round(rmse, 4),
        "aligned_max_abs_error_deg": round(max_abs, 4),
        "peak_current": max(currents) if currents else None,
        "actual_duration_s": (
            max(0.0, float(rows[-1]["time_s"]) - float(rows[0]["time_s"]))
            if len(rows) >= 2
            else 0.0
        ),
        "position_direction_passed": position_direction_passed,
        "signed_actual_peak_deg": signed_actual_peak,
        "direction_required_deg": direction_required,
        "failure_reasons": failure_reasons,
        **delay_metrics,
    }
    csv_path, json_path, html_path = write_outputs(output_dir, metadata, rows, summary)
    def display_width(text: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in text
        )

    def pad_display(text: str, width: int) -> str:
        return text + " " * max(0, width - display_width(text))

    def metric_line(label: str, value: str, requirement: str, verdict: str) -> str:
        return (
            f"  {pad_display(label, 16)}│ "
            f"{pad_display(value, 20)}│ "
            f"{pad_display(requirement, 28)}│ {verdict}"
        )

    print("\n判定指标：")
    print(metric_line("指标名称", "测量结果", "判定要求", "判定"))
    print("  " + "─" * 76)
    print(
        metric_line(
            "通信状态",
            f"通信异常 {communication_errors} 次",
            "异常次数 = 0",
            "通过" if communication_passed else "不通过",
        )
    )
    print(
        metric_line(
            "均方根误差",
            f"{rmse:.3f}°",
            f"≤ {args.max_rmse_deg:.1f}°",
            "通过" if rmse_passed else "不通过",
        )
    )
    print(
        metric_line(
            "最大绝对误差",
            f"{max_abs:.3f}°",
            f"≤ {args.max_error_deg:.1f}°",
            "通过" if max_error_passed else "不通过",
        )
    )
    print(metric_line("平均绝对误差", f"{mean_abs:.3f}°", "记录项", "—"))
    print(
        metric_line(
            "位置响应延时",
            f"{delay_metrics['position_delay_ms']} ms",
            "记录项；正值为输出滞后",
            "—",
        )
    )
    print(
        metric_line(
            "等效相位延时",
            f"{delay_metrics['phase_delay_deg']}°",
            "记录项",
            "—",
        )
    )
    print(
        metric_line(
            "位置反馈方向",
            f"达到 {signed_actual_peak:.3f}°",
            f"同指令方向 ≥ {direction_required:.3f}°",
            "通过" if position_direction_passed else "不通过",
        )
    )
    print(
        metric_line(
            "装配物理方向",
            "不判定",
            "需部件/整机多关节电流特征",
            "—",
        )
    )
    print(
        metric_line(
            "实际测试时长",
            f"{summary['actual_duration_s']:.3f} s",
            "记录项",
            "—",
        )
    )
    print(f"CSV原始数据：{csv_path.resolve()}")
    print(f"JSON汇总数据：{json_path.resolve()}")
    print_result_banner(
        result,
        report_path=html_path,
        failure_reasons=failure_reasons,
    )
    if dashboard is not None:
        dashboard.set_report(html_path)
        dashboard.set_downloads(
            [
                ("下载单舵机CSV原始数据", csv_path),
                ("下载JSON汇总", json_path),
            ]
        )
        dashboard.set_status(
            "测试结论：通过（PASS）"
            if result == "PASS"
            else "测试结论：不通过（FAIL）",
            "报告已生成",
        )
        if __import__("sys").stdin.isatty():
            try:
                input(
                    "请先在网页下载所需文件；完成后按 Enter "
                    "关闭页面并清理树莓派临时报告："
                )
            except KeyboardInterrupt:
                print("\n收到退出请求，开始清理临时报告。")
        dashboard.close()
    elif temporary_output and __import__("sys").stdin.isatty():
        try:
            input("完成后按 Enter 清理树莓派临时报告：")
        except KeyboardInterrupt:
            print("\n收到退出请求，开始清理临时报告。")
    cleanup_temporary_report(
        output_dir,
        temporary_output=temporary_output,
        keep_report=args.keep_report,
    )


def print_result_banner(result, *, report_path=None, failure_reasons=None):
    """在终端最后显著输出单舵机结论、原因和报告位置。"""

    border = "=" * 46
    print(border)
    print(
        "测试结论：通过（PASS）"
        if result == "PASS"
        else "测试结论：不通过（FAIL）"
    )
    for reason in failure_reasons or []:
        print(f"不通过原因：{reason}")
    if report_path is not None:
        report_path = Path(report_path).resolve()
        print(f"报告保存目录：{report_path.parent}")
        print(f"中文HTML报告：{report_path}")
    print(border)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_result_banner("FAIL", failure_reasons=["操作者中断测试"])
        raise SystemExit(130)
    except Exception as exc:
        print_result_banner(
            "FAIL",
            failure_reasons=[f"{type(exc).__name__}: {exc}"],
        )
        raise SystemExit(2)
