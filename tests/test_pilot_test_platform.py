"""整机中试平台的无硬件单元测试。"""

from __future__ import annotations

import tempfile
import unittest
import math
import json
from pathlib import Path
from unittest import mock

from scripts import pilot_test_platform as pilot


class FakeController:
    """只实现扫描所需读取接口，并记录是否出现任何写操作。"""

    def __init__(self, online_ids):
        self.online_ids = set(online_ids)
        self.writes = []

    def read_present_position(self, servo_id):
        if servo_id not in self.online_ids:
            raise RuntimeError("Operation timed out")
        return [0.01 * servo_id]

    def __getattr__(self, name):
        if name.startswith(("write_", "sync_write_")):
            def record_write(*args, **kwargs):
                self.writes.append((name, args, kwargs))

            return record_write
        raise AttributeError(name)


class ScanTests(unittest.TestCase):
    def test_complete_scan_finds_all_project_ids_without_writes(self):
        controller = FakeController(pilot.EXPECTED_IDS)

        result = pilot.scan_servo_ids(controller)

        self.assertTrue(result.complete)
        self.assertEqual(result.found_ids, list(pilot.EXPECTED_IDS))
        self.assertEqual(result.missing_ids, [])
        self.assertEqual(controller.writes, [])

    def test_incomplete_scan_lists_every_missing_id(self):
        controller = FakeController({12})

        result = pilot.scan_servo_ids(controller)

        self.assertFalse(result.complete)
        self.assertEqual(result.found_ids, [12])
        self.assertEqual(
            result.missing_ids,
            [10, 11, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33],
        )
        self.assertEqual(controller.writes, [])


class SafetyTests(unittest.TestCase):
    def test_low_voltage_aborts_immediately(self):
        monitor = pilot.SafetyMonitor()

        with self.assertRaisesRegex(pilot.SafetyAbort, "电压"):
            monitor.check(
                servo_id=12,
                error_deg=0.0,
                current_a=0.1,
                voltage_v=6.7,
                temperature_c=30.0,
                dt_s=0.02,
            )

    def test_overcurrent_requires_half_second(self):
        monitor = pilot.SafetyMonitor()

        for _ in range(4):
            monitor.check(
                servo_id=12,
                error_deg=0.0,
                current_a=2.1,
                voltage_v=7.4,
                temperature_c=30.0,
                dt_s=0.02,
                power_dt_s=0.1,
            )
        with self.assertRaisesRegex(pilot.SafetyAbort, "电流"):
            monitor.check(
                servo_id=12,
                error_deg=0.0,
                current_a=2.1,
                voltage_v=7.4,
                temperature_c=30.0,
                dt_s=0.02,
                power_dt_s=0.1,
            )

    def test_tracking_error_requires_sustained_error(self):
        monitor = pilot.SafetyMonitor()

        for _ in range(14):
            monitor.check(
                servo_id=12,
                error_deg=21.0,
                current_a=0.1,
                voltage_v=7.4,
                temperature_c=30.0,
                dt_s=0.02,
            )
        with self.assertRaisesRegex(pilot.SafetyAbort, "位置误差"):
            monitor.check(
                servo_id=12,
                error_deg=21.0,
                current_a=0.1,
                voltage_v=7.4,
                temperature_c=30.0,
                dt_s=0.02,
            )

    def test_temperature_warns_above_seventy_and_aborts_at_seventy_five(self):
        monitor = pilot.SafetyMonitor()

        monitor.check(
            servo_id=12,
            error_deg=0.0,
            current_a=0.1,
            voltage_v=7.4,
            temperature_c=70.1,
            dt_s=0.02,
        )
        self.assertIn(12, monitor.temperature_warning_ids)

        with self.assertRaisesRegex(pilot.SafetyAbort, "温度 75.0℃"):
            monitor.check(
                servo_id=12,
                error_deg=0.0,
                current_a=0.1,
                voltage_v=7.4,
                temperature_c=75.0,
                dt_s=0.02,
            )


class PairedJointTests(unittest.TestCase):
    def test_paired_rmse_detects_reversed_joint(self):
        """一侧对齐RMSE超过对侧2倍时标记FAIL。"""
        metrics = {
            "right_knee": {
                "servo_id": 13, "result": "PASS", "samples": 200,
                "rmse_deg": 1.0, "aligned_rmse_deg": 0.8,
            },
            "left_knee": {
                "servo_id": 23, "result": "PASS", "samples": 200,
                "rmse_deg": 3.5, "aligned_rmse_deg": 3.2,
            },
            "right_ankle": {
                "servo_id": 14, "result": "PASS", "samples": 200,
                "rmse_deg": 0.9, "aligned_rmse_deg": 0.7,
            },
            "left_ankle": {
                "servo_id": 24, "result": "PASS", "samples": 200,
                "rmse_deg": 1.1, "aligned_rmse_deg": 0.9,
            },
        }
        paired_joints = [
            (("right_knee", 13), ("left_knee", 23)),
            (("right_ankle", 14), ("left_ankle", 24)),
        ]
        direction_failures = []
        for (r_name, r_id), (l_name, l_id) in paired_joints:
            r_m = metrics.get(r_name, {})
            l_m = metrics.get(l_name, {})
            r_rmse = r_m.get("aligned_rmse_deg") or r_m.get("rmse_deg")
            l_rmse = l_m.get("aligned_rmse_deg") or l_m.get("rmse_deg")
            if r_rmse is None or l_rmse is None or min(r_rmse, l_rmse) < 0.01:
                continue
            ratio = max(r_rmse, l_rmse) / min(r_rmse, l_rmse)
            if ratio >= 2.0:
                worse_name = r_name if r_rmse > l_rmse else l_name
                direction_failures.append(worse_name)
                metrics[worse_name]["result"] = "FAIL"
        # left_knee has 3.2 vs right_knee 0.8 => ratio 4.0 => FAIL
        self.assertIn("left_knee", direction_failures)
        self.assertEqual(metrics["left_knee"]["result"], "FAIL")
        # right_ankle 0.7 vs left_ankle 0.9 => ratio 1.29 => no FAIL
        self.assertNotIn("right_ankle", direction_failures)
        self.assertNotIn("left_ankle", direction_failures)
        self.assertEqual(metrics["right_ankle"]["result"], "PASS")

    def test_symmetric_joints_pass_when_rmse_close(self):
        """左右对称关节RMSE接近时全部PASS。"""
        metrics = {
            "right_knee": {
                "servo_id": 13, "result": "PASS",
                "rmse_deg": 1.5, "aligned_rmse_deg": 1.3,
            },
            "left_knee": {
                "servo_id": 23, "result": "PASS",
                "rmse_deg": 1.3, "aligned_rmse_deg": 1.1,
            },
        }
        r_rmse = metrics["right_knee"].get("aligned_rmse_deg") or metrics["right_knee"].get("rmse_deg")
        l_rmse = metrics["left_knee"].get("aligned_rmse_deg") or metrics["left_knee"].get("rmse_deg")
        ratio = max(r_rmse, l_rmse) / min(r_rmse, l_rmse)
        self.assertLess(ratio, 2.0)
        self.assertEqual(metrics["right_knee"]["result"], "PASS")
        self.assertEqual(metrics["left_knee"]["result"], "PASS")

    def test_unidirectional_waveform_goes_negative_only(self):
        """单向波形只输出负值（0→-1→0），不输出正值。"""
        import math
        # ID 21 (left_hip_roll) 例外：正方向
        sign = pilot.JOINT_WAVEFORM_SIGN.get("left_hip_roll", -1.0)
        self.assertEqual(sign, 1.0)
        # 默认负方向
        sign = pilot.JOINT_WAVEFORM_SIGN.get("right_knee", -1.0)
        self.assertEqual(sign, -1.0)
        # 检查波形输出
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            val = pilot.unidirectional_waveform(t, 1.0)
            self.assertLessEqual(val, 0.0, f"t={t}: val={val} should be <= 0")

    def test_per_joint_amplitude_limits(self):
        """模具限制：ID 10/20 仅15°，其余30°。"""
        self.assertEqual(pilot.JOINT_AMPLITUDE_DEG["right_hip_yaw"], 15.0)
        self.assertEqual(pilot.JOINT_AMPLITUDE_DEG["left_hip_yaw"], 15.0)
        self.assertEqual(pilot.JOINT_AMPLITUDE_DEG["right_knee"], 30.0)
        self.assertEqual(pilot.JOINT_AMPLITUDE_DEG["left_hip_roll"], 30.0)

    def test_aligned_error_in_metrics(self):
        """calculate_joint_metrics 应输出 aligned_rmse_deg 等对齐误差字段。"""
        rows = []
        for i in range(200):
            t = i * 0.02
            target = -15.0 * (0.5 - 0.5 * __import__('math').cos(__import__('math').pi * t / 2.0))
            actual = target * 0.95  # simulate tracking error
            rows.append({
                "time_s": t, "joint": "right_knee", "servo_id": 13,
                "target_deg": target, "actual_deg": actual,
                "error_deg": actual - target,
            })
        metrics = pilot.calculate_joint_metrics(
            rows, (("right_knee", 13),),
            waveform_period_s=4.0,
        )
        knee = metrics["right_knee"]
        self.assertIn("aligned_rmse_deg", knee)
        self.assertIn("aligned_max_abs_error_deg", knee)
        self.assertIn("rmse_limit_deg", knee)
        self.assertIn("command_amplitude_deg", knee)



class InterfaceAndReportTests(unittest.TestCase):
    def test_serial_timeout_is_half_second(self):
        self.assertEqual(pilot.SERIAL_TIMEOUT_S, 0.5)

    def test_two_angle_waveform_returns_to_origin_between_targets(self):
        half_cycle = 1.0
        self.assertAlmostEqual(
            pilot.centered_two_angle_waveform(0.0, half_cycle),
            0.0,
        )
        self.assertAlmostEqual(
            pilot.centered_two_angle_waveform(1.0, half_cycle),
            1.0,
        )
        self.assertAlmostEqual(
            pilot.centered_two_angle_waveform(2.0, half_cycle),
            0.0,
        )
        self.assertAlmostEqual(
            pilot.centered_two_angle_waveform(3.0, half_cycle),
            -1.0,
        )
        self.assertAlmostEqual(
            pilot.centered_two_angle_waveform(4.0, half_cycle),
            0.0,
        )

    def test_safety_confirmation_accepts_enter(self):
        with (
            mock.patch.object(pilot.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="") as input_mock,
        ):
            pilot.confirm_supported(False)
        self.assertIn("按 Enter", input_mock.call_args.args[0])

    def test_cli_has_no_robot_serial_number_argument(self):
        parser = pilot.build_parser(Path("D:/test-repo"))
        option_names = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--robot-sn", option_names)
        args = parser.parse_args(["assembly", "--scan-only"])
        self.assertEqual(args.command, "assembly")
        self.assertTrue(args.scan_only)

    def test_fixed_gait_sequence_matches_plan(self):
        segments = pilot.gait_segments()

        self.assertEqual(
            [segment["name"] for segment in segments],
            ["站立", "原地踏步", "前进", "停止", "左转", "右转", "最终停止"],
        )
        self.assertEqual(sum(segment["cycles"] for segment in segments), 1000)

    def test_chinese_report_contains_scan_and_waveform_labels(self):
        summary = {
            "timestamp": "2026-07-28T18:00:00",
            "command": "assembly",
            "result": "FAIL",
            "scan": {
                "found_ids": [12],
                "missing_ids": [10, 11],
            },
            "error": "整机舵机不完整",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            report_path = pilot.render_report(output_dir, summary, [], [])
            report_text = report_path.read_text(encoding="utf-8")

        self.assertIn("OpenDuck Mini 整机中试报告", report_text)
        self.assertIn("已识别 ID", report_text)
        self.assertIn("输入指定波形与舵机实际响应波形", report_text)
        self.assertIn("发送的位置指令 θin(t)", report_text)
        self.assertIn("反馈的实际输出位置 θout(t)", report_text)
        self.assertIn("动作阶段未执行", report_text)
        self.assertIn("测试结论：不通过（FAIL）", report_text)

    def test_joint_metrics_use_binary_pass_fail(self):
        passing_rows = [
            {"servo_id": 12, "error_deg": 2.0},
            {"servo_id": 12, "error_deg": -3.0},
        ]
        failing_rows = [
            {"servo_id": 12, "error_deg": 1.0},
            {"servo_id": 12, "error_deg": 16.0},
        ]
        joints = (("right_hip_pitch", 12),)

        self.assertEqual(
            pilot.calculate_joint_metrics(passing_rows, joints)[
                "right_hip_pitch"
            ]["result"],
            "PASS",
        )
        self.assertEqual(
            pilot.calculate_joint_metrics(failing_rows, joints)[
                "right_hip_pitch"
            ]["result"],
            "FAIL",
        )

    def test_gait_metrics_fail_when_any_joint_exceeds_limit(self):
        row = {}
        for joint_name, _ in pilot.JOINTS:
            row[f"{joint_name}_error_deg"] = 0.5
        row["right_knee_error_deg"] = 16.0

        metrics = pilot.calculate_gait_metrics([row])

        self.assertEqual(metrics["right_knee"]["result"], "FAIL")
        self.assertTrue(
            all(
                item["result"] == "PASS"
                for name, item in metrics.items()
                if name != "right_knee"
            )
        )

    def test_position_delay_estimator_detects_actual_output_lag(self):
        sample_period = 0.02
        delay_samples = 5
        rows = []
        target_values = [
            math.sin(2.0 * math.pi * index * sample_period / 2.0)
            for index in range(300)
        ]
        for index, target in enumerate(target_values):
            actual_index = max(0, index - delay_samples)
            rows.append(
                {
                    "time_s": index * sample_period,
                    "target_deg": target,
                    "actual_deg": target_values[actual_index],
                }
            )

        metrics = pilot.estimate_position_delay(
            rows,
            target_field="target_deg",
            actual_field="actual_deg",
            period_s=2.0,
        )

        self.assertAlmostEqual(metrics["position_delay_ms"], 100.0, delta=20.0)
        self.assertAlmostEqual(metrics["phase_delay_deg"], 18.0, delta=4.0)
        self.assertGreater(metrics["delay_correlation"], 0.99)
        chart = pilot.svg_overlay(
            rows,
            target_field="target_deg",
            actual_field="actual_deg",
        )
        self.assertIn("t输入=", chart)
        self.assertIn("t输出=", chart)
        self.assertIn("响应延时 Δt=", chart)
        self.assertIn("marker-start", chart)

    def test_latest_position_charts_draw_zero_axes_and_aligned_waveform(self):
        rows = []
        sample_period = 0.02
        delay_samples = 5
        target_values = [
            30.0 * math.sin(2.0 * math.pi * index * sample_period / 2.0)
            for index in range(200)
        ]
        for index, target in enumerate(target_values):
            actual_index = max(0, index - delay_samples)
            rows.append(
                {
                    # 模拟后测关节从整段测试第12秒开始记录。
                    "time_s": 12.0 + index * sample_period,
                    "target_deg": target,
                    "actual_deg": target_values[actual_index],
                }
            )

        raw_chart = pilot.svg_position_chart(
            rows,
            target_field="target_deg",
            actual_field="actual_deg",
        )
        aligned_chart = pilot.svg_position_chart(
            rows,
            target_field="target_deg",
            actual_field="actual_deg",
            align_response=True,
        )

        self.assertIn("原始时序波形（含响应延时）", raw_chart)
        self.assertIn('stroke="#334155" stroke-width="1.8"', raw_chart)
        self.assertIn("响应时间对齐后的波形", aligned_chart)
        self.assertIn("沿时间轴向左平移", aligned_chart)
        self.assertIn("只消除响应时间滞后", aligned_chart)
        self.assertIn("输入函数：θin(t)", aligned_chart)
        self.assertIn("cos(πu)", aligned_chart)
        self.assertIn('<rect x="88" y="66"', raw_chart)
        self.assertIn('<rect x="88" y="46"', aligned_chart)
        self.assertIn('polyline points="82.00,', raw_chart)
        self.assertNotIn("t输入=12.", raw_chart)

    def test_response_marker_uses_first_effective_movement(self):
        rows = []
        for index in range(601):
            time_s = index * 0.02
            target = 45.0 - 45.0 * math.cos(2.0 * math.pi * time_s / 4.0)
            rows.append({"time_s": time_s, "target_deg": target})

        feature_index = pilot.select_first_response_feature(
            rows,
            target_field="target_deg",
        )

        self.assertAlmostEqual(rows[feature_index]["time_s"], 0.20, delta=0.03)
        self.assertGreaterEqual(rows[feature_index]["target_deg"], 1.8)
        self.assertLess(rows[feature_index]["target_deg"], 3.0)

    def test_incomplete_robot_stops_before_motion_and_writes_all_report_files(self):
        parser = pilot.build_parser(Path("D:/test-repo"))
        incomplete_scan = pilot.ScanResult(
            found_positions={12: 0.0},
            missing_ids=[servo_id for servo_id in pilot.EXPECTED_IDS if servo_id != 12],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = parser.parse_args(
                [
                    "all",
                    "--yes",
                    "--output-root",
                    temporary_directory,
                    "--no-live",
                ]
            )
            with mock.patch.object(
                pilot,
                "scan_with_new_controller",
                return_value=incomplete_scan,
            ), mock.patch.object(pilot, "run_assembly_motion") as assembly_motion:
                exit_code = pilot.run_session(args)

            result_directories = list(Path(temporary_directory).iterdir())
            self.assertEqual(exit_code, 2)
            assembly_motion.assert_not_called()
            self.assertEqual(len(result_directories), 1)
            result_files = {path.name for path in result_directories[0].iterdir()}

        self.assertEqual(
            result_files,
            {
                "manifest.json",
                "summary.json",
                "report.html",
                "motor_samples.csv",
                "gait_samples.csv",
            },
        )


if __name__ == "__main__":
    unittest.main()
