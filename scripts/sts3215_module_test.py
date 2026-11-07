"""OpenDuck Mini 左腿、右腿或头部组件检测。

连接一个已经组装好的组件后，程序自动扫描项目 ID 并识别：
- 右腿：10-14
- 左腿：20-24
- 头部：30-33

直接运行时自动扫描、识别组件并执行检测；--scan-only 仅用于维护排查。

方向基准流程：
1. identify_module()先确定右腿、左腿或头部；
2. 自动加载direction_references/<组件>.json；
3. run_assembly_motion()计算位置方向和多关节电流方向；
4. 明确反向时总体FAIL、退出码3，并在报告中写出疑似装反关节。
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts import pilot_test_platform as pilot
except ImportError:
    import pilot_test_platform as pilot


MODULES: dict[str, dict[str, Any]] = {
    "right_leg": {
        "label": "右腿",
        "ids": (10, 11, 12, 13, 14),
    },
    "left_leg": {
        "label": "左腿",
        "ids": (20, 21, 22, 23, 24),
    },
    "head": {
        "label": "头部",
        "ids": (30, 31, 32, 33),
    },
}
MODULE_TEST_AMPLITUDE_DEG = 15.0


@dataclass(frozen=True)
class ModuleSelection:
    """根据扫描结果确定的组件及其完整性。"""

    key: str
    label: str
    expected_ids: tuple[int, ...]
    found_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_ids and set(self.found_ids) == set(self.expected_ids)

    @property
    def joints(self) -> tuple[tuple[str, int], ...]:
        expected = set(self.expected_ids)
        return tuple(
            (joint_name, servo_id)
            for joint_name, servo_id in pilot.JOINTS
            if servo_id in expected
        )


def identify_module(found_ids: list[int]) -> ModuleSelection:
    """只接受一个完整组件或能够唯一归属到某组件的残缺 ID 集合。"""

    detected = set(found_ids)
    if not detected:
        raise RuntimeError("未扫描到项目舵机 ID")
    if detected == set(pilot.EXPECTED_IDS):
        raise RuntimeError(
            "当前扫描到14个整机舵机，请使用 pilot_test_platform.py"
        )

    candidates = []
    for key, definition in MODULES.items():
        expected = set(definition["ids"])
        if detected <= expected:
            candidates.append((key, definition))

    if len(candidates) != 1:
        raise RuntimeError(
            f"扫描到跨组件或无法归属的 ID：{sorted(detected)}；"
            "部件检测时只能连接一条完整左腿、右腿或头部总线"
        )

    key, definition = candidates[0]
    expected_ids = tuple(definition["ids"])
    found = tuple(servo_id for servo_id in expected_ids if servo_id in detected)
    missing = tuple(servo_id for servo_id in expected_ids if servo_id not in detected)
    return ModuleSelection(
        key=key,
        label=str(definition["label"]),
        expected_ids=expected_ids,
        found_ids=found,
        missing_ids=missing,
    )


def module_scan_from_full(
    full_scan: pilot.ScanResult,
    selection: ModuleSelection,
) -> pilot.ScanResult:
    """从全项目扫描中提取当前组件自己的已识别和缺失 ID。"""

    return pilot.ScanResult(
        expected_ids=selection.expected_ids,
        found_positions={
            servo_id: full_scan.found_positions[servo_id]
            for servo_id in selection.found_ids
        },
        missing_ids=list(selection.missing_ids),
        errors={
            servo_id: full_scan.errors.get(servo_id, "未识别")
            for servo_id in selection.missing_ids
        },
    )


def scan_all_project_ids() -> pilot.ScanResult:
    """打开总线并只读扫描全部项目 ID，然后释放串口。"""

    controller = pilot.open_controller()
    try:
        return pilot.scan_servo_ids(controller)
    finally:
        pilot.release_controller(controller)
        controller = None
        gc.collect()


def run(args: argparse.Namespace) -> int:
    """执行组件扫描或小幅波形检测，并始终生成中文报告。"""

    temporary_output = args.output_root is None
    output_root = (
        pilot.temporary_report_root("module")
        if temporary_output
        else Path(args.output_root)
    )
    output_dir = pilot.create_output_dir(output_root)
    motor_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": "module",
        "report_title": "OpenDuck Mini 腿/头部件中试报告",
        "result": "FAIL",
        "port": pilot.SERIAL_PORT,
        "baud": pilot.SERVO_BAUD_RATE,
        # 与整机报告共用阈值来源，避免把内部工程门槛写成厂家标准。
        "threshold_sources": pilot.threshold_sources(),
    }
    selection: ModuleSelection | None = None
    direction_reference: dict[str, Any] | None = None
    exit_code = 1
    dashboard = None

    if not args.no_live:
        try:
            try:
                from live_waveform_dashboard import LiveWaveformDashboard
            except ImportError:
                from scripts.live_waveform_dashboard import LiveWaveformDashboard
            dashboard = LiveWaveformDashboard(port=args.live_port)
            dashboard.start()
            dashboard.set_status("正在扫描舵机", "部件 ID 扫描")
        except Exception as exc:
            dashboard = None
            print(f"实时曲线启动失败，测试继续：{exc}")

    try:
        full_scan = scan_all_project_ids()
        summary["detected_project_ids"] = full_scan.found_ids
        # 在无法归属组件时，报告仍保留原始全项目扫描结果。
        summary["scan"] = pilot.scan_to_dict(full_scan)
        selection = identify_module(full_scan.found_ids)
        component_scan = module_scan_from_full(full_scan, selection)
        summary["module"] = {
            "key": selection.key,
            "name": selection.label,
            "test_amplitude_deg": MODULE_TEST_AMPLITUDE_DEG,
        }
        summary["scan"] = pilot.scan_to_dict(component_scan)

        direction_reference = None
        summary["direction_reference"] = {
            "loaded": direction_reference is not None,
            "path": (
                str(reference_path.resolve())
                if reference_path is not None
                else None
            ),
            "mode": (
                "explicit"
                if requested_reference
                else "automatic"
                if direction_reference is not None
                else "calibration"
                if save_reference_path
                else "unavailable"
            ),
            "note": (
                "使用已确认正确装配的组件方向基准"
                if direction_reference is not None
                else "正在采集组件基准"
                if save_reference_path
                else "默认组件基准不存在；只输出电流特征，不判定装反"
            ),
        }

        print(f"识别组件：{selection.label}")
        print(f"已识别 ID：{list(selection.found_ids)}")
        print(f"缺失 ID：{list(selection.missing_ids)}")
        if not selection.complete:
            raise RuntimeError(
                f"{selection.label} ID 不完整："
                f"已识别 {list(selection.found_ids)}；"
                f"缺失 {list(selection.missing_ids)}"
            )

        if args.scan_only:
            summary["result"] = "PASS"
            summary["note"] = "组件只读扫描完成，未写 PID、未发送动作"
            exit_code = 0
            return exit_code

        pilot.confirm_supported(args.yes)
        controller = pilot.open_controller()
        try:
            # 已识别组件后只复查该组件 ID，避免对10个无关ID重复等待超时。
            component_scan = pilot.scan_servo_ids(
                controller,
                selection.expected_ids,
            )
            if not component_scan.complete:
                raise RuntimeError("动作前复查发现当前组件舵机缺失")
            motor_rows, assembly_summary = pilot.run_assembly_motion(
                controller,
                component_scan,
                joints=selection.joints,
                expected_ids=selection.expected_ids,
                amplitude_by_joint={
                    joint_name: MODULE_TEST_AMPLITUDE_DEG
                    for joint_name, _ in selection.joints
                },
                direction_reference=direction_reference,
                sample_callback=(
                    dashboard.publish_motor_row
                    if dashboard is not None
                    else None
                ),
            )
            summary["assembly"] = assembly_summary
            if save_reference_path:
                print("[方向基准] 方向基准功能已移除，请使用对称关节对比判定方向")
                summary["direction_reference"]["saved_path"] = str(
                    saved_path.resolve()
                )
                print(f"[方向基准] 已保存：{saved_path.resolve()}")
            summary["result"] = assembly_summary["result"]
            direction_failures = assembly_summary.get(
                "direction_failures",
                [],
            )
            if direction_failures:
                # 报告、终端横幅和退出码同时明确说明装配方向错误。
                summary["note"] = (
                    "装配方向检测失败："
                    + "；".join(direction_failures)
                )
            exit_code = 0 if summary["result"] == "PASS" else 3
            return exit_code
        finally:
            pilot.disable_torque(controller, selection.expected_ids)
            pilot.release_controller(controller)
            controller = None
            gc.collect()
    except KeyboardInterrupt:
        summary["error"] = "操作者中断测试"
        summary["result"] = "FAIL"
        return 130
    except pilot.SafetyAbort as exc:
        if exc.motor_rows:
            motor_rows = list(exc.motor_rows)
            partial_joints = selection.joints if selection is not None else ()
            summary["assembly"] = {
                "result": "FAIL",
                "joint_metrics": pilot.calculate_joint_metrics(
                    motor_rows,
                    partial_joints,
                    waveform_period_s=4.0,
                ),
                "samples": len(motor_rows),
                "note": "安全停机前的部分数据",
            }
        summary["error"] = f"安全停机：{exc}"
        summary["result"] = "FAIL"
        return 4
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["result"] = "FAIL"
        return 2
    finally:
        pilot.write_csv(
            output_dir / "motor_samples.csv",
            pilot.MOTOR_CSV_FIELDS,
            motor_rows,
        )
        pilot.write_json(
            output_dir / "manifest.json",
            {
                "format_version": 1,
                "created_at": summary["timestamp"],
                "command": "module",
                "output_files": {
                    "summary": "summary.json",
                    "report": "report.html",
                    "motor_samples": "motor_samples.csv",
                },
            },
        )
        pilot.write_json(output_dir / "summary.json", summary)
        report_joints = selection.joints if selection is not None else pilot.JOINTS
        report_path = pilot.render_report(
            output_dir,
            summary,
            motor_rows,
            [],
            report_joints=report_joints,
        )
        pilot.print_metric_summary(summary)
        if summary.get("error"):
            print(f"终止原因：{summary['error']}")
        pilot.print_result_banner(
            summary["result"],
            report_path=report_path,
            error=summary.get("error") or summary.get("note"),
        )
        if dashboard is not None:
            dashboard.set_report(report_path)
            dashboard.set_downloads(
                [
                    ("下载部件CSV原始数据", output_dir / "motor_samples.csv"),
                    ("下载JSON汇总", output_dir / "summary.json"),
                ]
            )
            dashboard.set_status(
                f"测试结论：{pilot.result_text(summary['result'])}",
                "报告已生成",
            )
            if pilot.sys.stdin.isatty():
                try:
                    input(
                        "请先在网页下载所需文件；完成后按 Enter "
                        "关闭页面并清理树莓派临时报告："
                    )
                except KeyboardInterrupt:
                    print("\n收到退出请求，开始清理临时报告。")
            dashboard.close()
        elif temporary_output and pilot.sys.stdin.isatty():
            try:
                input("完成后按 Enter 清理树莓派临时报告：")
            except KeyboardInterrupt:
                print("\n收到退出请求，开始清理临时报告。")
        pilot.cleanup_temporary_report(
            output_dir,
            temporary_output=temporary_output,
            keep_report=args.keep_report,
        )


def build_parser(repo_root: Path | None = None) -> argparse.ArgumentParser:
    """构造组件测试命令行。"""

    root = repo_root or Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="自动识别并检测 OpenDuck Mini 左腿、右腿或头部组件"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="只扫描并确认组件 ID，不写 PID、不发送动作",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认组件已可靠固定，用于非交互终端",
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
    parser.add_argument(
        "--direction-reference",
        default=None,
        help=(
            "覆盖组件方向基准JSON；不填写时按识别结果自动加载"
            "direction_references/<组件>.json"
        ),
    )
    parser.add_argument(
        "--save-direction-reference",
        default=None,
        help=(
            "将本次电流特征保存为方向基准JSON；"
            "仅允许在人工确认组件装配方向正确时使用"
        ),
    )
    return parser


def main() -> int:
    """命令行入口。"""

    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
