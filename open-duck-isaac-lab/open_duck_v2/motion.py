"""모방학습 motion reference를 검증하고 정책 주기로 보간한다."""

from pathlib import Path
import numpy as np
from .profile import JOINT_NAMES, LIMITS, POLICY_HZ


def load_motion(path, target_fps=POLICY_HZ):
    """NPZ motion을 canonical 관절 순서의 ``target_fps`` 시계열로 읽는다.

    파일 필드, 좌표 frame, 관절 이름, shape, 유한값, 관절 한계와 속도를 모두
    확인한다. 누락되거나 잘못된 reference를 임의의 대체 motion으로 바꾸지
    않고 예외를 발생시킨다.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        required = {
            "joint_names",
            "joint_pos",
            "fps",
            "root_height",
            "root_lin_vel",
            "joint_frame",
            "root_lin_vel_frame",
        }
        if not required.issubset(data.files):
            raise ValueError(f"Motion fields missing: {required - set(data.files)}")
        if (
            str(data["joint_frame"]) != "canonical-v2"
            or str(data["root_lin_vel_frame"]) != "body"
        ):
            raise ValueError("Motion coordinate frames must be canonical-v2 and body")
        names = [str(n) for n in data["joint_names"]]
        if len(names) != 14 or set(names) != set(JOINT_NAMES):
            raise ValueError("Motion joint names must match v2 uniquely")
        fps = float(data["fps"])
        if (
            not np.isfinite(fps)
            or fps <= 0
            or not np.isfinite(target_fps)
            or target_fps <= 0
        ):
            raise ValueError("Invalid motion FPS")
        pos = np.asarray(data["joint_pos"], dtype=np.float32)
        h = np.asarray(data["root_height"], dtype=np.float32)
        v = np.asarray(data["root_lin_vel"], dtype=np.float32)
    if (
        pos.ndim != 2
        or pos.shape[1] != 14
        or len(pos) < 2
        or h.shape != (len(pos),)
        or v.shape != (len(pos), 3)
    ):
        raise ValueError("Inconsistent motion array shapes")
    if not all(np.isfinite(x).all() for x in (pos, h, v)):
        raise ValueError("Non-finite motion values")
    # 파일 내부 순서와 무관하게 정책의 canonical JOINT_NAMES 순서로 재배열한다.
    pos = pos[:, [names.index(n) for n in JOINT_NAMES]]
    bounds = np.asarray(LIMITS)
    if ((pos < bounds[:, 0] + 0.019) | (pos > bounds[:, 1] - 0.019)).any():
        raise ValueError("Reference exceeds v2 soft joint stops")
    if ((h < 0.08) | (h > 0.35)).any():
        raise ValueError("Reference base height is outside the v2 range")
    # 원본의 비정수 FPS도 시간축을 기준으로 정확히 목표 clock에 옮긴다.
    source = np.arange(len(pos)) / fps
    target = np.arange(int(np.floor(source[-1] * target_fps)) + 1) / target_fps
    if len(target) < 2:
        raise ValueError("Motion is shorter than one policy interval")

    def interpolate(x):
        """첫 축이 시간인 2차원 배열을 공통 target time으로 선형 보간한다."""

        return np.stack(
            [np.interp(target, source, x[:, i]) for i in range(x.shape[1])], -1
        ).astype(np.float32)

    pos = interpolate(pos)
    # 보간 뒤 속도를 다시 계산해야 목표 정책 clock의 속도와 일치한다.
    velocity = np.gradient(pos, 1 / target_fps, axis=0).astype(np.float32)
    if np.max(np.abs(velocity)) > 5.24:
        raise ValueError("Reference joint speed exceeds 5.24 rad/s")
    return dict(
        joint_pos=pos,
        joint_vel=velocity,
        root_height=np.interp(target, source, h).astype(np.float32),
        root_lin_vel=interpolate(v),
        fps=float(target_fps),
        joint_names=np.array(JOINT_NAMES),
        joint_frame=np.array("canonical-v2"),
        root_lin_vel_frame=np.array("body"),
    )
