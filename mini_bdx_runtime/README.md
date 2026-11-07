# Open Duck 하드웨어 및 정책 추론 라이브러리

이 폴더는 [보행 루프](../scripts/v2_rl_walk_mujoco.py)에서 사용하는 하드웨어, 제어 입력, 관측 전처리 및 추론 모듈이다.

| 모듈 | 입력 | 출력 |
| --- | --- | --- |
| [imu.py](mini_bdx_runtime/imu.py), [feet_contacts.py](mini_bdx_runtime/feet_contacts.py) | IMU와 발 접촉 센서 | 자이로·가속도·발 접촉 관측 |
| [xbox_controller.py](mini_bdx_runtime/xbox_controller.py), [keyboard_controller.py](mini_bdx_runtime/keyboard_controller.py) | 게임패드 또는 키 이벤트 | 보행 명령, 버튼과 보조 입력 |
| [duck_config.py](mini_bdx_runtime/duck_config.py), [rustypot_position_hwi.py](mini_bdx_runtime/rustypot_position_hwi.py) | 관절 이름·ID·오프셋, 목표 관절각 | 서보 명령과 위치·속도 피드백 |
| [onnx_infer.py](mini_bdx_runtime/onnx_infer.py) | ONNX 파일과 관측 벡터 | 정책 행동 벡터 |
| [rl_utils.py](mini_bdx_runtime/rl_utils.py) | 관절 벡터, 액션, 스케일 | 관절 순서 변환, 관절 이름별 목표, 선택적 필터 |

보행 스크립트는 gyro, accelerometer, 명령, 초기 자세 기준 위치, 속도 × 0.05, 직전 3개 행동, 모터 목표, 발 접촉, 보행 위상을 순서대로 결합한다. ONNX 추론 결과를 `init_pos + action_scale × action`으로 위치 목표에 매핑하고 관절 이름별 서보 명령을 보낸다. 모델을 교체할 때 관측 순서·길이와 관절 순서, 액션 스케일을 함께 맞춘다. 학습된 정책 파일이 추론 입력이며, 이 런타임에서 정책 학습을 수행하는 단계는 없다.
