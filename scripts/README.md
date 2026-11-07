# 보행 실행·화살표 인식·하드웨어 시험

## 보행과 카메라

[v2_rl_walk_mujoco.py](v2_rl_walk_mujoco.py)는 관측 → ONNX 추론 → 관절 위치 명령 → 센서 피드백을 제어 주기마다 반복한다. 관측과 출력 형식은 [런타임 README](../mini_bdx_runtime/README.md)에 있다. 실행 시 모델, 관절 설정, 제어 주기, PID와 액션 스케일을 지정한다.

[recognize_arrow_camera_v2.py](recognize_arrow_camera_v2.py)는 카메라 영상의 방향과 ROI를 맞춘 후 임계값으로 이진화한다. 가장 큰 유효 외곽선을 골라 정규화한 마스크를 실물 샘플 또는 합성 화살표 템플릿과 IoU로 비교할 수 있다. 회전된 샘플을 비교하는 경로, 끝점 기하 또는 TFLite 모델을 사용하는 검출 경로도 있다. 최근 프레임의 다수결과 신뢰도 기준을 통과한 라벨을 사용하며 JSON에 원시·안정화 결과를 함께 출력한다.

- [run_arrow_open_duck_runtime.py](run_arrow_open_duck_runtime.py)는 검출 라벨을 키 명령으로 변환한다. 기본 매핑은 전진 W, 뒤 화살표 E, 좌 A, 우 D, 미검출 Space다.
- [run_arrow_xbox_assist_runtime.py](run_arrow_xbox_assist_runtime.py)는 Xbox를 기본으로 사용하고 실행 상태에서 안정적으로 검출된 좌·우 화살표를 시간 제한 회전 명령으로 바꾼 후 일시정지한다.
- 모터 없이 인식만 확인하려면 [루트 README](../README.md)의 `--vision-only` 예제를 사용한다.

## 시험과 기록

| 실행 코드 | 입력 | 결과 |
| --- | --- | --- |
| [pilot_test_platform.py](pilot_test_platform.py) | 모터 ID, 조립 방향 기준, 관절·보행 시험 프로필 | 파형 추종 오차, 지연, 전류와 안전 판정, CSV·JSON·HTML |
| [sts3215_pilot_test.py](sts3215_pilot_test.py), [sts3215_module_test.py](sts3215_module_test.py) | 단일 서보 또는 조립 모듈 | 목표와 실제 응답 시계열 및 요약 |
| [record_5090_v01_robot_trial.py](record_5090_v01_robot_trial.py) | ONNX 정책, 관절 설정, 고정 명령 시나리오 | 관측·액션·목표·피드백 기록 |
| [calibrate_imu.py](calibrate_imu.py), [find_soft_offsets.py](find_soft_offsets.py) | 실측 센서와 기구 자세 | IMU 보정 및 관절 오프셋 준비 자료 |

파일별 실행 인자는 `--help`가 구현된 도구의 도움말과 [시험 절차 문서](../docs/README.md)를 확인한다. 시험 결과는 모델과 설정, 전압 및 지지 조건을 함께 맞춰 비교한다.

## 시험 지표 계산

파일럿 시험의 관절 목표는 반코사인 파형으로 부드럽게 이동시키고, 목표 및 실제 관절각·전류를 시간순으로 수집한다. `estimate_position_delay()`는 후보 시차별 상관도를 비교해 위치 응답 지연을 구한다. `compute_aligned_errors()`는 이 지연을 보상해 목표와 실제 시점을 맞춘 후 RMSE와 최대 오차를 다시 계산한다. 대칭축 간 정렬 오차 비교는 조립 방향 이상을 진단하는 지표다. 저전압·전류·온도·지속 추종 오차 판정은 파형 정밀도와 별도의 중단 조건으로 관리된다. [시험 구현](pilot_test_platform.py)
