# v2 기본 구성 설정과 검증

## 우선순위별 수정

| 검토 항목 | 적용 내용 | 검증 상태 |
|---|---|---|
| F01 모터 | explicit PD, 3.35 N·m, kp 17.8, damping 0.60, Coulomb 마찰 0.052 N·m, armature 0.028 | 공식 참조값과 PhysX 검사. 실제 식별값 입력은 남음 |
| F02 센서 입력 | actor 101, 정답 선속도·지형은 critic 222에만 포함 | 실제 평지·험지 shape 및 유한성 확인 |
| F03 계약 | 50Hz, L5-H4-R5 순서, URDF↔canonical 영점, scale 0.25 | 32자세 FK 비교, CPU 계약 검사, GPU 재생 |
| F04 모터 랜덤화 | 모델·solver 한계 함께 갱신, strict readback | 실제 PhysX 통과 |
| F05 높이 | root·발의 높이를 근처 지형 hit 기준으로 계산 | CPU 평행 이동 불변성, GPU 험지 실행 |
| F06 관절 한계 | stop 여유 0.02rad, 목표 변화율 ≤5.24rad/s | 무작위 action 검사, GPU 실행 |
| F07 비이상성 | action/IMU 지연 0~3 ticks, 잡음·bias, ±0.5° play operator | CPU 검사. 실측 시간·유격·노이즈 보정은 남음 |
| F08 랜덤화 | 질량·관성 동시 변경, CoM·마찰·gain·armature, readback | CPU 부분 리셋과 실제 PhysX 검사 |
| F09 등록 | 명시적 import launcher | 실제 PPO 실행 |
| F10 재생 | debug boolean, 제한 스텝, headless 입력, no_grad/reset 충돌 수정 | 128스텝 완료, 넘어짐 3회 |
| F11 다운로드 | 유효한 고정 zip URL, hash·BVH 검사 | 실제 자료 다운로드/retarget |
| F12 retarget | knee/UpLeg 분리, canonical bounds, 속도에 맞는 시간 늘림 | CPU 회귀와 실제 CMU 파일 |
| F13 motion | 이름 재정렬, 좌표계 명시, FPS 보간 | 100→50Hz 검사 |
| F14 지면 | sand를 rigid grid로 개명, self collision 활성화 | GPU 험지 실행. 모래/TPU 상세 물성은 별도 |
| F15 재현/배포 | 버전 기록, PPO 설정 명시, normalizer 포함 export, 계약 hash | 학습·저장·TorchScript·ONNX 출력 비교 |
| F16 위상 | cos/sin을 actor에 포함, clip 주기 전달 | CPU 위상 검사와 모방학습 GPU 실행 |

## 모터와 좌표

3.35 N·m 등은 공식 Playground BAM 기반 **시뮬레이션 참조값**입니다. 실물의 연속 허용 토크를 보증하지 않습니다. 버스 PID 정수 설정과 N·m/rad 단위 kp는 같은 수치 체계가 아니므로 직접 대입하지 않습니다.

실물 영점이 보정된 canonical 관절 좌표를 정책에서 사용합니다. `q_canonical = q_urdf + offset`이며 14개 변환은 `open_duck_v2/profile.py`에 있습니다. 공식 MJCF와 32개 자세를 비교한 최대 위치 차이는 약 1.13µm, 회전 차이는 약 1.16×10⁻⁵rad입니다. 원본 질량·관성·메시를 보존했고 USD의 실제 질량 합계는 약 2.107141kg입니다.

```text
L hip_yaw/roll/pitch/knee/ankle : 20 21 22 23 24
neck_pitch, head_pitch/yaw/roll: 30 31 32 33
R hip_yaw/roll/pitch/knee/ankle : 10 11 12 13 14
```

## 관측

raw 101개 순서는 `config/policy_contract.json`에 고정합니다.

```text
gyro(3), accelerometer(3), commands(7),
joint_pos-home(14), joint_vel*0.05(14),
last/previous/older action(각14), commanded motor targets(14),
left/right foot contacts(2), phase cos/sin(2)
```

명령은 vx, vy, yaw rate와 neck/head 4개입니다. 기본 학습에서 머리 명령은 0입니다. 높이 명령과 지형 정답은 actor에 넣지 않습니다. IMU는 trunk 축에 맞추며 가속도는 specific force입니다. 수평 정지 상태의 Z는 약 +9.81m/s²이며 실제 BNO055 remap·축·bias를 확인해야 합니다.

지연은 50Hz tick 단위로 최대 60ms입니다. play operator는 유격 근사이며 물리적 자유 관절 유격 전체를 재현하지는 않습니다. 토크-속도 곡선 근사 옵션은 측정 근거를 얻은 뒤 켭니다.

## 센서 로그

JSONL 한 줄에 다음 필드가 필요합니다. 관절 값에는 로봇별 제로점 보정이 적용돼 있어야 합니다.

```json
{
  "timestamp_s": 0.0,
  "joint_frame": "canonical-v2",
  "contract_version": "open-duck-mini-v2-101-v1",
  "joint_names": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"],
  "gyro": [0, 0, 0],
  "accelerometer": [0, 0, 9.81],
  "commands": [0, 0, 0, 0, 0, 0, 0],
  "joint_pos": [0.002, 0.053, -0.63, 1.368, -0.784, 0, 0, 0, 0, -0.003, -0.065, 0.635, 1.379, -0.796],
  "joint_vel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "contacts": [1, 1]
}
```

각도·각속도는 rad/rad/s, gyro는 rad/s, 가속도는 m/s²입니다. 재생 도구는 20ms 주기의 ±5ms 허용 범위를 검사합니다. 로그 관측으로 얻은 목표각과 같은 입력에 대한 시뮬레이터 목표각을 비교하세요. 모터 명령을 전송하지 않습니다.

## 남은 실물 검증

측정 기록은 아직 없습니다. 템플릿은 `incomplete`이며 `hardware_calibrated`는 false입니다. 중량, 부하별 서보 응답·포화·마찰·속도, 통신/IMU 지연, 영점·부호, IMU 축과 발 좌우를 기록해야 합니다. 값이 학습 랜덤화 범위를 벗어나면 모델/범위를 수정하고 재학습합니다.

전압 강하·열 제한·유연한 발바닥·모래 지면의 상세 물성은 이번 소프트웨어 검사로 확정하지 않았습니다. 충분히 학습한 정책을 시뮬레이션에서 평가한 뒤 실측 로그 출력 일치 검사와 지지 상태의 작은 동작부터 확인해야 합니다.

3회 PPO 학습은 저장·복원·export·재생 경로 검사입니다. 128스텝에서 3회 넘어졌으므로 실물용 정책으로 평가하지 않습니다. 충분한 학습 후 다른 seed, 명령 추종 오차, 넘어짐 비율, 포화율과 지연 변동을 평가해야 합니다.

GPU 조회는 Codex 샌드박스에서 실패했지만 시스템 권한에서는 정상 동작했습니다. 이전 검토의 “드라이버 미작동” 설명은 2026-11-04 실제 GPU 검사로 정정했습니다.


## 2026-11-10 학습 중 추가 확인·수정

- **F17 병렬 재질 상한:** 매번 새로운 마찰·반발 값을 만드는 방식은 512개 환경에서 PhysX의 64K 고유 재질 한계를 초과했습니다. 32개 재질 풀을 만들고 모든 충돌 형상과 리셋에서 재사용합니다. 부분 리셋과 값 재사용 회귀 검사를 추가했습니다.
- **F18 초기 발바닥 관통:** HOME 자세의 전체 STL 충돌 형상 최저점은 trunk 기준 -0.166829954m였습니다. 기존 초기 높이 0.15m는 약 16.83mm 관통을 만들었습니다. 초기 높이를 0.172m로 변경하고 `spawn_height`가 실제 리셋에 적용되도록 수정했습니다. 목표 몸통 높이는 0.17m이며 메시 기반 지면 여유 회귀 검사가 적용됩니다.
- **F19 기구 자기 충돌:** 모든 CAD 메시로 자기 충돌을 켜면 가까운 기구 링크가 실제 동작을 방해했습니다. 같은 정책의 자기 충돌 on/off 비교에서 토크 포화와 떨림 차이가 확인됐습니다. 2관절 이내의 조상 링크 25쌍을 필터링한 결과가 자기 충돌 off 진단과 유사했고, 좌우 다리·머리와 몸통 등 먼 부위의 충돌은 유지합니다. 원본 USD는 보존하고 로드 시 FilteredPairsAPI를 적용합니다.

최종 `model_699.pt`는 학습 경로가 실행됐다는 결과이며 전진 보행 성공 정책으로 판정하지 않았습니다. 기준 물성의 전진 8회 모두 20초를 유지했지만 XY 속도 RMSE 약 0.217m/s, yaw MAE 약 1.408rad/s로 목표 명령 추종에 미달했습니다. 정지 명령에도 움직임이 남습니다. 자세 유지·발 끌림·방향 추종을 영상과 함께 평가해야 합니다.

고정 명령 평가는 첫 에피소드만 사용하고 리셋으로 시간을 늘리지 않습니다. 발 접촉은 3개 20ms 샘플이 유지될 때 상태를 변경합니다. 발 들림은 안정 비접촉 구간의 링크 원점 높이를 접촉 중 중앙값과 비교한 근사값이며 실제 접촉면 높이와 구분합니다. `tools/assess_flat_walk.py`는 생존만 한 정지 상태를 보행 성공으로 세지 않습니다.


## 직진·정지 재학습과 머리·목 (2026-11-10)

- **F20 방향과 정지 보상:** 이동 속도의 크기 대신 명령 방향 투영과 목표 속도 오차를 평가합니다. 정지 명령에서는 잔여 이동·회전과 제자리 동작을 감점합니다. 보행 중 자연스러운 좌우 흔들림을 제거하려고 다리를 경직시키지 않도록 보상용 속도만 0.4초 EMA로 처리합니다.
- **F21 머리 동역학:** 원본 CAD 머리·목 4링크의 합계는 약 0.581535kg(전체 2.107141kg의 27.6%)입니다. 질량·관성·메시를 보존합니다. 공식 actor의 목 전체 고정 비교에서는 전진 생존이 8/8에서 0/8로 떨어져, 최종 후보는 균형 보상 움직임을 유지하는 `stabilized` 모드를 사용합니다. 머리 목표각 변화율은 1rad/s로 제한하며 실제 관절 속도 자체의 강제 상한과 구분합니다.
- **F22 위상·정책 초기화:** 공식 `Open_Duck_Mini`의 `BEST_WALK_ONNX_2.onnx`를 동일 출력 SiLU/tanh actor로 변환한 뒤 PPO 미세 학습의 초기값으로 사용합니다. 공개 모델을 새로 학습한 결과로 표기하지 않습니다. 입력 정규화도 보존하고 출력 일치를 검사합니다. 공식 gait 주기는 0.54초이며 체크포인트와 export sidecar에 기록합니다.
- **F23 느린 경로 이탈:** `StraightV4` 보상은 몸체 gyro Z와 실제 world yaw 변화가 같지 않을 수 있음을 반영합니다. world yaw 차분과 누적 방향 오차, 출발 직선에서의 측면 이탈을 학습 보상에 추가합니다. world 위치·방향 정답은 보상 계산에만 사용하며 actor의 raw 101개 입력은 유지합니다. 이 task는 전진·정지용이며 측진·후진·회전의 성능을 의미하지 않습니다.
- **F24 독립 판정:** `tools/assess_straight_walk.py`가 한 보행 주기(0.54초) 평균 속도 오차, 실제 world 이동, 누적 yaw, 양발 디딤·들림을 함께 검사합니다. 원래의 순간 속도 오차도 보고서에 남깁니다. 3초 정지→12초 전진→5초 정지 시험은 리셋 없이 수행합니다. 각 명령은 모든 항목을 통과한 시험이 90% 이상이어야 통과합니다. 설정한 수치는 이 실험의 공학적 목표이며 표준 인증 기준은 아닙니다.

재학습 데이터·실험 설정·머리 고정 비교 원자료는 `validation/straight_walk_2026-11-10/`에 있습니다. 학습 시 기준 물성에서 먼저 명령 추종을 맞추며, 랜덤화 스트레스 시험은 별도 표기합니다. 실물 서보·발바닥·지연 계측을 완료한 상태로 간주하지 않습니다.


`StraightV5`는 actor 101·critic 222 차원을 유지하며, 평지에서 비활성인 지형 context 117개를 `heading_error, lateral_path_error, zeros(115)`로 정의합니다. 경로 상태가 포함된 보상에 대한 critic의 관측 누락을 해결하기 위한 학습 전용 문맥입니다. actor에 world 위치·방향을 넣지 않습니다. 이 프로필에서는 험지/높이 스캔을 거부하며 checkpoint에 `critic_context=flat_path`를 기록합니다. 한쪽 발의 낮은 들림을 보완하기 위해 swing 발 링크 목표 높이를 0.065m, clearance 보상 가중치를 3으로 설정했습니다.


**F25 접촉/발 들림 보상 위상:** 공식 actor의 실측 시계열에서 왼발 들림 정점은 cycle 약 0.52, 오른발은 0.0 부근입니다. 기존 contact/swing 보상은 약 1/4주기 늦어 공식 정책과 충돌했습니다. `StraightV6`는 actor 입력 위상(0.54초 clock)을 유지하고 보상 접촉 위상에만 +0.25cycle 오프셋을 적용합니다. 초기 정책에서 접촉 보상 일치율이 약 30%에 불과했던 원인을 분리했으며 swing 정점에 stance를 요구하지 않는 회귀 검사를 추가했습니다. 이전 실험의 위상과 평가 원자료는 보존합니다.


**F26 선택 가능한 IMU 방향 유지 제어:** `heading_hold_kp>0`인 체크포인트는 기존 actor(101 입력) 앞에서 canonical trunk IMU의 상대 yaw를 이용해 전진 중 yaw-rate 명령을 보정합니다. 위치/속도를 강제로 바꾸지 않고 관절 출력만 사용합니다. 정지 명령에는 방향 보정·측면 trim을 적용하지 않습니다. 기준 gain 2.0s⁻¹, yaw-rate 보정 상한 0.35rad/s, 오차 필터 0.25초입니다. `heading_hold_*`와 `lateral_command_trim`을 checkpoint·export sidecar에 저장하고 재생에서도 복원합니다.

이 구성은 gyro/acceleration 외에 **IMU 자세 quaternion**을 명령 전처리에서 요구합니다. 센서 JSONL에 `"imu_quaternion_wxyz": [w,x,y,z]`를 추가해야 하며 단위 quaternion과 유한성을 검사합니다. 원시 BNO055 축/마운트에서 canonical trunk 축으로 변환한 값을 사용해야 합니다. 일부 기존 런타임의 `Imu.get_data()`는 xyzw를 반환하므로 순서를 명시적으로 바꾸세요. 실제 Raspberry Pi의 드라이버·모터 코드는 이 작업에서 변경하지 않았습니다. 실물의 축·순서·bias·지연과 초기 기준 자세를 확인하는 단계는 남아 있습니다.

시뮬레이션에서는 가상의 trunk IMU 자세를 사용합니다. 기준 물성 시험은 이상적 자세 측정이며, 랜덤화 시험은 yaw 잡음 ±0.01rad, 초기 bias ±0.02rad와 0–3tick 지연을 포함합니다. 초기 자세를 기준으로 하므로 고정 yaw bias는 상쇄됩니다. 실제 장시간 yaw drift나 자기장 교란을 계측·식별한 것으로 간주하지 않습니다. standalone ONNX/TorchScript actor를 호출할 때도 이 50Hz 명령 전처리와 모터 목표 처리를 함께 적용해야 합니다.


전진 명령 배율 `forward_command_scale=0.85`는 PPO 정책의 기준 물성 속도 편차를 보정한 값입니다. 외부 목표는 계속 0.1m/s이며 내부 actor 명령만 0.085m/s로 바꿉니다. `lateral_command_trim=0`으로 측면 고정 bias는 추가하지 않았습니다. `tools/configure_policy_controller.py`는 신경망 가중치를 바꾸지 않고 이 제어 설정과 보정에 사용한 평가 파일·SHA-256을 별도 기록합니다. 최종 성능은 **PPO로 학습한 보행 + IMU 방향 유지 + 전진 명령 보정**의 결합 결과입니다.
