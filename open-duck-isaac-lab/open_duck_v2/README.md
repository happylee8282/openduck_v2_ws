# `open_duck_v2` 패키지 개요

이 패키지는 Open Duck Mini v2의 **정책 계약, 좌표계, 제어기, 정책 로더와
sim-to-real 보조 로직**을 모아 둔 공통 코어다. Isaac Lab 환경 구현은
`open_duck_complex_terrain/`에 있지만, 시뮬레이터와 실물 런타임이 동일하게
해석해야 하는 값은 이 패키지에 둔다.

핵심 원칙은 다음과 같다.

- 관절 순서, 단위, 좌표계와 관측 순서를 버전이 있는 계약으로 고정한다.
- actor의 14차원 출력을 곧바로 관절에 쓰지 않고 속도·한계·지연·백래시를
  적용한 뒤 URDF 좌표로 변환한다.
- 학습 체크포인트가 현재 계약과 정확히 일치할 때만 로드한다.
- 전진, 후진, 자세 정책은 서로 다른 역할을 유지하고 전환기는 관절 목표만
  연속적으로 연결한다.
- 시뮬레이션 기준값과 실제 하드웨어에서 측정한 값을 구분한다.

## 전체 데이터 흐름

```text
센서/명령
  ├─ IMU gyro + specific force
  ├─ 7차원 명령
  ├─ 14관절 위치/속도
  └─ 발 접촉, gait phase
          │
          ▼
pack_actor_observation() ── 101차원 계약 ──► Actor
                                                │ 14차원 [-1, 1]
                                                ▼
                                           V2Control
                              action delay / rate limit / backlash
                                                │ URDF 관절 목표
                                                ▼
                                      V2ServoActuator / PhysX
```

자세 정책은 기본 101차원 뒤에 목표 높이와 높이 속도 2개를 추가한 103차원
계약을 사용한다. 멀티스킬 런타임은 전진·후진·자세 actor 가운데 하나를
선택하고, 전환 시 action history와 관절 목표가 불연속적으로 뛰지 않도록
상태를 재배치한다.

## 파일별 책임

| 파일 | 책임 |
|---|---|
| `profile.py` | 관절 순서, 모터 ID, 좌표 offset, home, 한계, 주기와 101차원 계약 |
| `control.py` | 관측 packing, 지연 버퍼, action→관절 목표 변환, servo 토크 수식 |
| `actuators.py` | 위 servo 수식을 Isaac Lab actuator 인터페이스에 연결 |
| `runtime.py` | RSL-RL actor 로드, 계약 검증, TorchScript/ONNX export |
| `heading.py` | IMU world heading을 이용한 직진 명령 보정 |
| `straight_rewards.py` | 직진·정지 명령 추종에 필요한 순수 tensor 보상 항 |
| `randomization.py` | 질량, 관성, CoM, 재질과 actuator domain randomization |
| `collision.py` | 가까운 기구학 조상끼리만 self-collision을 제외 |
| `motion.py` | 모방학습용 canonical motion 파일 검증과 50Hz 보간 |
| `posture.py` | IK 자세 참조, 높이 궤적과 residual 자세 제어 |
| `posture_runtime.py` | 103차원 자세 체크포인트와 참조 파일의 결합 검증 |
| `multiskill.py` | 전진·후진·자세 정책 선택과 안전한 전환 상태기계 |
| `training_policy.py` | RSL-RL actor 출력 제한, warm-start와 critic 준비 단계 |
| `calibration.py` | 실물 계측 기록의 완전성과 학습 범위 초과 여부 검사 |

## 반드시 유지해야 하는 계약

1. 정책 관절 순서는 `profile.JOINT_NAMES`와 같아야 한다.
2. canonical 각도는 `q_canonical = q_urdf + offset`이다.
3. 보행 actor 입력은 101차원, 출력은 14차원이다.
4. 자세 actor 입력은 103차원이며 참조 JSON의 SHA-256까지 계약에 포함한다.
5. IMU quaternion은 canonical trunk 축의 `wxyz` 순서다.
6. 정책 주기는 50Hz이고 기본 물리 주기는 500Hz다.
7. 체크포인트의 계약·normalizer·activation metadata가 현재 코드와 다르면
   이름이나 차원만 맞춰 우회하지 않고 로드를 거부한다.

## 하드웨어 적용 시 주의

`MOTOR_REFERENCE`는 기준 모델이며 실측 calibration이 아니다. 실제 질량,
전압, 지연, 관절 영점·부호, IMU 축과 servo 응답을 측정한 뒤
`calibration.py`로 기록을 검사해야 한다. 현재 정책이 명목 시뮬레이션에서
동작한다는 사실만으로 실제 로봇의 안전성과 성능이 보장되지는 않는다.
