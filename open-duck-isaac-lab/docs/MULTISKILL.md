# 전진·정지 유지, 후진과 양발 낮은 자세 / 기립

작업 폴더는 `/home/happy/openduck_ws/open-duck-isaac-lab`입니다. Isaac Lab 2.2 / Isaac Sim 5.0 / RSL-RL 2.3.3 PPO를 사용했습니다. 이 문서는 v2 기본 STS3215 7.4V 구성의 **기준 물성·평지 시뮬레이션** 결과입니다. 실물 시험이나 새 동작의 물성·잡음·지연 랜덤화 검증을 완료한 결과는 아닙니다.

## 실행

```bash
cd /home/happy/openduck_ws
./duck.sh play-skills
./duck.sh play-skills --demo
```

그래픽 세션의 Isaac 창에 포커스를 두고 W 전진, S 후진, 키 해제 또는 Space 정지, C 낮은 자세 유지, R 기립, Esc 종료입니다. C는 누른 뒤 놓아도 낮은 자세를 유지합니다. 낮은 자세에서 W/S를 누르면 먼저 기립하고 양발 접촉·관절 정지를 확인한 뒤 걷습니다. 방향을 바로 반대로 바꾸면 먼저 1초간 정지합니다. 기존 `./duck.sh play`는 보존된 전진·정지 전용 정책을 계속 사용합니다.

일반 SSH에서는 GUI 창이 사용자 화면으로 자동 전달되지 않습니다. 실행 경로 검사에는 `./duck.sh play-skills --headless --steps 128 --report validation/skills_smoke.json`을 사용합니다.

## 최종 구조

| 파일 | 역할 |
|---|---|
| `config/multiskill_policy.json` | 선택된 세 체크포인트·자세 참조의 경로/SHA-256, 검증 범위와 제어 설정 기록 |
| `policies/flat_straight_v2.pt` | 기존 전진·정지 actor를 수정 없이 보존, 101입력→14출력 |
| `policies/backward_v2.pt` | 전진 정책에서 후진 PPO로 별도 학습, 101입력→14출력 |
| `policies/posture_v2.pt` | IK 참조 위의 균형 보상 PPO, 103입력→14출력 |
| `config/posture_reference_v1.json` | 고정된 20개 높이의 14관절 목표; 자세 정책이 파일 해시 검사 |
| `open_duck_v2/multiskill.py` | 정책 선택, 방향 전환 정지, 후진→정지 1초 연결, 자세 전환과 이력 변환 |
| `open_duck_v2/posture.py` | 높이 보간, 4초 quintic 궤적, ±0.12rad 보상과 모터 목표 제한 |
| `open_duck_v2/posture_runtime.py` | 자세 전용 103입력·참조 해시 계약 검사와 actor/normalizer 로딩 |
| `open_duck_complex_terrain/posture_env.py` | 자세 PPO 환경, 높이·몸통 자세·관절·양발 접촉·미끄럼 보상 |
| `open_duck_complex_terrain/open_duck_env_cfg.py` | 후진 학습 명령 −0.12~−0.04m/s, 정지 비율 15% |
| `tools/train_skill.py` | 후진/자세 별도 학습 실행 |
| `tools/play_multiskill.py` | 키보드 / 52초 자동 데모 |
| `tools/evaluate_multiskill.py`, `tools/assess_multiskill.py` | 실시간 물리 적분 재생, 원자료 기록, 사전 설정 기준 판정 |
| `tools/validate_skill_exports.py` | 실제 재생 입력에서 RSL-RL·TorchScript·ONNX 출력 비교 |
| `tools/build_posture_reference.py` | 원본 URDF와 기록된 정지 자세로 제한 범위 내 참조 궤적 계산 |
| `tools/plot_multiskill_validation.py` | 검증 지표와 선택 체크포인트까지 학습 곡선·CSV 생성 |

manifest의 `controller`는 사용한 설정의 기록입니다. 실제 스케줄러 값은 `multiskill.py`에 있고, 후진 명령 보정 값은 체크포인트의 `infos.control_profile`에 있습니다. manifest 숫자만 바꿔도 제어기가 바뀌는 구조는 아닙니다.

## 구현 및 학습

전진 정책 SHA-256은 `8ed31af1c07788fe1c4612af80e04bc1027163fd8b8bc05468a18111d113d284`로 이전 결과와 같습니다. 전진만 재생하면 기존 `V2Control.process`를 직접 실행합니다. 별도 후진/자세 학습이 이 가중치를 덮어쓰지 않습니다.

후진은 `Isaac-Open-Duck-Backward-Direct-v0`에서 학습했습니다. 음수 명령의 gait 속도 정규화가 잘못 작동하지 않도록 상한 대신 명령 범위의 절댓값 최대를 사용했습니다. 기존 정책은 −0.1m/s를 줘도 거의 정지했으며 후진 PPO를 적용한 후 실제 후진이 생겼습니다. 선택 모델은 `logs/rsl_rl/open_duck_v2_backward/2026-11-10_04-25-43_backward_v1/model_100.pt`입니다. 선택 시점까지 101회 수집 × 512환경 × 32스텝 = 1,654,784개 새 샘플을 사용했고, 초기 40회는 critic 준비, 이후 61회는 actor도 학습했습니다. actor는 SiLU [512,256,128] / tanh, 학습률 1e-4, 탐색 표준편차 0.12입니다.

후진 정책에는 후처리로 x 명령 배율 0.55, y 보정 +0.006m/s, IMU 방향 유지 kp=2 / 최대 0.35rad/s / 필터 0.25초를 적용했습니다. 명령·방향 보정은 추가 신경망 학습과 구분합니다. 후진에서 정지 actor로 즉시 바꾸면 yaw가 커지는 문제가 있어 1초 동안 두 actor 출력을 연결했습니다. 전진에서 정지할 때는 기존 처리를 유지합니다.

자세는 양발을 고정하는 IK 참조와 PPO 잔차 제어를 결합했습니다. 서 있는 몸통 높이 참조는 0.165629m, 낮은 목표는 0.130m입니다. 머리·목의 질량을 포함한 전체 CoM을 계산하고, 내려갈 때 CoM을 발 지지 영역 안에서 뒤로 20mm 이동하도록 참조를 계산했습니다. 낮은 자세에서는 몸통이 약 25.5° 기울며 머리·목도 함께 보상합니다. 수직으로만 내려가는 자세나 바닥에 엉덩이를 대는 동작은 아닙니다. 더 깊게 앉으면 기존 무릎 범위에 가까워지므로 이 높이를 사용했습니다.

관절 참조는 기존 관절 제한에서 0.022rad 여유를 둔 20개 knot입니다. 4초 내려가기, 3초 낮은 자세 유지, 4초 올라가기, 앞뒤 정지로 16초 주기입니다. PPO는 참조에 ±0.12rad 이내 보상을 더합니다. 정지 구간에 보행 위상이 흔들리지 않도록 자세 actor의 위상 입력은 [1,0]입니다. 높이와 높이 속도 두 입력을 추가하여 actor 103 / critic 126의 별도 계약을 사용합니다. 기존 101입력 정책과 서로 잘못 로딩하면 거부합니다.

자세 선택 모델은 `logs/rsl_rl/open_duck_v2_posture/2026-11-10_04-33-25_posture_residual_v1/model_50.pt`입니다. 51회 × 256환경 × 32스텝 = 417,792개 샘플을 사용했습니다. 참조 위의 초기 보상 출력을 0으로 시작해 ELU [256,256,128] / tanh를 학습했고, 학습률 5e-5 / 탐색 표준편차 0.04입니다. 전체 실험에서는 선택 시점 이후 후보도 학습했지만 최종 정책에는 선택한 가중치만 사용합니다.

전체 질량 2.107141kg, 머리·목 0.581535kg, 관성, 발/몸통/머리 충돌, 관절 제한과 모터 모델은 유지했습니다. PD kp17.8 / damping0.6 / torque3.35Nm, 마찰·백래시, 50Hz 제어 / 500Hz 물리, 다리 목표 속도 제한 5.24rad/s / 머리 1rad/s를 적용합니다. 재생 중 루트 위치·자세를 덮어쓰거나 넘어지면 재시작해서 결과를 이어붙이지 않습니다.

## 검증 결과

seed62/63에서 각 경우 8회씩, 다섯 경우 총 80회 모두 기준을 통과했고 넘어짐은 없었습니다. seed62에서 후진 y 보정과 전환을 조정한 뒤 값을 고정하여 seed63을 별도로 실행했습니다. 반복 자세 시험은 16회 × 3주기 = 48주기이며 전체 전환에도 각 1주기가 포함됩니다.

| 경우 | 통과 | 측정 |
|---|---:|---|
| 정지 20초 | 16/16 | 전진 정책 그대로 사용 |
| 전진 20초 | 16/16 | 2초 이후 평균 vx +0.1076m/s; world 옆 이동 평균 +6.7mm |
| 후진 20초 | 16/16 | 2초 이후 평균 vx −0.1030m/s; world 옆 이동 평균 −96.3mm, 최대 절댓값 136.9mm |
| 낮은 자세·기립 48초 | 16/16 | 높이 평균 164.8→128.3→164.5mm; 복귀 오차 평균 0.31mm, 최대 1.11mm |
| 52초 전체 동작 | 16/16 | 기립 후 전진 재개 포함; 최대 방향 오차 6.52° |

자세 시험의 낮은 유지 구간은 양발 접촉 100%, 초기 2초 안정화 후 발의 수평 이동은 최대 1.29mm, 높이 RMSE 평균 2.26mm, 토크 포화 비율 0%였습니다. 후진에는 옆 이동이 남아 있습니다. 완전한 직선 추종으로 표현하지 않으며, 긴 거리에서는 누적 편차가 커질 수 있습니다.

보행 판정은 기존 전진 기준을 유지했습니다: 20초 생존, 2초 이후 명령 방향 속도 0.08~0.12m/s, 한 주기 XY RMSE ≤0.04m/s, yaw rate MAE ≤0.06rad/s, 이동 거리 명령의 80~120%, world 옆 이동 ≤0.15m, 최종 heading ≤0.12rad, 최대 heading ≤0.15rad, 양발 각각 3회 이상 안정 착지와 10mm 이상 들림입니다. 자세는 25mm 이상 낮추기, 복귀 높이 오차 ≤10mm, 낮은 구간 양발 접촉 ≥95%, 발 이동 ≤25mm, 높이 RMSE ≤10mm, 몸통 기울기 ≤0.70rad 등입니다. 이 기준은 프로젝트의 공학적 판정이며 인증 기준은 아닙니다.

별도 seed64의 52초 실제 렌더링 재생도 1/1회 통과했습니다. `validation/multiskill_2026-11-10/video_demo/playback.mp4`에 원본 영상이 있습니다.

기준/판정/원자료는 `validation/multiskill_2026-11-10/smooth_seed62/`와 `smooth_seed63/`에 있습니다. `rollout.npz`는 raw 속도, pose, 관절, 발 접촉/위치, 명령, 실제 actor 입력/출력, 모터 목표를 보존합니다. TensorBoard 원본은 각 학습 run에 있고, 그래프와 CSV는 선택한 iteration100/50까지만 표시합니다.

회귀 검사 47개 통과. 세 actor 각각 실제 시뮬레이터 입력 2,048개에서 RSL-RL 출력 오차 0, 기록된 GPU 출력 대비 최대 오차 5.67e-7 이하, TorchScript 오차 0, ONNX 64입력 대비 최대 오차 4.33e-7 이하입니다. 단독 ONNX에는 정책 선택·IMU 방향 처리·참조 궤적·모터 목표 처리가 포함되지 않으므로 전체 런타임이 함께 필요합니다.

## 재현

```bash
cd /home/happy/openduck_ws/open-duck-isaac-lab
./run.sh train-skill backward --headless
./run.sh train-skill posture --headless
# 학습은 별도 실험을 만듭니다. 실행용 정책을 자동 덮어쓰지 않습니다.
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/multiskill_policy.json --cases stand forward backward posture demo --trials 8 --eval-seed 63 --output validation/my_multiskill
./run.sh python tools/assess_multiskill.py --evaluation validation/my_multiskill/evaluation.json --output validation/my_multiskill/assessment.json
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/multiskill_policy.json --cases demo --trials 1 --eval-seed 64 --video --output validation/my_multiskill_video
./run.sh python tools/validate_skill_exports.py --rollout validation/my_multiskill/rollout.npz --output validation/my_multiskill_exports
./run.sh test
```

소스 ZIP에는 정책/로그/생성 USD가 제외됩니다. 새 동작을 복원하려면 세 `.pt`를 `policies/`에 두고 참조 JSON과 manifest를 함께 사용하세요. 다른 폴더로 복사하면 USD를 다시 변환해야 합니다. 실물 모터를 직접 구동하는 프로그램은 이번 작업에 포함하지 않았습니다.
