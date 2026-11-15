# Open Duck Mini v2 Isaac Lab

기본 STS3215 7.4V v2 구성을 기준으로 관절 좌표, 모터 모델, 정책 입력과 제어 주기를 수정한 브랜치입니다. 원본 기준은 `2e85abe5f22b61e9e121f74f91823a814e02734b`, 작업 브랜치는 `fix/v2-sim2real`입니다.

[CLI 한 줄씩 따라 하는 튜토리얼](CLI_TUTORIAL.md): 기존 모델 재생 → 짧은 학습 테스트 → 전진 본학습 → 새 후보 재생·평가·영상 저장. 후진·자세 학습도 실제 명령으로 따라 할 수 있습니다.

[전체 패키지 실행과 RL 인수인계 가이드](PACKAGE_RL_GUIDE.md): 처음 실행하는 방법, 관측·행동·보상과 PPO, 새 학습·평가·후보 선택, 다른 PC 복원과 실물 적용 준비를 순서대로 설명합니다.

**2026-11-04 실제 검증:** RTX 5070에서 USD 변환, 평지 250스텝, 험지·모방학습 각 64스텝, 랜덤화 PhysX readback, 소규모 PPO 학습 3회와 정책 재생 128스텝을 확인했습니다. TorchScript·ONNX 변환도 검사했습니다. 재생 시험에서 3회 넘어졌으므로 이 체크포인트는 학습 경로 검사용입니다. 실물용 보행 정책 학습을 완료한 상태는 아닙니다.

실물 계측값은 아직 없습니다. `config/calibration.template.json`의 미측정 값은 검증 도구가 `incomplete`로 처리합니다. [설정·변경과 남은 계측](SIM2REAL.md), [실제 검증 요약](../validation/summary.json)을 확인하세요.

## 2026-11-10 후진·양발 낮은 자세 / 기립 추가 (최신)

기존 전진·정지 가중치를 보존하고 후진 PPO와 IK 참조 위의 자세 보상 PPO를 별도로 학습했습니다. 기준 물성 평지에서 seed62/63 × 정지·전진·후진·자세·전체 전환 × 각8회 = **80/80회가 기준을 통과했습니다.** 반복 자세는 양발을 지면에 둔 상태로 평균 높이 164.8→128.3→164.5mm입니다. 후진 정상 구간 평균 속도는 −0.1030m/s이며 20초에 평균 96.3mm의 옆 이동이 남습니다. 새 동작의 실물·랜덤화 강건성은 검증하지 않았습니다.

```bash
cd ~/openduck_ws
./duck.sh play-skills          # W 전진 / S 후진 / 해제·Space 정지 / C 낮은 자세 / R 기립
./duck.sh play-skills --demo   # 52초: 정지→전진→정지→후진→정지→낮은 자세→기립→전진→정지
```

선택 파일은 `config/multiskill_policy.json`으로 고정합니다. 기존 `./duck.sh play`는 전진·정지 전용입니다. [구현·파일 구조·학습 설정·판정 기준·재현 방법](MULTISKILL.md), `validation/multiskill_2026-11-10/smooth_seed62/assessment.json`과 `smooth_seed63/assessment.json`에서 세부 결과를 확인할 수 있습니다.

## 2026-11-10 직진·정지 개선 결과 (보존된 정책)

**기준 물성에서 정지·직진·정지 전환 48/48회가 기준을 통과했습니다.** 선택 정책은 `policies/flat_straight_v2.pt`이며 `config/validated_flat_policy.json`의 SHA-256으로 고정합니다. `./duck.sh play`는 이후 저장된 실험 체크포인트 대신 이 정책을 사용합니다.

선택 정책은 공식 Mini의 `BEST_WALK_ONNX_2`를 동일 출력 SiLU/tanh actor로 가져온 뒤 PPO로 미세 학습한 `aligned_gait_heading_v6/model_100.pt`가 기반입니다. 선택 시점까지 새 상호작용 3,309,568개(초기 40회 critic 준비)를 사용했습니다. 전체 실험은 600회까지 수행했지만 나중 가중치로 바꾸지 않았습니다. **최종 성능에는 학습된 보행 정책, IMU 자세를 이용한 방향 유지 제어, 전진 명령 배율 0.85가 함께 포함됩니다.** 후처리 설정을 새 신경망 학습으로 표기하지 않습니다.

| 검사 | 별도 seed44 결과 |
|---|---|
| 정지 / 직진 / 정지→직진→정지 | 각 8/8회 통과, 각 시험 20초 |
| 직진 명령 0.100m/s | 평균 0.1064m/s |
| 20초 직진 world 이동 | 전진 2.133m, 옆 이동 평균 7.5mm (범위 -8.9~23.9mm) |
| 방향 | 최종 오차 평균 -2.82°, 전체 시험 최대 절대 오차 4.86° |
| 보행 1주기 XY RMSE | 0.00990m/s; raw 순간 RMSE 0.13327m/s도 보존 |
| 정지 명령 / 걷다가 정지 | 정상 구간 평균 평면 속도 0.00054 / 0.00060m/s |
| 양발 들림 | 왼발 약 22.3mm, 오른발 약 13.4mm (링크 기준 근사) |

머리·목 질량 0.581535kg(전체 2.107141kg의 27.6%)과 원본 관성을 유지했습니다. 목 전체 고정은 균형을 악화시켜 제한된 보상 동작을 유지합니다. 발 접촉 보상이 공식 gait와 약 1/4주기 어긋난 문제를 수정한 뒤 양발 들림이 개선됐습니다. IMU 방향 유지는 관절 명령만 바꾸며 루트 위치·자세를 강제로 고정하지 않습니다. 실제 BNO055 자세를 canonical wxyz로 전달하는 명령 전처리가 필요합니다.

**실물 전이는 미완료입니다.** 질량·마찰·토크·잡음·0–60ms 지연을 함께 바꾼 스트레스 시험은 정지 8/8, 직진 5/8, 전환 5/8회 통과했습니다(전환 생존 6/8). 이 범위에서는 정상 보행을 보장할 수 없습니다. 실측 계측과 강건화 학습이 남아 있습니다.

```bash
cd ~/openduck_ws
./duck.sh play
./duck.sh train-straight --headless
# 물성·센서·지연 랜덤화를 포함하는 별도 강건화 학습
./duck.sh train-straight --headless --randomized
```

재생은 원격 PC의 그래픽 세션에서 W로 전진, 키 해제 또는 Space로 정지합니다. 이 정책은 전진·정지 전용이며 후진·측진·회전 명령 성능은 검증하지 않았습니다. 일반 SSH에서 실행한 창이 화면에 자동 표시되지는 않습니다.

원자료는 `validation/straight_walk_2026-11-10/`에 있습니다. `selected_policy.json`은 계보·설정, `candidate_seed44/assessment.json`은 시험별 판정, `final_video/playback.mp4`는 실제 3초 정지→12초 직진→5초 정지 렌더링, `selected_curves/`는 선택 시점까지의 학습 지표입니다. 내보내기 출력 일치는 실제 재생 입력 2,048개로 검증했고 RSL-RL/TorchScript 오차 0, ONNX 최대 오차 약 3.58×10⁻⁷입니다. CPU 회귀 검사 40개가 통과했습니다.

```bash
./duck.sh python tools/evaluate_flat.py --headless --checkpoint policies/flat_straight_v2.pt --commands stand forward --include-transition --trials 8 --duration 20 --eval-seed 44 --output validation/my_straight_test
./duck.sh python tools/assess_straight_walk.py --evaluation validation/my_straight_test/evaluation.json --output validation/my_straight_test/assessment.json
./duck.sh python tools/evaluate_flat.py --headless --checkpoint policies/flat_straight_v2.pt --sequence stop_go_stop --trials 1 --duration 20 --video --output validation/my_video
```

## 2026-11-10 이전 평지 정책 실패 기록


512개 병렬 환경에서 PPO 학습을 진행했습니다. 최종 선택 정책은 `logs/rsl_rl/open_duck_v2_101_velocity/2026-11-10_00-50-18_flat_corrected/model_699.pt`입니다. 계보상 약 1,150만 샘플을 사용했으며, 마지막 구간은 초기 배치·충돌 필터를 수정한 후의 400회 학습입니다.

**현재 정책은 정상적인 평지 전진 보행 기준에 미달합니다.** 기준 물성에서 전진 시험 8회 모두 20초를 유지했으나, 목표 0.1m/s에 대해 평균 전진 0.111m/s, 측면 -0.134m/s, XY RMSE 0.217m/s, yaw MAE 1.408rad/s였습니다. 직진·정지 명령 추종과 발 디딤을 함께 검사해야 하며 생존·보상 증가만으로 성공으로 판정하지 않습니다.

학습·재생·평가 자료는 `validation/flat_walk_2026-11-10/`에 있습니다. `assessment.json`은 시험별 판정, `final_video/playback.mp4`는 실제 Isaac Sim 재생, `corrected_curves/learning_curves.png`는 TensorBoard 수치에서 생성한 학습 곡선입니다. 평가 도구는 다음과 같이 실행합니다.

```bash
./run.sh python tools/evaluate_flat.py --headless --checkpoint /absolute/path/model.pt --commands stand forward --trials 8 --duration 20 --output validation/my_evaluation
./run.sh python tools/evaluate_flat.py --headless --checkpoint /absolute/path/model.pt --commands forward --trials 1 --duration 20 --video --output validation/my_video
```

이번 과정에서 재질 32개 풀 재사용, 메시 기반 초기 높이 0.172m(지면 여유 약 5.17mm), 가까운 기구 조상 링크 25쌍의 충돌 필터를 추가했습니다. 좌우 다리 등 다른 부위의 자기 충돌은 유지합니다. CPU 회귀 검사 25개가 통과했습니다.

## 원격 PC의 터미널에서 실행

2026-11-09에 작업 폴더를 `/home/happy/openduck_ws/open-duck-isaac-lab`로 이동했습니다. 소스, Git 작업 상태, USD, 모션 파일과 학습 체크포인트를 함께 이동했습니다. 이전 경로는 이 폴더를 가리키는 심볼릭 링크입니다.

```bash
cd ~/openduck_ws
./duck.sh check --headless --steps 32
./duck.sh train --headless --num_envs 32 --max_iterations 3 --seed 42
./duck.sh play --headless --steps 128
```

`duck.sh`는 프로젝트의 `run.sh`를 가리키며, `(base)` 프롬프트에서도 Isaac Lab용 Python을 직접 사용합니다. `conda activate`는 필수가 아닙니다. `play`는 검증 정책 manifest가 있으면 해당 정책을 사용하며, 없을 때는 가장 최근 체크포인트를 선택합니다. `--checkpoint /absolute/path/model.pt`로 선택할 수도 있습니다. 위 학습 3회는 실행 확인용입니다.

`--headless`는 창 없이 원격 GPU에서 실행합니다. 창을 보려면 원격 PC의 그래픽 세션에 접속한 터미널에서 `./duck.sh play`를 실행하세요. 일반 SSH 터미널에서 실행한 창이 사용자의 화면에 자동 표시되지는 않습니다. 전체 명령은 `./duck.sh help`로 확인합니다.

## 환경

검증 조합: Python 3.11, Isaac Sim 5.0.0.0, Isaac Lab 2.2.0, RSL-RL 2.3.3, PyTorch 2.7.0+cu128, RTX 5070, 드라이버 580.178.04. 버전은 `validation/environment.json`에 기록합니다.

이 저장소는 **Isaac Lab의 Python 3.11**로 실행합니다. 상위 Mini/Playground의 Python 3.12 MuJoCo 환경과 별도입니다. 현재 컴퓨터에서는:

```bash
cd /home/happy/openduck_ws/open-duck-isaac-lab
export ISAACLAB_PATH=/home/happy/IsaacLab
export ISAAC_PYTHON=/home/happy/miniconda3/envs/env_isaaclab/bin/python
```

다른 컴퓨터에서는 설치한 Isaac Lab과 Python 경로를 사용하세요. 저장소 루트에서 별도 설치 없이 도구를 실행할 수 있습니다. 패키지 설치를 원하면 Isaac 환경에서 `python -m pip install -e . --no-deps`를 실행합니다.

## 모델과 검사

```bash
"$ISAAC_PYTHON" tools/inspect_urdf.py
"$ISAAC_PYTHON" tools/convert_urdf_to_training_usd.py --headless
"$ISAAC_PYTHON" tools/validate_assets.py --output validation/assets.json
"$ISAAC_PYTHON" tools/validate_isaac.py --headless --output validation/isaac.json
"$ISAAC_PYTHON" tools/validate_isaac.py --headless --task rough --steps 64 --output validation/isaac_rough.json
PYTHONPATH="$PWD" "$ISAAC_PYTHON" -m unittest discover -s tests -v
```

변환한 `Robots/open_duck_mini_training.usd`가 현재 작업 폴더에 있습니다. 생성물은 Git에 포함하지 않습니다. 다른 위치로 복사하거나 URDF를 수정하면 다시 변환하세요. 실제 PhysX 검사는 GPU가 보이지 않으면 `blocked`, 오류면 `failed`, 수행된 검사가 통과하면 `passed`를 기록합니다. CPU test double 결과는 실제 GPU 결과와 구분합니다.

## 학습

```bash
"$ISAAC_PYTHON" tools/launch.py list
"$ISAAC_PYTHON" tools/launch.py train --headless --num_envs 32 --max_iterations 3 --seed 42
```

위 명령은 학습 경로 검사입니다. 실제 성능을 학습하려면 충분한 반복 수와 여러 seed의 평가가 필요합니다. 기본은 평지이며 험지는 다음 task를 지정합니다.

```bash
"$ISAAC_PYTHON" tools/launch.py train --headless --task Isaac-Open-Duck-Rough-Direct-v0 --num_envs 64
```

`launch.py`가 task를 명시적으로 등록하므로 Isaac Lab 원본을 수정할 필요가 없습니다. 검증한 경로는 RSL-RL PPO입니다. 기존 AMP/SB3/RL-Games YAML은 활성 registry에서 제외했습니다.

체크포인트는 `logs/rsl_rl/open_duck_v2_101_velocity/<run>/`에 저장합니다. save hook이 정책 계약, 정규화 여부, phase 주기를 기록합니다. 기존 158/66차원 정책은 입력 의미가 달라 재학습해야 합니다.

## 내보내기와 재생

```bash
"$ISAAC_PYTHON" isaac_duck_teleop/scripts/export_open_duck_policy.py --checkpoint /absolute/path/model.pt --output /absolute/path/policy.pt
"$ISAAC_PYTHON" isaac_duck_teleop/scripts/export_open_duck_policy.py --checkpoint /absolute/path/model.pt --output /absolute/path/policy.onnx
"$ISAAC_PYTHON" isaac_duck_teleop/scripts/play_open_duck_teleop.py --project-root "$PWD" --checkpoint /absolute/path/model.pt --usd-path "$PWD/Robots/open_duck_mini_training.usd"
```

ONNX 생성에는 `onnx`, 출력 비교에는 `onnxruntime`이 필요합니다. 학습한 normalizer는 내보낸 모델 안에 포함합니다. 입력은 raw 101차원이며 예전 계약의 체크포인트는 거부합니다.

W/S 전후, A/D 좌우, Q/E 회전, Space 속도 초기화, Esc 종료입니다. 높이 키는 제거했습니다. 기본 정책에는 높이 명령이 없습니다. 자동 검사는 `--headless --steps 128 --report validation/playback.json`을 추가합니다.

## 모방학습

```bash
"$ISAAC_PYTHON" tools/download_cmu_mocap.py --clip cmu_05_01 --output data/motions/05_01.bvh
"$ISAAC_PYTHON" tools/retarget_bvh_to_open_duck.py data/motions/05_01.bvh --output data/motions/open_duck_cmu_imitation.npz --target-fps 50
"$ISAAC_PYTHON" tools/validate_isaac.py --headless --task imitation --steps 64 --output validation/isaac_imitation.json
```

고정 커밋·SHA-256으로 [CMU MoCap](http://mocap.cs.cmu.edu/)의 [BVH 변환 미러](https://github.com/Shriinivas/cmubvh)를 확인하고 출처를 기록합니다. 정확한 50Hz 보간, 이름 기반 매칭, canonical 관절 범위와 목표 속도를 검사합니다. 너무 빠르면 시간축을 늘립니다. 파일 누락 시 사인파를 자동 대입하지 않습니다.

이 retargeter는 관절/위상 기반 실험이며 전체 접촉·IK·동역학 파이프라인은 별도 보완 항목입니다.

## 실물 계측과 로그

```bash
"$ISAAC_PYTHON" tools/validate_calibration.py config/calibration.template.json --output validation/calibration.json
"$ISAAC_PYTHON" tools/replay_hardware_log.py --checkpoint /absolute/path/model.pt --input /absolute/path/sensors.jsonl --output validation/replayed_targets.jsonl
```

실측 전 템플릿은 `incomplete`입니다. 실제 중량, 서보 응답·포화·마찰, 지연, 관절 영점·부호, IMU 축과 발 좌우를 기록해야 합니다. 로그 재생은 모터 I/O 없이 관측과 목표각을 검사합니다. 상세 형식은 [SIM2REAL.md](SIM2REAL.md)에 있습니다.

소스 ZIP은 로그·생성 USD·체크포인트를 포함하지 않습니다. 별도로 제공된 `flat_straight_v2.pt`를 `policies/`에 두면 manifest SHA-256을 확인해 사용합니다. 다른 폴더나 PC에서는 USD를 다시 변환하고 Isaac Python 경로를 맞추세요.

원본 라이선스는 [BSD 3-Clause](../LICENSE)입니다. `open_duck_back/`는 과거 실험 보관 영역입니다.
