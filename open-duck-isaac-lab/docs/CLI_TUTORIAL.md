# Open Duck CLI 한 줄씩 따라 하는 튜토리얼

2026년 11월 15일 · GitHub 배포본 · Linux GPU PC 기준

터미널에 무엇을 입력해야 하는지 순서대로 설명한다. 대상은 `~/openduck_v2_ws/open-duck-isaac-lab`의 Open Duck Mini v2 Isaac Lab 패키지다. Isaac Lab 환경 설치 후 Release의 검증 정책을 내려받아 사용한다. 먼저 기존 모델을 보고, 짧은 학습 테스트 후 새 전진 모델을 학습·재생·평가한다. 후진과 낮은 자세 학습은 뒤의 선택 과정에서 진행한다.

각 코드 상자는 **명령 한 줄**이다. 한 줄을 붙여 넣고 Enter를 누른다. 명령이 끝나 터미널 프롬프트가 다시 나온 뒤 다음 명령을 입력한다. `(base) happy@...$` 같은 프롬프트는 복사하지 않는다. 긴 명령이 화면에서 여러 줄로 보이더라도 하나의 명령이다. 명령 복사는 HTML의 복사 버튼이나 TXT 원본을 사용하면 실제 한 줄이 유지된다.


## GitHub에서 처음 받았을 때

먼저 [공개 배포본 설치 안내](PUBLIC_SETUP.md)대로 저장소 복제, Isaac Lab Python 경로 설정,
정책 다운로드, USD 변환을 완료한다. 이후 아래 1단계부터 진행한다.
아래 예시는 저장소를 `~/openduck_v2_ws`에 복제한 경우다. 다른 위치라면 첫 `cd`만 바꾼다.
원래 개발 PC의 작업본 `~/openduck_ws`와 공개 배포본은 별도 폴더다.

## 1 시작할 폴더로 이동

### 1단계 폴더로 이동

```bash
cd ~/openduck_v2_ws/open-duck-isaac-lab
```

### 2단계 현재 위치 확인

```bash
pwd
```

출력이 `~/openduck_v2_ws/open-duck-isaac-lab`이면 맞다. 이 튜토리얼은 이 폴더에서 `./run.sh`를 사용하는 방식으로 통일한다. `~/openduck_v2_ws`에서 쓰는 `./duck.sh`도 이 스크립트를 가리킨다.

### 3단계 명령 목록 확인

```bash
./run.sh help
```

`play-skills`, `train-straight`, `train-skill`, `test`, `check` 등이 표시된다. `(base)` 상태에서도 실행할 수 있다. `run.sh`가 Isaac Lab의 Python 3.11을 직접 선택한다.

## 2 이미 학습된 로봇부터 보기

### 4단계 자동 데모 보기

```bash
./run.sh play-skills --demo
```

Isaac 창에서 약 52초 동안 정지→전진→정지→후진→정지→낮은 자세→기립→다시 전진→정지를 재생한다. 처음 실행은 자산·셰이더 로딩 때문에 시간이 더 걸릴 수 있다. 이 명령은 저장된 정책을 재생한다.

### 5단계 직접 조작

```bash
./run.sh play-skills
```

창을 클릭해 키보드 포커스를 주고 다음 키를 누른다.

| 키 | 동작 |
|---|---|
| W | 전진 |
| S | 후진 |
| 키 해제 또는 Space | 보행 정지 |
| C | 몸을 낮추고 유지 |
| R | 일어서기 |
| Esc | 재생 종료 |

C를 놓아도 낮은 자세를 유지한다. 낮은 자세에서 W/S를 누르면 먼저 일어선 뒤 걷는다. 실행이 끝나 터미널 프롬프트가 돌아오면 다음 단계를 진행한다.

### SSH만 사용해 창을 볼 수 없을 때

아래는 창 없이 약 2.56초 실행 경로만 확인하는 선택 명령이다.

```bash
./run.sh play-skills --headless --steps 128 --report validation/cli_play_smoke.json
```

`steps: 128`, `falls: 0`과 같은 실행 결과를 확인한다. GUI는 GPU PC의 그래픽 세션에 열린다. 일반 SSH에서 실행한다고 사용자 PC 화면에 자동 표시되는 것은 아니다. 화면 없이 영상을 저장하는 방법은 8장에 있다.

## 3 학습 전에 환경 확인

### 6단계 GPU 확인

```bash
nvidia-smi
```

GPU 이름과 메모리가 표시되어야 한다. 다른 작업이 메모리를 많이 사용 중이면 새 학습의 환경 수를 줄인다.

### 7단계 CPU 회귀 검사

```bash
./run.sh test
```

마지막이 `OK`인지 확인한다. 이 문서 작성 시 기존 검사 47개가 통과했다.

### 8단계 실제 Isaac 물리 경로 검사

```bash
./run.sh check --headless --steps 32 --output validation/cli_isaac_check.json
```

결과의 `status`가 `passed`인지 확인한다. 이는 USD·관측·질량·토크·물리 연결 검사이며, 새 학습 정책이 보행에 성공했다는 판정은 아니다.

모델 파일이 없거나 폴더를 옮겨 USD 경로 오류가 날 때만 아래 명령으로 다시 변환한다.

```bash
./run.sh convert --headless
```

변환이 끝나면 8단계 검사를 다시 실행한다.

## 4 짧은 학습 테스트

### 9단계 3회만 실행

```bash
./run.sh train-straight --headless --num_envs 32 --max_iterations 3 --run_name cli_smoke_v1 agent.save_interval=1
```

학습 루프가 실행되고 `logs/rsl_rl/open_duck_v2_straight/` 아래에 `cli_smoke_v1`로 끝나는 run 폴더와 `model_*.pt`가 생성되는지 확인한다.

| 인자 | 뜻 |
|---|---|
| `--headless` | 학습 창 없이 실행 |
| `--num_envs 32` | 로봇 32개를 병렬로 실행 |
| `--max_iterations 3` | PPO 갱신 루프를 3회 실행 |
| `--run_name cli_smoke_v1` | 실험을 구분할 이름 |
| `agent.save_interval=1` | 이 짧은 테스트에서는 매회 저장 |

이 단계는 실행·저장 경로를 보는 테스트다. 전진 helper는 기존 actor에서 시작하며 초기 40회는 critic 준비 구간이므로, 3회만으로 새 보행 actor가 학습됐다고 해석하지 않는다.

### 10단계 생성된 실험 확인

```bash
ls -td logs/rsl_rl/open_duck_v2_straight/*_cli_smoke_v1
```

날짜·시각이 붙은 실제 run 폴더가 출력된다. 학습이 끝나 프롬프트가 돌아오면 다음 장으로 진행한다.

## 5 전진과 정지 본학습

### 11단계 새 전진 실험 실행

```bash
./run.sh train-straight --headless --num_envs 256 --max_iterations 120 --seed 81 --run_name cli_forward_v1
```

기존 검증 actor를 초기값으로 사용하는 새 실험이다. `policies/flat_straight_v2.pt`와 기본 실행 manifest는 그대로 보존하고, 새 후보는 `logs/`에 저장한다. 120회는 따라 하기용 학습량이며 새 모델의 성공을 보장하는 횟수는 아니다.

터미널에는 iteration, reward, episode length, 저장 경로가 표시된다. 완료될 때까지 기다린다. 중단하려면 해당 터미널에서 Ctrl+C를 누른다. 저장 시점 이후의 경험은 남지 않을 수 있다.

### GPU 메모리가 부족할 때

오류로 실행이 끝났다면 환경 수를 줄이고 다른 실험 이름으로 다시 시작한다. 아래는 11단계의 대안이다.

```bash
./run.sh train-straight --headless --num_envs 64 --max_iterations 120 --seed 81 --run_name cli_forward_small_v1
```

이 대안을 선택했다면 뒤의 후보 생성 명령에서도 `--run-name cli_forward_small_v1`을 사용한다. 학습 환경 수를 줄이면 iteration당 새 샘플 수도 줄어든다.

## 6 방금 학습한 모델을 재생

### 12단계 날짜 경로를 직접 찾는 대신 후보 JSON 생성

```bash
./run.sh python tools/make_tutorial_bundle.py --skill forward --run-name cli_forward_v1 --output config/cli_forward_v1.json
```

같은 이름의 가장 최근 run에서 숫자가 가장 큰 `model_*.pt`를 찾아 새 후보 JSON을 만든다. 출력의 `checkpoint`에서 실제 선택 파일을 확인한다. 이 도구는 모델 가중치를 바꾸지 않으며, 기존 기본 manifest에 실험 모델을 적용하지 않는다.

최신 파일은 재생을 시작하기 위한 선택일 뿐 가장 좋은 파일이라는 뜻은 아니다. 특정 iteration100을 비교하려면 12단계의 대안으로 다음 명령을 사용한다. 해당 run에 `model_100.pt`가 있어야 한다.

```bash
./run.sh python tools/make_tutorial_bundle.py --skill forward --run-name cli_forward_v1 --iteration 100 --output config/cli_forward_iter100.json
```

학습 명령은 `--run_name`, 후보 생성 도구는 `--run-name`이라는 표기 차이가 있다.

### 13단계 새 후보를 자동 데모로 재생

```bash
./run.sh play-skills --bundle config/cli_forward_v1.json --demo
```

이제 전진·정지는 방금 학습한 후보, 후진·자세는 기존 검증 정책을 사용한다. 새 후보가 못 걷거나 전환에서 넘어지면 그대로 실패 결과로 확인한다.

### 14단계 새 후보 직접 조작

```bash
./run.sh play-skills --bundle config/cli_forward_v1.json
```

W/S/C/R/Space/Esc는 기존과 같다. `--bundle`을 빼고 실행하면 기본 검증 묶음을 사용하므로 새 학습 결과를 보고 있는지 구분한다.

## 7 수치로 새 모델 평가

### 15단계 정지와 전진을 각 8회 검사

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_forward_v1.json --cases stand forward --trials 8 --eval-seed 91 --output validation/cli_forward_seed91
```

첫 에피소드에서 각 20초를 기록한다. 완료되면 `evaluation.json`과 `rollout.npz`가 생성된다.

### 16단계 통과 여부 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_forward_seed91/evaluation.json --output validation/cli_forward_seed91/assessment.json
```

터미널의 `accepted`와 경우별 `accepted / trials`를 확인한다. 이 8회 묶음에서는 각 경우 8/8이어야 통과한다. `accepted: false`이면 학습·보정·후보 선택을 다시 검토한다. 평가 파일 생성과 통과 판정은 다르다.

### 17단계 전체 동작 전환도 검사

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_forward_v1.json --cases stand forward backward posture demo --trials 8 --eval-seed 92 --output validation/cli_forward_full_seed92
```

### 18단계 전체 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_forward_full_seed92/evaluation.json --output validation/cli_forward_full_seed92/assessment.json
```

새 전진 actor가 후진·정지·자세와의 전환도 유지하는지 확인한다. 더 확실히 비교하려면 다른 seed와 다른 output 폴더로 반복한다. 시간 길이를 줄인 시험은 원래 20초·48초·52초 기준의 완전한 검증과 구분한다.

## 8 영상 저장과 확인

### 19단계 창 없이 52초 영상 녹화

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_forward_v1.json --cases demo --trials 1 --eval-seed 93 --video --output validation/cli_forward_video
```

`validation/cli_forward_video/playback.mp4`가 생성된다. `--video`는 case 하나, trial 하나일 때 사용한다. 로봇이 중간에 넘어졌다면 영상과 기록도 그 실패를 포함한다.

### 20단계 영상 시험도 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_forward_video/evaluation.json --output validation/cli_forward_video/assessment.json
```

### 21단계 GPU PC에서 영상 열기

```bash
xdg-open validation/cli_forward_video/playback.mp4
```

그래픽 세션이 있어야 플레이어 창이 열린다. SSH만 사용 중이면 이 MP4를 사용자 PC로 복사해 재생한다.

## 9 학습 지표 보는 별도 터미널

학습을 실행한 터미널 A는 그대로 두고 터미널 B를 새로 연다.

### B1단계 동일한 폴더로 이동

```bash
cd ~/openduck_v2_ws/open-duck-isaac-lab
```

### B2단계 TensorBoard 실행

```bash
./run.sh python -m tensorboard.main --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

이 터미널은 계속 켜 둔다. GPU PC의 브라우저 주소창에 `http://127.0.0.1:6006`을 입력한다. 원격 PC에서 볼 때는 SSH 포트 전달을 사용한다.

먼저 `Train/mean_reward`, `Train/mean_episode_length`, `Stats/velocity_along_command`, `Stats/heading_error_abs_rad`를 본다. 자세 학습에서는 `Stats/height_error_abs_m`, `Stats/both_feet_contact`를 본다. reward가 올라도 실제 명령 추종에 실패할 수 있으므로 7장의 평가를 함께 한다.

TensorBoard를 종료할 때는 터미널 B에서 Ctrl+C를 누른다. 학습 자체는 터미널 A에서 별도 실행 중이면 계속 진행된다.

## 10 선택 과정 후진 학습

전진 과정과 독립적으로 실행할 수 있다. 이 장에서 만드는 bundle의 전진·자세 슬롯은 기본 검증 모델을 사용한다.

### 후진 1단계 새 실험 학습

```bash
./run.sh train-skill backward --headless --num_envs 256 --max_iterations 120 --seed 82 --run_name cli_backward_v1
```

### 후진 2단계 후보 연결

```bash
./run.sh python tools/make_tutorial_bundle.py --skill backward --run-name cli_backward_v1 --output config/cli_backward_v1.json
```

### 후진 3단계 재생

```bash
./run.sh play-skills --bundle config/cli_backward_v1.json --demo
```

### 후진 4단계 20초 후진 평가

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_backward_v1.json --cases backward --trials 8 --eval-seed 91 --output validation/cli_backward_raw_seed91
```

### 후진 5단계 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_backward_raw_seed91/evaluation.json --output validation/cli_backward_raw_seed91/assessment.json
```

학습 원본은 현재 검증 후진 모델의 명령 보정이 그대로 적용된 파일이 아니다. 너무 빠르거나 옆으로 흐르면 아래 보정 과정을 비교 실험할 수 있다.

### 선택 과정 후진 명령 보정

다음은 기존 검증 모델에서 사용한 배율0.55, 측면 보정+0.006m/s를 새 후보에 비교해 보는 예시다. 새 actor에도 적합하다고 보장하는 값은 아니므로 적용 후 다시 평가한다.

먼저 현재 선택한 원본 체크포인트 경로를 변수에 저장한다. 이 명령은 파일을 변경하지 않는다.

```bash
DUCK_BACKWARD_CKPT="$(./run.sh python tools/make_tutorial_bundle.py --skill backward --run-name cli_backward_v1 --print-checkpoint)"
```

선택한 경로를 확인한다.

```bash
printf '%s\n' "$DUCK_BACKWARD_CKPT"
```

신경망 가중치는 유지하고 명령 보정 정보가 들어 있는 별도 파일을 만든다. 후진 4단계에서 만든 평가 JSON이 필요하다.

```bash
./run.sh python tools/configure_policy_controller.py --checkpoint "$DUCK_BACKWARD_CKPT" --output work/cli_backward_calibrated.pt --heading-kp 2 --yaw-rate-limit 0.35 --heading-filter-tau 0.25 --forward-scale 0.55 --lateral-trim 0.006 --calibration-evaluation validation/cli_backward_raw_seed91/evaluation.json
```

이 보정 파일을 새 후보 JSON으로 연결한다.

```bash
./run.sh python tools/make_tutorial_bundle.py --skill backward --checkpoint work/cli_backward_calibrated.pt --output config/cli_backward_calibrated.json
```

보정 후 평가한다.

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_backward_calibrated.json --cases backward --trials 8 --eval-seed 92 --output validation/cli_backward_calibrated_seed92
```

판정한다.

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_backward_calibrated_seed92/evaluation.json --output validation/cli_backward_calibrated_seed92/assessment.json
```

원본·보정 후보의 지표를 비교하고, 선택할 후보는 전체 전환과 다른 seed에서도 검사한다. 보정 파일 생성은 추가 PPO 학습과 구분한다.

## 11 선택 과정 낮은 자세와 기립 학습

### 자세 1단계 학습

```bash
./run.sh train-skill posture --headless --num_envs 128 --max_iterations 100 --seed 83 --run_name cli_posture_v1
```

기존 `config/posture_reference_v1.json` 참조 위의 PPO 잔차를 학습한다. 이 참조 파일은 기존 정책 계약에 연결되어 있으므로 현재 파일을 덮어쓰지 않는다.

### 자세 2단계 후보 연결

```bash
./run.sh python tools/make_tutorial_bundle.py --skill posture --run-name cli_posture_v1 --output config/cli_posture_v1.json
```

### 자세 3단계 재생

```bash
./run.sh play-skills --bundle config/cli_posture_v1.json --demo
```

### 자세 4단계 각 3주기씩 평가

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_posture_v1.json --cases posture --trials 8 --eval-seed 91 --output validation/cli_posture_seed91
```

### 자세 5단계 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_posture_seed91/evaluation.json --output validation/cli_posture_seed91/assessment.json
```

양발 접촉, 25mm 이상 낮추기, 높이 복귀와 발 이동 등을 검사한다. 103입력 자세 정책은 101입력 전진 전용 재생 도구에 직접 넣지 않고 이 bundle 경로를 사용한다.

## 12 선택 과정 새 정책 세 개를 한 묶음으로 검사

이 장은 전진·후진·자세를 각각 학습하고 필요한 파일을 만든 뒤 진행한다. 아래 예시는 보정 후 후진 파일 `work/cli_backward_calibrated.pt`를 사용한다. 보정 파일을 만들지 않았다면 이 예시를 그대로 실행하지 않고 선택한 실제 후진 체크포인트를 지정한다.

### 합치기 1단계 새 전진 bundle에 새 후진 연결

```bash
./run.sh python tools/make_tutorial_bundle.py --skill backward --checkpoint work/cli_backward_calibrated.pt --base-bundle config/cli_forward_v1.json --output config/cli_forward_backward_v1.json
```

### 합치기 2단계 새 자세도 연결

```bash
./run.sh python tools/make_tutorial_bundle.py --skill posture --run-name cli_posture_v1 --base-bundle config/cli_forward_backward_v1.json --output config/cli_all_v1.json
```

### 합치기 3단계 새 세 정책 재생

```bash
./run.sh play-skills --bundle config/cli_all_v1.json --demo
```

### 합치기 4단계 전체 40회 평가

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_all_v1.json --cases stand forward backward posture demo --trials 8 --eval-seed 93 --output validation/cli_all_seed93
```

### 합치기 5단계 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_all_seed93/evaluation.json --output validation/cli_all_seed93/assessment.json
```

### 합치기 6단계 다른 초기 조건에서 전체 40회 평가

```bash
./run.sh python tools/evaluate_multiskill.py --headless --bundle config/cli_all_v1.json --cases stand forward backward posture demo --trials 8 --eval-seed 94 --output validation/cli_all_seed94
```

### 합치기 7단계 판정

```bash
./run.sh python tools/assess_multiskill.py --evaluation validation/cli_all_seed94/evaluation.json --output validation/cli_all_seed94/assessment.json
```

새 후보가 이전 검증 결과와 같은 성능을 내는지는 이 시험 결과로 판단한다. 명령을 입력하거나 학습을 완료한 사실만으로 성공을 가정하지 않는다.

## 13 내보내기와 기본 모델로 돌아가기

### 세 actor 출력 내보내기 및 일치 검사

전체 평가를 수행한 후 다음 명령으로 실제 입력에 대한 RSL-RL·TorchScript·ONNX 출력 일치를 확인한다. 위 12장의 전체 시험 자료가 있어야 한다.

```bash
./run.sh python tools/validate_skill_exports.py --bundle config/cli_all_v1.json --rollout validation/cli_all_seed93/rollout.npz --output validation/cli_all_exports
```

### 기본 검증 정책 다시 보기

```bash
./run.sh play-skills --demo
```

튜토리얼의 후보 생성은 기본 선택을 바꾸지 않는다. `--bundle config/cli_...json`을 지정한 경우에만 해당 후보를 사용한다. 후보 도구는 `config/multiskill_policy.json`과 `config/validated_flat_policy.json`에 쓰는 것을 거부한다.

## 14 자주 막히는 곳

| 상황 | 확인과 다음 행동 |
|---|---|
| 명령 다음 줄에 아무것도 입력할 수 없음 | 학습·재생·TensorBoard가 아직 실행 중인지 본다. 기다리거나 해당 프로세스를 종료하거나 다른 터미널을 연다. |
| 창이 안 보임 | `--headless` 여부와 GPU PC의 그래픽 세션을 확인한다. SSH에서는 영상 녹화 과정을 사용한다. |
| `Candidate already exists` | 이미 만든 후보를 보호하는 동작이다. `--output config/cli_forward_v2.json`처럼 새 이름을 쓰고 뒤의 `--bundle`도 그 이름으로 맞춘다. |
| `No run named` | 해당 학습이 실행·저장됐는지, 학습의 `--run_name`과 후보 생성의 `--run-name` 값이 같은지 확인한다. |
| `No saved checkpoint` | 지정 iteration 파일이 없거나 저장 전에 중단했다. `--iteration`을 빼거나 실제 저장 번호를 선택한다. |
| `Checkpoint declares ... requested ...` | 전진·후진·자세 중 다른 종류의 모델을 연결했다. `--skill`과 체크포인트 종류를 맞춘다. |
| GPU 메모리 부족 | `--num_envs`를 줄이고 다른 실험 이름으로 재시도한다. 다른 사용자의 작업은 임의 종료하지 않는다. |
| `contract mismatch` | 101/103입력 종류, 관절 계약, 참조 JSON 해시를 확인한다. 계약 숫자만 바꿔서 우회하지 않는다. |
| 학습했는데 기존처럼 움직임 | 기본 `play-skills`가 아니라 새 후보의 `--bundle`을 지정했는지 확인한다. |
| 같은 평가 폴더에 다시 실행 | 기존 원자료를 보존하려면 새 output 이름을 사용한다. 예: `validation/cli_forward_seed95` |
| 원격 연결을 끊으면 학습이 종료됨 | 끊기 전에 저장된 checkpoint를 확인한다. 세션을 유지해야 하면 별도의 tmux 사용 절차로 실행한다. |

기본적으로 시뮬레이션만 실행한다. 현재 성공 범위는 기준 물성의 평지다. 새 후진·자세의 실물·전체 랜덤화 검증은 별도 작업이며, 실물에 ONNX만 복사해 같은 동작이 완성되는 구조는 아니다.

## 15 오늘 처음 한다면 여기까지

처음에는 1~2장의 기존 정책 재생을 확인한다. 이후 3장의 환경 검사와 4장의 짧은 테스트를 끝내고, 5~8장에서 전진 한 종류를 학습·재생·평가·녹화한다. 이 순서를 익힌 뒤 선택 과정의 후진·자세·전체 묶음으로 넓힌다.

개념과 패키지 구조를 더 읽으려면 [전체 RL 가이드](PACKAGE_RL_GUIDE.md), 동작 구현은 [멀티스킬 문서](MULTISKILL.md), 계측은 [Sim2Real 문서](SIM2REAL.md)를 참고한다. 이 튜토리얼의 새 명령은 `tools/make_tutorial_bundle.py`이며 CPU에서 실제 세 정책의 계약·해시·덮어쓰기 방지를 확인했다. 여기 적힌 새 실험 이름으로 본학습을 대신 실행한 것은 아니므로, 실행 후 자신의 평가 결과를 확인한다.
