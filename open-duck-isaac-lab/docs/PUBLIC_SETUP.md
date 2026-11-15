# GitHub 배포본 설치와 첫 실행

2026-11-15 · Linux NVIDIA GPU PC · 기본 Open Duck Mini v2

이 저장소는 Raspberry Pi 런타임과 Isaac Lab 학습 패키지를 함께 보관한다.
GPU 학습·재생은 `open-duck-isaac-lab/`에서 한다. 기존 실물 실행 명령과 혼용하지 않는다.

## 1 저장소 복제

```bash
git clone https://github.com/happylee8282/openduck_v2_ws.git ~/openduck_v2_ws
```

```bash
cd ~/openduck_v2_ws
```

## 2 Isaac 환경 준비

실제로 검증한 조합은 Python 3.11, Isaac Sim 5.0.0.0, Isaac Lab 2.2.0,
RSL-RL 2.3.3, PyTorch 2.7.0+cu128, RTX 5070 / 드라이버 580.178.04다.
환경이 없다면 [Isaac Lab 2.2 설치 문서](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html)를 먼저 따른다.
저장소 복제는 Isaac Sim과 GPU 드라이버를 설치하지 않는다.

현재 PC의 기존 `env_isaaclab` 환경을 쓰는 예시다. 다른 PC에서는 실제 설치 경로로 바꾼다.

```bash
export ISAAC_PYTHON="$HOME/miniconda3/envs/env_isaaclab/bin/python"
```

```bash
export ISAACLAB_PATH="$HOME/IsaacLab"
```

```bash
"$ISAAC_PYTHON" --version
```

```bash
nvidia-smi
```

## 3 검증 정책 받기

```bash
python3 open-duck-isaac-lab/tools/download_policies.py
```

GitHub Release `v0.2.0-isaac-multiskill`의 고정 ZIP을 받고, ZIP 및 세 `.pt`의 SHA-256을 검사하여
`open-duck-isaac-lab/policies/`에 설치한다. 기존 파일이 같은 해시면 건너뛰며,
다른 파일이면 덮어쓰지 않고 중단한다. 설정 JSON은 저장소에 있는 값을 유지한다.

직접 [정책 ZIP](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill)을 받은 경우에는 다음처럼 지정한다.

```bash
python3 open-duck-isaac-lab/tools/download_policies.py --archive ~/Downloads/open-duck-v2-multiskill-policies.zip
```

```bash
python3 open-duck-isaac-lab/tools/download_policies.py --check-only
```

## 4 자산 변환과 검사

```bash
./duck.sh help
```

```bash
./duck.sh convert --headless
```

```bash
./duck.sh test
```

```bash
./duck.sh check --headless --steps 32 --output validation/public_setup_check.json
```

`validation/` 출력은 하위 학습 패키지에 저장된다. USD는 로컬 경로를 포함할 수 있으므로
새 PC나 위치에서는 변환한다. `test`는 CPU 코드 검사이며 `check`가 실제 Isaac 물리 경로를 검사한다.

## 5 재생과 학습

```bash
./duck.sh play-skills --demo
```

52초 정지→전진→정지→후진→정지→낮은 자세→기립→전진→정지를 재생한다.
직접 조작은 `./duck.sh play-skills`, W/S 전후진, 해제·Space 정지, C 낮추기, R 기립이다.
창은 GPU PC의 그래픽 세션에 열린다. SSH만 쓴다면 아래처럼 창 없이 실행한다.

```bash
./duck.sh play-skills --headless --steps 128 --report validation/public_play_smoke.json
```

새 학습은 [한 줄씩 CLI 튜토리얼](CLI_TUTORIAL.md)의 3장부터 진행한다.
그 문서는 하위 `open-duck-isaac-lab/`에서 `./run.sh`를 쓰며, 저장소 루트의 `./duck.sh`와 같은 진입점이다.

## 6 검증 결과와 실물 적용 범위

평지 기준 물성의 기존 80회 시험과 공개 시 CPU 검사 결과를 함께 제공한다.
실물 검증, 새 후진·자세 동작의 물성 랜덤화 검증, Raspberry Pi 3정책 통합은 완료되지 않았다.
이번 공개는 원격 실물 로봇을 실행하거나 모터를 구동하지 않는다.
가이드 PDF, 10분 발표자료, 설명 대본과 실제 52초 영상은 [Release](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill)에서 받는다.
