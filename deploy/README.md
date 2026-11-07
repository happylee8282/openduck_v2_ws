# Raspberry Pi 런타임 배포

[open-duck-runtime.service](open-duck-runtime.service)는 systemd에서 로봇 사용자로 보행 프로세스를 실행하는 템플릿이다. [run_duck.sh](../run_duck.sh)의 foreground 실행을 사용해 systemd가 실제 프로세스 수명과 로그를 추적하도록 한다. [wait_for_audio.py](wait_for_audio.py)는 오디오 준비를 확인하는 배포 보조 스크립트다.

입력은 로봇의 설치 경로, Python 환경, ONNX 모델과 관절 설정, 직렬·조이스틱 장치다. 출력은 서비스 상태와 저널, 실행 중인 보행 프로세스다. 서비스 파일을 적용하기 전에 루트의 `./run_duck.sh check`로 모델과 장치가 잡히는지 확인한다. 설치·활성화 명령은 [루트 README](../README.md)의 Boot Autostart 절에 있다.
