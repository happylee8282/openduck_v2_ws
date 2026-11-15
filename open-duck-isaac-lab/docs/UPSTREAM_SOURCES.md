# 출처와 배포 범위

| 출처 | 사용 범위 | 기준 커밋/버전 |
|---|---|---|
| [zhangzijie-pro/open-duck-isaac-lab](https://github.com/zhangzijie-pro/open-duck-isaac-lab) | Isaac Lab 환경과 로봇 자산을 포함한 출발 소스 | `2e85abe5f22b61e9e121f74f91823a814e02734b` |
| [apirrone/Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) | 로봇 설계 및 공개 `BEST_WALK_ONNX_2` actor 초기값 | `b23317a485b3cec7d8417f352478778b3475173c` |
| [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) | 좌표·HOME·BAM 기반 모터 참조 | `b9be205ac64488c23504ca42e5ec790337adeec3` |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.2.0) | 물리 및 Direct RL 실행 환경 | 2.2.0 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl/tree/v2.3.3) | PPO 학습 | 2.3.3 |

상류 Isaac 패키지의 [BSD 3-Clause 라이선스](../LICENSE)를 유지한다.
공식 Mini의 Apache 2.0 라이선스는 [원문](OPEN_DUCK_MINI_LICENSE.txt)을 함께 제공한다.
Mini 설계·초기 정책의 원저작자와 상류 기여자를 표기하며, 이 배포가 공식 프로젝트의 보증을 뜻하지 않는다.
Isaac Sim/Isaac Lab 설치물과 외부 라이브러리는 이 저장소에서 재배포하지 않는다.

이 저장소 루트의 기존 Raspberry Pi 런타임과 학습 패키지는 별도 실행 체계다.
세 학습 정책을 실물 런타임에 통합하여 검증한 상태는 아니다.
