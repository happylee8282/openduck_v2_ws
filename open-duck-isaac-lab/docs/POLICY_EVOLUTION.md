# Open Duck 보행 개발 3단계와 체크포인트 비교

정리일 2026년 11월 15일 · 측정 원자료 2026년 11월 10일

시간 순서로는 **초기 실패 → 전진·정지 개선 → 후진·자세 확장**의 세 단계다. 최종 실행 정책도 세 개이며 **전진·정지, 후진, 낮은 자세·기립**을 담당한다. 두 번째 단계의 전진 정책은 세 번째 단계에서도 같은 파일과 가중치를 그대로 사용한다.

아래 수치는 저장된 실제 Isaac Lab 평가 JSON·체크포인트·학습 YAML에서 다시 집계했다. 오늘 새로 재학습하거나 재생을 실행해 만든 성능 수치는 아니다. 코드 링크는 **현재 누적 수정본의 관련 구현 위치**이고, 당시 설정의 기준은 각 run의 보존된 `params/env.yaml`, `params/agent.yaml` 및 `.pt`의 `infos`다. 별도의 역사적 코드 커밋 세 개가 존재하는 것으로 해석하지 않는다.

## 1 세 시점의 결과와 파일

| 시점 | 의미 | 실제 사용 파일 | 원자료 |
| --- | --- | --- | --- |
| ① 초기 실패 | 발을 움직이며 버티지만 옆 밀림·회전·정지 실패 | [model_699.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | [초기 판정](../validation/flat_walk_2026-11-10/assessment.json) |
| ② 전진·정지 개선 | 3초 정지→12초 전진→5초 정지, 두 seed에서48/48 통과 | [flat_straight_v2.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | [독립 seed44 판정](../validation/straight_walk_2026-11-10/candidate_seed44/assessment.json) |
| ③ 후진·자세 확장 | 전진 보존 + 별도 후진 PPO + IK·PPO 자세,80/80 통과 | [세 정책 manifest](../config/multiskill_policy.json) | [최종 seed63 판정](../validation/multiskill_2026-11-10/smooth_seed63/assessment.json) |

①의 앞6초는 미리보기 길이다. 초기 정책의 정량 평가는20초×8회 자료를 사용한다. ②의20초 영상은 한 trial이고,48/48은 두 seed×정지·전진·정지전환×각8회를 합한 수다. ③의80/80은 두 seed×다섯 경우×각8회다.

| 영상 | 실제 파일 |
| --- | --- |
| ① 초기 실패20초 | [재생 MP4](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) |
| ② 정지3초·전진12초·정지5초 | [재생 MP4](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) |
| ③ 후진·낮은 자세·기립 포함52초 | [재생 MP4](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) |

## 2 실제 학습 설정

②는 ①의 model699를 이어 학습한 것이 아니다. 공식 Mini의 `BEST_WALK_ONNX_2`를 같은 출력을 내는 actor로 가져와 미세 학습했다. ③의 후진은 ②를 초기값으로 사용했고, 자세는 IK 참조 위에서 초기 잔차0으로 시작했다.

| 설정 | ① 초기 실패 | ② 전진·정지 | ③ 후진 | ③ 낮은 자세·기립 |
| --- | --- | --- | --- | --- |
| 선택 checkpoint index | 699 | 100 | 100 | 50 |
| 병렬 환경 | 512 | 1024 | 512 | 256 |
| rollout 스텝/환경 | 32 | 32 | 32 | 32 |
| 학습률 | 3e-4 adaptive | 1e-4 fixed | 1e-4 fixed | 5e-5 fixed |
| Actor 입력→출력 | 101→14 | 101→14 | 101→14 | 103→14 |
| Critic 입력 | 222 | 222 | 222 | 126 |
| Actor hidden | 256,256,128 | 512,256,128 | 512,256,128 | 256,256,128 |
| Actor 활성화/평균 출력 | ELU / clamp | SiLU / tanh | SiLU / tanh | ELU / tanh |
| 입력 정규화 | 없음 | 사용 | 사용 | 사용 |
| 학습 탐색 std | 초기0.8, 학습 가능 | 다리0.05 / 머리0.01 고정 | 다리0.12 / 머리0.01 고정 | 다리0.04 / 머리0.01 고정 |
| 학습 x명령 m/s | +0.04~+0.12 | +0.04~+0.12 | −0.12~−0.04 | 0, 높이 명령 별도 |
| 정지 명령 확률 | 10% | 25% | 15% | 100% |
| 학습 물성·잡음 랜덤화 | 켜짐 | 기준 물성·잡음 끔 | 기준 물성·잡음 끔 | 기준 물성·잡음 끔 |
| 정책 시간 기준 | 보행0.8초 | 보행0.54초 | 보행0.54초 | 동작16초, actor위상[1,0] |
| 선택 시점 새 샘플 | 계보 약1150만 | 3,309,568 | 1,654,784 | 417,792 |

새 샘플은 이 작업에서 추가로 수집한 양이다. 공식 초기 actor의 사전 학습량을 포함한 전체 학습량으로 해석하지 않는다. ①은 직전 run의 model300에서400회 이어 학습했다. ②는 전체600회 실험 중100을 선택했다. 후진·자세도 선택 시점 이후 후보를 만들었지만 그 가중치로 바꾸지 않았다.

공통 PPO는 clip0.2, gamma0.99, GAE lambda0.95, epoch5, minibatch4, gradient norm1.0이다. ②·후진 warm start의 첫40 update는 critic 준비이며 이후 actor도 갱신했다. YAML의 `activation: elu`는 초기 공통 구성이고, warm-start hook이 actor만 SiLU로 교체하므로 최종 actor 활성화는 `.pt`의 `infos`와 실제 모듈을 기준으로 확인했다.

| 시점 | 당시 환경 설정 | 당시 PPO 설정 |
| --- | --- | --- |
| ① | [params/env.yaml](evidence/training_params/old/env.yaml) | [params/agent.yaml](evidence/training_params/old/agent.yaml) |
| ② | [params/env.yaml](evidence/training_params/forward/env.yaml) | [params/agent.yaml](evidence/training_params/forward/agent.yaml) |
| ③ 후진 | [params/env.yaml](evidence/training_params/backward/env.yaml) | [params/agent.yaml](evidence/training_params/backward/agent.yaml) |
| ③ 자세 | [params/env.yaml](evidence/training_params/posture/env.yaml) | [params/agent.yaml](evidence/training_params/posture/agent.yaml) |

이 보고서 폴더의 `params/old`, `params/forward`, `params/backward`, `params/posture`에도 당시 YAML을 그대로 복사했다.

## 3 유지한 물성과 바꾼 재생 설정

총질량2.107141kg, 머리·목0.581535kg(27.6%), 관성·관절 범위·먼 부위 자기충돌을 유지했다. 모터 참조는 kp17.8, damping0.6, 최대토크3.35Nm, Coulomb마찰0.052Nm, armature0.028kg·m², 백래시 반폭0.5°다. 정책50Hz와 물리500Hz도 유지했다.

머리를 가볍게 만들어 성공한 결과가 아니다. 같은 공식 actor·같은 seed43의 머리 진단에서 정책의 목·머리 움직임을 허용하면 전진20초8/8 생존했고, 고정 명령 방식으로 억제하면0/8 생존했다. 이 진단은 머리의 균형 기여를 보여 주며 모든 이탈의 원인이 머리 하나라고 증명한 것은 아니다.

| 설정 | ① 초기 | ② 최종 전진 | ③ 후진·자세 |
| --- | --- | --- | --- |
| 머리/목 제어 | 14관절 정책 제어, 전용 안정화 없음 | stabilized, 균형 보상 허용 | 동일, 자세에서는 IK+잔차 |
| 머리 목표각 변화율 | 공통5.24rad/s 제한 | 1rad/s 전용 제한 | 1rad/s |
| 보행 발 접촉 보상 위상 | 기존 기준 | +0.25cycle = 135ms | 후진 동일 |
| 발 들림 보상 설정 | 목표0.060m / 가중치0.6 | 목표0.065m / 가중치3 | 후진 동일 |
| Heading 유지 | 전용 처리 없음 | kp2 / 최대0.35rad/s / 필터0.25s | 후진 동일, 자세 전용 actor는0 |
| 외부 명령→actor x명령 | 배율1 | 배율0.85: +0.1→+0.085m/s | 후진0.55: −0.1→−0.055m/s |
| y명령 trim | 0 | 0 | 후진+0.006m/s |
| 자세 잔차 | 없음 | 없음 | IK 관절 참조 ±0.12rad |
| 전환 | 별도3정책 전환 없음 | 전진·정지 | 후진→정지1초 crossfade, 높이변경4초 |

발 접촉의0.25cycle 보정은 **보상 시점**에 적용하며 actor 입력 위상을135ms 밀어 버린다는 뜻이 아니다. 현재 재생 설정은 `config/validated_flat_policy.json`, `config/multiskill_policy.json`과 checkpoint의 `infos.control_profile`에 기록된다. manifest의 `controller` 기록만 수정해도 코드의 하드코딩된 전환 값이 바뀌는 것은 아니다.

## 4 초기 실패에서 전진 성공까지 수치 비교

초기는 seed43, 최종 독립 확인은 seed44이며 각 경우20초×8회다. 주기 평균 지표는 두 경우 모두0.54초 boxcar와 초기2초 제외라는 같은 계산법이다. 아래 감소율은 저장된 집계값의 기술적 비교이며 하나의 변경만 분리한 인과 추정은 아니다.

| 측정 | ① 초기 | ② 최종 독립 seed44 |
| --- | --- | --- |
| 평균 body vx, m/s | 0.111270 | 0.106388 |
| 평균 body vy, m/s | -0.133984 | 0.004400 |
| 순간 XY RMSE, m/s | 0.217224 | 0.133267 (38.65% 감소) |
| 0.54초 평균 XY RMSE, m/s | 0.140511 | 0.009900 (92.95% 감소) |
| 순간 yaw MAE, rad/s | 1.408048 | 0.156281 |
| 0.54초 평균 yaw MAE, rad/s | 0.419033 | 0.028092 |
| 20초 world X 순변위 평균, m | 0.415153 | 2.133412 |
| 20초 world Y 순변위 평균, m | 0.211006 | 0.007464 |
| 최종 누적 heading 변화 평균, ° | 455.86 | -2.82 |
| 정지 중 평면 속도, m/s (2초 후) | 0.22503805 | 0.00053746 (99.76% 감소) |
| 행동 clip 비율 | 73.4705% | 0.0009% |
| 전진 통과 | 0/8 | 8/8 |

body vx/vy는 로봇 몸통 좌표이고 world X/Y는 출발 지면 좌표다. body vx가0.1m/s 부근이어도 계속 회전하면 실제 world 전진은 작을 수 있다. 누적 heading은 unwrap하여 한 바퀴도 합산한다. 초기455.86°는 360°를 뺀 끝 방향만 표시한 수치가 아니다.

정지→전진→정지 시험에서 최종 정지 구간(17~20초)의 평균 평면 속도는 독립 seed44에서0.000605m/s였다. 표의 정지0.000537m/s는 별도20초 정지 시험의 초기2초 이후 값이다. 순간 RMSE와 주기 평균 RMSE를 혼용해 개선율을 계산하지 않았다.

## 5 사이에 무엇을 바꿨고 어떻게 달라졌나

전진 중간 표는 모두 seed43, 전진20초×8회다. V3~V6는 다른 학습 후보이고, V6 이후 IMU/배율 행은 같은 model100 신경망의 명령 전처리 비교다. 각 설정이 항상 더 좋아진 것은 아니다.

| 실험 | 방법 한 줄 | vx m/s | 옆 순변위 m | 최종 heading ° | 전진 통과 |
| --- | --- | --- | --- | --- | --- |
| 초기 실패 | 구형 PPO 후보 평가 | 0.1113 | 0.2110 | 455.86 | 0/8 |
| 공식 초기 정책 | 공식 ONNX actor와 정규화 가져오기 | 0.0833 | 0.3572 | -6.66 | 0/8 |
| V3 100 | 정지·방향 보상, 머리 안정화 | 0.1374 | 0.6297 | 30.37 | 0/8 |
| V3 300 | V3 학습을 더 진행 | 0.0918 | 0.2700 | 8.30 | 0/8 |
| V4 100 | 누적 yaw·옆 경로 보상 추가 | 0.0917 | -0.9703 | -61.09 | 0/8 |
| V5 100 | critic에 경로 상태, 발 들림 보상 강화 | 0.0930 | 0.2760 | 22.27 | 0/8 |
| V6 100 보정 전 | 접촉 보상 +0.25cycle 정렬 | 0.1174 | -1.2671 | -71.08 | 0/8 |
| V6 + IMU 방향 유지 | IMU kp2로 yaw 명령 보정 | 0.1174 | -0.0067 | -2.80 | 8/8 |
| V6 + 명령 배율0.85 | 내부 x명령 배율1→0.85 | 0.1066 | 0.0090 | -2.80 | 8/8 |

V5→V6에서 오른발 들림 근사값은8.39→13.40mm로 커졌지만 V6 원본은 회전·옆 이동 때문에 여전히0/8이었다. IMU 보정 후 옆 이동은−1.2671→−0.00670m, heading은−71.08→−2.80°가 됐다. 이어 배율0.85로 실제 속도를0.1174→0.1066m/s로 줄이자 정지전환도6/8→8/8이 됐다. 독립 seed44까지 합쳐48/48이다.

## 6 후진과 전환을 추가한 과정

후진은 전진 정책에서 별도 PPO로 학습했다. 아래에서 배율·trim·crossfade는 학습 가중치를 새로 바꾼 작업이 아니다. 초기 후진 원본 screen만12초×4회이며 이후 수치는20초×8회다.

| 실험 | seed / 시간 | 평균 vx m/s | 평균 옆 순변위 m | 후진 통과 |
| --- | --- | --- | --- | --- |
| 전진 정책에 후진 명령 | 61 / 20s | 0.0011 | 0.0127 | 생존8/8, 거의 정지 |
| 후진 PPO 원본 | 61 / 12s | -0.1330 | -0.0030 | 0/4 |
| 배율0.75 | 61 / 20s | -0.1193 | -0.0268 | 2/8 |
| 배율0.40 | 61 / 20s | -0.0841 | -0.1306 | 8/8 |
| 배율0.55 | 62 / 20s | -0.1012 | -0.1673 | 3/8 |
| 측면 trim 추가 | 62 / 20s | -0.1010 | -0.0946 | 8/8 |
| 정지 연결 보완 | 62 / 20s | -0.1010 | -0.0946 | 8/8 |
| 독립 seed63 | 63 / 20s | -0.1017 | -0.0981 | 8/8 |

배율0.55와 trim0의 seed62 후진은3/8 통과였다. trim+0.006을 적용하자 옆 이동은−167.3→−94.6mm로 줄고 후진8/8이 됐다. 그러나 전체52초 전환은7/8이었고 최대 heading8.95°가 기준8.59°를 넘었다. 후진→정지 출력을1초 연결한 뒤 같은 seed62에서 최대6.52°, 전체전환8/8이 됐다. seed63에서도 다섯 경우 각8/8로 확인했다.

## 7 낮은 자세와 기립의 설정과 결과

IK 참조는 양발 위치와 전체 질량 중심을 고려해 만들었다. 기준점 높이165.629→130mm, 참조 knot20개, CoM x변화 약−20mm, 관절 stop 여유0.022rad를 사용했다. PPO는 참조에 ±0.12rad 잔차를 더한다. 내려가기4초·낮은 유지3초·올라가기4초와 앞뒤 정지로16초 주기다. 실물에 직접 몸체 위치를 쓰는 것이 아니라 관절 목표와 물리 힘으로 움직인다.

자세 학습의 초기 높이는 참조 standing+3mm인168.629mm이며, 보행/복합 재생의 기본 초기 높이172mm와 구분한다. 아래는 루트/몸통 기준점 높이이며 머리까지의 전체 로봇 키가 아니다.

| 자세 측정 | 최종값 |
| --- | --- |
| 시험 | 16/16회, 각3주기 =48주기 |
| 평균 높이 변화 | 164.8 → 128.3 → 164.5 mm |
| 복귀 높이 오차 평균 | 0.314mm |
| 낮은 유지 구간 양발 접촉 | 100% |
| 초기2초 이후 최대 발 수평 이동 | 1.287mm |

최종 전체80회는 정지16/16·전진16/16·후진16/16·자세16/16·전체전환16/16이다. 이때 전진 평균vx는 전체20초 기준0.1065m/s, 초기2초 제외0.1076m/s다. 후진은 각각−0.1013과−0.1030m/s다. 두 평균 구간의 차이를 정책 변경으로 해석하지 않는다. 후진20초 옆 순변위 평균−96.3mm는 아직 남아 있다.

## 8 실제 코드와 설정 위치

| 목적 | 현재 구현 위치 |
| --- | --- |
| 공통 좌표·주기·모터 기준 | [profile.py](../open_duck_v2/profile.py#L95) |
| 초기/기본 보상과 상태 처리 | [open_duck_env.py::_get_rewards](../open_duck_complex_terrain/open_duck_env.py#L531) |
| 정지·방향 추종 보상 | [straight_rewards.py](../open_duck_v2/straight_rewards.py#L10) / [보행 보상](../open_duck_complex_terrain/open_duck_env.py#L713) |
| V3 머리·V4 경로·V5 critic·V6 위상 설정 | [open_duck_env_cfg.py](../open_duck_complex_terrain/open_duck_env_cfg.py#L287) |
| critic에 heading/옆 경로 상태 | [open_duck_env.py](../open_duck_complex_terrain/open_duck_env.py#L219) |
| 머리 목표 변화율 제한 | [control.py](../open_duck_v2/control.py#L240) |
| IMU heading·x배율·y trim | [heading.py](../open_duck_v2/heading.py#L73) |
| 신경망을 유지한 보정 metadata 생성 | [configure_policy_controller.py](../tools/configure_policy_controller.py#L20) |
| 공식 actor import / warm start hook | [import_official_policy.py](../tools/import_official_policy.py) / [training_policy.py](../open_duck_v2/training_policy.py#L32) |
| 후진 명령·학습 task 설정 | [OpenDuckBackwardEnvCfg](../open_duck_complex_terrain/open_duck_env_cfg.py#L326) |
| 후진→정지 crossfade | [multiskill.py](../open_duck_v2/multiskill.py#L253) |
| IK 참조 계산과 목표 | [build_posture_reference.py](../tools/build_posture_reference.py#L28) / [고정 참조 JSON](../config/posture_reference_v1.json) |
| 자세 주기·잔차 제어 | [posture.py](../open_duck_v2/posture.py#L104) |
| 자세 PPO 환경·보상 | [posture_env.py](../open_duck_complex_terrain/posture_env.py#L37) |
| 전진/후진/자세 학습 실행값 | [train_straight.py](../tools/train_straight.py#L46) / [train_skill.py](../tools/train_skill.py#L20) |
| 현재 선택 정책·실행 설정 | [전진 manifest](../config/validated_flat_policy.json) / [세 정책 manifest](../config/multiskill_policy.json) |

## 9 pt 파일의 의미

`.pt`는 PyTorch 저장 파일이다. 현재 선택된 세 `.pt`는 actor만 담은 그림·영상 파일이 아니라 학습 checkpoint다. 파일에는 actor/critic 가중치, optimizer, iteration, 정책 계약과 제어 profile, 새 정책의 관측 정규화 상태가 들어 있다. 실행할 때는 actor·normalizer·제어 profile을 사용하며 학습 재개에는 critic·optimizer도 필요하다.

| 파일 | 담당 동작·의미 |
| --- | --- |
| [초기 model_699.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | ① 실패 정책의 학습 checkpoint. 현재 기본 실행으로 선택하지 않음 |
| [flat_straight_v2.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | ②에서 선택한 전진·정지 actor101→14. ③에서도 같은 SHA-256으로 보존 |
| [backward_v2.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | ③에서 별도 후진 PPO로 학습한 actor101→14 + x배율0.55/y trim0.006 metadata |
| [posture_v2.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | ③의 IK 참조 위 균형 잔차 actor103→14. 참조 JSON 해시와 함께 로드 |
| [official_teacher.pt](https://github.com/happylee8282/openduck_v2_ws/releases/tag/v0.2.0-isaac-multiskill) | 공식 공개 ONNX actor를 가져온 초기값. 이번 미세 학습의 출발점 |
| 각 run 폴더의 model_100.pt, model_50.pt | 숫자는 그 run의 저장 iteration index. 폴더가 다르면 같은100이라도 다른 가중치이며 더 큰 숫자가 더 좋은 정책이라는 뜻은 아님 |

선택 전진과 후진의 원본 신경망 tensor를 최종 `.pt`와 직접 비교한 결과 모두 동일했다. Heading/명령 배율/trim을 붙일 때 바뀐 부분은 `infos.control_profile` 등 metadata다. 세 정책 전환·목표각 제한·PD·IK 참조 처리는 Python 런타임에도 있으므로 `.pt` 한 파일만으로 현재 전체 동작이 완성되는 구조는 아니다.

## 10 범위와 재사용 자료

현재 통과는 기준 물성 평지 시뮬레이션이다. 전진의 전체 랜덤화 스트레스 시험은 정지8/8, 전진5/8, 정지전환5/8 통과였으며 새 후진·자세의 전체 랜덤화·실물 검증은 미완료다. 여러 설정과 초기 정책이 동시에 달라진 구간은 단일 원인의 효과로 분리해서 주장하지 않는다.

- `numeric_evidence.json`: 모든 수치·설정·체크포인트 경로·SHA와 정의.
- `forward_development.csv`: 초기·공식 초기값·V3~V6·보정·독립 평가 수치.
- `backward_development.csv`: 후진 baseline·학습·배율·trim·전환 비교.
- `params/`: 당시 환경/PPO YAML 원본 사본.
- `source/collect_evidence.py`: 원자료 집계 코드.
