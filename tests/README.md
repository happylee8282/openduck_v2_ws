# 시험 플랫폼 소프트웨어 검증

[test_pilot_test_platform.py](test_pilot_test_platform.py)는 파일로 제공되는 [pilot_test_platform.py](../scripts/pilot_test_platform.py)의 측정 및 보고 로직을 검증한다. 테스트의 입력은 코드에 정의한 시험 샘플과 설정이고, 출력은 테스트 러너의 성공·실패 결과다. 실기 검증에는 별도 하드웨어 시험 절차가 필요하다.

```bash
python -m pytest tests/test_pilot_test_platform.py
```

## 확인하는 동작

모의 입력으로 모터 ID 누락, 저전압·과전류·과열 중단, 반코사인 파형의 구간별 값, 시차 정렬 후 오차, 역방향 조립 후보, HTML 리포트 생성을 확인한다. 하드웨어 동작 확인이 필요한 항목과 구분해 사용한다.
