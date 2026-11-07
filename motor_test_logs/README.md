# 단일 모터 시험 기록

각 `motor_<ID>_<시간>` 폴더는 특정 STS3215 서보의 시험 세션이다. 입력은 서보 ID, 시험 명령과 측정 조건이고, 출력은 `samples.csv` 시계열, `summary.json` 집계, `report.html`이다. 다른 날짜의 시험을 비교할 때 모터 ID, 목표 파형과 구동 조건을 확인한 뒤 목표 대비 실제 응답을 비교한다. 수집 코드는 [sts3215_pilot_test.py](../scripts/sts3215_pilot_test.py)에 있다.
