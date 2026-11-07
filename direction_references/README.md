# 조립 방향 검사용 기준 JSON

[robot.json](robot.json)은 전체 로봇 기준이다. [head.json](head.json), [left_leg.json](left_leg.json), [right_leg.json](right_leg.json)은 조립 단위별로 필요한 기대 ID와 관절별 기준을 담는다. JSON의 `expected_ids`는 모듈에서 찾을 서보 목록이며, `joint_profiles`에는 관절 ID와 기준 전류 시그니처가 들어 있다.

시험에 같은 방향과 지지 조건의 조립체, 기준 JSON, 실측 전류를 입력한다. 시험 도구가 관절 구동 시 측정된 방향 응답을 기준과 비교해 판정과 근거를 출력한다. 단품 프로필을 전체 로봇 시험에 적용할 때는 ID 집합과 기준 조건을 확인한다. 관련 시험 문서는 [docs](../docs/README.md)에 있다.
