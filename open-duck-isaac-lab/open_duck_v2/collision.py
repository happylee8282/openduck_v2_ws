"""가까운 기구학 조상 충돌만 제외하는 self-collision 보조 함수.

URDF 트리에서 몇 개의 가동 관절만 떨어진 인접 부품은 조립 상태에서 항상
가까워 불필요한 접촉을 만들 수 있다. 반면 좌우 다리나 원격 body 사이의
충돌은 넘어짐·자기 충돌 학습에 필요하므로 그대로 유지한다.
"""

import xml.etree.ElementTree as ET


def mechanical_neighbor_pairs(urdf_path, distance=2):
    """URDF에서 ``distance``개 이하 가동 관절로 연결된 collider 쌍을 찾는다.

    fixed joint는 기구학적 거리 증가에 포함하지 않는다. 반환 쌍은 중복과
    방향성을 제거하고 정렬하여 검증 결과가 결정적이도록 한다.
    """

    if distance < 0:
        raise ValueError("Negative joint distance")
    root = ET.parse(urdf_path).getroot()
    colliders = {
        link.get("name")
        for link in root.findall("link")
        if link.find("collision") is not None
    }
    # child -> (parent, 이 edge가 가동 관절인지) 형태로 역방향 트리를 만든다.
    parents = {
        joint.find("child").get("link"): (
            joint.find("parent").get("link"),
            joint.get("type") != "fixed",
        )
        for joint in root.findall("joint")
    }
    pairs = set()
    for child in colliders:
        node, depth = child, 0
        # 각 collider에서 root 방향으로 올라가며 가까운 조상 collider를 모은다.
        while node in parents:
            node, moving = parents[node]
            depth += int(moving)
            if depth > distance:
                break
            if node in colliders:
                pairs.add(tuple(sorted((child, node))))
    return sorted(pairs)


def apply_mechanical_neighbor_filters(stage, robot_path, urdf_path, distance=2):
    """찾은 인접 쌍을 USD ``FilteredPairsAPI`` 관계로 stage에 적용한다.

    URDF에는 있지만 변환 USD에 body가 없으면 조용히 건너뛰지 않고 자산
    변환 오류로 보고한다.
    """

    # pxr는 Isaac/Kit가 실행되는 경로에서만 필요하므로 지연 import한다.
    from pxr import UsdPhysics

    pairs = mechanical_neighbor_pairs(urdf_path, distance)
    for left, right in pairs:
        prim = stage.GetPrimAtPath(f"{robot_path}/{left}")
        target = stage.GetPrimAtPath(f"{robot_path}/{right}")
        if not prim.IsValid() or not target.IsValid():
            raise RuntimeError(
                f"Cannot filter missing collision bodies {left}, {right}"
            )
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(
            target.GetPath()
        )
    return pairs
