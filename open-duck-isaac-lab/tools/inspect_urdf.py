from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


URDF_PATH = Path(__file__).resolve().parents[1] / "Robots" / "open_duck_mini.urdf"


def main():
    root = ET.parse(URDF_PATH).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints = root.findall("joint")

    children = {
        joint.find("child").attrib["link"]
        for joint in joints
        if joint.find("child") is not None
    }
    root_links = [link for link in links if link not in children]

    print(f"URDF: {URDF_PATH}")
    print(f"Links: {len(links)}")
    print(f"Joints: {len(joints)}")
    print(f"Root links: {root_links}")
    print()

    print("Movable joints:")
    for joint in joints:
        joint_type = joint.attrib.get("type")
        if joint_type == "fixed":
            continue
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        limit = joint.find("limit")
        limit_text = ""
        if limit is not None:
            limit_text = (
                f" lower={limit.attrib.get('lower')} upper={limit.attrib.get('upper')}"
            )
        print(
            f"  {joint.attrib['name']} ({joint_type}): {parent} -> {child}{limit_text}"
        )

    print()
    print("Fixed joints:")
    for joint in joints:
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        print(f"  {joint.attrib['name']}: {parent} -> {child}")


if __name__ == "__main__":
    main()
