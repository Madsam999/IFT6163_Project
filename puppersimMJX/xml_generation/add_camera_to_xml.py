#!/usr/bin/env python3
"""Add or replace a body-mounted camera in an existing stable MuJoCo XML.

Default behavior matches this project request:
- Reads stable XML: puppersim/data/pupper_v2_final_stable.xml
- Adds camera on base_link named front_cam
- Uses provided default pose:
  pos="0.0001 -0.147 -0.0064"
  forward-axis="-y" (produces xyaxes="-1 0 0 0 0 1")
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET
from typing import Iterable


def _parse_vec3(text: str) -> list[float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got: {text!r}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _vec_to_str(v: Iterable[float], ndigits: int = 6) -> str:
    return " ".join(f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".") for x in v)


def _norm(v: list[float]) -> list[float]:
    mag = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if mag <= 1e-12:
        raise ValueError("Zero-length vector.")
    return [v[0] / mag, v[1] / mag, v[2] / mag]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _parse_forward_axis(axis_text: str) -> list[float]:
    mapping = {
        "+x": [1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "+y": [0.0, 1.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, 1.0],
        "-z": [0.0, 0.0, -1.0],
    }
    key = axis_text.strip().lower()
    if key not in mapping:
        raise ValueError(f"Invalid --forward-axis: {axis_text}. Use one of {sorted(mapping)}")
    return mapping[key]


def _xyaxes_from_forward(forward: list[float]) -> str:
    """Build camera xyaxes so optical axis looks along `forward`.

    MuJoCo camera looks along -Z in local camera frame.
    """
    f = _norm(forward)
    z = [-f[0], -f[1], -f[2]]

    up_ref = [0.0, 0.0, 1.0]
    if abs(z[2]) > 0.98:
        up_ref = [1.0, 0.0, 0.0]

    x = _norm(_cross(up_ref, z))
    y = _cross(z, x)
    return _vec_to_str([x[0], x[1], x[2], y[0], y[1], y[2]])


def _add_or_replace_camera(
    xml_path: pathlib.Path,
    output_path: pathlib.Path,
    camera_name: str,
    camera_pos: list[float],
    forward: list[float],
    fovy_deg: float,
    force: bool,
) -> None:
    if not xml_path.exists():
        raise FileNotFoundError(f"Input XML not found: {xml_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}. Use --force.")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> in XML.")
    base_link = worldbody.find("./body[@name='base_link']")
    if base_link is None:
        raise ValueError("No base_link body in XML.")

    # Remove same camera name from base_link/worldbody to avoid duplicates.
    for parent in (base_link, worldbody):
        for cam in list(parent.findall("./camera")):
            if (cam.get("name") or "") == camera_name:
                parent.remove(cam)

    ET.SubElement(
        base_link,
        "camera",
        name=camera_name,
        pos=_vec_to_str(camera_pos),
        xyaxes=_xyaxes_from_forward(forward),
        fovy=f"{float(fovy_deg):.4f}".rstrip("0").rstrip("."),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--xml-path",
        type=pathlib.Path,
        default=pathlib.Path("puppersim/data/pupper_v2_final_stable.xml"),
        help="Input stable XML.",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output XML path. If omitted, writes in-place to --xml-path.",
    )
    p.add_argument("--force", action="store_true", help="Allow overwriting output XML.")

    p.add_argument("--camera-name", type=str, default="front_cam")
    p.add_argument(
        "--camera-pos",
        type=str,
        default="0.0001 -0.147 -0.0064",
        help="Camera XYZ in meters in base_link frame.",
    )
    p.add_argument(
        "--forward-axis",
        type=str,
        default="-y",
        help="One of +x,-x,+y,-y,+z,-z.",
    )
    p.add_argument("--fovy", type=float, default=70.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    output_path = args.output if args.output is not None else args.xml_path
    camera_pos = _parse_vec3(args.camera_pos)
    forward = _parse_forward_axis(args.forward_axis)

    _add_or_replace_camera(
        xml_path=args.xml_path,
        output_path=output_path,
        camera_name=args.camera_name,
        camera_pos=camera_pos,
        forward=forward,
        fovy_deg=float(args.fovy),
        force=bool(args.force),
    )

    print(f"Wrote camera XML: {output_path}")
    print(f"camera_name={args.camera_name}")
    print(f"camera_pos_base_link(m)={_vec_to_str(camera_pos)}")
    print(f"camera_forward_base_link={_vec_to_str(forward)}")


if __name__ == "__main__":
    main()
