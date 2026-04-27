#!/usr/bin/env python3
"""Generate AprilTag textures and an MJCF room with two wall tags."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from puppersimMJX import get_assets_path

DEFAULT_TAG_HALF = 0.20


def _write_tag_quad_obj(path: Path) -> None:
    """Writes a unit quad mesh with explicit UVs for exact tag texture mapping."""
    content = """# AprilTag textured quad
v -0.5 -0.5 0.0
v  0.5 -0.5 0.0
v  0.5  0.5 0.0
v -0.5  0.5 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _draw_apriltag(path: Path, tag_id: int, marker_px: int = 512, quiet_zone_px: int = 64) -> None:
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError("OpenCV aruco AprilTag dictionary unavailable. Install opencv-contrib-python.")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), marker_px)

    canvas_size = marker_px + 2 * quiet_zone_px
    canvas_gray = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    canvas_gray[quiet_zone_px : quiet_zone_px + marker_px, quiet_zone_px : quiet_zone_px + marker_px] = marker
    # MuJoCo texture loading is more reliable with explicit RGB images.
    canvas_rgb = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(path), canvas_rgb):
        raise RuntimeError(f"Failed to write {path}")

    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(canvas_gray)
    detected = set(int(x) for x in ids.flatten()) if ids is not None else set()
    if int(tag_id) not in detected or not corners:
        raise RuntimeError(f"Generated marker {tag_id} failed OpenCV detection sanity check.")


def _ensure_asset(asset: ET.Element, tag: str, name: str, **attrs: str) -> None:
    for child in asset.findall(tag):
        if child.get("name") == name:
            for k, v in attrs.items():
                child.set(k, v)
            return
    ET.SubElement(asset, tag, name=name, **attrs)


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _remove_existing(worldbody: ET.Element) -> None:
    for child in list(worldbody):
        name = child.get("name", "")
        if name.startswith("room_") or name.startswith("apriltag_"):
            worldbody.remove(child)


def _add_room_and_tags(worldbody: ET.Element, *, include_bad_tag: bool) -> None:
    room_half = 2.0
    wall_t = 0.03
    wall_h = 0.6

    ET.SubElement(
        worldbody,
        "geom",
        name="room_wall_front",
        type="box",
        pos=f"0 {-room_half} {wall_h}",
        size=f"{room_half} {wall_t} {wall_h}",
        material="room_wall_mat",
        contype="1",
        conaffinity="1",
        condim="3",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="room_wall_back",
        type="box",
        pos=f"0 {room_half} {wall_h}",
        size=f"{room_half} {wall_t} {wall_h}",
        material="room_wall_mat",
        contype="1",
        conaffinity="1",
        condim="3",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="room_wall_left",
        type="box",
        pos=f"{-room_half} 0 {wall_h}",
        size=f"{wall_t} {room_half} {wall_h}",
        material="room_wall_mat",
        contype="1",
        conaffinity="1",
        condim="3",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="room_wall_right",
        type="box",
        pos=f"{room_half} 0 {wall_h}",
        size=f"{wall_t} {room_half} {wall_h}",
        material="room_wall_mat",
        contype="1",
        conaffinity="1",
        condim="3",
    )

    tag_y = -room_half + wall_t + 0.004
    tag_z = 0.45
    front_wall_quat = "0.70710678 -0.70710678 0 0"

    tag_x_offset = 0.90

    ET.SubElement(
        worldbody,
        "geom",
        name="apriltag_good_panel",
        type="mesh",
        mesh="apriltag_quad",
        pos=f"{-tag_x_offset:.2f} {tag_y} {tag_z}",
        quat=front_wall_quat,
        material="apriltag_good_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )
    if include_bad_tag:
        ET.SubElement(
            worldbody,
            "geom",
            name="apriltag_bad_panel",
            type="mesh",
            mesh="apriltag_quad",
            pos=f"{tag_x_offset:.2f} {tag_y} {tag_z}",
            quat=front_wall_quat,
            material="apriltag_bad_mat",
            contype="0",
            conaffinity="0",
            group="1",
        )


def _build_xml(
    input_xml: Path,
    output_xml: Path,
    good_tex: str,
    bad_tex: str,
    quad_mesh_obj: str,
    *,
    tag_half: float,
    include_bad_tag: bool,
) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Invalid MJCF: missing <asset> or <worldbody>.")

    # apriltag_quad.obj has half-extents of 0.5, so scale by 2*tag_half.
    _ensure_asset(
        asset,
        "mesh",
        "apriltag_quad",
        file=quad_mesh_obj,
        scale=f"{2.0 * tag_half} {2.0 * tag_half} 1.0",
    )
    for child in asset.findall("mesh"):
        if child.get("name") == "apriltag_quad":
            # Required for zero-thickness visual quad mesh used for exact UV mapping.
            child.set("inertia", "shell")
            break
    _ensure_asset(asset, "texture", "apriltag_good_tex", type="2d", file=good_tex)
    _ensure_asset(asset, "texture", "apriltag_bad_tex", type="2d", file=bad_tex)
    _ensure_asset(
        asset,
        "material",
        "apriltag_good_mat",
        texture="apriltag_good_tex",
        texuniform="true",
        texrepeat="1 1",
        reflectance="0",
        shininess="0",
        specular="0",
    )
    _ensure_asset(
        asset,
        "material",
        "apriltag_bad_mat",
        texture="apriltag_bad_tex",
        texuniform="true",
        texrepeat="1 1",
        reflectance="0",
        shininess="0",
        specular="0",
    )
    _ensure_asset(asset, "material", "room_wall_mat", rgba="0.92 0.92 0.92 1")

    _remove_existing(worldbody)
    _add_room_and_tags(worldbody, include_bad_tag=include_bad_tag)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def main() -> None:
    assets_path = get_assets_path()
    p = argparse.ArgumentParser()
    p.add_argument("--input-xml", type=Path, default=assets_path / "pupper_v2_final_stable_cam.xml")
    p.add_argument("--output-xml", type=Path, default=assets_path / "pupper_v2_apriltag_room.xml")
    p.add_argument("--textures-dir", type=Path, default=assets_path / "textures")
    p.add_argument("--meshes-dir", type=Path, default=assets_path / "meshes")
    p.add_argument("--good-tag-id", type=int, default=101)
    p.add_argument("--bad-tag-id", type=int, default=287)
    p.add_argument("--tag-half", type=float, default=DEFAULT_TAG_HALF)
    p.add_argument("--include-bad-tag", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    args.textures_dir.mkdir(parents=True, exist_ok=True)
    args.meshes_dir.mkdir(parents=True, exist_ok=True)
    good_tex = args.textures_dir / f"apriltag_36h11_id{args.good_tag_id:03d}.png"
    bad_tex = args.textures_dir / f"apriltag_36h11_id{args.bad_tag_id:03d}.png"
    quad_mesh = args.meshes_dir / "apriltag_quad.obj"

    _draw_apriltag(good_tex, tag_id=args.good_tag_id)
    _draw_apriltag(bad_tex, tag_id=args.bad_tag_id)
    _write_tag_quad_obj(quad_mesh)

    _build_xml(
        input_xml=args.input_xml,
        output_xml=args.output_xml,
        good_tex=_relpath(good_tex, args.output_xml.parent),
        bad_tex=_relpath(bad_tex, args.output_xml.parent),
        quad_mesh_obj=quad_mesh.name,
        tag_half=float(args.tag_half),
        include_bad_tag=bool(args.include_bad_tag),
    )
    print(f"Wrote: {args.output_xml}")
    print(f"Textures: {good_tex}, {bad_tex}")


if __name__ == "__main__":
    main()
