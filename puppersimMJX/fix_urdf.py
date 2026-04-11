import argparse
import pathlib
import xml.etree.ElementTree as ET


def _normalize_mesh_filename(
    filename: str,
    package_prefix: str,
    convert_mesh_ext_to_stl: bool,
) -> str:
    norm = filename.strip().replace("\\", "/")

    # Preserve explicit package:// URIs.
    if norm.startswith("package://"):
        out = norm
    else:
        # Strip common relative prefixes.
        while norm.startswith("../"):
            norm = norm[3:]
        if norm.startswith("./"):
            norm = norm[2:]
        if package_prefix:
            out = f"{package_prefix.rstrip('/')}/{norm.lstrip('/')}"
        else:
            out = norm

    out = out.replace(" ", "")
    if convert_mesh_ext_to_stl and out.lower().endswith(".dae"):
        out = out[:-4] + ".stl"
    return out


def fix_urdf(
    urdf_file_path: pathlib.Path,
    output_file_path: pathlib.Path,
    package_prefix: str,
    meshdir: str,
    add_world_joint: bool,
    convert_mesh_ext_to_stl: bool,
):
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

    # Rewrite mesh paths.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh.set(
            "filename",
            _normalize_mesh_filename(
                filename=filename,
                package_prefix=package_prefix,
                convert_mesh_ext_to_stl=convert_mesh_ext_to_stl,
            ),
        )

    # Add world link/joint once (if requested).
    if add_world_joint:
        # Infer root link as: links that are never a joint child.
        link_names = [link.attrib.get("name", "") for link in root.findall("./link")]
        child_links = [child.attrib.get("link", "") for child in root.findall("./joint/child")]
        root_candidates = [name for name in link_names if name and name not in set(child_links)]
        root_link_name = root_candidates[0] if root_candidates else (link_names[0] if link_names else "base_link")

        has_world_link = any(
            child.tag == "link" and child.attrib.get("name") == "world" for child in list(root)
        )
        if not has_world_link:
            ET.SubElement(root, "link", name="world")

        has_world_joint = any(
            child.tag == "joint" and child.attrib.get("name") == "world_to_body" for child in list(root)
        )
        if not has_world_joint:
            world_to_robot_joint = ET.SubElement(root, "joint", name="world_to_body", type="floating")
            ET.SubElement(world_to_robot_joint, "parent", link="world")
            ET.SubElement(world_to_robot_joint, "child", link=root_link_name)

    # Ensure <mujoco><compiler .../></mujoco> tag.
    mujoco_tag = root.find("mujoco")
    if mujoco_tag is None:
        mujoco_tag = ET.Element("mujoco")
        root.insert(0, mujoco_tag)
    compiler_tag = mujoco_tag.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.SubElement(
            mujoco_tag,
            "compiler",
            attrib={"meshdir": meshdir, "discardvisual": "false"},
        )
    else:
        compiler_tag.set("meshdir", meshdir)
        compiler_tag.set("discardvisual", compiler_tag.attrib.get("discardvisual", "false"))

    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix URDF mesh paths and inject MuJoCo-specific tags.")
    parser.add_argument("--urdf_path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output_path",
        type=pathlib.Path,
        default=None,
        help="Output path. Defaults to <urdf_path>.fixed.urdf",
    )
    parser.add_argument(
        "--package_prefix",
        type=str,
        default="",
        help="Optional prefix for mesh filenames (example: package://pupper_v3_description/description).",
    )
    parser.add_argument(
        "--meshdir",
        type=str,
        default="../meshes/stl/",
        help="Value for <mujoco><compiler meshdir=...>",
    )
    parser.add_argument(
        "--add_world_joint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add world link and floating joint world_to_body if missing (default: true).",
    )
    parser.add_argument(
        "--convert_mesh_ext_to_stl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert .dae mesh refs to .stl (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_path = args.output_path or args.urdf_path.with_suffix(".fixed.urdf")
    fix_urdf(
        urdf_file_path=args.urdf_path,
        output_file_path=output_path,
        package_prefix=args.package_prefix,
        meshdir=args.meshdir,
        add_world_joint=args.add_world_joint,
        convert_mesh_ext_to_stl=args.convert_mesh_ext_to_stl,
    )
    print(f"Wrote fixed URDF to: {output_path}")
