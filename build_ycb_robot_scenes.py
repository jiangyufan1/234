#!/usr/bin/env python
"""Merge prepared YCB object XMLs into the three-finger robot MJCF template."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PROJECT = Path(r"E:\project\p1\deepgelsight")
DEFAULT_TEMPLATE = PROJECT / "xml" / "ur3_with_threefinger_gelsight_table_tomato_underactuated.xml"
DEFAULT_OBJECT_ROOT = PROJECT / "object" / "ycb"
DEFAULT_OUTPUT = PROJECT / "xml" / "generated_ycb"
DEFAULT_OBJECT_COUNT = 10
# A compact, shape-diverse starter set. Pass --objects for an exact custom set.
DEFAULT_OBJECTS = (
    "apple", "banana", "cracker_box", "master_chef_can", "mug",
    "mustard_bottle", "power_drill", "sponge", "sugar_box", "tennis_ball",
)
OBJECT_BODY_NAME = "tomato_soup_can"
OBJECT_FREEJOINT_NAME = "tomato_soup_can_free"
VISUAL_GEOM_NAME = "tomato_visual_geom"
VISUAL_MESH_NAME = "tomato_visual_mesh"
VISUAL_MATERIAL_NAME = "tomato_mat"
CENTER_SITE_NAME = "tomato_center_site"


def set_mesh_collision(mesh_geom: ET.Element, is_visual: bool, collision_index: int | None = None) -> None:
    if is_visual:
        mesh_geom.set("name", VISUAL_GEOM_NAME)
        mesh_geom.set("contype", "0")
        mesh_geom.set("conaffinity", "0")
        mesh_geom.set("group", "1")
        mesh_geom.set("mesh", VISUAL_MESH_NAME)
        mesh_geom.set("material", VISUAL_MATERIAL_NAME)
        return
    if collision_index is None:
        raise ValueError("Collision mesh index is required")
    mesh_geom.set("name", f"tomato_collision_{collision_index}")
    mesh_geom.set("contype", "4")
    mesh_geom.set("conaffinity", "3")
    mesh_geom.set("condim", "4")
    mesh_geom.set("friction", "0.65 0.02 0.001")
    mesh_geom.set("group", "3")


def build_scene(template: Path, object_root: Path, object_name: str, output: Path) -> Path:
    object_xml = object_root / f"{object_name}.xml"
    object_dir = object_root / object_name
    visual_file = object_dir / "textured.obj"
    texture_file = object_dir / "texture_map.png"
    collision_files = sorted(object_dir.glob("textured_coacd_*.stl"))
    missing = [str(p) for p in (object_xml, visual_file, texture_file) if not p.is_file()]
    if missing or not collision_files:
        raise FileNotFoundError(
            f"{object_name}: missing required assets: {missing or ['textured_coacd_*.stl']}"
        )

    base_tree = ET.parse(template)
    base_root = base_tree.getroot()
    source_root = ET.parse(object_xml).getroot()
    base_asset = base_root.find("asset")
    base_worldbody = base_root.find("worldbody")
    source_asset = source_root.find("asset")
    source_body = source_root.find(f".//body[@name='{object_name}']")
    if base_asset is None or base_worldbody is None or source_asset is None or source_body is None:
        raise ValueError(f"Could not find required MJCF sections for {object_name}")

    source_mesh_files = {
        item.get("name"): item.get("file", "")
        for item in source_asset.findall("mesh")
    }
    used_collision_files = []
    for geom in source_body.findall("geom"):
        if "visual" in geom.get("name", ""):
            continue
        mesh_file = source_mesh_files.get(geom.get("mesh"), "")
        if "coacd" not in mesh_file:
            continue
        source_file = (object_root / mesh_file).resolve()
        if not source_file.is_file():
            source_file = (object_dir / Path(mesh_file).name).resolve()
        if not source_file.is_file():
            raise FileNotFoundError(f"{object_name}: collision mesh missing: {mesh_file}")
        used_collision_files.append(source_file)
    if not used_collision_files:
        raise FileNotFoundError(f"{object_name}: object XML references no CoACD collision meshes")

    # Keep the robot's original mesh directory available from generated files.
    compiler = base_root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(template.parent.resolve()))

    # Remove the tomato can assets and model body from the robot template.
    for child in list(base_asset):
        name = child.get("name", "")
        if name.startswith("tomato_"):
            base_asset.remove(child)
    old_body = base_worldbody.find(f"body[@name='{OBJECT_BODY_NAME}']")
    if old_body is None:
        raise ValueError(f"Robot template has no {OBJECT_BODY_NAME} body")
    original_pos = old_body.get("pos", "0.45 0 0.825")
    base_worldbody.remove(old_body)

    # Copy object visual/collision meshes and texture into the robot model.
    mesh_name_map: dict[str, str] = {}
    collision_mesh_index = 0
    for item in source_asset:
        kind = item.tag
        if kind == "mesh":
            source_name = item.get("name", "")
            file_attr = item.get("file", "")
            if not file_attr:
                continue
            source_file = (object_root / file_attr).resolve()
            if not source_file.is_file():
                # Some XMLs include object_name/ prefixed paths while residing beside it.
                source_file = (object_dir / Path(file_attr).name).resolve()
            if not source_file.is_file():
                raise FileNotFoundError(f"{object_name}: mesh missing: {file_attr}")
            item.set("file", str(source_file))
            if "visual" in source_name:
                target_name = VISUAL_MESH_NAME
            else:
                target_name = f"tomato_col{collision_mesh_index}_mesh"
                collision_mesh_index += 1
            mesh_name_map[source_name] = target_name
            item.set("name", target_name)
            base_asset.append(item)
        elif kind == "texture":
            source_name = item.get("name", "")
            item.set("name", "tomato_texture")
            image_file = item.get("file")
            if image_file:
                resolved = (object_root / image_file).resolve()
                if not resolved.is_file():
                    resolved = texture_file.resolve()
                item.set("file", str(resolved))
            base_asset.append(item)
        elif kind == "material":
            item.set("name", VISUAL_MATERIAL_NAME)
            if item.get("texture"):
                item.set("texture", "tomato_texture")
            base_asset.append(item)

    if VISUAL_MESH_NAME not in mesh_name_map.values():
        # Normalize older/alternate YCB XML naming.
        mesh = ET.Element("mesh", {"name": VISUAL_MESH_NAME, "file": str(visual_file.resolve())})
        base_asset.append(mesh)
        mesh_name_map["__visual_fallback__"] = VISUAL_MESH_NAME
    material = base_asset.find(f"material[@name='{VISUAL_MATERIAL_NAME}']")
    if material is None:
        ET.SubElement(base_asset, "material", {"name": VISUAL_MATERIAL_NAME, "texture": "tomato_texture"})
    if base_asset.find("texture[@name='tomato_texture']") is None:
        ET.SubElement(base_asset, "texture", {
            "name": "tomato_texture", "type": "2d", "file": str(texture_file.resolve())
        })

    body = ET.fromstring(ET.tostring(source_body, encoding="unicode"))
    body.set("name", OBJECT_BODY_NAME)
    body.set("pos", original_pos)
    body.attrib.pop("euler", None)
    freejoint = body.find("freejoint")
    if freejoint is None:
        raise ValueError(f"{object_name}: object body has no freejoint")
    freejoint.set("name", OBJECT_FREEJOINT_NAME)

    geom_entries = [g for g in list(body) if g.tag == "geom"]
    visual_geoms = [g for g in geom_entries if "visual" in g.get("name", "")]
    if not visual_geoms:
        # Usually the first mesh geom is visual in this prepared YCB collection.
        visual_geoms = [g for g in geom_entries if g.get("mesh") in mesh_name_map.values()][:1]
    if not visual_geoms:
        raise ValueError(f"{object_name}: could not identify visual geom")
    for geom in geom_entries:
        source_mesh = geom.get("mesh")
        if source_mesh in mesh_name_map:
            geom.set("mesh", mesh_name_map[source_mesh])
        if geom in visual_geoms:
            set_mesh_collision(geom, True)
        elif geom.get("type", "mesh") == "mesh":
            set_mesh_collision(geom, False, collision_mesh_index - len(collision_files) + 0)

    # Assign stable sequential collision ids independent of source XML's naming.
    collision_geoms = [g for g in geom_entries if g not in visual_geoms and g.get("type", "mesh") == "mesh"]
    if len(collision_geoms) != len(used_collision_files):
        raise ValueError(
            f"{object_name}: XML has {len(collision_geoms)} collision geoms, "
            f"but references {len(used_collision_files)} collision meshes"
        )
    for index, geom in enumerate(collision_geoms):
        set_mesh_collision(geom, False, index)

    # All YCB objects in this prepared library are centered at their inertial origin.
    if body.find(f"site[@name='{CENTER_SITE_NAME}']") is None:
        ET.SubElement(body, "site", {
            "name": CENTER_SITE_NAME, "type": "sphere", "pos": "0 0 0",
            "size": "0.0035", "rgba": "1 0 1 0.9",
        })
    base_worldbody.append(body)

    output.parent.mkdir(parents=True, exist_ok=True)
    base_tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects", nargs="*", help="Object names; overrides the 10-object default")
    parser.add_argument("--object-count", type=int, default=DEFAULT_OBJECT_COUNT,
                        help="Number of default objects to build; 0 means all prepared objects")
    parser.add_argument("--all-objects", action="store_true",
                        help="Build every prepared object (same as --object-count 0)")
    parser.add_argument("--compile-check", action="store_true",
                        help="Compile generated robot scenes with installed MuJoCo")
    args = parser.parse_args()

    if not args.template.is_file():
        parser.error(f"Robot template not found: {args.template}")
    available = sorted(
        p.stem for p in args.object_root.glob("*.xml")
        if (args.object_root / p.stem / "textured.obj").is_file()
    )
    if args.objects:
        names = args.objects
    elif args.all_objects or args.object_count == 0:
        names = available
    else:
        if args.object_count < 1:
            parser.error("--object-count must be positive, or 0 for all objects")
        preferred = [name for name in DEFAULT_OBJECTS if name in available]
        names = (preferred + [name for name in available if name not in preferred])[:args.object_count]
    if not names:
        parser.error(f"No prepared YCB objects found in {args.object_root}")

    failures = []
    for name in names:
        try:
            output = build_scene(args.template, args.object_root, name,
                                 args.output_dir / f"ur3_threefinger_{name}.xml")
            if args.compile_check:
                import mujoco
                mujoco.MjModel.from_xml_path(str(output))
            print(f"OK {name}: {output}")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"Generated: {len(names) - len(failures)}/{len(names)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
