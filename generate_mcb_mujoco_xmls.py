from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = SCRIPT_DIR / "dataset_org_norm"
BASE_XML = PROJECT_DIR / "xml" / "ur3_with_threefinger_gelsight_table_tomato.xml"
OUTPUT_DIR = SCRIPT_DIR / "mujoco_xml"
OBJECT_POS = (0.45, 0.0, 0.80)
PART_SIZE_MM = 50.0
PART_MASS_KG = 0.05

# Ten mechanically useful tactile categories, with two distinct MCB objects
# per category. The official train/test split keeps the second object held out.
CLASSES = [
    ("hex_nut", "Hexagonal nuts"),
    ("cap_nut", "Cap nuts"),
    ("castle_nut", "Castle nuts"),
    ("cylindrical_pin", "Cylindrical pins"),
    ("grooved_pin", "Grooved pins"),
    ("taper_pin", "Taper pins"),
    ("split_pin", "Split pins"),
    ("countersunk_screw", "Screws and bolts with countersunk head"),
    ("hex_head_screw", "Screws and bolts with hexagonal head"),
    ("spring_washer", "Spring washers"),
]
SPLITS = ("train", "test")
OLD_MESH_NAMES = {
    "tomato_visual_mesh",
    "tomato_col0_mesh",
    "tomato_col1_mesh",
    "tomato_col2_mesh",
    "tomato_col3_mesh",
    "tomato_col4_mesh",
}


def obj_bounds(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 4 and fields[0] == "v":
                try:
                    vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
                except ValueError:
                    continue
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices.max(axis=0) - vertices.min(axis=0)


def generate_one(class_id: str, category: str, split: str) -> dict:
    category_dir = DATASET_DIR / split / category
    objects = sorted(category_dir.glob("*.obj"))
    if not objects:
        raise FileNotFoundError(f"No OBJ files in {category_dir}")
    source = objects[0]
    extents = obj_bounds(source)
    longest = float(extents.max())
    if not math.isfinite(longest) or longest <= 0:
        raise ValueError(f"Invalid mesh bounds for {source}: {extents}")

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(BASE_XML, parser=parser)
    root = tree.getroot()
    root.set("model", f"ur3_mcb_{class_id}_{split}")

    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"Missing <compiler> in {BASE_XML}")
    # Keep references to the UR3 and gripper mesh files valid from this output folder.
    compiler.set("meshdir", BASE_XML.parent.as_posix())

    option = root.find("option")
    if option is None:
        raise ValueError(f"Missing <option> in {BASE_XML}")
    option.set("gravity", "0 0 0")

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"Missing <asset> in {BASE_XML}")
    for node in list(asset.findall("mesh")):
        if node.get("name") in OLD_MESH_NAMES:
            asset.remove(node)

    scale = (PART_SIZE_MM / 1000.0) / longest
    part_geom = f"mcb_{class_id}_{split}_mesh"
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": part_geom,
            "file": source.resolve().as_posix(),
            "scale": f"{scale:.12g} {scale:.12g} {scale:.12g}",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {BASE_XML}")
    old_body = worldbody.find("./body[@name='tomato_soup_can']")
    if old_body is None:
        raise ValueError(f"Missing tomato_soup_can body in {BASE_XML}")
    worldbody.remove(old_body)

    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "mcb_part", "pos": " ".join(f"{v:.6f}" for v in OBJECT_POS)},
    )
    ET.SubElement(body, "freejoint", {"name": "mcb_part_free"})
    ixx = PART_MASS_KG * ((extents[1] * scale) ** 2 + (extents[2] * scale) ** 2) / 12.0
    iyy = PART_MASS_KG * ((extents[0] * scale) ** 2 + (extents[2] * scale) ** 2) / 12.0
    izz = PART_MASS_KG * ((extents[0] * scale) ** 2 + (extents[1] * scale) ** 2) / 12.0
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{PART_MASS_KG:.8g}",
            "diaginertia": f"{max(ixx, 1e-8):.8g} {max(iyy, 1e-8):.8g} {max(izz, 1e-8):.8g}",
        },
    )
    common = {
        "type": "mesh",
        "mesh": part_geom,
        "density": "0",
    }
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": "mcb_part_visual",
            "rgba": "0.20 0.55 0.78 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": "mcb_part_collision",
            "rgba": "0.15 0.45 0.80 0.18",
            "contype": "4",
            "conaffinity": "3",
            "condim": "4",
            "friction": "0.65 0.02 0.001",
            "group": "3",
        },
    )
    ET.SubElement(body, "site", {"name": "mcb_part_center", "type": "sphere", "pos": "0 0 0", "size": "0.002", "rgba": "1 0 1 0.9"})

    out_name = f"{class_id}_{split}_{source.stem}.xml"
    output_path = OUTPUT_DIR / out_name
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "label": class_id,
        "category": category,
        "split": split,
        "object_id": source.stem,
        "source_obj": source.resolve().as_posix(),
        "xml_file": output_path.resolve().as_posix(),
        "source_extents_normalized": " ".join(f"{v:.8g}" for v in extents),
        "source_longest_extent": f"{longest:.8g}",
        "mesh_scale": f"{scale:.12g}",
        "simulated_longest_mm": f"{PART_SIZE_MM:.1f}",
        "gravity_m_s2": "0 0 0",
    }


def main() -> None:
    if not BASE_XML.is_file():
        raise FileNotFoundError(f"Base robot XML not found: {BASE_XML}")
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"MCB dataset not found: {DATASET_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for class_id, category in CLASSES:
        for split in SPLITS:
            records.append(generate_one(class_id, category, split))

    manifest = OUTPUT_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} MJCF files in: {OUTPUT_DIR}")
    print(f"Manifest: {manifest}")
    print(f"Gravity: 0 0 0; normalized meshes scaled to {PART_SIZE_MM:.1f} mm longest edge.")
    for row in records:
        print(f"{row['label']:20s} {row['split']:5s} {row['object_id']} -> {Path(row['xml_file']).name}")


if __name__ == "__main__":
    main()
