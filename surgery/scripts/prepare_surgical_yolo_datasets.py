"""
Convert the surgical datasets into YOLO-ready tool and anatomy datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import yaml


TOOLS_CLASSES = [
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
    "specimen_bag",
]

ANATOMY_CLASSES = [
    "gallbladder",
    "cystic_duct",
    "cystic_artery",
    "cystic_plate",
    "hepatocystic_triangle",
]

M2CAI_CLASS_MAP = {
    "grasper": "grasper",
    "bipolar": "bipolar",
    "hook": "hook",
    "scissors": "scissors",
    "clipper": "clipper",
    "irrigator": "irrigator",
    "specimenbag": "specimen_bag",
}

ENDOSCAPES_CLASS_MAP = {
    "gallbladder": "gallbladder",
    "cysticduct": "cystic_duct",
    "cysticartery": "cystic_artery",
    "cysticplate": "cystic_plate",
    "hepatocystictriangledissection": "hepatocystic_triangle",
}

CHOLECSEG8K_CLASS_IDS = {
    8: "cystic_duct",
    10: "gallbladder",
}

CHOLEC80_BOXES_CLASS_MAP = {
    "grasper": "grasper",
    "bipolar": "bipolar",
    "hook": "hook",
    "scissors": "scissors",
    "clipper": "clipper",
    "irrigator": "irrigator",
    "specimenbag": "specimen_bag",
}


def normalize_label(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def canonicalize_label(name: str, mapping: Dict[str, str]) -> Optional[str]:
    key = normalize_label(name).replace("__", "_")
    key = key.replace("specimen_bag", "specimenbag")
    key = key.replace("cystic_duct", "cysticduct")
    key = key.replace("cystic_artery", "cysticartery")
    key = key.replace("cystic_plate", "cysticplate")
    key = key.replace("hepatocystic_triangle_dissection", "hepatocystictriangledissection")
    key = key.replace("hepatocystic_triangle", "hepatocystictriangledissection")
    return mapping.get(key)


def ensure_dataset_tree(root: Path):
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_label_file(
    path: Path,
    boxes: Iterable[Tuple[int, float, float, float, float]],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for class_id, x_center, y_center, width, height in boxes:
            handle.write(
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    x_center = (x1 + box_w / 2.0) / width
    y_center = (y1 + box_h / 2.0) / height
    return x_center, y_center, box_w / width, box_h / height


def xywh_to_yolo(
    x: float,
    y: float,
    box_w: float,
    box_h: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x_center = (x + box_w / 2.0) / width
    y_center = (y + box_h / 2.0) / height
    return x_center, y_center, box_w / width, box_h / height


def emit_example(
    source_image: Path,
    split: str,
    dataset_prefix: str,
    boxes: List[Tuple[int, float, float, float, float]],
    output_root: Path,
):
    image_name = f"{dataset_prefix}__{source_image.stem}{source_image.suffix.lower()}"
    dest_image = output_root / "images" / split / image_name
    dest_label = output_root / "labels" / split / f"{dataset_prefix}__{source_image.stem}.txt"
    hardlink_or_copy(source_image, dest_image)
    write_yolo_label_file(dest_label, boxes)


def write_dataset_yaml(path: Path, dataset_root: Path, class_names: List[str]):
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def resolve_root_with_marker(base_dir: Path, marker_name: str) -> Path:
    if (base_dir / marker_name).exists():
        return base_dir
    for candidate in base_dir.rglob(marker_name):
        return candidate.parent
    return base_dir


def prepare_m2cai(root_dir: Path, output_root: Path) -> Dict[str, int]:
    dataset_root = resolve_root_with_marker(
        root_dir / "m2cai16_tool_locations" / "raw",
        "JPEGImages",
    )
    images_dir = dataset_root / "JPEGImages"
    annotations_dir = dataset_root / "Annotations"
    counts = {"train": 0, "val": 0, "test": 0}
    class_to_id = {name: idx for idx, name in enumerate(TOOLS_CLASSES)}

    for split_name in ("train", "val", "test"):
        split_file = dataset_root / f"{split_name}.txt"
        if not split_file.exists():
            continue
        frame_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for frame_id in frame_ids:
            xml_path = annotations_dir / f"{frame_id}.xml"
            image_path = images_dir / f"{frame_id}.jpg"
            if not xml_path.exists() or not image_path.exists():
                continue

            root = ET.parse(xml_path).getroot()
            size_node = root.find("size")
            width = int(size_node.findtext("width"))
            height = int(size_node.findtext("height"))
            boxes: List[Tuple[int, float, float, float, float]] = []
            for obj in root.findall("object"):
                label = canonicalize_label(obj.findtext("name", ""), M2CAI_CLASS_MAP)
                if label is None:
                    continue
                bbox = obj.find("bndbox")
                x1 = float(bbox.findtext("xmin"))
                y1 = float(bbox.findtext("ymin"))
                x2 = float(bbox.findtext("xmax"))
                y2 = float(bbox.findtext("ymax"))
                boxes.append((class_to_id[label], *xyxy_to_yolo(x1, y1, x2, y2, width, height)))

            emit_example(image_path, split_name, "m2cai", boxes, output_root)
            counts[split_name] += 1

    return counts


def prepare_endoscapes(root_dir: Path, output_root: Path) -> Dict[str, int]:
    dataset_root = resolve_root_with_marker(root_dir / "endoscapes" / "raw", "train")
    counts = {"train": 0, "val": 0, "test": 0}
    class_to_id = {name: idx for idx, name in enumerate(ANATOMY_CLASSES)}

    for split_name in ("train", "val", "test"):
        split_dir = dataset_root / split_name
        annotation_path = split_dir / "annotation_coco.json"
        if not annotation_path.exists():
            continue

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {item["id"]: item for item in payload["images"]}
        categories = {item["id"]: canonicalize_label(item["name"], ENDOSCAPES_CLASS_MAP) for item in payload["categories"]}
        grouped: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)

        for annotation in payload["annotations"]:
            label = categories.get(annotation["category_id"])
            if label is None:
                continue
            image_info = images[annotation["image_id"]]
            width = int(image_info["width"])
            height = int(image_info["height"])
            x, y, box_w, box_h = annotation["bbox"]
            grouped[annotation["image_id"]].append(
                (class_to_id[label], *xywh_to_yolo(float(x), float(y), float(box_w), float(box_h), width, height))
            )

        for image_id, image_info in images.items():
            image_path = split_dir / image_info["file_name"]
            if not image_path.exists():
                continue
            emit_example(
                image_path,
                split_name,
                "endoscapes",
                grouped.get(image_id, []),
                output_root,
            )
            counts[split_name] += 1

    return counts


def find_cholecseg8k_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(dataset_root.rglob("*_endo.png")):
        parent = image_path.parent
        stem = image_path.stem.replace("_endo", "")
        candidates = list(parent.glob(f"{stem}*watershed*.png"))
        if not candidates:
            candidates = list(parent.glob(f"{stem}*mask*.png"))
        if not candidates:
            continue
        mask_path = sorted(candidates)[0]
        pairs.append((image_path, mask_path))
    return pairs


def split_clip_names(clip_names: List[str]) -> Dict[str, str]:
    total = len(clip_names)
    train_cut = int(total * 0.7)
    val_cut = int(total * 0.85)
    mapping = {}
    for index, clip_name in enumerate(sorted(clip_names)):
        if index < train_cut:
            mapping[clip_name] = "train"
        elif index < val_cut:
            mapping[clip_name] = "val"
        else:
            mapping[clip_name] = "test"
    return mapping


def prepare_cholecseg8k(root_dir: Path, output_root: Path, min_component_area: int) -> Dict[str, int]:
    dataset_root = root_dir / "cholecseg8k" / "raw"
    pairs = find_cholecseg8k_pairs(dataset_root)
    clip_names = sorted({image_path.parent.name for image_path, _ in pairs})
    clip_to_split = split_clip_names(clip_names)
    class_to_id = {name: idx for idx, name in enumerate(ANATOMY_CLASSES)}
    counts = {"train": 0, "val": 0, "test": 0}

    for image_path, mask_path in pairs:
        split_name = clip_to_split[image_path.parent.name]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        height, width = mask.shape[:2]
        boxes: List[Tuple[int, float, float, float, float]] = []

        for class_id, label in CHOLECSEG8K_CLASS_IDS.items():
            binary = (mask == class_id).astype("uint8")
            if binary.max() == 0:
                continue
            num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for component_idx in range(1, num_components):
                x, y, box_w, box_h, area = stats[component_idx]
                if int(area) < min_component_area:
                    continue
                boxes.append(
                    (class_to_id[label], *xywh_to_yolo(float(x), float(y), float(box_w), float(box_h), width, height))
                )

        emit_example(image_path, split_name, "cholecseg8k", boxes, output_root)
        counts[split_name] += 1

    return counts


def prepare_cholec80_boxes(root_dir: Path, output_root: Path) -> Dict[str, int]:
    dataset_root = root_dir / "cholec80_boxes" / "raw"
    csv_path = root_dir / "cholec80_boxes" / "downloads" / "ROI_Labels.csv"
    image_root = resolve_root_with_marker(dataset_root, "Images")
    counts = {"train": 0, "val": 0, "test": 0}
    class_to_id = {name: idx for idx, name in enumerate(TOOLS_CLASSES)}

    if not csv_path.exists() or not image_root.exists():
        return counts

    grouped_rows: Dict[str, List[dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            grouped_rows[row["FrameName"]].append(row)

    for frame_name, rows in grouped_rows.items():
        surgery_num = int(rows[0]["Surgery_num"])
        split_name = "train" if surgery_num in (41, 42, 43) else "val" if surgery_num == 44 else "test"
        relative_dir = rows[0]["Dir"].strip("\\/").replace("\\", "/")
        image_path = dataset_root / relative_dir / frame_name
        if not image_path.exists():
            image_path = image_root / f"Video_{surgery_num}" / frame_name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes: List[Tuple[int, float, float, float, float]] = []
        for row in rows:
            label = canonicalize_label(row["ToolName"], CHOLEC80_BOXES_CLASS_MAP)
            if label is None:
                continue
            x = float(row["BBox_X"])
            y = float(row["BBox_Y"])
            box_w = float(row["BBox_Width"])
            box_h = float(row["BBox_Height"])
            boxes.append((class_to_id[label], *xywh_to_yolo(x, y, box_w, box_h, width, height)))

        emit_example(image_path, split_name, "cholec80boxes", boxes, output_root)
        counts[split_name] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare tool and anatomy YOLO datasets from the downloaded surgical sources"
    )
    parser.add_argument("--root-dir", default="data/datasets", help="Dataset root produced by the downloader")
    parser.add_argument(
        "--output-dir",
        default="data/datasets/prepared",
        help="Destination directory for the YOLO-ready datasets",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=100,
        help="Minimum connected-component area for CholecSeg8k pseudo-boxes",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    tools_root = output_dir / "yolo_tools"
    anatomy_root = output_dir / "yolo_anatomy"
    ensure_dataset_tree(tools_root)
    ensure_dataset_tree(anatomy_root)

    summary = {
        "tools": {
            "m2cai16_tool_locations": prepare_m2cai(root_dir, tools_root),
            "cholec80_boxes": prepare_cholec80_boxes(root_dir, tools_root),
        },
        "anatomy": {
            "endoscapes": prepare_endoscapes(root_dir, anatomy_root),
            "cholecseg8k": prepare_cholecseg8k(root_dir, anatomy_root, args.min_component_area),
        },
    }

    write_dataset_yaml(output_dir / "yolo_tools_dataset.yaml", tools_root, TOOLS_CLASSES)
    write_dataset_yaml(output_dir / "yolo_anatomy_dataset.yaml", anatomy_root, ANATOMY_CLASSES)
    summary_path = output_dir / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {summary_path}")


if __name__ == "__main__":
    main()
