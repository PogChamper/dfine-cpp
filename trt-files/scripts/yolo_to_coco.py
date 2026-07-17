#!/usr/bin/env python3
"""Convert a D-FINE-seg YOLO dataset split to COCO detection annotations.

The input is the training layout produced by D-FINE-seg's `make split`:

    dataset/
    ├── images/       # all images
    ├── labels/       # one `class_id xc yc w h` (normalized) .txt per image
    └── <split>.csv   # image names, one per line

Class order must match the checkpoint's training `label_to_name` mapping; a
label file may be absent for images without objects. The output is the
`instances_<split>.json` that the evaluation tooling consumes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from evaluation_report import atomic_json, sha256_file


def _fail(message: str):
    raise SystemExit(f"[yolo2coco]: {message}")


def resolve_class_names(value: str) -> list[str]:
    """A file (one name per line) or a comma list, matching `--class-names` elsewhere."""
    path = Path(value)
    names = (
        [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else [item.strip() for item in value.split(",") if item.strip()]
    )
    if not names or len(names) != len(set(names)):
        _fail("--class-names must list unique, non-empty names")
    return names


def split_names(path: Path) -> list[str]:
    if not path.is_file():
        _fail(f"split file does not exist: {path}")
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        _fail(f"split file must list unique image names: {path}")
    return names


def convert(dataset: Path, split: str, class_names: list[str]) -> dict:
    names = split_names(dataset / f"{split}.csv")

    images = []
    annotations = []
    annotation_id = 1
    for image_id, name in enumerate(names, start=1):
        image_path = dataset / "images" / name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            _fail(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})

        label_path = dataset / "labels" / f"{Path(name).stem}.txt"
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                _fail(f"{label_path}:{line_number}: expected `class xc yc w h`")
            try:
                category, xc, yc, box_width, box_height = map(float, fields)
            except ValueError:
                _fail(f"{label_path}:{line_number}: fields must be numbers")
            category_index = int(category)
            if category != category_index or not 0 <= category_index < len(class_names):
                _fail(f"{label_path}:{line_number}: class id outside the declared names")
            pixel_width = box_width * width
            pixel_height = box_height * height
            if pixel_width <= 0 or pixel_height <= 0:
                _fail(f"{label_path}:{line_number}: non-positive box")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_index + 1,
                    "bbox": [
                        (xc - box_width / 2) * width,
                        (yc - box_height / 2) * height,
                        pixel_width,
                        pixel_height,
                    ],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    if not annotations:
        _fail(f"{dataset}/{split}: no boxes; COCO evaluation needs a labeled split")
    return {
        "info": {
            "description": "YOLO split converted by yolo_to_coco.py",
            "dataset": dataset.name,
            "split": split,
            "split_csv_sha256": sha256_file(dataset / f"{split}.csv"),
            "converter_sha256": sha256_file(Path(__file__).resolve()),
            "class_names": class_names,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "object"}
            for index, name in enumerate(class_names, start=1)
        ],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a D-FINE-seg YOLO dataset split to COCO annotations"
    )
    parser.add_argument("--dataset", required=True, help="directory with images/, labels/, <split>.csv")
    parser.add_argument("--split", default="val", help="split name; reads <split>.csv (default: val)")
    parser.add_argument(
        "--class-names",
        required=True,
        help="training class order: a file (one per line) or a comma list",
    )
    parser.add_argument("--output", required=True, help="COCO annotations output path")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(args) -> int:
    dataset = Path(args.dataset)
    if not (dataset / "images").is_dir() or not (dataset / "labels").is_dir():
        _fail(f"dataset needs images/ and labels/: {dataset}")
    class_names = resolve_class_names(args.class_names)
    payload = convert(dataset, args.split, class_names)
    atomic_json(args.output, payload, overwrite=args.overwrite, sort_keys=True)
    print(
        f"[yolo2coco] wrote {args.output}: {len(payload['images'])} images, "
        f"{len(payload['annotations'])} boxes, {len(class_names)} classes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
