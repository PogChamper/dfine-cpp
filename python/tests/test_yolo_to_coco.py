from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "trt-files/scripts/yolo_to_coco.py"
SPEC = importlib.util.spec_from_file_location("yolo_to_coco", SCRIPT)
YOLO2COCO = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(YOLO2COCO)
finally:
    sys.path.remove(str(SCRIPT.parent))


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    cv2.imwrite(str(dataset / "images" / "one.jpg"), np.zeros((6, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(dataset / "images" / "two.jpg"), np.zeros((10, 4, 3), dtype=np.uint8))
    (dataset / "labels" / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n")
    # `two` has no label file: an image without objects stays in the split.
    (dataset / "val.csv").write_text("one.jpg\ntwo.jpg\n")
    return dataset


def test_convert_produces_coco_boxes_and_provenance(tmp_path):
    dataset = _dataset(tmp_path)
    payload = YOLO2COCO.convert(dataset, "val", ["stray", "target"])

    assert [image["file_name"] for image in payload["images"]] == ["one.jpg", "two.jpg"]
    assert payload["images"][0] == {"id": 1, "file_name": "one.jpg", "width": 8, "height": 6}
    first, second = payload["annotations"]
    # 8x6 image: xc=.5, yc=.5, w=.5, h=.5 -> xywh (2, 1.5, 4, 3).
    assert first["image_id"] == 1
    assert first["category_id"] == 1
    assert first["bbox"] == [2.0, 1.5, 4.0, 3.0]
    assert first["area"] == 12.0
    assert second["category_id"] == 2
    assert [category["name"] for category in payload["categories"]] == ["stray", "target"]
    assert payload["categories"][0]["id"] == 1
    assert payload["info"]["split_csv_sha256"]
    assert payload["info"]["class_names"] == ["stray", "target"]


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ("0 0.5 0.5 0.5\n", "expected"),
        ("2 0.5 0.5 0.5 0.5\n", "class id"),
        ("0 0.5 0.5 0 0.5\n", "non-positive"),
        ("x 0.5 0.5 0.5 0.5\n", "numbers"),
    ],
)
def test_convert_rejects_malformed_labels(tmp_path, label, match):
    dataset = _dataset(tmp_path)
    (dataset / "labels" / "one.txt").write_text(label)
    with pytest.raises(SystemExit, match=match):
        YOLO2COCO.convert(dataset, "val", ["stray", "target"])


def test_convert_rejects_broken_splits(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "val.csv").write_text("one.jpg\none.jpg\n")
    with pytest.raises(SystemExit, match="unique"):
        YOLO2COCO.convert(dataset, "val", ["stray", "target"])

    (dataset / "val.csv").write_text("missing.jpg\n")
    with pytest.raises(SystemExit, match="cannot read image"):
        YOLO2COCO.convert(dataset, "val", ["stray", "target"])

    with pytest.raises(SystemExit, match="split file does not exist"):
        YOLO2COCO.convert(dataset, "test", ["stray", "target"])


def test_convert_requires_at_least_one_box(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "labels" / "one.txt").unlink()
    with pytest.raises(SystemExit, match="no boxes"):
        YOLO2COCO.convert(dataset, "val", ["stray", "target"])


def test_class_names_accept_file_or_comma_list(tmp_path):
    names = tmp_path / "names.txt"
    names.write_text("stray\ntarget\n")
    assert YOLO2COCO.resolve_class_names(str(names)) == ["stray", "target"]
    assert YOLO2COCO.resolve_class_names("stray, target") == ["stray", "target"]
    with pytest.raises(SystemExit, match="unique"):
        YOLO2COCO.resolve_class_names("stray,stray")


def test_main_writes_atomically_and_respects_overwrite(tmp_path):
    dataset = _dataset(tmp_path)
    output = tmp_path / "instances_val.json"
    args = YOLO2COCO.parse_args(
        [
            "--dataset",
            str(dataset),
            "--split",
            "val",
            "--class-names",
            "stray,target",
            "--output",
            str(output),
        ]
    )
    assert YOLO2COCO.main(args) == 0
    written = json.loads(output.read_text())
    assert len(written["annotations"]) == 2

    with pytest.raises(ValueError, match="already exists"):
        YOLO2COCO.main(args)
