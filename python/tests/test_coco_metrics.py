from __future__ import annotations

import importlib.util
import json
import weakref
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
COCO = pytest.importorskip("pycocotools.coco").COCO
COCOeval = pytest.importorskip("pycocotools.cocoeval").COCOeval

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def metrics_module():
    path = REPO / "trt-files/scripts/coco_metrics.py"
    spec = importlib.util.spec_from_file_location("coco_metrics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coco(images, categories, annotations):
    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    coco.createIndex()
    return coco


def _dense_dataset(count=150):
    annotations = []
    detections = []
    for index in range(count):
        bbox = [(index % 15) * 25, (index // 15) * 25, 10, 10]
        annotations.append(
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "area": 100,
                "iscrowd": 0,
            }
        )
        detections.append(
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "score": 1.0 - index * 1e-5,
            }
        )
    coco = _coco(
        [{"id": 1, "width": 400, "height": 400, "file_name": "dense.jpg"}],
        [{"id": 1, "name": "cell"}],
        annotations,
    )
    return coco, detections


def test_dense_dataset_uses_standard_coco_max_dets(metrics_module):
    coco, detections = _dense_dataset()
    metrics = metrics_module.evaluate_bbox(coco, detections, [1], summarize=False)
    ground_truth = metrics_module.ground_truth_summary(coco, [1])

    assert metrics["AR100"] == pytest.approx(2 / 3)
    assert metrics["max_dets"] == [1, 10, 100]
    assert list(metrics["AP_by_iou"]) == [f"{iou:.2f}" for iou in np.arange(0.5, 1.0, 0.05)]
    assert metrics["per_class"] == [
        {
            "category_id": 1,
            "name": "cell",
            "gt_instances": 150,
            "AP": metrics["AP"],
        }
    ]
    assert ground_truth["gt_instances"] == 150
    assert ground_truth["per_image"]["over_100"] == 1
    assert ground_truth["per_image"]["histogram"][7] == {
        "range": "101-200",
        "images": 1,
    }
    json.dumps({"map": metrics, "ground_truth": ground_truth})


def test_standard_coco_metrics_remain_compatible(metrics_module):
    coco, detections = _dense_dataset()
    metrics = metrics_module.evaluate_bbox(coco, detections, [1], summarize=False)

    standard = COCOeval(coco, coco.loadRes(detections), iouType="bbox")
    standard.params.imgIds = [1]
    standard.evaluate()
    standard.accumulate()
    standard.summarize()

    for name, index in (
        ("AP", 0),
        ("AP50", 1),
        ("AP75", 2),
        ("APs", 3),
        ("AR1", 6),
        ("AR10", 7),
        ("AR100", 8),
        ("ARs", 9),
    ):
        assert metrics[name] == pytest.approx(standard.stats[index])
    assert metrics["APm"] is None
    assert metrics["APl"] is None
    assert metrics["ARm"] is None
    assert metrics["ARl"] is None


def test_per_class_counts_and_density_exclude_crowd(metrics_module):
    images = [
        {"id": 1, "width": 100, "height": 100, "file_name": "one.jpg"},
        {"id": 2, "width": 100, "height": 100, "file_name": "two.jpg"},
        {"id": 3, "width": 100, "height": 100, "file_name": "negative.jpg"},
    ]
    categories = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [20, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 2,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 4,
            "image_id": 2,
            "category_id": 2,
            "bbox": [20, 0, 10, 10],
            "area": 100,
            "iscrowd": 1,
        },
    ]
    detections = [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 0.9,
        }
        for ann in annotations
        if not ann["iscrowd"]
    ]
    coco = _coco(images, categories, annotations)

    metrics = metrics_module.evaluate_bbox(coco, detections, [1, 2, 3], summarize=False)
    ground_truth = metrics_module.ground_truth_summary(coco, [1, 2, 3])

    assert [(row["name"], row["gt_instances"], row["AP"]) for row in metrics["per_class"]] == [
        ("alpha", 2, pytest.approx(1.0)),
        ("beta", 1, pytest.approx(1.0)),
    ]
    assert ground_truth["gt_instances"] == 3
    assert ground_truth["crowd_instances"] == 1
    assert ground_truth["per_image"]["min"] == 0
    assert ground_truth["per_image"]["median"] == 1.0
    assert ground_truth["per_image"]["max"] == 2
    histogram = {row["range"]: row["images"] for row in ground_truth["per_image"]["histogram"]}
    assert histogram["0"] == 1
    assert histogram["1"] == 1
    assert histogram["2-5"] == 1


def test_model_space_area_bins_are_reported_separately(metrics_module):
    images = [{"id": 1, "width": 1000, "height": 1000, "file_name": "large-canvas.jpg"}]
    categories = [{"id": 1, "name": "object"}]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 40, 40],
            "area": 1600,
            "iscrowd": 0,
        }
    ]
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 40, 40],
            "score": 0.99,
        }
    ]
    coco = _coco(images, categories, annotations)

    metrics = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
    )

    assert metrics["APs"] is None
    assert metrics["APm"] == pytest.approx(1.0)
    assert metrics["GT_by_area"] == {"small": 0, "medium": 1, "large": 0}
    assert metrics["model_space_area"] == {
        "input_h": 640,
        "input_w": 640,
        "resize": "stretch",
        "APs": pytest.approx(1.0),
        "APm": None,
        "APl": None,
        "ARs": pytest.approx(1.0),
        "ARm": None,
        "ARl": None,
        "GT_by_area": {"small": 1, "medium": 0, "large": 0},
    }


@pytest.mark.parametrize(
    ("resize", "anchor", "upscale", "expected_bbox", "expected_area"),
    [
        ("stretch", "center", True, [32.0, 128.0, 96.0, 256.0], 24576.0),
        ("letterbox", "center", True, [32.0, 224.0, 96.0, 128.0], 12288.0),
        ("letterbox", "topleft", True, [32.0, 64.0, 96.0, 128.0], 12288.0),
        ("letterbox", "center", False, [230.0, 290.0, 30.0, 40.0], 1200.0),
        ("letterbox", "topleft", False, [10.0, 20.0, 30.0, 40.0], 1200.0),
    ],
)
def test_model_space_transform_matches_runtime_geometry(
    metrics_module,
    resize,
    anchor,
    upscale,
    expected_bbox,
    expected_area,
):
    coco = _coco(
        [{"id": 1, "width": 200, "height": 100, "file_name": "frame.jpg"}],
        [{"id": 1, "name": "object"}],
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            }
        ],
    )
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 20, 30, 40],
            "score": 0.9,
        }
    ]

    scaled, scaled_detections = metrics_module._scaled_dataset(
        coco,
        detections,
        [1],
        (640, 640),
        resize,
        anchor,
        upscale,
    )

    annotation = scaled.loadAnns([1])[0]
    assert annotation["bbox"] == pytest.approx(expected_bbox)
    assert annotation["area"] == pytest.approx(expected_area)
    assert scaled_detections[0]["bbox"] == pytest.approx(expected_bbox)


def test_scaled_detections_exclude_evaluator_fields(metrics_module):
    coco, detections = _dense_dataset(count=1)
    detections[0].update(
        {
            "area": 100.0,
            "id": 9,
            "iscrowd": 0,
            "segmentation": [[0, 0, 0, 10, 10, 10, 10, 0]],
        }
    )

    _, scaled_detections = metrics_module._scaled_dataset(
        coco,
        detections,
        [1],
        (640, 640),
        "stretch",
        "center",
        True,
    )

    assert scaled_detections == [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": pytest.approx([0.0, 0.0, 16.0, 16.0]),
            "score": 1.0,
        }
    ]


def test_model_space_releases_canonical_evaluator_before_scaling(metrics_module, monkeypatch):
    coco, detections = _dense_dataset(count=1)
    run_evaluator = metrics_module._run_evaluator
    scale_dataset = metrics_module._scaled_dataset
    evaluator_refs = []

    def tracked_evaluator(*args, **kwargs):
        evaluator = run_evaluator(*args, **kwargs)
        evaluator_refs.append(weakref.ref(evaluator))
        return evaluator

    def tracked_scaling(*args, **kwargs):
        assert len(evaluator_refs) == 1
        assert evaluator_refs[0]() is None
        return scale_dataset(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "_run_evaluator", tracked_evaluator)
    monkeypatch.setattr(metrics_module, "_scaled_dataset", tracked_scaling)

    metrics = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
    )

    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["model_space_area"]["APs"] == pytest.approx(1.0)
    assert len(evaluator_refs) == 2


def test_centered_letterbox_uses_rounded_content_extent(metrics_module):
    transform = metrics_module._model_space_transform(
        {"id": 7, "width": 100, "height": 33},
        (640, 640),
        "letterbox",
        "center",
        True,
    )

    assert transform == pytest.approx((6.4, 6.4, 0.0, 214.0))


def test_letterbox_upscale_changes_only_model_space_area_bins(metrics_module):
    coco = _coco(
        [{"id": 1, "width": 200, "height": 100, "file_name": "frame.jpg"}],
        [{"id": 1, "name": "object"}],
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            }
        ],
    )
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 20, 30, 40],
            "score": 0.9,
        }
    ]

    upscale = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
        model_resize="letterbox",
    )
    no_upscale = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
        model_resize="letterbox",
        letterbox_upscale=False,
    )

    assert (
        upscale["GT_by_area"]
        == no_upscale["GT_by_area"]
        == {
            "small": 0,
            "medium": 1,
            "large": 0,
        }
    )
    assert upscale["model_space_area"]["GT_by_area"] == {
        "small": 0,
        "medium": 0,
        "large": 1,
    }
    assert no_upscale["model_space_area"]["GT_by_area"] == {
        "small": 0,
        "medium": 1,
        "large": 0,
    }


def test_letterbox_pad_is_recorded_without_changing_geometry(metrics_module):
    coco, detections = _dense_dataset(count=1)
    default = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
        model_resize="letterbox",
    )
    custom = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
        model_resize="letterbox",
        letterbox_pad=0,
    )

    assert default["model_space_area"]["letterbox_pad"] == 114
    assert custom["model_space_area"]["letterbox_pad"] == 0
    assert default["AP"] == custom["AP"]


def test_model_space_configuration_does_not_change_canonical_metrics(metrics_module):
    coco, detections = _dense_dataset()
    stretch = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
    )
    letterbox = metrics_module.evaluate_bbox(
        coco,
        detections,
        [1],
        summarize=False,
        model_hw=(640, 640),
        model_resize="letterbox",
        letterbox_anchor="center",
        letterbox_upscale=False,
    )

    for key in (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
        "AP_by_iou",
        "per_class",
        "GT_by_area",
    ):
        assert letterbox[key] == stretch[key]


@pytest.mark.parametrize(
    ("resize", "anchor", "upscale", "match"),
    [
        ("crop", "center", True, "resize"),
        ("letterbox", "bottom", True, "anchor"),
        ("letterbox", "center", 1, "boolean"),
    ],
)
def test_model_space_transform_rejects_invalid_geometry(
    metrics_module, resize, anchor, upscale, match
):
    with pytest.raises(ValueError, match=match):
        metrics_module._model_space_transform(
            {"id": 1, "width": 100, "height": 100},
            (640, 640),
            resize,
            anchor,
            upscale,
        )
