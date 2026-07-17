#!/usr/bin/env python3
"""Shared COCO bbox metrics and ground-truth summaries."""

from __future__ import annotations

import gc
from collections import Counter
from copy import deepcopy

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STANDARD_MAX_DETS = (1, 10, 100)
_DENSITY_RANGES = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-300", 201, 300),
    ("301+", 301, None),
)


def _mean_valid(values) -> float | None:
    values = np.asarray(values)
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def _selected_annotations(coco, img_ids: list[int]) -> tuple[list[dict], list[dict]]:
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
    evaluated = [ann for ann in annotations if not ann.get("iscrowd", 0)]
    crowd = [ann for ann in annotations if ann.get("iscrowd", 0)]
    return evaluated, crowd


def _area_counts(annotations: list[dict]) -> dict[str, int]:
    areas = np.asarray(
        [float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])) for ann in annotations]
    )
    return {
        "small": int((areas < 32**2).sum()),
        "medium": int(((areas >= 32**2) & (areas < 96**2)).sum()),
        "large": int((areas >= 96**2).sum()),
    }


def _model_space_transform(
    image: dict,
    model_hw: tuple[int, int],
    resize: str,
    letterbox_anchor: str,
    letterbox_upscale: bool,
) -> tuple[float, float, float, float]:
    target_height, target_width = model_hw
    if target_height <= 0 or target_width <= 0:
        raise ValueError("model-space dimensions must be positive")
    source_width = int(image["width"])
    source_height = int(image["height"])
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"image {image['id']} dimensions must be positive")
    if resize == "stretch":
        return target_width / source_width, target_height / source_height, 0.0, 0.0
    if resize != "letterbox":
        raise ValueError("model-space resize must be 'stretch' or 'letterbox'")
    if letterbox_anchor not in {"center", "topleft"}:
        raise ValueError("letterbox anchor must be 'center' or 'topleft'")
    if type(letterbox_upscale) is not bool:
        raise ValueError("letterbox upscale must be a boolean")

    scale = min(
        np.float32(target_width) / np.float32(source_width),
        np.float32(target_height) / np.float32(source_height),
    )
    if not letterbox_upscale and scale > np.float32(1.0):
        scale = np.float32(1.0)
    content_width = int(np.float32(np.float32(source_width) * scale + np.float32(0.5)))
    content_height = int(np.float32(np.float32(source_height) * scale + np.float32(0.5)))
    content_width = min(target_width, max(1, content_width))
    content_height = min(target_height, max(1, content_height))
    offset_x = 0 if letterbox_anchor == "topleft" else (target_width - content_width) // 2
    offset_y = 0 if letterbox_anchor == "topleft" else (target_height - content_height) // 2
    return float(scale), float(scale), float(offset_x), float(offset_y)


def _scaled_dataset(
    coco,
    detections: list[dict],
    image_ids: list[int],
    model_hw: tuple[int, int],
    resize: str,
    letterbox_anchor: str,
    letterbox_upscale: bool,
):
    target_height, target_width = model_hw
    images = [deepcopy(image) for image in coco.loadImgs(image_ids)]
    transforms = {}
    for image in images:
        transforms[int(image["id"])] = _model_space_transform(
            image,
            model_hw,
            resize,
            letterbox_anchor,
            letterbox_upscale,
        )
        image["width"] = target_width
        image["height"] = target_height

    annotations = []
    for source in coco.loadAnns(coco.getAnnIds(imgIds=image_ids)):
        annotation = deepcopy(source)
        scale_x, scale_y, offset_x, offset_y = transforms[int(annotation["image_id"])]
        x, y, width, height = annotation["bbox"]
        annotation["bbox"] = [
            float(x) * scale_x + offset_x,
            float(y) * scale_y + offset_y,
            float(width) * scale_x,
            float(height) * scale_y,
        ]
        annotation["area"] = float(
            annotation.get("area", float(width) * float(height)) * scale_x * scale_y
        )
        annotations.append(annotation)

    scaled = COCO()
    scaled.dataset = {
        "info": deepcopy(coco.dataset.get("info", {})),
        "licenses": deepcopy(coco.dataset.get("licenses", [])),
        "images": images,
        "categories": deepcopy(coco.loadCats(coco.getCatIds())),
        "annotations": annotations,
    }
    scaled.createIndex()

    scaled_detections = []
    for source in detections:
        scale_x, scale_y, offset_x, offset_y = transforms[int(source["image_id"])]
        x, y, width, height = source["bbox"]
        scaled_detections.append(
            {
                "image_id": source["image_id"],
                "category_id": source["category_id"],
                "bbox": [
                    float(x) * scale_x + offset_x,
                    float(y) * scale_y + offset_y,
                    float(width) * scale_x,
                    float(height) * scale_y,
                ],
                "score": source["score"],
            }
        )
    return scaled, scaled_detections


def _run_evaluator(coco, detections: list[dict], image_ids: list[int], summarize: bool):
    coco_dt = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_dt, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = list(STANDARD_MAX_DETS)
    evaluator.evaluate()
    evaluator.accumulate()
    if summarize:
        evaluator.summarize()
    return evaluator


def _area_metrics(evaluator) -> dict[str, float | None]:
    params = evaluator.params
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    area_index = {label: index for index, label in enumerate(params.areaRngLbl)}
    max_det_index = {value: index for index, value in enumerate(params.maxDets)}
    return {
        f"AP{area[0]}": _mean_valid(precision[:, :, :, area_index[area], max_det_index[100]])
        for area in ("small", "medium", "large")
    } | {
        f"AR{area[0]}": _mean_valid(recall[:, :, area_index[area], max_det_index[100]])
        for area in ("small", "medium", "large")
    }


def _bbox_metrics(coco, evaluator, image_ids: list[int]) -> dict:
    params = evaluator.params
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    area_index = {label: index for index, label in enumerate(params.areaRngLbl)}
    max_det_index = {value: index for index, value in enumerate(params.maxDets)}
    all_area = area_index["all"]
    max_det_100 = max_det_index[100]

    def ap(*, iou_index=None, area="all"):
        values = precision[:, :, :, area_index[area], max_det_100]
        if iou_index is not None:
            values = values[iou_index]
        return _mean_valid(values)

    def ar(max_dets: int, *, area="all"):
        values = recall[:, :, area_index[area], max_det_index[max_dets]]
        return _mean_valid(values)

    ap_by_iou = {
        f"{float(iou):.2f}": ap(iou_index=index) for index, iou in enumerate(params.iouThrs)
    }
    categories = {cat["id"]: cat for cat in coco.loadCats(params.catIds)}
    evaluated, _ = _selected_annotations(coco, image_ids)
    gt_by_category = Counter(int(ann["category_id"]) for ann in evaluated)
    per_class = []
    for index, category_id in enumerate(params.catIds):
        category_id = int(category_id)
        category = categories[category_id]
        per_class.append(
            {
                "category_id": category_id,
                "name": str(category.get("name", category_id)),
                "gt_instances": gt_by_category[category_id],
                "AP": _mean_valid(precision[:, :, index, all_area, max_det_100]),
            }
        )

    return {
        "AP": ap(),
        "AP50": ap_by_iou["0.50"],
        "AP75": ap_by_iou["0.75"],
        "APs": ap(area="small"),
        "APm": ap(area="medium"),
        "APl": ap(area="large"),
        "AR1": ar(1),
        "AR10": ar(10),
        "AR100": ar(100),
        "ARs": ar(100, area="small"),
        "ARm": ar(100, area="medium"),
        "ARl": ar(100, area="large"),
        "max_dets": list(params.maxDets),
        "AP_by_iou": ap_by_iou,
        "per_class": per_class,
        "GT_by_area": _area_counts(evaluated),
    }


def ground_truth_summary(coco, img_ids) -> dict:
    """Summarize the non-crowd GT population used by bbox evaluation."""
    image_ids = [int(image_id) for image_id in img_ids]
    if not image_ids:
        raise ValueError("COCO evaluation requires at least one image")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("COCO image ids must be unique")

    evaluated, crowd = _selected_annotations(coco, image_ids)
    by_image = Counter(int(ann["image_id"]) for ann in evaluated)
    counts = np.asarray([by_image[image_id] for image_id in image_ids], dtype=np.int64)
    histogram = []
    for label, lower, upper in _DENSITY_RANGES:
        selected = counts >= lower
        if upper is not None:
            selected &= counts <= upper
        histogram.append({"range": label, "images": int(selected.sum())})

    return {
        "images": len(image_ids),
        "gt_instances": int(counts.sum()),
        "crowd_instances": len(crowd),
        "per_image": {
            "min": int(counts.min()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
            "over_100": int((counts > 100).sum()),
            "histogram": histogram,
        },
    }


def evaluate_bbox(
    coco,
    detections: list[dict],
    img_ids,
    *,
    summarize: bool = True,
    model_hw: tuple[int, int] | None = None,
    model_resize: str = "stretch",
    letterbox_anchor: str = "center",
    letterbox_upscale: bool = True,
    letterbox_pad: int = 114,
) -> dict:
    """Evaluate standard COCO bbox metrics and detailed AP slices."""
    image_ids = [int(image_id) for image_id in img_ids]
    evaluator = _run_evaluator(coco, detections, image_ids, summarize)
    metrics = _bbox_metrics(coco, evaluator, image_ids)
    if model_hw is not None:
        if model_resize == "letterbox" and (
            type(letterbox_pad) is not int or not 0 <= letterbox_pad <= 255
        ):
            raise ValueError("letterbox pad must be an integer in [0, 255]")
        del evaluator
        gc.collect()
        scaled_coco, scaled_detections = _scaled_dataset(
            coco,
            detections,
            image_ids,
            model_hw,
            model_resize,
            letterbox_anchor,
            letterbox_upscale,
        )
        scaled_evaluator = _run_evaluator(
            scaled_coco, scaled_detections, image_ids, summarize=False
        )
        scaled_annotations, _ = _selected_annotations(scaled_coco, image_ids)
        scaled_area_metrics = _area_metrics(scaled_evaluator)
        del scaled_evaluator
        metrics["model_space_area"] = {
            "input_h": model_hw[0],
            "input_w": model_hw[1],
            "resize": model_resize,
            **scaled_area_metrics,
            "GT_by_area": _area_counts(scaled_annotations),
        }
        if model_resize == "letterbox":
            metrics["model_space_area"].update(
                {
                    "letterbox_anchor": letterbox_anchor,
                    "letterbox_pad": letterbox_pad,
                    "letterbox_upscale": letterbox_upscale,
                }
            )
    return metrics
