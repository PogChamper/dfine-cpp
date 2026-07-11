#!/usr/bin/env python3
"""Compare D-FINE export backends against the PyTorch .pt reference on real images.

Backends (all consume the IDENTICAL preprocessed 640x640 /255 RGB tensor):
  torch   : the .pt checkpoint in PyTorch            -> REFERENCE / etalon
  ort     : ONNXRuntime-GPU on the faithful ONNX     -> should match torch
  trt_basic   : TensorRT engine, GridSample deformable core (standard/naive export)
  trt_correct : TensorRT engine, explicit gather-bilinear core (the D-FINE-cpp fix)

For each backend B we score B's predictions treating the .pt predictions as ground
truth, two complementary ways:

  (A) Query-aligned parity. All backends share the same 300-query decoder with
      identical query->object semantics, so query index q means the same object
      across backends. For every .pt "signal" query (max-class score >= parity_thr)
      we compare box[q] directly: IoU, normalised-cxcywh L1, and |dscore|. This is
      the project's established parity method (parity_check.py) and isolates pure
      box drift.

  (B) COCO mAP vs reference. Treat .pt's confident detections (score >= gt_thr) as
      GT annotations; treat each backend's sigmoid->top-k detections as predictions;
      score with pycocotools. AP=1.0 means the backend reproduces .pt exactly. This
      is the standard "how well does B reproduce A" number, matched by IoU (no query
      alignment assumed). Reported class-aware and class-agnostic.

Preprocessing is stretch-to-640 + /255 + RGB + NCHW (no mean/std) -- the D-FINE-seg
inference contract. It only needs to be identical across backends for the comparison
to be valid, which it is.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# --------------------------- preprocessing & geometry ---------------------------

def preprocess(bgr: np.ndarray, img: int) -> torch.Tensor:
    resized = cv2.resize(bgr, (img, img), interpolation=cv2.INTER_LINEAR)  # stretch
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def iou_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise IoU of two [N,4] xyxy arrays."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a + area_b - inter + 1e-9)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[M,N] IoU between a[M,4] and b[N,4] xyxy. For alignment-free best-match scoring."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


# --------------------------- backends ---------------------------

class TorchBackend:
    def __init__(self, args):
        sys.path.insert(0, args.dfine_src)
        from src.d_fine.dfine import build_model
        from src.d_fine.utils import load_tuning_state
        m = build_model(args.model_name, num_classes=args.num_classes, enable_mask_head=False,
                        device="cuda", img_size=(args.img_size, args.img_size), in_channels=3,
                        pretrained_model_path=None, pretrained_backbone=False)
        m = load_tuning_state(m, args.checkpoint).cuda()
        m.deploy()
        m.eval()
        self.m = m

    @torch.no_grad()
    def __call__(self, x):
        o = self.m(x.cuda())
        return o["pred_logits"].float().cpu().numpy(), o["pred_boxes"].float().cpu().numpy()


class OrtBackend:
    def __init__(self, onnx_path):
        import cuda_env
        self.sess, providers = cuda_env.make_session(onnx_path)
        print(f"[cmp] ort providers: {providers}")

    def __call__(self, x):
        o = self.sess.run(["logits", "boxes"], {"images": x.numpy()})
        return o[0], o[1]


class EngineBackend:
    def __init__(self, engine_path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        inp = x.cuda().contiguous()
        self.ctx.set_input_shape("images", tuple(inp.shape))
        self.ctx.set_tensor_address("images", inp.data_ptr())
        out = {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                buf = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
                out[n] = buf
                self.ctx.set_tensor_address(n, buf.data_ptr())
        # the H2D copy (x.cuda()) was enqueued on the current stream; make the engine
        # stream wait for it before launching, else the engine can read a partial input.
        self.stream.wait_stream(torch.cuda.current_stream())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out["logits"].cpu().numpy(), out["boxes"].cpu().numpy()


# --------------------------- detection decode (for mAP) ---------------------------

def decode_dets(logits, boxes, W, H, num_classes, topk, thr):
    """sigmoid -> top-k over (query x class) -> pixel xywh, labels, scores (filtered by thr)."""
    prob = sigmoid(logits[0])           # [Q, C]
    flat = prob.reshape(-1)
    k = min(topk, flat.shape[0])
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx]
    keep = scores >= thr
    idx, scores = idx[keep], scores[keep]
    labels = idx % num_classes
    q = idx // num_classes
    b = boxes[0, q]
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xywh = np.stack([(cx - 0.5 * w) * W, (cy - 0.5 * h) * H, w * W, h * H], axis=1)
    return xywh, labels, scores, q


# --------------------------- image sampling ---------------------------

def sample_images(root: Path, limit: int):
    folders = sorted(glob.glob(str(root / "recognition_*")))
    if limit and limit < len(folders):
        stride = len(folders) / float(limit)          # spread across the whole set
        folders = [folders[int(i * stride)] for i in range(limit)]
    out = []
    for f in folders:
        mains = sorted(glob.glob(str(Path(f) / "photos" / "*_Main.jpg")))
        if mains:
            out.append(mains[0])
    return out


# --------------------------- COCO mAP-vs-reference ---------------------------

def _dedup_per_query(items, score_key):
    """Keep one item per (image_id, qid) -- the highest score (or only one, for GT).
    Removes the multi-class-per-query duplicates that otherwise inflate class-agnostic FPs."""
    best = {}
    for x in items:
        key = (x["image_id"], x["qid"])
        if key not in best or x.get(score_key, 1.0) > best[key].get(score_key, 1.0):
            best[key] = x
    return list(best.values())


def coco_ap(gt_anns, gt_imgs, dets, num_classes, agnostic):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not dets or not gt_anns:
        return None
    cats = [{"id": 1, "name": "obj"}] if agnostic else [{"id": c + 1, "name": str(c)} for c in range(num_classes)]
    ga = gt_anns
    d = [dict(x) for x in dets]
    if agnostic:
        # collapse classes, then dedup so one physical query is one box (not C boxes)
        ga = [{**x, "category_id": 1} for x in _dedup_per_query(gt_anns, "area")]
        d = [{**x, "category_id": 1} for x in _dedup_per_query(d, "score")]
    # reassign annotation ids after dedup
    ga = [{**x, "id": i + 1} for i, x in enumerate(ga)]
    g = {"images": gt_imgs, "annotations": ga, "categories": cats}
    coco = COCO()
    coco.dataset = g
    coco.createIndex()
    dt = coco.loadRes(d)
    ev = COCOeval(coco, dt, iouType="bbox")
    ev.params.imgIds = sorted({im["id"] for im in gt_imgs})
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}


# --------------------------- main ---------------------------

def main(args):
    root = Path(args.dataset)
    images = sample_images(root, args.limit)
    print(f"[cmp] sampled {len(images)} Main.jpg images from {root}")

    backends = {"torch": TorchBackend(args)}
    if "ort" in args.backends:
        backends["ort"] = OrtBackend(args.onnx)
    if "trt_basic" in args.backends:
        backends["trt_basic"] = EngineBackend(args.basic_engine)
    if "trt_basic_tf32" in args.backends and args.basic_tf32_engine:
        backends["trt_basic_tf32"] = EngineBackend(args.basic_tf32_engine)
    if "trt_correct" in args.backends:
        backends["trt_correct"] = EngineBackend(args.correct_engine)
    cmp_names = [b for b in backends if b != "torch"]
    print(f"[cmp] backends: {list(backends)}  |  comparing {cmp_names} vs torch(.pt)")

    C = args.num_classes
    # (A) query-aligned accumulators
    A = {b: {"iou": [], "l1": [], "dscore": [], "clsmatch": [], "n_ref": 0,
             # alignment-free best-match IoU + whether the best match is the same query index
             "miou": [], "align_hit": [],
             # medium-confidence band [det_thr, parity_thr): does drift grow off the easy set?
             "med_iou": [], "med_l1": [], "med_n": 0} for b in cmp_names}
    # (B) mAP accumulators: GT from torch, dets per backend
    gt_anns, gt_imgs = [], []
    dets = {b: [] for b in cmp_names}
    ann_id = 1
    timing = {b: 0.0 for b in backends}

    t0 = time.time()
    for n, path in enumerate(images):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, Wd = bgr.shape[:2]
        x = preprocess(bgr, args.img_size)
        iid = n + 1

        raw = {}
        for name, be in backends.items():
            ts = time.time()
            log, box = be(x)
            timing[name] += time.time() - ts
            raw[name] = (log, box)

        # ---- reference (.pt) ----
        ref_log, ref_box = raw["torch"]
        ref_prob = sigmoid(ref_log[0])               # [Q,C]
        ref_qscore = ref_prob.max(axis=1)            # [Q]
        ref_qcls = ref_prob.argmax(axis=1)           # [Q]

        # (A) signal queries for parity
        sig_q = np.where(ref_qscore >= args.parity_thr)[0]
        med_q = np.where((ref_qscore >= args.det_thr) & (ref_qscore < args.parity_thr))[0]
        ref_box_xyxy = cxcywh_to_xyxy(ref_box[0])    # [Q,4] normalised

        # (B) GT from .pt confident dets
        gt_imgs.append({"id": iid, "width": Wd, "height": H})
        g_xywh, g_lab, g_sc, g_q = decode_dets(ref_log, ref_box, Wd, H, C, args.topk, args.gt_thr)
        for bb, lb, qq in zip(g_xywh, g_lab, g_q):
            gt_anns.append({"id": ann_id, "image_id": iid, "category_id": int(lb) + 1, "qid": int(qq),
                            "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                            "area": float(bb[2] * bb[3]), "iscrowd": 0})
            ann_id += 1

        for b in cmp_names:
            log, box = raw[b]
            prob = sigmoid(log[0])
            # (A)
            bx = cxcywh_to_xyxy(box[0])
            if len(sig_q):
                iou = iou_pairs(ref_box_xyxy[sig_q], bx[sig_q])          # query-aligned IoU
                l1 = np.mean(np.abs(ref_box[0][sig_q] - box[0][sig_q]), axis=1)
                ds = np.abs(ref_prob[sig_q, ref_qcls[sig_q]] - prob[sig_q, ref_qcls[sig_q]])
                cm = (prob[sig_q].argmax(axis=1) == ref_qcls[sig_q]).astype(np.float32)
                # alignment-free: best-matching engine box for each ref signal box, over ALL queries
                M = iou_matrix(ref_box_xyxy[sig_q], bx)                  # [len(sig_q), 300]
                best_j = M.argmax(axis=1)
                miou = M[np.arange(len(sig_q)), best_j]
                align_hit = (best_j == sig_q).astype(np.float32)        # is the same index the best match?
                A[b]["iou"].extend(iou.tolist())
                A[b]["l1"].extend(l1.tolist())
                A[b]["dscore"].extend(ds.tolist())
                A[b]["clsmatch"].extend(cm.tolist())
                A[b]["miou"].extend(miou.tolist())
                A[b]["align_hit"].extend(align_hit.tolist())
                A[b]["n_ref"] += len(sig_q)
            if len(med_q):
                A[b]["med_iou"].extend(iou_pairs(ref_box_xyxy[med_q], bx[med_q]).tolist())
                A[b]["med_l1"].extend(np.mean(np.abs(ref_box[0][med_q] - box[0][med_q]), axis=1).tolist())
                A[b]["med_n"] += len(med_q)
            # (B)
            d_xywh, d_lab, d_sc, d_q = decode_dets(log, box, Wd, H, C, args.topk, args.det_thr)
            for bb, lb, sc, qq in zip(d_xywh, d_lab, d_sc, d_q):
                dets[b].append({"image_id": iid, "category_id": int(lb) + 1, "qid": int(qq),
                                "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                "score": float(sc)})

        if (n + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f"[cmp]   {n+1}/{len(images)}  ({dt:.1f}s, {1000*dt/(n+1):.1f} ms/img)  "
                  f"gt_anns={len(gt_anns)}")

    # ---------------- summarize ----------------
    def stats(v):
        v = np.asarray(v, dtype=np.float64)
        if not len(v):
            return {}
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "ge0.9": float((v >= 0.9).mean()), "ge0.7": float((v >= 0.7).mean()),
                "ge0.5": float((v >= 0.5).mean())}

    report = {"n_images": len(gt_imgs), "n_gt_anns": len(gt_anns),
              "params": {"parity_thr": args.parity_thr, "gt_thr": args.gt_thr,
                         "det_thr": args.det_thr, "topk": args.topk, "img_size": args.img_size},
              "timing_ms_per_img": {b: 1000 * t / max(1, len(gt_imgs)) for b, t in timing.items()},
              "query_aligned": {}, "map_vs_ref": {}}

    print("\n" + "=" * 92)
    print(f"COMPARISON vs PyTorch .pt reference   |  images={len(gt_imgs)}  ref_GT_dets={len(gt_anns)}"
          f"  (gt_thr={args.gt_thr}, parity_thr={args.parity_thr})")
    print("=" * 92)
    print("\n(A) QUERY-ALIGNED PARITY  (per .pt signal query, box[q] vs box[q])")
    print(f"{'backend':<14} {'n_q':>7} {'IoU.mean':>9} {'IoU.med':>8} {'IoU>=0.9':>9} "
          f"{'boxL1':>9} {'dscore':>8} {'cls%':>6} | {'mIoU':>7} {'align%':>7}")
    for b in cmp_names:
        iou_s = stats(A[b]["iou"])
        l1 = float(np.mean(A[b]["l1"])) if A[b]["l1"] else float("nan")
        dsc = float(np.mean(A[b]["dscore"])) if A[b]["dscore"] else float("nan")
        cls = float(np.mean(A[b]["clsmatch"])) if A[b]["clsmatch"] else float("nan")
        miou = float(np.mean(A[b]["miou"])) if A[b]["miou"] else float("nan")
        align = float(np.mean(A[b]["align_hit"])) if A[b]["align_hit"] else float("nan")
        report["query_aligned"][b] = {"iou": iou_s, "box_l1": l1, "dscore": dsc, "cls_match": cls,
                                      "matched_iou": miou, "align_rate": align, "n_query": A[b]["n_ref"]}
        if iou_s:
            print(f"{b:<14} {A[b]['n_ref']:>7d} {iou_s['mean']:>9.4f} {iou_s['median']:>8.4f} "
                  f"{iou_s['ge0.9']:>9.3f} {l1:>9.5f} {dsc:>8.4f} {100*cls:>5.1f} | "
                  f"{miou:>7.4f} {100*align:>6.2f}")
    print("  mIoU = alignment-free best-match IoU (each ref box vs ALL engine boxes);")
    print("  align% = how often the same query index IS that best match (query-alignment validity).")

    print(f"\n(A') MEDIUM-CONFIDENCE BAND  [{args.det_thr} <= .pt score < {args.parity_thr}] "
          f"(query-aligned; the harder, non-saturating set)")
    print(f"{'backend':<14} {'n_q':>7} {'IoU.mean':>9} {'IoU.med':>8} {'boxL1':>9}")
    for b in cmp_names:
        ms = stats(A[b]["med_iou"])
        ml1 = float(np.mean(A[b]["med_l1"])) if A[b]["med_l1"] else float("nan")
        report["query_aligned"][b]["medium_band"] = {"iou": ms, "box_l1": ml1, "n_query": A[b]["med_n"]}
        if ms:
            print(f"{b:<14} {A[b]['med_n']:>7d} {ms['mean']:>9.4f} {ms['median']:>8.4f} {ml1:>9.5f}")

    print("\n(B) COCO mAP vs .pt reference  (.pt confident dets = GT; AP=1.0 -> perfect reproduction)")
    print(f"{'backend':<13} {'AP@[.5:.95]':>12} {'AP@.5':>8} {'AP@.75':>8} | {'agn.AP':>8} {'agn.AP50':>9}")
    for b in cmp_names:
        ca = coco_ap(gt_anns, gt_imgs, dets[b], C, agnostic=False)
        ag = coco_ap(gt_anns, gt_imgs, dets[b], C, agnostic=True)
        report["map_vs_ref"][b] = {"class_aware": ca, "class_agnostic": ag, "n_dets": len(dets[b])}
        if ca and ag:
            print(f"{b:<13} {ca['AP']:>12.4f} {ca['AP50']:>8.4f} {ca['AP75']:>8.4f} | "
                  f"{ag['AP']:>8.4f} {ag['AP50']:>9.4f}")

    print("\ntiming (ms/img):", {b: round(v, 2) for b, v in report["timing_ms_per_img"].items()})
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[cmp] wrote {args.out}")


def parse_args():
    repo = SCRIPTS.parents[1]
    eng = repo / "trt-files" / "engines"
    onnx = repo / "trt-files" / "onnx"
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/mnt/d/datasets/RRS_Dataset_21k_with_meta/export_20251204_123546")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--model-name", default="s")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--parity-thr", type=float, default=0.30, help="(A) min .pt max-class score for a signal query")
    p.add_argument("--gt-thr", type=float, default=0.40, help="(B) min .pt score to count as reference GT det")
    p.add_argument("--det-thr", type=float, default=0.05, help="(B) min score to keep a backend prediction")
    p.add_argument("--checkpoint", default="/home/dxdxxd/projects/custom-dfine/second-staff/full_detector_1506_ex_2104.pt")
    p.add_argument("--dfine-src", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
    p.add_argument("--onnx", default=str(onnx / "dfine_s_food_explicit.onnx"))
    p.add_argument("--basic-engine", default=str(eng / "dfine_s_food_basic.engine"))
    p.add_argument("--basic-tf32-engine", default=str(eng / "dfine_s_food_basic_tf32.engine"))
    p.add_argument("--correct-engine", default=str(eng / "dfine_s_food_explicit.engine"))
    p.add_argument("--backends", nargs="+",
                   default=["ort", "trt_basic", "trt_basic_tf32", "trt_correct"])
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--out", default=str(Path(tempfile.gettempdir()) / "dfine-compare-report.json")
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
