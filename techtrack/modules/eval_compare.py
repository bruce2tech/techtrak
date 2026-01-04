import cv2
import pandas as pd
from glob import glob
import os
from typing import List, Tuple, Dict
from inference.model    import Detector
from utils.metrics  import (
    evaluate_detections,
    calculate_precision_recall_curve,
    calculate_map_x_point_interpolated,
)
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2
import time
import datetime
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MODULE_DIR = os.path.dirname(__file__)               # e.g. .../techtrack/modules
PROJ_ROOT  = os.path.abspath(os.path.join(MODULE_DIR, '..', '..'))
MODEL_DIR  = os.path.join(
    PROJ_ROOT,
    'assignment-2-test',    # the folder containing "storage"
    'storage',
    'yolo_models'         # your particular model subfolder
)
GROUND_TRUTH = os.path.join(PROJ_ROOT, 'logistics')

def compute_class_ap(precision: List[float], recall: List[float]) -> float:
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]
    return ap

def evaluate_model(
    name: str,
    detector,
    file_names: List[str],
    gt_boxes: List[List[Tuple[int,int,int,int]]],
    gt_classes: List[List[int]],
    num_classes: int,
    iou_thres: float = 0.5
) -> Dict:
    """
    Runs inference on every image with `detector`, then evaluates
    with your metrics.py functions and returns a dict of summary stats.
    """
    # run inference & collect detections
    all_boxes, all_cls, all_scores, all_cls_scores = [], [], [], []
    for fn in file_names:
        frame = cv2.imread(fn)
        outs  = detector.predict(frame)
        b, c, s, cs = detector.post_process(outs)
        all_boxes.append(b)
        all_cls.append(c)
        all_scores.append(s)
        all_cls_scores.append(cs)

    # get the matched y_true & pred_scores
    y_true, pred_scores = evaluate_detections(
        all_boxes, all_cls, all_scores, all_cls_scores,
        gt_boxes, gt_classes,
        map_iou_threshold=iou_thres,
        eval_type="class_scores",
        num_classes=num_classes 
    )

    # PR + per-class AP
    precision, recall, _ = calculate_precision_recall_curve(
        y_true, pred_scores, num_classes=num_classes
    )



    class_aps = {}
    for cls in range(num_classes):
        class_aps[cls] = compute_class_ap(precision[cls], recall[cls])

    mean_ap = np.mean(list(class_aps.values()))
    metrics = {
        "model": name,
        # "mAP@0.5": map_val
        "mAP@0.5": mean_ap,
        **{f"AP_cls{cls}": ap for cls, ap in class_aps.items()}
    }
    return metrics, precision, recall

def load_yolo_annotations(dataset_dir):
    file_names, gt_boxes, gt_classes = [], [], []
    for img_path in sorted(glob(os.path.join(dataset_dir, "*.jpg"))):
        ann_path = os.path.splitext(img_path)[0] + ".txt"
        img      = cv2.imread(img_path)
        h, w     = img.shape[:2]

        boxes, classes = [], []
        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                for line in f:
                    cls, xc, yc, nw, nh = map(float, line.split())
                    cx, cy = xc * w, yc * h
                    bw, bh = nw * w, nh * h
                    x1 = cx - bw/2
                    y1 = cy - bh/2
                    boxes.append((int(x1), int(y1), int(bw), int(bh)))
                    classes.append(int(cls))

        file_names.append(img_path)
        gt_boxes.append(boxes)
        gt_classes.append(classes)

    return file_names, gt_boxes, gt_classes

def get_coco_metrics(gt_json: str, pred_json: str) -> Dict[str, float]:
    coco_gt   = COCO(gt_json)
    coco_dt   = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    stats = list(coco_eval.stats or [])
    if len(stats) < 10:
        print("⚠️ COCOeval.stats too short—returning zeros")
        return {'mAP@.50': 0.0, 'AR@.50': 0.0}
    return {
      'mAP@.50': stats[1],
      'AR@.50':  stats[9],
    }

def dump_gt_to_coco_json(
    file_names, gt_boxes, gt_classes, out_json, class_names
):
    images, annotations = [], []
    ann_id = 1
    for img_id, fn in enumerate(file_names, start=1):
        img = cv2.imread(fn)
        h, w = img.shape[:2]
        images.append({
          "id": img_id,
          "file_name": os.path.basename(fn),
          "height": h,
          "width":  w
        })
        for (x,y,ww,hh), cls in zip(gt_boxes[img_id-1], gt_classes[img_id-1]):
            annotations.append({
              "id": ann_id,
              "image_id": img_id,
              # shift to 1-based
              "category_id": cls + 1,
              "bbox": [x,y,ww,hh],
              "area": ww*hh,
              "iscrowd": 0
            })
            ann_id += 1

    # 1-based categories
    categories = [{"id": i+1, "name": n}
                  for i,n in enumerate(class_names)]

    coco = {
      "info": {},          # minimal required
      "licenses": [],      # minimal required
      "images": images,
      "annotations": annotations,
      "categories": categories
    }
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

def dump_preds_to_coco_json(
    detector,
    file_names: List[str],
    out_json: str
) -> None:
    """
    Runs detector on each image in file_names, collects bboxes, scores, and class IDs,
    then writes a COCO‐style list of detection dicts to out_json.
    """
    coco_results = []
    for img_id, fn in enumerate(file_names, start=1):
        frame = cv2.imread(fn)
        bboxes, class_ids, scores, _ = detector.post_process(detector.predict(frame))
        for box, cid, score in zip(bboxes, class_ids, scores):
            x, y, w, h = box
            coco_results.append({
                "image_id":     img_id,
                "category_id":  int(cid) + 1,
                "bbox":         [x, y, w, h],
                "score":        float(score)
            })

    # write JSON
    with open(out_json, "w") as f:
        json.dump(coco_results, f)

if __name__ == "__main__":
    
    start = time.perf_counter()

    # ——— load your logistics images + YOLO txt GT ———
    GROUND_TRUTH = os.path.join(PROJ_ROOT, 'logistics')
    file_names, gt_boxes, gt_classes = load_yolo_annotations(GROUND_TRUTH)
    
    # only use the first N images for now        
    N = 20
    file_names = file_names[:N]
    gt_boxes   = gt_boxes[:N]
    gt_classes = gt_classes[:N]
    print(f"→ Evaluating on {len(file_names)} images (first {N})")

    # ——— read class names ———
    names_path = os.path.join(MODEL_DIR, "logistics.names")
    class_names = [l.strip() for l in open(names_path)]

    # ——— dump COCO‐GT once ———
    gt_json = "gt_annotations.json"
    dump_gt_to_coco_json(file_names, gt_boxes, gt_classes, gt_json, class_names)

    # ——— compare each model ———
    configs = [
      { "name":"yolov4-tiny1",
        "weights":os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_1.weights"),
        "cfg":    os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_1.cfg"),
        "names":  names_path,
        "thr":    0.5 },
      { "name":"yolov4-tiny2",
        "weights":os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_2.weights"),
        "cfg":    os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_2.cfg"),
        "names":  names_path,
        "thr":    0.5 },
    ]

    # ——— run your custom AP script ———
    results = []
    num_classes = len(class_names)
    for cfg in configs:
        det = Detector(cfg["weights"], cfg["cfg"], cfg["names"], score_threshold=cfg["thr"])
        metrics, prec, rec = evaluate_model(
            cfg["name"], det,
            file_names, gt_boxes, gt_classes,
            num_classes, iou_thres=0.5
        )
        results.append(metrics)
        # store these for PR-curve plotting
        if cfg["name"] == configs[0]["name"]:
            precision1, recall1 = prec, rec
        else:
            precision2, recall2 = prec, rec

    df = pd.DataFrame(results).set_index("model")
    print(df, "\n")

    # ——— per-class AP comparison plot ———
    ap_cols     = sorted(c for c in df.columns if c.startswith("AP_cls"))
    cls_idx     = np.arange(len(ap_cols))
    aps1, aps2  = df.loc["yolov4-tiny1", ap_cols].values, df.loc["yolov4-tiny2", ap_cols].values
    fig, ax     = plt.subplots(figsize=(12,5))
    w = 0.4
    ax.bar(cls_idx - w/2, aps1, w, label="Model 1")
    ax.bar(cls_idx + w/2, aps2, w, label="Model 2")
    ax.set_xticks(cls_idx)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("AP@0.5"); ax.legend(loc="upper right")
    ax.set_title("Per-class AP@0.5")
    plt.tight_layout()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"per_class_comparison_{ts}.png")


    # ——— now compute & plot COCO AR@0.5 ———
    ar_results = []
    for cfg in configs:
        det = Detector(cfg["weights"], cfg["cfg"], cfg["names"], score_threshold=cfg["thr"])
        pred_json = f"preds_{cfg['name']}.json"
        dump_preds_to_coco_json(det, file_names, pred_json) 
        coco_m = get_coco_metrics(gt_json, pred_json)
        ar_results.append({ "model":cfg["name"], "AR@.50": coco_m["AR@.50"] })

    ar_df = pd.DataFrame(ar_results).set_index("model")
    print(ar_df, "\n")

    fig, ax = plt.subplots()
    ax.bar(ar_df.index, ar_df["AR@.50"])
    ax.set_ylabel("AR@0.5"); ax.set_title("Average Recall @ IoU=0.5")
    for i,val in enumerate(ar_df["AR@.50"]):
        ax.text(i, val+0.01, f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"avg_recall_{ts}.png")


    # ——— precision–recall curve for one chosen class (e.g. class 3) ———
    # reuse the PR data from evaluate_model if you stored it, or
    # re-call calculate_precision_recall_curve here...
   
    # if you saved precision1, recall1 & precision2, recall2, you can:
    cls = 3
    plt.figure()
    plt.plot(recall1[cls], precision1[cls], label="Model 1")
    plt.plot(recall2[cls], precision2[cls], label="Model 2", linestyle="--")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve for class: {class_names[cls]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pr_curve_cls{cls}_{ts}.png")

    end = time.perf_counter()
    print(f"\n⏱ Total time: {end - start:.1f}s")