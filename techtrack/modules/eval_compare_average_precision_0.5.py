# eval_models.py

import os
import cv2
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from typing import List, Tuple

from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score

from inference.model import Detector

# ─── CONFIGS ─────────────────────────────────────────────────────────────────
MODULE_DIR = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(MODULE_DIR, '..', '..'))
MODEL_DIR  = os.path.join(PROJ_ROOT, 'assignment-2-test', 'storage', 'yolo_models')
DATA_DIR   = os.path.join(PROJ_ROOT, 'logistics')  # your .jpg + .txt folder

CONFIGS = [
    {
      "name":    "yolov4-tiny1",
      "weights": os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_1.weights"),
      "cfg":     os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_1.cfg"),
      "names":   os.path.join(MODEL_DIR, "logistics.names"),
      "thr":     0.5
    },
    {
      "name":    "yolov4-tiny2",
      "weights": os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_2.weights"),
      "cfg":     os.path.join(MODEL_DIR, "yolov4-tiny-logistics_size_416_2.cfg"),
      "names":   os.path.join(MODEL_DIR, "logistics.names"),
      "thr":     0.5
    },
]

# only evaluate one class for now
TARGET_CLASS = 1  # change as needed (0-based index)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_yolo_annotations(dataset_dir: str
) -> Tuple[List[str], List[List[Tuple[int,int,int,int]]], List[List[int]]]:
    """
    Returns lists: file_names, gt_boxes, gt_classes
    """
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
                    x1, y1 = cx - bw/2, cy - bh/2
                    boxes.append((int(x1), int(y1), int(bw), int(bh)))
                    classes.append(int(cls))
        file_names.append(img_path)
        gt_boxes.append(boxes)
        gt_classes.append(classes)
    return file_names, gt_boxes, gt_classes

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.perf_counter()

    # 1) load and sample
    file_names, gt_boxes, gt_classes = load_yolo_annotations(DATA_DIR)

    idxs = [i for i, clist in enumerate(gt_classes) if TARGET_CLASS in clist]
    print(f"→ Evaluating on {len(file_names)} images with class {TARGET_CLASS})\n")

    # 2) load class names
    with open(CONFIGS[0]["names"]) as f:
        class_names = [l.strip() for l in f]
    print("Classes:", class_names, "\n")

    threshold = 0.5
    num_classes = len(class_names)

    # prepare storage
    ap_results = {cfg["name"]: [] for cfg in CONFIGS}

    results = []
    pr_curves = {}

    # 3) loop over models
    for cfg in CONFIGS:
        print(f"Model: {cfg['name']}")
        det = Detector(
            weights_path   = cfg["weights"],
            config_path    = cfg["cfg"],
            class_path     = cfg["names"],
            score_threshold= cfg["thr"]
        )
        
        for cls in range(num_classes):
            y_true = [int(cls in clist) for clist in gt_classes]
            y_scores = []
            for fn in file_names:
                frame = cv2.imread(fn)
                b, c, s, _ = det.post_process(det.predict(frame))
                cls_scores = [score for lab, score in zip(c, s) if lab == cls]
                y_scores.append(max(cls_scores) if cls_scores else 0.0)
            
            ap = average_precision_score(y_true, y_scores)
            ap_results[cfg["name"]].append(ap)

            # for fn, gts, gtc in zip(file_names, gt_boxes, gt_classes):
            #     # ground-truth: positive if ANY GT box of target class exists
            #     y_true.append(int(TARGET_CLASS in gtc))

            #     # run detector & collect max score for target class
            #     outs = det.predict(cv2.imread(fn))
            #     b, c, s, cs = det.post_process(outs)
            #     cls_scores = [score for label, score in zip(c, s) if label == TARGET_CLASS]
            #     y_scores.append(max(cls_scores) if cls_scores else 0.0)

            # # compute PR & AP
            # precision, recall, _ = precision_recall_curve(y_true, y_scores)
            # ap = average_precision_score(y_true, y_scores)
            # print(f" → AP for class {TARGET_CLASS} '{class_names[TARGET_CLASS]}': {ap:.3f}\n")

            # results.append({"model": cfg["name"], "AP": ap})
            # pr_curves[cfg["name"]] = (precision, recall)

    # now plot
    labels = class_names
    x = np.arange(num_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x - width/2, ap_results[CONFIGS[0]["name"]], width, label=CONFIGS[0]["name"])
    ax.bar(x + width/2, ap_results[CONFIGS[1]["name"]], width, label=CONFIGS[1]["name"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"AP@0.5")
    ax.set_title(f"Per-class Average Precision @ IoU=0.5 Model 1 vs Model 2")
    ax.legend()
    plt.tight_layout()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"per_class_AP_comparison.png"
    plt.savefig(out_path)
    print(f"Saved precision comparison to {out_path}")

    # # 4) summarize & bar chart
    # df = pd.DataFrame(results).set_index("model")
    # print(df, "\n")

    # fig, ax = plt.subplots()
    # ax.bar(df.index, df["AP"])
    # ax.set_ylabel(f"AP@0.5 (class {TARGET_CLASS})")
    # ax.set_title("Single-Class AP Comparison")
    # for i, v in enumerate(df["AP"]):
    #     ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    # plt.tight_layout()
    # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_bar = f"single_class_ap_{TARGET_CLASS}_{ts}.png"
    # plt.savefig(out_bar)
    # print(f"→ Saved bar chart to {out_bar}\n")

    # # 5) plot PR curves
    # fig, ax = plt.subplots()
    # for name,(prec,rec) in pr_curves.items():
    #     ax.plot(rec, prec, label=name)
    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # ax.set_title(f"Precision–Recall Curve (class {TARGET_CLASS})")
    # ax.legend()
    # plt.tight_layout()
    # out_pr = f"pr_curve_class{TARGET_CLASS}_{ts}.png"
    # plt.savefig(out_pr)
    # print(f"→ Saved PR curve to {out_pr}\n")

    # # 6) done
    elapsed = time.perf_counter() - start
    print(f"⏱  Done in {elapsed:.1f}s")
