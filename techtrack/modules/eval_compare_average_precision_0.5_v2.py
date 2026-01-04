# eval_models.py

import pickle
import os
import cv2
import time
from glob import glob
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ─── 2) Worker function ─────────────────────────────────────────
def evaluate_single_image(args):
    """
    Runs detector on one image and returns the per–class max scores (for one model).
    """
    fn, cfg = args
    # instantiate Detector *inside* each process
    det = Detector(
        weights_path   = cfg["weights"],
        config_path    = cfg["cfg"],
        class_path     = cfg["names"],
        score_threshold= cfg["thr"]
    )

    img = cv2.imread(fn)
    b, c, s, _ = det.post_process(det.predict(img))

    # For each class, record the max score in this image (0 if none)
    num_classes = len(open(cfg["names"]).read().splitlines())
    max_scores = [0.0] * num_classes
    for label, score in zip(c, s):
        if score > max_scores[label]:
            max_scores[label] = score

    return os.path.basename(fn), max_scores  # return filename + list of scores


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
    # num_classes = len(class_names)

    # # prepare storage
    # ap_results = {cfg["name"]: [] for cfg in CONFIGS}

    results = []
    pr_curves = {}

    # We'll build a dict mapping model_name → per-image max-score vectors
    all_model_scores = {cfg["name"]: {} for cfg in CONFIGS}

    # For each model, spin up a pool and evaluate all images in parallel
    for cfg in CONFIGS:
        print(f"Evaluating images for model {cfg['name']} in parallel…")

        # Prepare the list of (fn, cfg) tuples
        tasks = [(fn, cfg) for fn in file_names]

        # Use as many processes as you have CPU cores
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
            # exe.map returns results in order, but as_completed lets you get them asap
            futures = {exe.submit(evaluate_single_image, t): t[0] for t in tasks}
            for future in as_completed(futures):
                fn = futures[future]
                _, max_scores = future.result()
                all_model_scores[cfg["name"]][fn] = max_scores

        print(f" → Done model {cfg['name']}\n")

    # ─── 4) Build y_true & y_scores per class from the results ───────   
    num_classes = len(class_names)
    metrics_data = {}

    for model_name, img_scores in all_model_scores.items():
        model_data = {}
        for cls in range(num_classes):
            # build y_true over full dataset
            y_true = [ int(cls in clist) for clist in gt_classes ]
            # build y_scores from your dict (in same order as file_names)
            y_scores = [ img_scores[fn][cls] for fn in file_names ]
            model_data[cls] = {
                "y_true":  y_true,
                "y_scores": y_scores
            }
        metrics_data[model_name] = model_data

    # save everything
    with open("raw_detection_data.pkl", "wb") as f:
        pickle.dump({
            "class_names": class_names,
            "file_names":  file_names,
            "metrics_data": metrics_data
        }, f)

    # # 6) done
    elapsed = time.perf_counter() - start
    print(f"⏱  Done in {elapsed:.1f}s")
