# viz_metrics.py
import pickle, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

def load_metrics(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["class_names"], data["file_names"], data["metrics_data"]

def compute_ap(y_true, y_scores):
    # Handles degenerate cases gracefully
    if len(set(y_true)) == 1:
        # AP is undefined when only one class present; return 0
        return 0.0
    return float(average_precision_score(y_true, y_scores))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to the saved metrics pickle")
    ap.add_argument("--model1", default="yolov4-tiny1", help="Model 1 name key")
    ap.add_argument("--class_index", type=int, default=0, help="Class index for PR curves")
    ap.add_argument("--outdir", default="viz_out", help="Directory for charts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    class_names, file_names, metrics_data = load_metrics(args.pkl)

    # Filter keys for Model 1 with each augmentation
    aug_order = ["none", "gaussian", "flip", "brightness"]
    keys = [f"{args.model1}__{aug}" for aug in aug_order if f"{args.model1}__{aug}" in metrics_data]
    if not keys:
        raise SystemExit(f"No keys found for model '{args.model1}' in {args.pkl}")

    # ---- Compute AP per augmentation per class ----
    rows = []
    for key in keys:
        class_dict = metrics_data[key]
        for cls_idx in range(len(class_names)):
            y_true   = class_dict[cls_idx]["y_true"]
            y_scores = class_dict[cls_idx]["y_scores"]
            ap_val = compute_ap(y_true, y_scores)
            rows.append({"model_aug": key, "class_idx": cls_idx, "class_name": class_names[cls_idx], "AP": ap_val})

    df = pd.DataFrame(rows)
    # Macro mAP over classes for each augmentation
    macro = df.groupby("model_aug")["AP"].mean().reset_index().rename(columns={"AP": "mAP_macro"})

    # Save summary CSV
    csv_path = os.path.join(args.outdir, "summary_model1_augstudy.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")
    print("\nMacro mAP by augmentation:")
    print(macro)

    # ---- Bar chart: macro mAP by augmentation ----
    fig = plt.figure()
    x = np.arange(len(keys))
    y = [float(macro.loc[macro["model_aug"]==k, "mAP_macro"].values[0]) for k in keys]
    plt.bar(x, y)
    plt.xticks(x, [k.split("__")[1] for k in keys], rotation=0)
    plt.ylabel("Macro mAP (AP averaged over classes)")
    plt.title(f"{args.model1}: Impact of Augmentations")
    bar_path = os.path.join(args.outdir, "macro_map_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {bar_path}")

    # ---- PR curves for a single class (default: class_index) ----
    cls_idx = args.class_index
    fig = plt.figure()
    for key in keys:
        y_true   = metrics_data[key][cls_idx]["y_true"]
        y_scores = metrics_data[key][cls_idx]["y_scores"]
        if len(set(y_true)) < 2:
            # skip degenerate PR
            continue
        P, R, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(R, P, label=key.split("__")[1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    title_cls = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"cls{cls_idx}"
    plt.title(f"{args.model1} PR Curves — Class: {title_cls}")
    plt.legend()
    pr_path = os.path.join(args.outdir, f"pr_curves_class_{cls_idx}.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {pr_path}")

if __name__ == "__main__":
    main()
