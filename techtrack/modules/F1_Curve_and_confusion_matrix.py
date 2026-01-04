import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    f1_score
)
import seaborn as sns  # for pretty heatmaps

# 1) Load your saved data
with open("raw_detection_data.pkl","rb") as f:
    data = pickle.load(f)

class_names  = data["class_names"]      # list of strings
file_names   = data["file_names"]       # list of image paths
metrics_data = data["metrics_data"]     # dict[model][cls]→{"y_true","y_scores"}

models = list(metrics_data.keys())
num_classes = len(class_names)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Pick a class to analyze (or loop over all)
cls = 2   # e.g. class index 2 (“cardboard box”)
label = class_names[cls]

for model in models:
    arr = metrics_data[model][cls]
    y_true   = np.array(arr["y_true"])
    y_scores = np.array(arr["y_scores"])

    # --- F1 curve -------------------------------------------------------------
    # You get precision, recall @ each threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 at each threshold:
    # Note thresholds has length (n_points -1), so align F1 to thresholds
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1 = f1[:-1]  # drop the last point which corresponds to threshold = 0

    # Plot F1 vs threshold
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, f1, label=model)
    best = thresholds[np.argmax(f1)]
    plt.axvline(best, color="gray", linestyle="--",
                label=f"best thr={best:.2f}")
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 score")
    plt.title(f"F1‐Curve for class '{label}' ({model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"F1_curve_{model}_cls{cls}.png")
    print(f"Saved F1 curve for {model}, class {label}")
    plt.close()

    # --- Confusion matrix at chosen threshold --------------------------------
    thr = best   # or any fixed value, e.g. 0.5
    y_pred = (y_scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred=0","pred=1"],
                yticklabels=["true=0","true=1"])
    plt.title(f"CM @thr={thr:.2f} for '{label}' ({model})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"confmat_{model}_cls{cls}.png")
    print(f"Saved confusion matrix for {model}, class {label}")
    plt.close()