import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# 1) Load the saved metrics_data (make sure path is correct)
with open("raw_detection_data.pkl", "rb") as f:
    saved = pickle.load(f)

class_names  = saved["class_names"]
metrics_data = saved["metrics_data"]  # {model: {class_idx: {"y_true","y_scores"}}}
models       = list(metrics_data.keys())
num_classes  = len(class_names)

# 2) Build a flat table of per-class metrics
rows = []
for model in models:
    for cls in range(num_classes):
        arr      = metrics_data[model][cls]
        y_true   = np.array(arr["y_true"])
        y_scores = np.array(arr["y_scores"])

        ap = average_precision_score(y_true, y_scores)
        thr = 0.5
        y_pred = (y_scores >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc = float("nan")

        rows.append({
            "Model":        model,
            "Class":        class_names[cls],
            "AP@0.5":       ap,
            "Precision@0.5":p,
            "Recall@0.5":   r,
            "F1@0.5":       f1,
            "ROC-AUC":      roc
        })

df = pd.DataFrame(rows)

# 3) Compute overall mAP per model
mAPs = df.groupby("Model")["AP@0.5"].mean().rename("mAP@0.5").reset_index()

# 4) Write to Excel
output_file = "model_comparison_metrics.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name="Per-Class Metrics", index=False)
    mAPs.to_excel(writer, sheet_name="Overall mAP", index=False)

print(f"✅ Written all metrics to Excel: {output_file}")