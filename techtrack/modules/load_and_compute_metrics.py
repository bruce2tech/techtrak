import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# 1) Load the saved data
with open("raw_detection_data.pkl", "rb") as f:
    saved = pickle.load(f)

class_names  = saved["class_names"]
file_names   = saved["file_names"]
metrics_data = saved["metrics_data"]  # {model: {class: {y_true, y_scores}}}

models = list(metrics_data.keys())
num_classes = len(class_names)

# 2) Prepare DataFrames for AP, Recall, F1, ROC-AUC
rows = []
for model in models:
    print(f"\n=== Metrics for model: {model} ===")
    for cls in range(num_classes):
        arr = metrics_data[model][cls]
        y_true   = np.array(arr["y_true"])
        y_scores = np.array(arr["y_scores"])

        # average precision (area under PR curve)
        ap = average_precision_score(y_true, y_scores)
        print(f"Class {cls:2d} ({class_names[cls]:15s})  AP@0.5    = {ap:.4f}")

        # precision, recall, f1 at fixed threshold
        thr = 0.5
        y_pred = (y_scores >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Class {cls:2d} ({class_names[cls]:15s})  Precision = {p:.4f}, Recall = {r:.4f}, F1 = {f1:.4f}")

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc_auc = np.nan
        print(f"Class {cls:2d} ({class_names[cls]:15s})  ROC-AUC   = {roc_auc:.4f}")
        
        rows.append({
            "model": model,
            "class": class_names[cls],
            "AP@0.5": ap,
            "Precision@0.5": p,
            "Recall@0.5": r,
            "F1@0.5": f1,
            "ROC-AUC": roc_auc
        })
df = pd.DataFrame(rows)
df.set_index(["model", "class"], inplace=True)
print(df.head())

# index = class_names, columns = model names
ap_df = df["AP@0.5"].unstack(level=0)

# 1) compute mAP per model
mAPs = ap_df.mean(axis=0)
print("✔️  Mean AP per model (mAP@0.5):")
for model_name, m in mAPs.items():
    print(f"   {model_name}: {m:.4f}")

# 2) append as a new row called "mAP"
ap_df.loc["mAP"] = mAPs
print("\n✔️  Added “mAP” row to ap_df:")
print(ap_df.loc["mAP"])


# df = pd.DataFrame(rows)
# df.set_index(["model", "class"], inplace=True)
# print(df.head())



# # index = class_names, columns = model names
# ap_df = df["AP@0.5"].unstack(level=0)

# # 1) compute mAP per model
# mAPs = ap_df.mean(axis=0)

# # 2) append as a new row called "mAP"
# ap_df.loc["mAP"] = mAPs

# # optional: if you want mAP to appear at the end
# # you can reorder the index so 'mAP' is last
# classes_plus_map = list(class_names) + ["mAP"]
# ap_df = ap_df.reindex(classes_plus_map)

# # 3) plot
# ap_df.plot.bar(figsize=(12,6))
# plt.ylabel("AP@0.5")
# plt.title("Per-Class AP@0.5 plus overall mAP")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("per_class_AP_with_mAP.png")

# # a) Bar chart of per-class mAP (AP@0.5)
# ap_df = df["AP@0.5"].unstack(level=0)  # classes × models
# ap_df.plot.bar(figsize=(12,6))
# plt.ylabel("AP@0.5")
# plt.title("Per-Class Average Precision @ IoU=0.5")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("per_class_AP.png")

# # b) Bar chart of average Recall@0.5
# rec_df = df["Recall@0.5"].unstack(level=0)
# rec_df.plot.bar(figsize=(12,6))
# plt.ylabel("Recall@0.5")
# plt.title("Per-Class Recall @ Confidence=0.5")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("per_class_Recall.png")


# # c) Bar chart of per-class F1@0.5 or ROC-AUC
# f1_df = df["F1@0.5"].unstack(level=0)
# f1_df.plot.bar(figsize=(12,6))
# plt.ylabel("F1@0.5")
# plt.title("Per-Class F1 Score @ Confidence=0.5")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("per_class_F1.png")

# roc_df = df["ROC-AUC"].unstack(level=0)
# roc_df.plot.bar(figsize=(12,6))
# plt.ylabel("ROC-AUC")
# plt.title("Per-Class ROC AUC")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("per_class_ROC_AUC.png")

# # d) Overlay PR curves for a chosen class
# for i in range(20):
#     cls =   # e.g. safety vest
#     plt.figure(figsize=(6,6))
#     for model in models:
#         arr = metrics_data[model][cls]
#         prec, rec, _ = precision_recall_curve(arr["y_true"], arr["y_scores"])
#         plt.plot(rec, prec, label=model)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(f"PR Curve for {class_names[cls]}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"PR_curve_{class_names[cls]}.png")
