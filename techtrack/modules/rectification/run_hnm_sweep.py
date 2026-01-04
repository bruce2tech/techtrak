import os, time, argparse
import numpy as np
import pandas as pd

# adjust these imports to your package layout
from ..utils.loss import Loss
from ..inference.model import Detector
from ..inference.nms import NMS
from .hard_negative_mining import HardNegativeMiner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1,2,3,5,10])
    ap.add_argument("--pos_ref", choices=["gt","tp"], default="gt")
    ap.add_argument("--outdir", default="hnm_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) set up components
    loss = Loss(
        iou_threshold=args.iou_thr,
        lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.5, lambda_cls=0.5,
        num_classes=len(open(args.names).read().splitlines())
    )
    det = Detector(args.weights, args.cfg, args.names, score_threshold=args.score_thr)
    try:
        nms = NMS()             # no constructor args
    except TypeError:
        # If NMS is actually a function (not a class), just keep a reference to it
        nms = NMS
    miner = HardNegativeMiner(model=det, nms=nms, measure=loss, dataset_dir=args.data_dir)

    # 2) build miner table once (don’t modify class; call sample to trigger table build)
    # This calls the private __construct_table via the provided method.
    _ = miner.sample_hard_negatives(num_hard_negatives=1, criteria="total_loss")  # primes miner.table

    # 3) lambda sweep (external to the class so we don’t edit it)
    rows = []
    for _, row in miner.table.iterrows():
        N_gt = int(row.get("N_gt", 0))
        N_tp = int(row.get("N_tp", 0))
        N_neg_avail = int(row.get("N_neg_avail", 0))
        neg_hard = row.get("neg_hardness_desc", []) or []
        # ensure it's a list (not a string) if your DataFrame coerces it
        if isinstance(neg_hard, str):
            # fallback if it was stringified; try to eval safely
            import ast
            neg_hard = ast.literal_eval(neg_hard)

        # neg_hard is expected to be sorted desc by Loss.compute
        for lam in args.lambdas:
            N_pos_ref = max(N_tp if args.pos_ref == "tp" else N_gt, 1)
            k = min(int(lam * N_pos_ref), N_neg_avail)
            sum_hard = float(np.sum(neg_hard[:k])) if k > 0 else 0.0
            rows.append({
                "image_file": row["image_file"],
                "annotation_file": row["annotation_file"],
                "lambda": float(lam),
                "N_gt": N_gt,
                "N_tp": N_tp,
                "N_neg_avail": N_neg_avail,
                "N_neg_kept": int(k),
                "sum_hardness_kept": sum_hard,
                "pos_ref": args.pos_ref,
            })

    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"hnm_sweep_{args.pos_ref}_{stamp}.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    # quick aggregation for readability
    try:
        df["gt_bin"] = pd.qcut(df["N_gt"], q=min(10, df["N_gt"].nunique()), duplicates="drop")
        summary = (df.groupby(["lambda","gt_bin"])
                     .agg(N_neg_kept_mean=("N_neg_kept","mean"),
                          N_images=("image_file","count"))
                     .reset_index())
        sum_csv = os.path.join(args.outdir, f"hnm_summary_{args.pos_ref}_{stamp}.csv")
        summary.to_csv(sum_csv, index=False)
        print("Wrote:", sum_csv)
    except Exception as e:
        print("Summary binning skipped:", e)

if __name__ == "__main__":
    main()