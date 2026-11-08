import os
import json
import argparse
import numpy as np

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

def metrics_from_cm(cm):
    n = np.sum(cm)
    acc = np.trace(cm) / n
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    tp = np.diag(cm)

    precision = np.divide(tp, col_sums, out=np.zeros_like(tp, dtype=float), where=col_sums!=0)
    recall = np.divide(tp, row_sums, out=np.zeros_like(tp, dtype=float), where=row_sums!=0)
    denom = precision + recall
    per_class_f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    mf1 = per_class_f1.mean()

    po = acc
    pe = (row_sums * col_sums).sum() / (n ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

    return {"acc": acc, "mf1": mf1, "kappa": kappa, "per_class": per_class_f1.tolist()}

def save_results_from_cms(results_path: str, dataset_name: str, dataset_path: str, num_folds: int) -> dict:
    seq_folds, cnn_folds = [], []
    missing = []

    for fold in range(num_folds):
        cm_path = os.path.join(results_path, f'fold_{fold}_cm.npz')
        if not os.path.isfile(cm_path):
            missing.append(fold)
            continue
        cms = np.load(cm_path)
        seq_folds.append(metrics_from_cm(cms['seq_cm']))
        cnn_folds.append(metrics_from_cm(cms['cnn_cm']))

    if missing:
        print(f"WARNING: no cm.npz for folds {missing} — excluded, not retrained.")

    def block(fold_list):
        means = lambda key: [r[key] for r in fold_list]
        return {
            "accuracy_mean": float(np.mean(means("acc"))),
            "mf1_mean": float(np.mean(means("mf1"))),
            "kappa_mean": float(np.mean(means("kappa"))),
            "per_class_f1": {n: float(np.mean([r["per_class"][i] for r in fold_list]))
                              for i, n in enumerate(CLASS_NAMES)},
            "accuracy_per_fold": means("acc"),
            "mf1_per_fold": means("mf1"),
        }

    results = {
        "dataset": dataset_name,
        "dataset path": dataset_path,
        "cnn": block(cnn_folds),
        "seq": block(seq_folds),
        "recovered_from_cm": True,
        "folds_recovered": num_folds - len(missing),
        "folds_expected": num_folds,
        "missing_folds": missing,
    }

    out = os.path.join(results_path, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Seq MF1: {results['seq']['mf1_mean']:.4f}")
    print(f"Recovered {results['folds_recovered']}/{results['folds_expected']} folds → {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_folds', type=int, required=True)
    args = parser.parse_args()

    save_results_from_cms(args.results_path, args.dataset_name, args.dataset_path, args.num_folds)