"""Pre-registered significance tests on per-fold MF1 for the MorpheusNet ablations.

Procedure per (dataset, ablation, epoch duration) cell:
  - width:  Page's L trend test if pooled MF1 increases monotonically with width
            (0.5 < 0.75 < 1.0), otherwise Friedman
  - blocks: Friedman (non-directional)
  - follow-up: pairwise Wilcoxon signed-rank, Holm-Bonferroni corrected
               (interpret only if the omnibus test has p <= 0.05)
  - effect size: Kendall's W (from the Friedman statistic)

Output: significance_morpheusnet.csv (one row per cell)
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, page_trend_test, wilcoxon, t

FOLDS = {"SEDF20": 20, "SEDF78": 10, "Physio-2018": 5, "DOD-H": 25, "PHYSIO": 5}
LEVELS = {"width": ["05", "075", "1"], "blocks": [1, 2, 3]}  # ordered low -> high
EPOCHS = [5, 10, 30]
# Pooled MF1 (Table 21), order [0.5, 0.75, 1.0]. Only used to pick Page vs. Friedman
# for the width axis, since results.json stores only the mean of per-fold MF1.
POOLED_WIDTH_MF1 = {
    "SEDF20":      {5: [71.32, 71.79, 71.72], 10: [72.58, 73.43, 73.36], 30: [73.73, 76.08, 74.72]},
    "SEDF78":      {5: [68.37, 69.09, 69.28], 10: [71.11, 71.55, 71.10], 30: [73.30, 73.80, 74.11]},
    "PHYSIO": {5: [63.44, 63.82, 64.46], 10: [66.68, 67.63, 66.59], 30: [67.98, 68.73, 68.42]},
    "DOD-H":       {5: [71.58, 71.56, 70.10], 10: [72.82, 71.83, 72.82], 30: [75.14, 75.57, 75.28]},
}

MARGIN = 1.0  # equivalence margin in MF1 percentage points (fix before looking at results)
BASELINE = {"width": "1", "blocks": 3}

def update_config(base, override):
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            update_config(base[key], value)
        else:
            base[key] = value
    return base


def load_mf1(base_config, dataset, epoch, ablation, level):
    """Return the per-fold MF1 list for one configuration, or None if missing."""
    # width=1.0 and blocks=3 are the same baseline run
    config_level = 3 if (ablation == "width" and level == "1") else level
    if dataset == "PHYSIO":
        override_path = f"../configs/ablations/{dataset}_{epoch}_{config_level}_og_data_new_resnet_relu.json"
    else:
        override_path = f"../configs/ablations/{dataset}_{epoch}_{config_level}.json"
    if not os.path.exists(override_path):
        return None

    with open(base_config) as f:
        config = json.load(f)
    with open(override_path) as f:
        update_config(config, json.load(f))

    if dataset == "SEDF20" or dataset == "SEDF78":
        results_path = os.path.join("..", "results", config["name"] + "_subj", "results.json")
    elif dataset == "DOD-H":
        results_path = os.path.join("..", "results", config["name"] + "_new_data", "results.json")
    else:
        results_path = os.path.join("..", "results", config["name"] , "results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        seq = json.load(f)["seq"]
    return seq["mf1_per_fold"][:FOLDS[dataset]]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running_max = max(running_max, (m - rank) * pvals[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted


def test_cell(per_fold, dataset, epoch, ablation):
    levels = LEVELS[ablation]
    data = np.array([per_fold[level] for level in levels]).T  # shape: folds x levels
    n, k = data.shape

    # Omnibus test
    chi2, p_friedman = friedmanchisquare(*data.T)
    monotonic = False
    if ablation == "width":
        low, mid, high = POOLED_WIDTH_MF1[dataset][epoch]
        monotonic = low < mid < high
        print(f"dataset={dataset} epoch={epoch}s width pooled MF1: {low:.2f} < {mid:.2f} < {high:.2f} -> monotonic={monotonic}")
    if monotonic:
        page = page_trend_test(data, predicted_ranks=[1, 2, 3])
        row = {"test": "Page", "stat": page.statistic, "p": page.pvalue}
    else:
        row = {"test": "Friedman", "stat": chi2, "p": p_friedman}
    row["kendall_w"] = chi2 / (n * (k - 1))

    # Pairwise follow-ups, Holm-corrected
    pairs = list(itertools.combinations(range(k), 2))
    raw_p = [wilcoxon(data[:, a], data[:, b]).pvalue for a, b in pairs]
    for (a, b), p_adj in zip(pairs, holm(raw_p)):
        row[f"p_holm_{levels[a]}_vs_{levels[b]}"] = p_adj
    return row

def equivalence_cell(per_fold, ablation, margin=MARGIN):
    """Each reduced level vs. baseline: mean paired MF1 difference, 90% CI, Wilcoxon TOST."""
    base = np.asarray(per_fold[BASELINE[ablation]], dtype=float)
    scale = 100 if base.max() <= 1 else 1  # MF1 stored as 0-1 -> percentage points
    rows = []
    for level in LEVELS[ablation]:
        if level == BASELINE[ablation]:
            continue
        diff = (np.asarray(per_fold[level], dtype=float) - base) * scale
        n = len(diff)
        mean = diff.mean()
        half_width = t.ppf(0.95, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        # TOST: reject "diff <= -margin" and reject "diff >= +margin"
        p_lower = wilcoxon(diff + margin, alternative="greater").pvalue
        p_upper = wilcoxon(diff - margin, alternative="less").pvalue
        p_tost = max(p_lower, p_upper)
        rows.append({"level": level, "mean_diff_pp": mean,
                     "ci90_low": mean - half_width, "ci90_high": mean + half_width,
                     "p_tost": p_tost, "equivalent": p_tost <= 0.05})
    return rows
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config path")
    parser.add_argument("--datasets", nargs="+", default=list(FOLDS))
    parser.add_argument("--save", action="store_true", help="save CSV output")
    args = parser.parse_args()

    rows, eq_rows = [], []
    for dataset in args.datasets:
        for ablation in LEVELS:
            for epoch in EPOCHS:
                per_fold = {}
                for level in LEVELS[ablation]:
                    result = load_mf1(args.config, dataset, epoch, ablation, level)
                    if result is not None:
                        per_fold[level] = result

                if len(per_fold) < len(LEVELS[ablation]):
                    print(f"Missing runs: {dataset} {ablation} E={epoch}s -> skipped")
                    continue
                rows.append({"dataset": dataset, "ablation": ablation, "epoch_s": epoch,
                             "n_folds": len(per_fold[LEVELS[ablation][0]]),
                             **test_cell(per_fold, dataset, epoch, ablation)})
                for eq in equivalence_cell(per_fold, ablation):
                    eq_rows.append({"dataset": dataset, "ablation": ablation,
                                    "epoch_s": epoch, **eq})

    df = pd.DataFrame(rows)
    df_eq = pd.DataFrame(eq_rows)
    if args.save:
        df.to_csv(f"significance_morpheusnet_{dataset}.csv", index=False)
        df_eq.to_csv(f"equivalence_morpheusnet_{dataset}.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nEquivalence vs. baseline (margin = +/-{MARGIN} pp)")
    print(df_eq.round(3).to_string(index=False))