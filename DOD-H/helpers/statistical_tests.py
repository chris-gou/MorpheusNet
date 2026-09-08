import numpy as np
import json
import argparse
from scipy.stats import wilcoxon, friedmanchisquare, page_trend_test
from sklearn.metrics import accuracy_score, f1_score
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import pandas as pd
import os

def update_config(base_config, override_config):
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            update_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def save_fold_results(results_path, dataset, ablation, ablation_value, epoch_duration, fold, acc, mf1, kappa, wf1, n1f1, n2f1, n3f1, rf1, cm):
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    else:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        data = {}

    data.setdefault(dataset, {}).setdefault(ablation, {}).setdefault(str(ablation_value), {}).setdefault(str(epoch_duration), {})
    data[dataset][ablation][str(ablation_value)][str(epoch_duration)][str(fold)] = {"acc": acc, "mf1": mf1, "kappa": kappa, "wf1": wf1, "n1f1": n1f1, "n2f1": n2f1, "n3f1": n3f1, "rf1": rf1, "cm": cm.tolist()}

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_metric_array(data_path, metric, fold=None):
    with open(data_path) as f:
        data = json.load(f)
    d = data["seq"][f'{metric}']
    if fold is not None:
        d = d[fold]
    return d

def evaluate_fold(args, fold, config):
    evaluator = OneFoldEvaluator(args, fold, config)
    y_true, y_pred, _, _, _ = evaluator.run()
    y_pred_labels = y_pred.argmax(axis=1)  

    fold_mf1 = f1_score(y_true, y_pred_labels, average='macro')
    fold_acc = accuracy_score(y_true, y_pred_labels)
    results[ablation_value].append(fold_mf1)
    result_dict = skmet.classification_report(y_true, y_pred_labels, digits=3, output_dict=True)
    cm = skmet.confusion_matrix(y_true, y_pred_labels)

    accuracy = round(result_dict['accuracy']*100, 1)
    macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1)
    kappa = round(skmet.cohen_kappa_score(y_true, y_pred_labels), 3)
    
    wpr = round(result_dict['0.0']['precision']*100, 1)
    wre = round(result_dict['0.0']['recall']*100, 1)
    wf1 = round(result_dict['0.0']['f1-score']*100, 1)
    
    n1pr = round(result_dict['1.0']['precision']*100, 1)
    n1re = round(result_dict['1.0']['recall']*100, 1)
    n1f1 = round(result_dict['1.0']['f1-score']*100, 1)

    n2pr = round(result_dict['2.0']['precision']*100, 1)
    n2re = round(result_dict['2.0']['recall']*100, 1)
    n2f1 = round(result_dict['2.0']['f1-score']*100, 1)
    
    n3pr = round(result_dict['3.0']['precision']*100, 1)
    n3re = round(result_dict['3.0']['recall']*100, 1)
    n3f1 = round(result_dict['3.0']['f1-score']*100, 1)
    
    rpr = round(result_dict['4.0']['precision']*100, 1)
    rre = round(result_dict['4.0']['recall']*100, 1)
    rf1 = round(result_dict['4.0']['f1-score']*100, 1)

    fold_results_path = os.path.join("checkpoints", config['name'], f"raw_results.json")
    save_fold_results(fold_results_path, dataset, ablation, ablation_value, epoch_duration, fold, fold_acc, fold_mf1, kappa, wf1, n1f1, n2f1, n3f1, rf1, cm)
    
    pass


def marginal_normality(df, group_cols=("dataset", "epoch_s", "ablation_level")):
    """Shapiro-Wilk W and p-value on raw MF1, per group."""
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        mf1 = g["mf1"].values
        w, p = (np.nan, np.nan) if len(mf1) < 3 else stats.shapiro(mf1)
        rows.append({**dict(zip(group_cols, keys)), "n": len(mf1), "W": w, "p": p})
    return pd.DataFrame(rows)


def pairwise_difference_normality(df, cell_cols=("dataset", "epoch_s")):
    """Shapiro-Wilk on within-fold differences between each pair of ablation
    levels, per (dataset, epoch_s) cell. This is the assumption RM-ANOVA
    actually needs."""
    rows = []
    for keys, g in df.groupby(list(cell_cols)):
        wide = g.pivot(index="fold", columns="ablation_level", values="mf1")
        levels = wide.columns.tolist()
        for a, b in itertools.combinations(levels, 2):
            diff = (wide[a] - wide[b]).dropna().values
            w, p = (np.nan, np.nan) if len(diff) < 3 else stats.shapiro(diff)
            rows.append({**dict(zip(cell_cols, keys)),
                         "pair": f"{a} vs {b}", "n": len(diff), "W": w, "p": p})
    return pd.DataFrame(rows)


def plot_cell(df, dataset, epoch_s, out_path=None):
    """Histogram + Q-Q plot per ablation level, for one (dataset, epoch_s) cell."""
    cell = df[(df.dataset == dataset) & (df.epoch_s == epoch_s)]
    levels = sorted(cell["ablation_level"].unique())
    fig, axes = plt.subplots(2, len(levels), figsize=(4 * len(levels), 6), squeeze=False)

    for i, level in enumerate(levels):
        mf1 = cell[cell.ablation_level == level]["mf1"].values
        axes[0, i].hist(mf1, bins=min(10, max(len(mf1), 1)), edgecolor="black")
        axes[0, i].set_title(f"{level} (n={len(mf1)})")
        axes[0, i].set_xlabel("MF1")
        stats.probplot(mf1, dist="norm", plot=axes[1, i])
        axes[1, i].set_title("Q-Q plot")

    fig.suptitle(f"{dataset}, {epoch_s}s epochs")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig

def dict_to_long_df(data, dataset_name):
    """Convert {epoch_s: {ablation_level: [[mf1, mf1, ...]]}} into the
    long-format DataFrame used below. Handles missing/empty runs (e.g. a
    level with no folds yet) by simply contributing no rows for that cell."""
    rows = []
    for epoch_s, levels in data.items():
        for level, folds in levels.items():
            mf1_scores = folds[0] if folds else []  # folds = [[...]] or []
            for fold_idx, mf1 in enumerate(mf1_scores):
                rows.append({"dataset": dataset_name, "epoch_s": int(epoch_s),
                             "ablation_level": level, "fold": fold_idx, "mf1": mf1})
    return pd.DataFrame(rows)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path OR configs folder path', required=True)
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--streaming', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU for streaming latency')
    parser.add_argument('--ablation', type=str, default=None, required=True, help='Ablation config file path for inference')
    parser.add_argument('--dataset', type=str, default=None, required=True, help='Dataset config file path for inference')
    parser.add_argument('--type', type=str, default=None, help='Evaluate or statistical test')
    args = parser.parse_args()
    print(f"Running statistical tests for dataset {args.dataset} with ablation {args.ablation}...")
    results_file = f"statistical_test_results_{args.dataset}_{args.ablation}.json"
    if args.ablation:
        if args.ablation == "blocks":
            ablation = 'blocks'
              # num_scales -> list of per-fold ACC
            ablation_values = [1, 2, 3]
        elif args.ablation == "width":
            ablation = 'width'
             # width -> list of per-fold ACC
            ablation_values = ["05", "075", "1"]

    epoch_durations = [5, 10, 30]
    stat_results = {5: [], 10: [], 30: []}

    if args.dataset == 'SEDF78':
        total_folds = 10
        dataset = 'SEDF78'
    if args.dataset == 'SEDF20':
        total_folds = 20
        dataset = 'SEDF20'
    elif args.dataset == 'PHYSIO':
        total_folds = 5
        dataset = 'Physio-2018' 
    elif args.dataset == 'DOD-H':
        total_folds = 25
        dataset = 'DOD-H'
    results_complete = {"5": {val: [] for val in ablation_values}, "10": {val: [] for val in ablation_values}, "30": {val: [] for val in ablation_values}}  # epoch_duration -> list of per-fold mf1

    for epoch_duration in epoch_durations:
        if ablation == 'blocks':
            results = {1: [], 2: [], 3: []}
        elif ablation == 'width':
            results = {"05": [], "075": [], "1": []}
        base_config = args.config
        for ablation_value in ablation_values:
            if ablation == 'blocks':
                override_config_path = f"../configs/ablations/{dataset}_{epoch_duration}_{ablation_value}.json"
            elif ablation == 'width':
                if ablation_value != "1":
                    override_config_path = f"../configs/ablations/{dataset}_{epoch_duration}_{ablation_value}.json"
                else:
                    override_config_path = f"../configs/ablations/{dataset}_{epoch_duration}_3.json"
            with open(base_config) as config_file:
                config = json.load(config_file)
            if override_config_path:
                if not os.path.exists(override_config_path):
                    print(f"Override config path {override_config_path} does not exist. Skipping this ablation value.")
                    continue
                with open(override_config_path) as f:
                    override_config = json.load(f)
                update_config(config, override_config)

            # if we need to evaluate
            if args.type == 'evaluate':
                # check if the results for this ablation value and epoch duration already exist
                fold_results_path = os.path.join("checkpoints", config['name'], f"raw_results.json")
                if os.path.exists(fold_results_path):
                    continue
                else:
                    print(f"Running inference for ablation value {ablation_value} (E={epoch_duration}s) on dataset {dataset}...")
                    for fold in range(1, total_folds + 1):
                        evaluate_fold(args, fold, config)
            else:
                print(f"Skipping inference for ablation value {ablation_value} (E={epoch_duration}s) on dataset {dataset} since --type is not 'evaluate'.")
                print(f"Checking config {config['name']} for results...")
                results_path = os.path.join("..", "results", config['name']+"_new_data", f"results.json")
                if not os.path.exists(results_path):
                    print(f"No results found for ablation value {ablation_value} (E={epoch_duration}s) on dataset {dataset}.")
                    continue
                else:
                    # print(results)
                    # print(results_complete)
                    for fold in range(0, total_folds):
                        acc = get_metric_array(results_path, metric='mf1_per_fold', fold=fold)
                        results[ablation_value].append(acc)
                    results_complete[str(epoch_duration)][ablation_value].append(results[ablation_value])

    df = dict_to_long_df(results_complete, dataset)
    # marginal_normality(df).to_csv("shapiro_marginal.csv", index=False)
    # pairwise_difference_normality(df).to_csv("shapiro_pairwise_diff.csv", index=False)
    print(df)
    for dataset in df["dataset"].unique():
        if args.type == 'evaluate':
            continue

        for epoch_duration in sorted(df["epoch_s"].unique()):
            cell = df[(df.dataset == dataset) & (df.epoch_s == epoch_duration)]
            results = {
                level: cell[cell.ablation_level == level].sort_values("fold")["mf1"].tolist()
                for level in cell["ablation_level"].unique()
            }

            if not results or any(len(v) == 0 for v in results.values()):
                print(f"No results found for dataset {dataset}, epoch_duration {epoch_duration}s.")
                continue

            print(f"Running statistical tests for dataset {dataset}, epoch_duration {epoch_duration}s...")

            if ablation == 'blocks':
                stat, p = friedmanchisquare(results[1], results[2], results[3])
                stat_results[epoch_duration].append(f"Friedman across block configs (E={epoch_duration}s): p={p:.4f}")
                stat, p = wilcoxon(results[2], results[3])
                stat_results[epoch_duration].append(f"block=2 vs block=3 (E={epoch_duration}s): p={p:.4f}")
                stat, p = wilcoxon(results[1], results[2])
                stat_results[epoch_duration].append(f"block=1 vs block=2 (E={epoch_duration}s): p={p:.4f}")
                stat, p = wilcoxon(results[1], results[3])
                stat_results[epoch_duration].append(f"block=1 vs block=3 (E={epoch_duration}s): p={p:.4f}")

            elif ablation == 'width':
                if dataset == 'SEDF78' or (dataset == 'Physio-2018' and epoch_duration == 10):
                    data = np.array([results["05"], results["075"], results["1"]]).T
                    page_result = page_trend_test(data, predicted_ranks=[1, 2, 3])
                    stat_results[epoch_duration].append(
                        f"Page trend test across width configs (E={epoch_duration}s): "
                        f"L={page_result.statistic:.1f}, p={page_result.pvalue:.4f}")
                else:
                    stat, p = friedmanchisquare(results["05"], results["075"], results["1"])
                    stat_results[epoch_duration].append(f"Friedman across width configs (E={epoch_duration}s): p={p:.4f}")

                stat, p = wilcoxon(results["05"], results["075"])
                stat_results[epoch_duration].append(f"width=0.5 vs width=0.75 (E={epoch_duration}s): p={p:.4f}")
                stat, p = wilcoxon(results["075"], results["1"])
                stat_results[epoch_duration].append(f"width=0.75 vs width=1 (E={epoch_duration}s): p={p:.4f}")
                stat, p = wilcoxon(results["05"], results["1"])
                stat_results[epoch_duration].append(f"width=0.5 vs width=1 (E={epoch_duration}s): p={p:.4f}")

        with open(results_file, 'w') as f:
            json.dump(stat_results, f)
            