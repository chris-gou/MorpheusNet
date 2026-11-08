import glob
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from terminaltables import SingleTable
from termcolor import colored


base = "/home/christina/Documents/MorpheusNet/DOD-H/results" 
widths = ["05", "075", "1"] # 3 represents 1.0 
blocks = [1,2,3]
dataset = "DOD-H"
epochs = [5, 10, 30]
runs_blocks = {f"e={e},b={b}": f"{dataset}_{e}_{b}_new_data" for e in epochs for b in blocks}
runs_widths = {f"e={e},w={w}": f"{dataset}_{e}_{w}_new_data" for e in epochs for w in widths}


def mf1(cm):
    return np.mean(2 * np.diag(cm) / (cm.sum(0) + cm.sum(1) + 1e-8))

def class_f1(cm):
    return (2 * np.diag(cm) / (cm.sum(0) + cm.sum(1) + 1e-8))*100

def accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)

def kappa(cm):
    n = np.sum(cm)
    po = np.sum(np.diag(cm)) / n
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n * n)
    return (po - pe) / (1 - pe)

def load(run):
    paths = sorted(glob.glob(f"{base}/{run}/fold_*_cm.npz"),
                   key=lambda s: int(s.split("fold_")[1].split("_")[0]))
    cms = [np.load(p)["seq_cm"] for p in paths]
    if len(cms) == 0:
        return 0, 0, 0, [0,0,0,0,0]
    return accuracy(sum(cms)), mf1(sum(cms)), kappa(sum(cms)), class_f1(sum(cms))

def holm(ps):
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        adj[i] = running
    return adj

def calculate_stats(res, factor, levels):
    for name, (acc, pooled, kappa_v, class_mf1) in res.items():
        print("====================" + f"{name} ====================")
        overall_data = [['ACC', 'MF1', '\u03BA'], [f"{acc*100:.6f}", f"{pooled*100:.6f}", f"{kappa_v:.6f}"]]
        print(SingleTable(overall_data, title=colored('Overall Metrics', 'magenta')).table)
        perclass_data = [['Class', 'MF1']]+[[f"{i}", f"{v:.6f}"] for i, v in enumerate(class_mf1)]
        print(SingleTable(perclass_data, title=colored('Per-Class MF1', 'magenta')).table)
    pairs = [(5, 10), (10, 30), (5, 30)]
    for lvl in levels:
        fold_mf1 = {e: res[f"e={e},{factor}={lvl}"][-1] for e in epochs}

        p_fried = friedmanchisquare(*fold_mf1.values()).pvalue
        print(f"\n{factor}={lvl}: Friedman p = {p_fried:.4f}")

        if p_fried < 0.05:  # post-hoc only after a significant omnibus test
            ps = np.array([wilcoxon(fold_mf1[a], fold_mf1[b]).pvalue for a, b in pairs])
            for (a, b), p, pa in zip(pairs, ps, holm(ps)):
                print(f"  e={a} vs e={b}: p = {p:.4f}, Holm = {pa:.4f}")


res_b = {name: load(run) for name, run in runs_blocks.items()}
res_w = {name: load(run) for name, run in runs_widths.items()}
calculate_stats(res_b, "b", blocks)
calculate_stats(res_w, "w", widths)