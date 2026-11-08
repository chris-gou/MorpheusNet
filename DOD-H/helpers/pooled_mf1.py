import glob
import os
import numpy as np

RUNS = {
    "e=5,b=1": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_5_05_og_data_new_resnet_relu",
    "e=5,b=2":  "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_5_075_og_data_new_resnet_relu",
    "e=5,b=3": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_5_3_og_data_new_resnet_relu",
    "e=10,b=1": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_10_05_og_data_new_resnet_relu",
    "e=10,b=2": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_10_075_og_data_new_resnet_relu",
    "e=10,b=3": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_10_3_og_data_new_resnet_relu",
    "e=30,b=1": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_30_05_og_data_new_resnet_relu",
    "e=30,b=2": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_30_075_og_data_new_resnet_relu",
    "e=30,b=3": "/home/christina/Documents/MorpheusNet/DOD-H/results/PHYSIO_30_3_og_data_new_resnet_relu",
}
CLASSES = ["W", "N1", "N2", "N3", "REM"]
file_name = "pooled_metrics.txt"
results_path = os.path.join('results', file_name)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

def metrics(cm):
    tp, n = np.diag(cm), cm.sum()
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    acc = tp.sum() / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    return f1, acc, (acc - pe) / (1 - pe)

for name, res in RUNS.items():
    print(f"\n{name}  ({os.path.basename(res)})")
    paths = glob.glob(f"{res}/fold_*_cm.npz")
    assert paths, f"no fold_*_cm.npz in {res}"
    cms = [np.load(p)["seq_cm"] for p in paths]

    f1, acc, kappa = metrics(sum(cms))
    per_fold = [metrics(cm)[0][cm.sum(1) > 0].mean() for cm in cms]

    latency_file = "streaming_total_latency_summary_perfold.json"
    latency_file_path = os.path.join(res, latency_file)
    if os.path.exists(latency_file_path):
        import json
        with open(latency_file_path, "r") as f:
            latency_summary = json.load(f)
        with open(results_path, "a") as f:
            f.write(f"\n{os.path.basename(res)}  ({len(paths)} folds, {sum(cms).sum()} epochs)\n")
            f.write(f"  latency summary: {latency_summary}\n")
    # print(f"\n{name}  ({len(paths)} folds, {sum(cms).sum()} epochs)")
    # print(f"  pooled ACC: {acc*100:.4f} | MF1: {f1.mean()*100:.4f} | kappa: {kappa:.4f}")
    # with open(results_path, "a") as f:
    #     f.write(f"\n{os.path.basename(res)}  ({len(paths)} folds, {sum(cms).sum()} epochs)\n")
    #     f.write(f" {acc*100:.4f} & {f1.mean()*100:.4f} & {kappa:.4f} \n")
    #     f.write("  " + "  ".join(f"{c}: {v:.4f}" for c, v in zip(CLASSES, f1)) + "\n")
    print(f" {acc*100:.4f} & {f1.mean()*100:.4f} & {kappa:.4f} \\\\")
    # print(f"  mean per-÷fold MF1: {np.mean(per_fold):.4f} ")
    print("  " + "  ".join(f"{c}: {v:.4f}" for c, v in zip(CLASSES, f1)))
    for f in f1:
        print(f"  {f*100:.4f} & ", end="")
    print("\n========================================")

print(f"\nResults written to {file_name}")