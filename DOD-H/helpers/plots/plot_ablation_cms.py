import glob
import math
import numpy as np
import matplotlib.pyplot as plt

KEYS = ['cnn_cm', 'seq_cm']
KEYS_NAME = {'cnn_cm': 'CNN', 'seq_cm': 'SEQ'}
labels = ["W", "N1", "N2", "N3", "REM"]
dataset = 'SEDF78'

def plot_grid(runs, out, ncols=3):
    for key in KEYS:
        key_name = KEYS_NAME[key]
        nrows = math.ceil(len(runs) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                                layout="constrained", squeeze=False)

        k = len(labels)

        for idx, (ax, (name, folder)) in enumerate(zip(axes.flat, runs.items())):
            print(f"Processing {out}: {name} in {folder}")
            total = np.load((f"{folder}/total_{key}.npy")) 
            pct = total / total.sum(axis=1, keepdims=True) * 100
            tp = np.diag(total)
            mf1 = np.mean(2 * tp / (total.sum(0) + total.sum(1)))

            im = ax.imshow(pct, cmap="RdPu", vmin=0, vmax=100)
            for i in range(k):
                for j in range(k):
                    color = "white" if pct[i, j] > 50 else "black"
                    ax.text(j, i, f"{pct[i, j]:.1f}%\n({int(total[i, j])})",
                            ha="center", va="center", color=color, fontsize=6.5)

            ax.set_title(f"{name}  (MF1 {mf1:.3f})", fontsize=9)
            ax.set_xticks(range(k), labels if idx // ncols == nrows - 1 else [])
            ax.set_yticks(range(k), labels if idx % ncols == 0 else [])

        for ax in axes.flat[len(runs):]:        # hide unused panels
            ax.axis("off")

        fig.suptitle(f"{dataset} - Confusion Matrix ({key_name}) - Ablation: {'width' if 'width' in out else 'blocks'}", fontsize=12)
        fig.supxlabel("Predicted")
        fig.supylabel("True")
        fig.colorbar(im, ax=axes, shrink=0.6, label="% of true class")
        fig.savefig(f"{dataset}-{out}-{key_name}.pdf")
        fig.savefig(f"{dataset}-{out}-{key_name}.png", dpi=300)


width_runs = {
    "e=5,w=0.5": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_05_new_data",
    "e=5,w=0.75": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_075_new_data",
    "e=5,w=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_3_new_data",
    "e=10,w=0.5": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_05_new_data",
    "e=10,w=0.75": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_075_new_data",
    "e=10,w=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_3_new_data",
    "e=30,w=0.5": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_05_new_data",
    "e=30,w=0.75": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_075_new_data",
    "e=30,w=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_3_new_data",
    
}
block_runs = {
    "e=5,b=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_1_new_data",
    "e=5,b=2": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_2_new_data",
    "e=5,b=3": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_5_3_new_data",
    "e=10,b=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_1_new_data",
    "e=10,b=2": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_2_new_data",
    "e=10,b=3": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_10_3_new_data",
    "e=30,b=1": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_1_new_data",
    "e=30,b=2": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_2_new_data",
    "e=30,b=3": f"/home/christina/Documents/MorpheusNet/DOD-H/results/{dataset}_30_3_new_data",
}

plot_grid(width_runs, "cm_grid_width")
plot_grid(block_runs, "cm_grid_blocks")