
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


CLASSES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_S = 30   # transition window in seconds (k = WINDOW_S / e)
MAX_S = 120     # x-range of the distance plot

RUNS = {
    "10 s, 1 b": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF78_10_1_subj", 10),
    "10 s, 2 b": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF78_10_2_subj", 10),
    "10 s, 3 b": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF78_10_3_subj", 10),
}

def near_mask(dist_s):
    """Option A: 30 s before to 30 s after a true transition."""
    return (dist_s >= -WINDOW_S) & (dist_s < WINDOW_S)

def load_recordings(results_dir):
    """One (y_true, y_pred) pair per recording. Sorted: glob order is not stable on NAS."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
    return [(np.load(f)["y_true"], np.load(f)["y_pred"]) for f in files]


def signed_distance(y):
    """Epochs to nearest true transition. <0 = before it, >=0 = after (0 = first epoch of new stage)."""
    b = np.flatnonzero(np.diff(y)) + 1
    idx = np.arange(len(y))
    if len(b) == 0:
        return np.full(len(y), np.inf)
    pos = np.searchsorted(b, idx, side="right")
    d_prev = np.where(pos > 0, idx - b[np.clip(pos - 1, 0, len(b) - 1)], np.inf)
    d_next = np.where(pos < len(b), idx - b[np.clip(pos, 0, len(b) - 1)], -np.inf)
    return np.where(d_prev <= -d_next, d_prev, d_next)


def pool(recordings, epoch_s):
    """Concatenate across recordings AFTER computing distances per recording."""
    y_true = np.concatenate([t for t, _ in recordings])
    y_pred = np.concatenate([p for _, p in recordings])
    dist_s = np.concatenate([signed_distance(t) for t, _ in recordings]) * epoch_s
    return y_true, y_pred, dist_s


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=range(5), average=None, zero_division=0)


def plot_class_f1(ax, y_true, y_pred, dist_s, title):
    near = near_mask(dist_s)
    f1_t = per_class_f1(y_true[near], y_pred[near])
    f1_s = per_class_f1(y_true[~near], y_pred[~near])
    print(f"{title}: {near.mean():.1%} near transition | "
          f"MF1 transition {f1_t.mean():.3f}, stable {f1_s.mean():.3f}")

    x = np.arange(5)
    bar_s = ax.bar(x - 0.2, f1_s * 100, 0.4, label="Stable")
    bar_t = ax.bar(x + 0.2, f1_t * 100, 0.4, label=f"Transition (±{WINDOW_S} s)")
    ax.bar_label(bar_s, fmt="%.1f", fontsize=7, padding=2)
    ax.bar_label(bar_t, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 110)  # headroom for labels
    ax.set_title(title)


def plot_distance_curve(ax, y_true, y_pred, dist_s, epoch_s, label):
    correct = y_true == y_pred
    bins = np.arange(-MAX_S, MAX_S + epoch_s, epoch_s)
    acc = [correct[dist_s == b].mean() * 100 for b in bins]
    ax.plot(bins, acc, marker="o", markersize=3, label=label)
    print(f"{label}: {np.sum(~np.isnan(acc))} points plotted, "
          f"overall accuracy {correct.mean():.3f}, "
          f"transition accuracy {correct[near_mask(dist_s)].mean():.3f}")

def plot_confusion(axes, y_true, y_pred, dist_s, title):
    near = near_mask(dist_s)
    for ax, mask, name in [(axes[0], ~near, "Stable"), (axes[1], near, "Transition")]:
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=range(5), normalize="true") * 100
        ax.imshow(cm, cmap="RdPu", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 50 else "black")
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Predicted (normalized %)")
        ax.set_title(f"{title}: {name}", fontsize=9)
    axes[0].set_ylabel("True (normalized %)")

def main():
    fig1, axes1 = plt.subplots(1, len(RUNS), figsize=(4 * len(RUNS), 3), squeeze=False) # transition f1 per class
    fig2, ax2 = plt.subplots(figsize=(5, 3)) # transition accuracy vs distance
    fig3, axes3 = plt.subplots(len(RUNS), 2, figsize=(6, 3 * len(RUNS)), squeeze=False) # transition confusion matrices


    for ax, (label, (results_dir, epoch_s)) in zip(axes1[0], RUNS.items()):
        y_true, y_pred, dist_s = pool(load_recordings(results_dir), epoch_s)
        plot_class_f1(ax, y_true, y_pred, dist_s, label)
        plot_distance_curve(ax2, y_true, y_pred, dist_s, epoch_s, label)

    axes1[0, 0].set_ylabel("F1 Score (%)")
    axes1[0, 0].legend(loc="upper right", fontsize=8)
    fig1.tight_layout()
    fig1.savefig("transition_f1_per_class.png", dpi=300)

    ax2.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("Time relative to true transition (s)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(title="Epoch length")
    fig2.tight_layout()
    fig2.savefig("transition_accuracy_vs_distance.png", dpi=300)


    for ax, cm_axes, (label, (results_dir, epoch_s)) in zip(axes1[0], axes3, RUNS.items()):
        y_true, y_pred, dist_s = pool(load_recordings(results_dir), epoch_s)
        # plot_class_f1(ax, y_true, y_pred, dist_s, label)
        # plot_distance_curve(ax2, y_true, y_pred, dist_s, epoch_s, label)
        plot_confusion(cm_axes, y_true, y_pred, dist_s, label)

    fig3.tight_layout()
    fig3.savefig("transition_confusion.png", dpi=300)


if __name__ == "__main__":
    main()