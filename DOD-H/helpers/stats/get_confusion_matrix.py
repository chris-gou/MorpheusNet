from sklearn.metrics import confusion_matrix
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

def get_confusion_matrix(npz_file):
    if not os.path.exists(npz_file):
        print(f"File {npz_file} does not exist.")
        return None

    data = np.load(npz_file)
    cnn_cm = data['cnn_cm']
    seq_cm = data['seq_cm']
    return cnn_cm, seq_cm

dataset = 'SEDF78'
total_folds = 10
results_path = '/home/christina/Documents/MorpheusNet/DOD-H/results'
KEYS = ['cnn_cm', 'seq_cm']
KEYS = ['seq_cm']
KEYS_NAME = {'cnn_cm': 'CNN', 'seq_cm': 'SEQ'}
labels = ["W", "N1", "N2", "N3", "REM"]

for folder in os.listdir(results_path):
    if dataset and '_subj' in folder:
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(results_path, folder)
        folder_name = os.path.basename(folder_path).replace('_subj', '')
        folder_name = folder_name.replace('_', ' ')
        cm_files = glob.glob(os.path.join(folder_path, 'fold_*_cm.npz'))
        if len(cm_files) < total_folds:
            print(f"Warning: Found only {len(cm_files)} confusion matrix files in {folder_path}, expected {total_folds}.")
            continue

        for key in KEYS:
            total = sum(np.load(f)[key] for f in cm_files)
            np.save(os.path.join(folder_path, f'total_{key}.npy'), total)

            tp = np.diag(total)
            f1 = 2 * tp / (total.sum(axis=0) + total.sum(axis=1))
            print(total)
            print("per-class F1:", f1.round(3))
            print("MF1:", f1.mean().round(3), " Acc:", (tp.sum() / total.sum()).round(3))

            # plot and save
            pct = total / total.sum(axis=1, keepdims=True) * 100  # row-normalised
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.imshow(pct, cmap="RdPu", vmin=0, vmax=100)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if pct[i, j] > 50 else "black"
                    ax.text(j, i, f"{pct[i, j]:.1f}%\n({int(total[i, j])})",
                            ha="center", va="center", color=color, fontsize=8)

            ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
                xticklabels=labels, yticklabels=labels,
                xlabel="Predicted", ylabel="True")
            ax.title.set_text(f"{folder_name} - Confusion Matrix ({KEYS_NAME[key]})")
            fig.colorbar(im, ax=ax, label="% of true class")
            fig.tight_layout()
            fig.savefig(f"{folder_path}/pooled_cm_{key}.png", dpi=300)
                        


# for fold in range(5):
#     if os.path.exists(os.path.join(file1, f'fold_{fold}_cm.npz')):
#         cnn_cm1, seq_cm1 = get_confusion_matrix(os.path.join(file1, f'fold_{fold}_cm.npz'))
#     else:
#         cnn_cm1, seq_cm1 = None, None
#     if os.path.exists(os.path.join(file2, f'fold_{fold}_cm.npz')):
#         cnn_cm2, seq_cm2 = get_confusion_matrix(os.path.join(file2, f'fold_{fold}_cm.npz'))
#     else:   
#         cnn_cm2, seq_cm2 = None, None
#     print(f"Confusion matrix cnn for fold {fold} in {file1} and {file2}:")
#     print(cnn_cm1)
#     print(cnn_cm2)

#     print(f"Confusion matrix seq for fold {fold} in {file1} and {file2}:")

#     print(seq_cm1)
#     print(seq_cm2)