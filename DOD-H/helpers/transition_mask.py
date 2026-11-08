import numpy as np
import glob
from sklearn.metrics import confusion_matrix, f1_score

RUNS = {
    "e=30,b=3": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF20_30_075_subj", 30),
    # "e=5,b=2": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_5_2_new_data", 5),
    # "e=5,b=3": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_5_3_new_data", 5),
    # "e=10,b=1": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF78_10_1_subj", 10),
    # "e=10,b=2": ("/home/christina/Documents/MorpheusNet/DOD-H/results/SEDF78_10_2_subj", 10),
    # "e=10,b=3": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_10_3_new_data", 10),
    # "e=30,b=1": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_30_1_new_data", 30),
    # "e=30,b=2": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_30_2_new_data", 30),
    # "e=30,b=3": ("/home/christina/Documents/MorpheusNet/DOD-H/results/DOD-H_30_3_new_data", 30),
}
CLASSES = ["W", "N1", "N2", "N3", "REM"]

def transition_mask(y, k=1):
    """True for epochs within k epochs of a stage change."""
    mask = np.zeros(len(y), bool)
    for c in np.flatnonzero(np.diff(y)) + 1:
        mask[max(c - k, 0):c + k] = True
    return mask

def load_preds(res):
    search_path = f"{res}/per_subject_preds/*.npz"
    paths = sorted(glob.glob(search_path))
    assert paths, f"no predictions in path {search_path}"
    print(f"Loading {len(paths)} predictions from {res}")
    print(f"Found {len(paths)} prediction files in {search_path}")
    print(np.load(paths[0]).files)
    y_true = [np.load(p)["y_true"] for p in paths]
    y_pred = [np.load(p)["y_pred"] for p in paths]
    return y_true, y_pred


for name, (res, e) in RUNS.items():
    y_true, y_pred = load_preds(res)
    k = int(30/e)

    # Build the mask per subject, then concatenate. Masking after concatenating
    # would create fake transitions at subject boundaries.
    m = np.concatenate([transition_mask(y, k) for y in y_true])
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)

    f1_tr = f1_score(yt[m], yp[m], average=None, labels=range(5), zero_division=0)
    f1_st = f1_score(yt[~m], yp[~m], average=None, labels=range(5), zero_division=0)

    print(f"\n{name}  ({m.mean():.1%} of epochs near a transition, k={k})")
    print(f"  transition MF1: {f1_tr.mean():.3f}   stable MF1: {f1_st.mean():.3f}")
    print("  per-class F1 (transition / stable):")
    for c, a, b in zip(CLASSES, f1_tr, f1_st):
        print(f"    {c:<4} {a:.3f} / {b:.3f}")

    cm = confusion_matrix(yt, yp, labels=range(5))
    print(f"  n: {len(yt)} | true counts: {cm.sum(axis=1)}")
    print(f"  overall MF1: {f1_score(yt, yp, average='macro'):.3f}")
    print(cm)
    print(f"`  overall MF1: {f1_score(yt, yp, average='macro'):.3f}")

print(confusion_matrix(yt, yp, labels=range(5)))
