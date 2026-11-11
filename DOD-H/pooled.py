import glob, sys
import numpy as np

run_dir = sys.argv[1]
cm = sum(np.load(p)["seq_cm"] for p in glob.glob(run_dir + "/fold_*_cm.npz"))

f1 = 2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))
acc = np.trace(cm) / cm.sum()
pe = np.sum(np.sum(cm, 0) * np.sum(cm, 1)) / cm.sum() ** 2
kappa = (acc - pe) / (1 - pe)

print(f"folds={len(glob.glob(f'{run_dir}/fold*_cm.npz'))}")
print(f"acc={acc*100:.4f}, kappa={kappa:.4f}, f1={f1.mean()*100:.4f}")
print(f"W/N1/N2/N3/REM:", " ".join(f"{v*100:.3f}" for v in f1))