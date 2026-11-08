import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

subjects = [919, 525, 567, 656, 926, 429, 869, 711, 174, 604, 866, 449, 778, 580, 76, 371, 885, 136,
 158, 290, 673, 321, 758, 70, 355, 359, 107, 265, 826, 139, 184, 708, 622, 942, 305, 810,
 306, 829, 864, 323]
all_activity = []   # create once, outside the per-file loop
all_stages = []      # create once, outside the per-file loop

for subj in subjects:
    print(f"Processing subject {subj}...")
    hyp_path = f'files/PHYSIO_30_3_og_data_new_resnet_subj_{subj}_hyp.npy'
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    cams = np.load(f'files/PHYSIO_30_3_og_data_new_resnet_subj_{subj}_cams_raw_clip.npy')

    hyp = np.load(hyp_path)
    mean_activity_per_epoch = cams.mean(axis=1)
    n1_vals = mean_activity_per_epoch[hyp[:,0]==1]
    n2_vals = mean_activity_per_epoch[hyp[:,0]==2]
    all_activity.append(mean_activity_per_epoch)
    all_stages.append(hyp[:, 0])

# for other_id, other_name in [(2, 'N2'), (3, 'N3'), (4, 'REM'), (0, 'W')]:
#     other_vals = mean_activity_per_epoch[hyp[:,0]==other_id]
#     u, p = stats.mannwhitneyu(n1_vals, other_vals)
#     print(f"N1 vs {other_name}: U={u:.1f}, p={p:.4f}")

# pvals = [0.0016, 0.0000, 0.0118, 0.0002]
# reject, pvals_corrected, _, _ = multipletests(pvals, method='holm')
# print(pvals_corrected)

pooled_activity = np.concatenate(all_activity)
pooled_stages = np.concatenate(all_stages)

n1_vals = pooled_activity[pooled_stages == 1]
for other_id, other_name in [(2,'N2'), (3,'N3'), (4,'REM'), (0,'W')]:
    other_vals = pooled_activity[pooled_stages == other_id]
    u, p = stats.mannwhitneyu(n1_vals, other_vals)
    print(f"N1 vs {other_name}: p={p:.4f}, n1={len(n1_vals)}, n2={len(other_vals)}")

pvals = [0.0000, 0.0000, 0.0032, 0.0000]
reject, pvals_corrected, _, _ = multipletests(pvals, method='holm')
print(pvals_corrected)