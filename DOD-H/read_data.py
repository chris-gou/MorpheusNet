import numpy as np
import glob
import os
from scipy.signal import welch

harm_data_path = '/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30/'
print(f"Reading file {harm_data_path + 'tr03-0100.npz'}")
a = np.load(harm_data_path + 'tr03-0100.npz', allow_pickle=True)
for k in a.files:
    print(k, a[k].shape, a[k].dtype, a[k].nbytes/1e6, "MB")

non_harm_data_path = '/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2/'
print(f"Reading file {non_harm_data_path + 'tr03-0100.npz'}")
b = np.load(non_harm_data_path + 'C3-M2_tr03-0100.npz', allow_pickle=True)
for k in b.files:
    print(k, b[k].shape, b[k].dtype, b[k].nbytes/1e6, "MB")


print(len(a['y']), len(b['y']))
local = a
mnt = b
y_local = a['y']
y_mnt = b['y']

best_offset, best_match = None, -1
# max_shift = 20
# for offset in range(-max_shift, max_shift + 1):
#     if offset >= 0:
#         a, b = y_local[offset:], y_mnt[:len(y_mnt) - offset]
#     else:
#         a, b = y_local[:len(y_local) + offset], y_mnt[-offset:]
#     n = min(len(a), len(b))
#     if n < 100:
#         continue
#     match = np.mean(a[:n] == b[:n])
#     if match > best_match:
#         best_match, best_offset = match, offset

# print(f"best offset: {best_offset}, agreement: {best_match:.4f}")
# print("agreement at offset 0:", np.mean(y_local[:min(len(y_local),len(y_mnt))] == y_mnt[:min(len(y_local),len(y_mnt))]))

# print("local mu/sigma:", np.mean(local['x']), np.std(local['x']))
# print("mnt mu/sigma:", np.mean(mnt['x']), np.std(mnt['x']))

local_files = sorted(glob.glob(os.path.join("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30", "*.npz")))
mnt_files   = sorted(glob.glob(os.path.join("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2", "*.npz")))

local_unsorted = glob.glob(os.path.join("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30", "*.npz"))
mnt_unsorted   = glob.glob(os.path.join("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2", "*.npz"))

print("local sorted == local unsorted order:", local_files == local_unsorted)
print("mnt sorted == mnt unsorted order:", mnt_files == mnt_unsorted)
print("local unsorted[:5]:", [os.path.basename(f) for f in local_unsorted[:5]])
print("mnt unsorted[:5]:", [os.path.basename(f) for f in mnt_unsorted[:5]])


import numpy as np

# def n1_fraction(npz_path):
#     y = np.load(npz_path, allow_pickle=True)['y']
#     return np.mean(y == 1)  

# local_n1 = [n1_fraction(f) for f in local_unsorted]
# mnt_n1   = [n1_fraction(f) for f in mnt_unsorted]

# fold_size = 993 // 5  
# for fold in range(5):
#     start, end = fold * fold_size, (fold + 1) * fold_size
#     print(f"fold {fold} local test-slice mean N1 frac: {np.mean(local_n1[start:end]):.4f}")
#     print(f"fold {fold} mnt   test-slice mean N1 frac: {np.mean(mnt_n1[start:end]):.4f}")

def avg_psd(files, n=30):
    psds = []
    for f in (files)[:n]:
        with np.load(f, allow_pickle=True) as d:
            x = d['x'][0].flatten()  # first epoch per subject
            freqs, psd = welch(x, fs=100, nperseg=256)
            psds.append(psd)
    return freqs, np.mean(psds, axis=0), np.std(psds, axis=0)

f_l, m_l, s_l = avg_psd(local_files)
f_m, m_m, s_m = avg_psd(mnt_files)

print("local mean PSD:", m_l[:5])
print("mnt mean PSD:", m_m[:5])


def get_psd_at_freqs(path, target_freqs=[0.5, 1, 2, 5, 10, 20, 30]):
    with np.load(path, allow_pickle=True) as d:
        x = d['x'][0].flatten()
    freqs, psd = welch(x, fs=100, nperseg=256)
    for tf in target_freqs:
        idx = np.argmin(np.abs(freqs - tf))
        print(f"  {freqs[idx]:.2f}Hz: psd={psd[idx]:.4f}")

print("local:")
get_psd_at_freqs("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30/tr03-0100.npz")
print("mnt:")
get_psd_at_freqs("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2/C3-M2_tr03-0100.npz")

def freq_values_per_subject(files, target_freqs=[0.5,1,2,5,10,15,20,30], n=30):
    vals = {f: [] for f in target_freqs}
    for fp in sorted(files)[:n]:
        with np.load(fp, allow_pickle=True) as d:
            x = d['x'][0].flatten()
        freqs, psd = welch(x, fs=100, nperseg=256)
        for tf in target_freqs:
            idx = np.argmin(np.abs(freqs - tf))
            vals[tf].append(psd[idx])
    return vals


local_vals = freq_values_per_subject(sorted(glob.glob("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30/*.npz")))
mnt_vals   = freq_values_per_subject(sorted(glob.glob("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2/*.npz")))

# for f in [0.5,1,2,5,10,15,20,30]:
#     l, m = np.array(local_vals[f]), np.array(mnt_vals[f])
#     print(f"{f}Hz: local median={np.median(l):.2f} mean={np.mean(l):.2f} max={np.max(l):.2f}")
#     print(f"       mnt   median={np.median(m):.2f} mean={np.mean(m):.2f} max={np.max(m):.2f}")


import numpy as np, glob, os

def get_subject_id(filename):
    base = os.path.basename(filename).replace('.npz', '')
    return base.split('_')[-1] if '_' in base else base

local_files = {get_subject_id(f): f for f in glob.glob("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30/*.npz")}
mnt_files   = {get_subject_id(f): f for f in glob.glob("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2/*.npz")}
new_files = {get_subject_id(f): f for f in glob.glob("/mnt/truenas_db/user/christina/process-test/PHYSIO2018_harmonized/npz/30/*/C3-M2/*.npz")}
common_ids = sorted(set(local_files) & set(mnt_files))[:20]

correlations = []
for sid in common_ids:
    with np.load(local_files[sid], allow_pickle=True) as dl, np.load(mnt_files[sid], allow_pickle=True) as dm:
        xl, xm = dl['x'], dm['x']
        n = min(len(xl), len(xm))
        for i in range(0, n, max(1, n // 20)):  # sample ~20 epochs per subject
            a, b = xl[i].flatten(), xm[i].flatten()
            if a.std() > 0 and b.std() > 0:
                corr = np.corrcoef(a, b)[0, 1]
                correlations.append(corr)

correlations = np.array(correlations)
print(f"n={len(correlations)}  mean corr={correlations.mean():.4f}  median={np.median(correlations):.4f}")
print(f"pct below 0.9: {(correlations < 0.9).mean()*100:.1f}%")
print(f"pct below 0.5: {(correlations < 0.5).mean()*100:.1f}%")

def best_lag_correlation(a, b, max_lag=1500):  # ±15s at 100Hz
    best_corr, best_lag = -2, None
    for lag in range(-max_lag, max_lag + 1, 25):  # coarse scan first, step=25 samples
        if lag >= 0:
            x, y = a[lag:], b[:len(b)-lag]
        else:
            x, y = a[:len(a)+lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 500:
            continue
        x, y = x[:n], y[:n]
        if x.std() > 0 and y.std() > 0:
            c = np.corrcoef(x, y)[0, 1]
            if c > best_corr:
                best_corr, best_lag = c, lag
    return best_lag, best_corr
with np.load(local_files['tr03-0100'], allow_pickle=True) as dl, np.load(mnt_files['tr03-0100'], allow_pickle=True) as dm:
    xl = dl['x'][50].flatten()  # pick a mid-recording epoch, avoid edge effects
    xm = dm['x'][50].flatten()

lag, corr = best_lag_correlation(xl, xm)
print(f"best lag: {lag} samples ({lag/100:.2f}s), corr at best lag: {corr:.4f}")
print(f"corr at lag=0: {np.corrcoef(xl, xm)[0,1]:.4f}")

with np.load(mnt_files['tr03-0100'], allow_pickle=True) as dm:
    print("rm_start_epochs:", dm['rm_start_epochs'])
    print("n_all_epochs:", dm['n_all_epochs'] if 'n_all_epochs' in dm.files else 'N/A')
    print("n_epochs:", dm['n_epochs'])

def best_lag_correlation_wide(a, b, max_lag_epochs=20, epoch_len=3000):
    max_lag = max_lag_epochs * epoch_len
    best_corr, best_lag = -2, None
    for lag in range(-max_lag, max_lag + 1, epoch_len // 4):  # step in quarter-epochs
        if lag >= 0:
            x, y = a[lag:], b[:len(b)-lag]
        else:
            x, y = a[:len(a)+lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 500:
            continue
        x, y = x[:n], y[:n]
        if x.std() > 0 and y.std() > 0:
            c = np.corrcoef(x, y)[0, 1]
            if c > best_corr:
                best_corr, best_lag = c, lag
    return best_lag, best_corr

with np.load(new_files['tr06-0677'], allow_pickle=True) as dl, np.load(mnt_files['tr06-0677'], allow_pickle=True) as dm:
    xl = dl['x'].flatten()  # full concatenated recording, not one epoch
    xm = dm['x'].flatten()

lag, corr = best_lag_correlation_wide(xl, xm)
print(f"best lag: {lag} samples ({lag/3000:.2f} epochs, {lag/100:.2f}s), corr: {corr:.4f}")