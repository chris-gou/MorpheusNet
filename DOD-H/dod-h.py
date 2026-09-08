import h5py
import numpy as np
from scipy.signal import resample_poly

def load_signal(filepath, channel='C3_M2', old_layout=False):
    """Load one channel's signal + labels from an HDF5 file, either layout."""
    with h5py.File(filepath, 'r') as f:
        signals_grp = f['signals']
        if old_layout:
            raw = signals_grp['eeg'][channel]
        else:
            raw = signals_grp[channel]

        signals = np.asarray(raw['data'], dtype=np.float32) if isinstance(raw, h5py.Group) \
            else np.asarray(raw, dtype=np.float32)

        y = np.asarray(f['y'], dtype=np.int64) if 'y' in f \
            else np.asarray(f['hypnogram'], dtype=np.int64)

        return signals, y


def compare_signals(old_path, new_path, channel='C3_M2'):
    old_fs, new_fs = 250, 100
    sig_old, y_old = load_signal(old_path, channel, old_layout=True)
    sig_new, y_new = load_signal(new_path, 'F3-RPA', old_layout=False)

    print(f"--- Shapes ---")
    print(f"old: signal {sig_old.shape}, labels {y_old.shape}")
    print(f"new: signal {sig_new.shape}, labels {y_new.shape}")

    sig_old_rs = resample_poly(sig_old, up=new_fs, down=old_fs)
    print(f"old resampled: {sig_old_rs.shape}")

    print(f"\n--- Signal stats ---")
    for name, s in [("old", sig_old), ("new", sig_new)]:
        print(f"{name}: mean={s.mean():.4f} std={s.std():.4f} "
              f"min={s.min():.4f} max={s.max():.4f} nan={np.isnan(s).any()}")

    print(f"\n--- Label agreement ---")
    if y_old.shape == y_new.shape:
        match = (y_old == y_new).mean()
        print(f"label match: {match:.4%}")
    else:
        print("label shapes differ, skipping direct comparison")

    print(f"\n--- Signal agreement ---")
    if sig_old.shape == sig_new.shape:
        diff = sig_old - sig_new
        print(f"max abs diff: {np.abs(diff).max():.6f}")
        print(f"mean abs diff: {np.abs(diff).mean():.6f}")
        corr = np.corrcoef(sig_old.flatten(), sig_new.flatten())[0, 1]
        print(f"correlation: {corr:.6f}")
        print(f"bit-identical: {np.array_equal(sig_old, sig_new)}")
    else:
        print("signal shapes differ, skipping direct diff/correlation")

    if sig_old_rs.shape != sig_new.shape:
        # trim to shortest length (edge effects from resampling)
        n = min(len(sig_old_rs), len(sig_new))
        sig_old_rs = sig_old_rs[:n]
        sig_new_trim = sig_new[:n]
    else:
        sig_new_trim = sig_new

    diff = sig_old_rs - sig_new_trim
    corr = np.corrcoef(sig_old_rs, sig_new_trim)[0, 1]

    print(f"\n--- Resampled comparison ---")
    print(f"correlation: {corr:.6f}")
    print(f"max abs diff: {np.abs(diff).max():.6f}")
    print(f"mean abs diff: {np.abs(diff).mean():.6f}")

    diff = sig_old_rs - sig_new_trim
    worst_idx = np.argmax(np.abs(diff))
    n = len(diff)

    print(f"worst diff at index {worst_idx} / {n} ({100*worst_idx/n:.2f}% through signal)")
    print(f"old value: {sig_old_rs[worst_idx]:.4f}, new value: {sig_new_trim[worst_idx]:.4f}")

    # Check how many samples exceed some threshold, e.g. 5x std
    threshold = 5 * sig_new_trim.std()
    n_outliers = np.sum(np.abs(diff) > threshold)
    print(f"samples with |diff| > {threshold:.2f}: {n_outliers} ({100*n_outliers/n:.4f}%)")

    # Where are the outliers concentrated? (edges vs. scattered)
    outlier_idx = np.where(np.abs(diff) > threshold)[0]
    if len(outlier_idx) > 0:
        print(f"outlier index range: {outlier_idx.min()} to {outlier_idx.max()}")

def get_file_fs(filepath, channel='F3-M2'):
    with h5py.File(filepath, 'r') as f:
        x = f['signals']['eeg'][channel]
        y = f['hypnogram']
        # x = f['signals'][channel]
        # signals = np.asarray(x['data'], dtype=np.float32)
        # y = f['y']
        n_samples = x.shape[0] if x.ndim == 1 else x.shape[-1]
        n_epochs = y.shape[0]
        natife_fs = n_samples / (n_epochs * 30)
        print(natife_fs)


if __name__ == "__main__":
    old_file = "/mnt/truenas_db/user/christina/DOD-H - Dreem Open Dataset - Healthy/095d6e40-5f19-55b6-a0ec-6e0ad3793da0.h5"
    new_file = "/home/christina/Documents/pillow/processed-datasets/DOD-H_harmonized/hdf5/095d6e40-5f19-55b6-a0ec-6e0ad3793da0.hdf5"
    compare_signals(old_file, new_file, channel='F3_M2')
    # get_file_fs(new_file, channel='F3_RPA')