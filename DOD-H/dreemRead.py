import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import h5py
from scipy.signal import butter, lfilter, resample, welch

# ======= HELPERS ========
# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# function to read the refrenced F3 signal, EOG and labels
def do(f1):
    sigs = f1['signals']
    eeg = sigs['eeg']    
    f3 = eeg['F3_M2']    
    eog = sigs['eog']    
    eog1 = eog['EOG1']    
    hyp = f1['hypnogram']
    return f3,eog1,hyp

def needs_bandpass(x, fs=100, lowcut=0.5, highcut=40, n_samples=20):
    "Check if signal needs bandpass filtering by checking if it contains frequencies outside the desired range."
    sig = x[:n_samples].reshape(-1) # check first n_samples
    sig = sig - np.mean(sig) # remove DC offset
    fr, p = welch(sig, fs=fs, nperseg=min(1024, len(sig)))

    outside_range = p[(fr > 0)&(fr<lowcut)].sum() + p[fr>highcut].sum() # sum of power outside the desired range
    ratio = outside_range / p.sum() if p.sum() > 0 else 0
    return ratio > 0.01 # if more than 1% of the power is outside the range, it needs bandpass filtering
    pass

# ======== MAIN FUNCS ========
# Function to extract single channel EEG data based on filetype (npz or h5)
def extract_data(ind, eeg_chan = 'F3_F4', path = '', epoch_length=30, files=None):
    if files is None:
        # find any h5 files in the directory and/or subdirectories, depends on data path
        h5 = (glob.glob(os.path.join(path, '*.h5'))     +
            glob.glob(os.path.join(path, '*.hdf5'))   +
            glob.glob(os.path.join(path, '*', '*.h5')) +
            glob.glob(os.path.join(path, '*', '*.hdf5')))
        
        if h5:
            x, hyp = extract_h5(ind, h5, epoch_length, channel=eeg_chan)
            return x, hyp
    
    if files is None: # no h5 files found or given, look for npz
        files = glob.glob(os.path.join(path, '*.npz')) + glob.glob(os.path.join(path, '*', '*.npz')) + glob.glob(os.path.join(path, '*','*', '*.npz'))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz or .h5 files found in {path}")
    # files = glob.glob(os.path.join(path, '*.npz'), recursive=True)
    # files = glob.glob(os.path.join(path, '**', eeg_chan, '*.npz'), recursive=True)
    if ind < 0 or ind >= len(files):
        print(f"Index {ind} is out of range for the available files. Total files: {len(files)}")
        # Skip invalid indices
        return None, None
    x, hyp = extract_npz(ind, files, epoch_length, eeg_chan)

    # data cleanup
    mask = hyp != -1 # exclude unknown labels
    x, hyp = x[mask], hyp[mask]
    mask = ~np.isnan(x).any(axis=(1,2,3)) & (hyp.flatten() != -1)
    x, hyp = x[mask], hyp[mask]
    return x, hyp

# function to extract the data from 2 different channels
def extract_data_bothChan(ind, eeg_chan = 'F3_F4', path = ''):
    os.chdir(path)
    files = os.listdir()
    filess = []
    for i in files:
        if i.split('.')[-1]=='h5':
            filess.append(i)
    files = filess
    f1 = h5py.File(files[ind])
    signals = f1['signals']
    eeg = signals['eeg']
    x = eeg['F3_F4']
    hyp = f1['hypnogram']
    x = np.array(x)
    x2 = eeg['F3_M2']
    hyp = f1['hypnogram']
    x2 = np.array(x2)
    return x, x2, hyp

def split_epochs(epochs_30s, hyp, epoch_duration, fs=100):
    window = int(epoch_duration*fs)
    hop = window
    base_len = 30 * fs

    starts = list(range(0, base_len-window+1, hop))
    x_out, y_out = [], []

    for i in range(len(epochs_30s)):
        for s in starts:
            w = epochs_30s[i:i+1, s:s+window]  # (1, 1, window, 1)
            w = w.reshape(1, 1, window, 1)   
            mu, sigma = w.mean(), w.std()
            if sigma > 0:
                w = (w - mu) / sigma
            x_out.append(w)
            y_out.append(hyp[i])
    return np.concatenate(x_out, axis=0), np.array(y_out)

# ======== NPZ ========
def extract_npz(ind, files, epoch_length, eeg_chan):
    # Function to extract data from npz files
    npz_file = np.load(files[ind], allow_pickle=True)
    channel_labels = np.atleast_1d(npz_file['ch_label'])
    x = npz_file['x']
    y = npz_file['y']
    if eeg_chan not in channel_labels:
        raise ValueError(f"Index: {ind}, File: {files[ind]}, Channel {eeg_chan} not found in the available channels: {channel_labels}")

    if needs_bandpass(epochs, fs=100): # we assume that data is already filtered but in case it isnt, apply the bandpass filter
        if ind < 5:
            print(f"[INFO]: bandpass filtering {eeg_chan} channel between 0.5 and 40 Hz")
        epochs = butter_bandpass_filter(epochs, 0.5, 40, 100)

    # Get the index of the desired channel
    chan_idx = np.where(channel_labels == eeg_chan)[0]
    
    # x is expected to be already epoched at 100 Hz (30 s -> 3000 samples).
    if x.ndim == 2 and x.shape[1] == int(epoch_length) * 100:
        epochs = x[:, None, :, None]                       # (n, 1, 3000, 1)
    elif x.ndim == 3 and x.shape[1] == int(epoch_length) * 100:               # (n, 3000, C)
        epochs = x[:, :, chan_idx][:, None, :, None]
    elif x.ndim == 3 and x.shape[2] == int(epoch_length) * 100:               # (n, C, 3000)
        epochs = x[:, chan_idx, :][:, None, :, None]
    # else:
    #     raise ValueError(f"Unexpected x shape (expected pre-epoched), got: {x.shape}")

    if epoch_length != 30  and x.shape[1] != int(epoch_length) * 100: # case where the data is not already epoched at the desired length, we need to split the epochs
        if ind < 5:
            print(f"[INFO]: splitting epoch into {epoch_length}s chunks") # print this for confirmation that it's going through
        epochs, y = split_epochs(x, y, epoch_length)

    
    # close file
    npz_file.close()
    return epochs.astype(np.float32), y

# ======== H5 ========
def extract_h5(ind, files, epoch_length=30, channel='F3_M2'):
    # Function to extract data from h5 files
    with h5py.File(files[ind], 'r') as f:
        if 'signals' in f and channel in f['signals']:
            signals = np.asarray(f[f'signals/eeg/{channel}/data'], dtype=np.float32)
            if 'y' in f:
                y30 = np.asarray(f['y'], dtype=np.int64)
            return _epoch_continuous(signals, y30, epoch_length, fs=100)
        else:
            signals = f['signals']
            eeg = signals['eeg']
            x = eeg[channel]
            hyp = f['hypnogram']
            x = np.array(x)
            hyp = np.array(hyp)
            return x, hyp
    pass

def _epoch_continuous(sig, y30, epoch_length, fs=100):
    """Reshape 1D 100-Hz signal into (n, 1, fs*epoch, 1) z-scored epochs.
    For 10s/5s, each 30s label is repeated 3 or 6 times (label inheritance)."""
    sig = butter_bandpass_filter(sig,0.5,40,fs)    #bandpassing between 0.5 and 40Hz
    spe     = epoch_length * fs
    n_short = max(1, 30 // epoch_length)
    n       = min(len(y30) * n_short, len(sig) // spe)

    x = sig[: n * spe].reshape(n, spe)
    y = np.repeat(y30, n_short)[:n]

    mask = np.isin(y, [0, 1, 2, 3, 4])           # keep W, N1, N2, N3, REM only
    x, y = x[mask], y[mask]

    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    x = x.reshape(-1, 1, spe, 1).astype(np.float32)
    fr, p = welch(sig, fs=fs, nperseg=1024)
    # print("[_epoch_continuous] post-filter <0.1 Hz:", round(p[fr < 0.1].sum() / p.sum(), 4))

    return x, y