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

def get_file_fs(filepath, channel='F3-M2',x=None, y=None, epoch_duration=30, file_type='h5'):
    if file_type == 'h5':
        if x is not None and y is not None:
            n_samples = x.shape[0] if x.ndim == 1 else x.shape[-1]
            n_epochs = y.shape[0]
            native_fs = n_samples / (n_epochs * 30)
            return native_fs
        
        with h5py.File(filepath, 'r') as f:
            signals_group = f['signals']
            if 'eeg' in signals_group and channel in signals_group['eeg']:
                x = signals_group['eeg'][channel]
            elif channel in signals_group:
                x = signals_group[channel]
            if 'y' in f:
                y = np.asarray(f['y'], dtype=np.int64)
            elif 'hypnogram' in f:
                y = np.asarray(f['hypnogram'], dtype=np.int64)

            n_samples = x.shape[0] if x.ndim == 1 else x.shape[-1]
            n_epochs = y.shape[0]
            native_fs = n_samples / (n_epochs * 30)
        return native_fs

    if file_type == 'npz':
        npz_file = np.load(filepath, allow_pickle=True)
        if 'fs' in npz_file:
            native_fs = npz_file['fs']
            npz_file.close()
            return native_fs
        x = npz_file['x']
        y = npz_file['y']
        if x.ndim == 2:
            samples_per_epoch = x.shape[1]
        elif x.ndim == 3:
            samples_per_epoch = x.shape[2]
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")
        native_fs = samples_per_epoch / 30
        npz_file.close()
        return native_fs

# function to read the refrenced F3 signal, EOG and labels
def do(f1):
    sigs = f1['signals']
    eeg = sigs['eeg']    
    f3 = eeg['F3_M2']    
    eog = sigs['eog']    
    eog1 = eog['EOG1']    
    hyp = f1['hypnogram']
    return f3,eog1,hyp

def needs_bandpass(x, fs=100, lowcut=0.3, highcut=35, n_samples=20):
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
def extract_data(ind, files, eeg_chan = 'F3_F4', path = '', epoch_length=30):
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz or .h5 files found in {path}")

    if ind < 0 or ind >= len(files):
        print(f"Index {ind} is out of range for the available files. Total files: {len(files)}")
        # Skip invalid indices
        return None, None

    if 'h5' in files[0] or 'hdf5' in files[0]:
        x, hyp = extract_h5(ind, files, epoch_length, channel=eeg_chan)
    else:
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

def split_epochs(epochs_30s, hyp, target_epoch_duration, fs=100):
    window = int(target_epoch_duration*fs)
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
        new_chann_name = f'eeg/{eeg_chan}'
        if new_chann_name not in channel_labels:
            raise ValueError(f"Index: {ind}, File: {files[ind]}, Channel {eeg_chan} not found in the available channels: {channel_labels}")
    original_fs = get_file_fs(files[ind], channel=eeg_chan, x=x, y=y, epoch_duration=epoch_length, file_type='npz')
    if original_fs != 100:
        if ind < 2:
            print(f"[INFO]: resampling {eeg_chan} channel from {original_fs} Hz to 100 Hz")
        x = resample(x, int(len(x) * 100 / original_fs))
    if needs_bandpass(x, fs=100): # we assume that data is already filtered but in case it isnt, apply the bandpass filter
        if ind < 2:
            print(f"[INFO]: bandpass filtering {eeg_chan} channel between 0.3 and 35 Hz")
        x = butter_bandpass_filter(x, 0.3, 35, 100)

    # Get the index of the desired channel
    chan_idx = np.where(channel_labels == eeg_chan)[0]
    
    # x is already epoched at 100 Hz (30 s -> 3000 samples, x s -> x*100 samples).
    if x.ndim == 2 and x.shape[1] == int(epoch_length) * 100:
        epochs = x[:, None, :, None]                       # (n, 1, 3000, 1)
    elif x.ndim == 3 and x.shape[1] == int(epoch_length) * 100:               # (n, 3000, C)
        epochs = x[:, :, chan_idx][:, None, :, None]
    elif x.ndim == 3 and x.shape[2] == int(epoch_length) * 100:               # (n, C, 3000)
        epochs = x[:, chan_idx, :][:, None, :, None]
    # else:
    #     raise ValueError(f"Unexpected x shape (expected pre-epoched), got: {x.shape}")

    if epoch_length != 30  and x.shape[1] != int(epoch_length) * 100: # case where the data is not already epoched at the desired length, we need to split the epochs
        # if ind < 2:
        #     print(f"[INFO]: splitting epoch into {epoch_length}s chunks") # print this for confirmation that it's going through
        epochs, y = split_epochs(x, y, epoch_length)

    # close file
    npz_file.close()
    return epochs.astype(np.float32), y

# ======== H5 ========
def extract_h5(ind, files, epoch_length=30, channel='F3_M2'):
    # Function to extract data from h5 files, handle all possible file layouts
    with h5py.File(files[ind], 'r') as f:
        if 'signals' not in f:
            raise KeyError(f"[ERROR]: 'signals' group not found in file {files[ind]}. Available keys: {list(f.keys())}")
        
        signals_group = f['signals']
        if 'eeg' in signals_group and channel in signals_group['eeg']:
            raw = signals_group['eeg'][channel]
        elif channel in signals_group:
            raw = signals_group[channel]
        else:
            raise KeyError(f"[ERROR]: Channel {channel} not found in file {files[ind]}. Available channels: {list(signals_group.keys())}")

        if isinstance(raw, h5py.Group):
            signals = np.asarray(raw['data'], dtype=np.float32)
        else:
            signals = np.asarray(raw, dtype=np.float32)
        x = raw['data'] if isinstance(raw, h5py.Group) else raw
        if 'y' in f:
            y30 = np.asarray(f['y'], dtype=np.int64)
        elif 'hypnogram' in f:
            y30 = np.asarray(f['hypnogram'], dtype=np.int64)

        original_fs = get_file_fs(files[ind], channel=channel, x=x, y=y30, epoch_duration=epoch_length)

        return _epoch_continuous(signals, y30, epoch_length, target_fs=100, original_fs=original_fs)

def _epoch_continuous(sig, y30, epoch_length, target_fs=100, original_fs=250):
    if epoch_length == 30:
        sig = butter_bandpass_filter(sig, 0.5, 40, original_fs)
        if original_fs != target_fs:
            sig = resample(sig, int(len(sig) * target_fs / original_fs))

        spe = epoch_length * 100 # samples per epoch
        n = len(sig) // spe
        x = sig[: n * spe].reshape(n, 1, spe, 1)

        for i in range(n):
            x[i] = (x[i] - np.mean(x[i])) / np.std(x[i])

        y = np.array(y30[:n]).reshape(n, 1)
        return x.astype(np.float32), y
    else:
        sig = butter_bandpass_filter(sig, 0.5, 40, original_fs)
        if original_fs != target_fs:
            sig = resample(sig, int(len(sig) * target_fs / original_fs))
        spe = epoch_length * 100 # samples per epoch
        n_shorter = 30 // epoch_length
        n = len(sig) // spe
        x = sig[: n * spe].reshape(n, 1, spe, 1)

        for i in range(n):
            x[i] = (x[i] - np.mean(x[i])) / np.std(x[i])

        y = np.repeat(y30, n_shorter)[:n].reshape(n, 1)
        return x.astype(np.float32), y
    