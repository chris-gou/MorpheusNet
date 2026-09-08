from scipy.signal import butter, lfilter, resample
import random
from scipy.io import loadmat
import numpy as np
from dreemRead import extract_data
from trainer import Configuration
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import glob 
import os 
import functools
import time
SEQ_LEN=12

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

def get_subject_id(filename):
    # e.g. 'tr07-0564.npz' -> 'tr07-0564', 'C3-M2_tr08-0111.npz' -> 'tr08-0111'
    base = os.path.basename(filename).replace('.npz', '')
    if '_' in base:
        base = base.split('_')[-1] # case where channel is in front of the file name
    return base

@functools.lru_cache(maxsize=None)
def get_files(path, channel=None):
    print(f"Scanning for sleep files in {path} (channel={channel})...")
    files = glob.glob(os.path.join(path, '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*', '*.npz'))
    
    if files is not None:
        print(f"Found {len(files)} npz files in {path}, proceeding with npz")
        return sorted(files, key=get_subject_id)
    
    if files is None: # npz not found, check h5 (currently the only other file option)
        files = (glob.glob(os.path.join(path, '*.h5'))     +
            glob.glob(os.path.join(path, '*.hdf5'))   +
            glob.glob(os.path.join(path, '*', '*.h5')) +
            glob.glob(os.path.join(path, '*', '*.hdf5')))
    if files is not None:
        print(f"Found {len(files)} h5/hdf5 files in {path}.")
        return sorted(files, key=get_subject_id)
    return None

# function to create indices for each fold
def get_fold_indices(fold_number, total_folds=25, seed=100, cv_type="loo", mat_path=None):
    if fold_number < 0 or fold_number >= total_folds:
        raise ValueError("Fold number is out of range")
    rng = random.Random(seed + fold_number)

    if mat_path is not None: # sedf datasets
        splits = loadmat(mat_path)
        if "train_sub" not in splits or "eval_sub" not in splits or "test_sub" not in splits:
            raise ValueError(f"Expected keys 'train_sub', 'eval_sub', 'test_sub' in {mat_path}, but got {list(splits.keys())}")
        train = [int(i) - 1 for i in splits["train_sub"][fold_number][0].flatten()]
        val   = [int(i) - 1 for i in splits["eval_sub"][fold_number][0].flatten()]
        test  = [int(i) - 1 for i in splits["test_sub"][fold_number][0].flatten()]
        return train, test, val

    if cv_type == "kfold": # physio
        num_subjects = 993
        all_indices = list(range(num_subjects))
        subjects_per_fold = len(all_indices) // total_folds
            
        start = fold_number * subjects_per_fold
        end = start + subjects_per_fold
        test_indices = all_indices[start:end]
        
        remaining = [i for i in all_indices if i not in test_indices]
        val_size = max(1, len(remaining) // 5)  # ~20% of remaining for val
        val_indices = rng.sample(remaining, val_size)
        train_indices = [i for i in remaining if i not in val_indices]
            
        return train_indices, test_indices, val_indices
    
    # LOO fallback
    # Calculate the test index (1 number)
    test_index = fold_number

    # Calculate random validation indices (7 distinct integers)
    all_indices = list(range(total_folds))
    all_indices.remove(test_index)  # Remove test index

    validation_indices = random.sample(all_indices, min(7, len(all_indices)))  # Sample 7 or fewer if not enough indices

    # Calculate the training indices (17 indices)
    training_indices = [
        i for i in range(total_folds) if i not in [test_index] + validation_indices
    ]

    return training_indices, [test_index], validation_indices
    
def load_subject(idx, cfg: Configuration, files: list = None) -> list[np.array]: 
    x, hyp = extract_data(idx, files, path=cfg.data_path, eeg_chan=cfg.dataset["eeg_channel"],
                            epoch_length=cfg.training["epoch_duration"])
    if x is None or hyp is None:
        return None, None  # Return None if data extraction failed

    if len(x.shape) > 3:
        if x.shape[2] != cfg.window_length:
            print(f"[WARNING] Subject {idx}: unexpected epoch shape {x.shape}, expected last dim {cfg.window_length}. Skipping this subject.")
            # resample data before splitting
            x = butter_bandpass_filter(x, 0.5, 40, 250)    # bandpassing between 0.5 and 40Hz
            x_r = resample(x, int(len(x)*100/250))
            n_epochs = len(x_r) // cfg.window_length
            x_r = x_r[:n_epochs * cfg.window_length]
            hyp = hyp[:n_epochs]
            epochs = np.reshape(x_r, (n_epochs, 1, cfg.window_length, 1))
            mu = epochs.mean(axis=(1, 2, 3), keepdims=True)
            sigma = epochs.std(axis=(1, 2, 3), keepdims=True)
            epochs = (epochs - mu) / (sigma + 1e-8)
            # for num in range(len(epochs)):
            #     epochs[num] = (epochs[num] - np.mean(epochs[num])) / np.std(epochs[num])
            hyp = np.array(hyp[:n_epochs]).reshape(n_epochs, 1)
        else:
            epochs = np.reshape(x, (len(x), 1, cfg.window_length, 1))
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                mu, sigma = np.mean(epochs), np.std(epochs)
                epochs = (epochs - mu) / (sigma + 1e-8)
            else:
                for num, y in enumerate(epochs):
                    epochs[num] = (y- np.mean(y)) / (np.std(y) + 1e-8)
            hyp = np.array(hyp).reshape(len(hyp), 1)
    else:
        # print(f"[INFO] Subject {idx}: epoch shape {x.shape} matches expected last dim {cfg.window_length}. No resampling needed.")
        # epochs = x
        # hyp    = np.array(hyp).reshape(len(hyp), 1)
        x = butter_bandpass_filter(x,0.5,40,250)    #bandpassing between 0.5 and 40Hz
        x_r = resample(x, int(len(x)*100/250))    #ownsampling to 100 Hz
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        # epochs = np.reshape(x, (len(x), (len(inds)), cfg.window_length, 1))
        if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
            mu, sigma = np.mean(epochs), np.std(epochs)
            epochs = (epochs - mu) / (sigma + 1e-8)
        else:
            for num, y in enumerate(epochs):
                epochs[num] = (y- np.mean(y)) / (np.std(y) + 1e-8)
        hyp = np.array(hyp).reshape(len(hyp), 1)
    return epochs, hyp

# ====== set creation =======

def create_set(indices, eeg_channel, data_path, training_params, cfg, files=None):
    x_set, y_set = [], []
    printed = False
    epoch_length = training_params.get("epoch_duration")
    for i in indices:
        x, hyp = load_subject(i, cfg, files=files)  # Use the load_subject function to get preprocessed data
        # data is extracted with butter already
        expected_last_dim = epoch_length * 100
        if not printed:
            print("raw shape", x.shape, "| expected last dim", epoch_length * 100,
                "->", "RESAMPLE branch" if x.shape[2] != epoch_length * 100 else "passthrough")
            printed = True
        if  x.shape[2] != expected_last_dim:
            raise ValueError(
                f"Subject {i}: unexpected epoch shape {x.shape}, "
                f"expected last dim {expected_last_dim}. "
                f"extract_data should already return correctly epoched, filtered, 100Hz data — "
                f"a resample branch here would risk double-filtering."
            )
        else:
            x_set.append(x)
            y_set.append(np.array(hyp).reshape((len(hyp),1)))
    x_set = np.vstack(x_set)
    y_set = np.vstack(y_set)

    return x_set, y_set

def create_seq_sets(indices, interpreter, cfg, fp32_submodel=None, files=None):
    x_seq, y_seq = [], []
    for idx in indices:
        epochs, hyp = load_subject(idx, cfg, files=files)
        if epochs is None or hyp is None:
            continue  # Skip if data extraction failed
        hyp = hyp.flatten()
        preds = run_tflite(interpreter, epochs, cfg, fp32_submodel=fp32_submodel)
        for i in range(SEQ_LEN, len(preds)): 
            x_seq.append(preds[i-SEQ_LEN:i])
            y_seq.append(hyp[i-1])
    return x_seq, y_seq

def run_tflite(interpreter, epochs, cfg, fp32_submodel=None):
    pred = []
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    input_scale, input_zero_point = input_details[0]['quantization']
    output_details = interpreter.get_output_details()
    output_index = output_details[0]["index"]
    output_scale, output_zero_point = output_details[0]['quantization']
    epochs = np.asarray(epochs).reshape((-1, 1, cfg.window_length, 1))
    if fp32_submodel is not None:
        input_all = fp32_submodel.predict(epochs, batch_size=128, verbose=0)
    else:
        input_all = epochs
        # input_data = i.reshape((1,1,cfg.window_length,1))
        # Get the input details (assuming a single input tensor)
        

    for input_data in input_all:
        input_data = input_data[np.newaxis, ...]
        quantized_input = (input_data / input_scale) + input_zero_point
        quantized_input = quantized_input.astype(np.int8)

        interpreter.set_tensor(input_index, quantized_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        pred.append(dequantized_output)

    return pred
    
def data_generator(indices, cfg, shuffle_buffer=5000, files=None):
    batch_size = cfg.training.get("cnn_batch_size", 512)
    processed_subjects = 0
    while True:
        x_batch, y_batch = [], []
        for i in indices:
            t0 = time.perf_counter()
            epochs, hyp = load_subject(i, cfg, files=files)
            t1 = time.perf_counter()
            processed_subjects += 1
            if epochs is None or hyp is None:
                continue  # Skip if data extraction failed
            for ep, label in zip(epochs, hyp):
                x_batch.append(ep)
                y_batch.append(label)
                if len(x_batch) >= shuffle_buffer:
                    x_batch, y_batch = shuffle(x_batch, y_batch)
                    while len(x_batch) >= batch_size:
                        yield np.array(x_batch[:batch_size]), to_categorical(np.array(y_batch[:batch_size]), 5)
                        x_batch, y_batch = x_batch[batch_size:], y_batch[batch_size:]
        # if x_batch:
        #     x_batch, y_batch = shuffle(x_batch, y_batch)
        #     yield np.array(x_batch), to_categorical(np.array(y_batch), 5)

# def seq_generator(indices, interpreter, cfg, batch_size):
#     """Yields (x_batch, y_batch) windows lazily, one subject at a time.
#     Never holds more than one subject's preds + one batch in memory."""
#     x_buf, y_buf = [], []
#     while True: 
#         for idx in indices:
#             epochs, hyp = load_subject(idx, cfg)
#             if epochs is None or hyp is None:
#                 continue
#             hyp = hyp.flatten()
#             preds = np.array(run_tflite(interpreter, epochs, cfg))  # one subject's worth only

#             for i in range(SEQ_LEN, len(preds)):
#                 window = preds[i-SEQ_LEN:i]
#                 window = np.array(window).reshape(-1)
#                 x_buf.append(window)
#                 y_buf.append(hyp[i])
#                 x_batch = np.array(x_buf).reshape(len(x_buf), 60, 1)
#                 if len(x_batch) == batch_size:
#                     yield x_batch, to_categorical(y_buf, num_classes=5)
#                     x_buf, y_buf = [], []


def count_seq_windows(indices, cfg):
    # sum len epochs - SEQ_LEN for each subject in indices
    total = 0
    for idx in indices:
        epochs, _ = load_subject(idx, cfg)
        if epochs is not None:
            total += len(epochs) - SEQ_LEN
    return total