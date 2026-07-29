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
def get_npz_files(path, channel=None):
    print(f"Scanning for .npz files in {path} (channel={channel})...")
    files = glob.glob(os.path.join(path, '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*', '*.npz'))
    return sorted(files, key=get_subject_id)

# function to create indecies for each fold
def get_fold_indices(fold_number, total_folds=25, seed=100, cv_type="loo", mat_path=None):
    
    if fold_number < 0 or fold_number >= total_folds:
        raise ValueError("Fold number is out of range")
    rng = random.Random(seed + fold_number)

    if mat_path is not None:
        splits = loadmat(mat_path)
        train = [int(i) - 1 for i in splits["train_sub"][fold_number][0].flatten()]
        val   = [int(i) - 1 for i in splits["train_check_sub"][fold_number][0].flatten()]
        test  = [int(i) - 1 for i in splits["test_sub"][fold_number][0].flatten()]
        return train, test, val

    if cv_type == "kfold":
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
    
    if fold_number == 0:
        raise ValueError("Fold number should be between 1 and total_folds (inclusive)")

    # LOO fallback
    # Calculate the test index (1 number)
    all_indices = list(range(1, total_folds + 1))

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
    x, hyp = extract_data(idx, path=cfg.data_path, eeg_chan=cfg.dataset["eeg_channel"],
                            epoch_length=cfg.training["epoch_duration"], files=files)
    if x is None or hyp is None:
        return None, None  # Return None if data extraction failed
    
    if x.shape[2] != cfg.window_length:
        # resample data before splitting
        x = butter_bandpass_filter(x, 0.5, 40, 250)    # bandpassing between 0.5 and 40Hz
        x_r = resample(x, int(len(x)*100/250))
        n_epochs = len(x_r) // cfg.window_length
        x_r = x_r[:n_epochs * cfg.window_length]
        hyp = hyp[:n_epochs]
        epochs = np.reshape(x_r, (n_epochs, 1, cfg.window_length, 1))
        for num in range(len(epochs)):
            epochs[num] = (epochs[num] - np.mean(epochs[num])) / np.std(epochs[num])
        hyp = np.array(hyp[:n_epochs]).reshape(n_epochs, 1)
    else:
        # epochs = x
        # hyp    = np.array(hyp).reshape(len(hyp), 1)
        epochs = np.reshape(x, (len(x), 1, cfg.window_length, 1))
        if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
            mu, sigma = np.mean(epochs), np.std(epochs)
            epochs = (epochs - mu) / (sigma + 1e-8)
        else:
            for num in range(len(epochs)):
                epochs[num] = (epochs[num] - np.mean(epochs[num])) / (np.std(epochs[num]) + 1e-8)
        hyp = np.array(hyp).reshape(len(hyp), 1)
    return epochs, hyp

def create_set(indices, eeg_channel, data_path, training_params):
    x_set, y_set = [], []
    printed = False
    epoch_length = training_params.get("epoch_duration")
    for i in indices:
        x,hyp = extract_data(i, path=data_path, eeg_chan=eeg_channel, epoch_length=epoch_length)  
        if not printed:
            print("raw shape", x.shape, "| expected last dim", epoch_length * 100,
                "->", "RESAMPLE branch" if x.shape[2] != epoch_length * 100 else "passthrough")
            printed = True
        if  x.shape[2] != epoch_length * 100:
            x = butter_bandpass_filter(x,0.5,40,250)    #bandpassing between 0.5 and 40Hz
            x_r = resample(x, int(len(x)*100/250))    #ownsampling to 100 Hz
            inds = np.arange(0,len(x_r),epoch_length*100)
            epochs = np.reshape(x_r,(len(inds),1,epoch_length*100,1))
            for num,i in enumerate(epochs):
                epochs[num]=(i-np.mean(i))/np.std(i)
            
            x_set.append(epochs)
            y_set.append(np.array(hyp).reshape((len(hyp),1)))
        else:
            x_set.append(x)
            y_set.append(np.array(hyp).reshape((len(hyp),1)))
    x_set = np.vstack(x_set)
    y_set = np.vstack(y_set)

    return x_set, y_set

def create_seq_sets(indices, interpreter, cfg):
    x_seq, y_seq = [], []
    for idx in indices:
        epochs, hyp = load_subject(idx, cfg)
        if epochs is None or hyp is None:
            continue  # Skip if data extraction failed
        hyp = hyp.flatten()
        preds = run_tflite(interpreter, epochs, cfg)
        for i in range(SEQ_LEN, len(epochs)):
            x_seq.append(preds[i-SEQ_LEN:i])
            y_seq.append(hyp[i])
    return x_seq, y_seq

def run_tflite(interpreter, epochs, cfg):
    pred = []
    for i in epochs:
        input_data = i.reshape((1,1,cfg.window_length,1))
        # Get the input details (assuming a single input tensor)
        input_details = interpreter.get_input_details()
        input_index = input_details[0]["index"]
        input_scale, input_zero_point = input_details[0]['quantization']
        quantized_input = (input_data / input_scale) + input_zero_point
        quantized_input = quantized_input.astype(np.int8)
        
        # Set the input tensor
        interpreter.set_tensor(input_index, quantized_input)
        
        interpreter.invoke()
        
        # Get the output details (assuming a single output tensor)
        output_details = interpreter.get_output_details()
        output_index = output_details[0]["index"]
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_index)
        output_details = interpreter.get_output_details()
        output_scale, output_zero_point = output_details[0]['quantization']

        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        pred.append(dequantized_output)
    return pred
    
def data_generator(indices, cfg, shuffle_buffer=5000, files=None):
    batch_size = cfg.training.get("cnn_batch_size", 512)
    processed_subjects = 0
    while True:
        x_batch, y_batch = [], []
        for i in indices:
            epochs, hyp = load_subject(i, cfg, files=files)
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