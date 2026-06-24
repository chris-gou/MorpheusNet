from scipy.signal import butter, lfilter, resample
import random
import loadmat
import numpy as np
from dreemRead import extract_data
from trainer import Configuration

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


# function to create indecies for each fold
def get_fold_indices(fold_number, total_folds=25, seed=100, cv_type="loo", mat_path=None):
    
    if fold_number < 0 or fold_number >= total_folds:
        raise ValueError("Fold number is out of range")
    rng = random.Random(seed + fold_number)
    all_indices = list(range(1, total_folds + 1))
    if mat_path is not None:
        splits = loadmat(mat_path)
        train = [int(i) - 1 for i in splits["train_sub"][fold][0].flatten()]
        val   = [int(i) - 1 for i in splits["train_check_sub"][fold][0].flatten()]
        test  = [int(i) - 1 for i in splits["test_sub"][fold][0].flatten()]
        return train, test, val

    if cv_type == "kfold":
        subjects_per_fold = len(all_indices) // total_folds
            
        start = (fold_number - 1) * subjects_per_fold
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

    
def load_subject(idx, cfg: Configuration) -> list[np.array]: 
    x, hyp = extract_data(idx, path=cfg.data_path, eeg_chan=cfg.eeg_channel,
                            epoch_length=cfg.epoch_duration)

    if x.shape[2] != cfg.window_length:
        # resample data before splitting
        x = butter_bandpass_filter(x)
        x_r = resample(x, int(len(x)*100/250))
        n_epochs = len(x_r) // cfg.window_length
        x_r = x_r[:n_epochs * cfg.window_length]
        hyp = hyp[:n_epochs]
        epochs = np.reshape(x_r, (n_epochs, 1, cfg.window_length, 1))
        for num in range(len(epochs)):
            epochs[num] = (epochs[num] - np.mean(epochs[num])) / np.std(epochs[num])
        hyp = np.array(hyp[:n_epochs]).reshape(n_epochs, 1)
    else:
        epochs = x
        hyp    = np.array(hyp).reshape(len(hyp), 1)
    return epochs, hyp

    recordings = []
    if dataset == 'Sedf20':
        for night in [1, 2]:
            filename = f"SC4{subject_id:02d}{night}E0.npz"
            filepath = os.path.join(data_path, filename)
            if not os.path.exists(filepath):
                continue
            npz = np.load(filepath, allow_pickle=True)
            x = npz['x']
            hyp = np.array(npz['y'])
            if x.ndim == 3:
                epochs_30 = x[:, np.newaxis, :, :]
            if epoch_duration < 30:
                epochs_30, hyp = split_epochs(epochs_30, hyp, epoch_duration)
            else:
                for i in range(len(epochs_30)):
                    mu, sigma = epochs_30[i].mean(), epochs_30[i].std()
                    if sigma > 0:
                        epochs_30[i] = (epochs_30[i] - mu) / sigma
            recordings.append((epochs_30, hyp))
            # hyp = hyp[:len(epochs_30)]
            # recordings.append((epochs_30, hyp))
    else:
        x, hyp = extract_data(subject_id, path=data_path, epoch_length=epoch_duration)
        hyp = np.array(hyp) 
        if x.ndim == 4:
            # print("Data already epoched, skipping preprocessing")
            epochs = x
            hyp = np.array(hyp)[:len(epochs)]
        else:
            x = butter_bandpass_filter(x, 0.5, 40, 250)
            x_r = resample(x, int(len(x)*100/250))
            n_epochs = len(x_r) // (epoch_duration*100)
            x_r = x_r[:n_epochs * 3000]
            # hyp = np.array(hyp)[:n_epochs]
            hyp = hyp[:n_epochs]
            epochs = np.reshape(x_r, (n_epochs, 1, epoch_duration*100, 1))
            for num in range(len(epochs)):
                epochs[num] = (epochs[num] - np.mean(epochs[num])) / np.std(epochs[num])
        # remove -1 hyp epochs
        mask = hyp != -1
        epochs, hyp = epochs[mask], hyp[mask] 
        recordings.append((epochs, hyp))
    return recordings

def create_set(indices, eeg_channel, data_path, training_params):
    x_set, y_set = [], []
    printed = False
    epoch_length = training_params.get("epoch_length")
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
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        pred.append(dequantized_output)
    return pred
    