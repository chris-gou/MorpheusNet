import tensorflow as tf
from nn_model import *
from dreemRead import *
from scipy.signal import resample
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter
import random
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score
import argparse
from utils import *
from configuration import Configuration
import math
from logger import ResourceLogger
from pyutils.helpers import notify # own function for sending notifications via ntfy.sh
import functools

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
tf.random.set_seed(100)

# ======= HELPERS =======
def count_epochs(indices, cfg):
    total = 0
    for i  in indices:
        epochs, _ = load_subject(i, cfg)
        total += len(epochs)
    return total

def get_subject_id(filename):
    # e.g. 'tr07-0564.npz' -> 'tr07-0564', 'C3-M2_tr08-0111.npz' -> 'tr08-0111'
    base = os.path.basename(filename).replace('.npz', '')
    if '_' in base:
        base = base.split('_')[-1] # case where channel is in front of the file name
    return base

@functools.lru_cache(maxsize=None)
def _get_npz_files(path, channel=None):
    print(f"Scanning for .npz files in {path} (channel={channel})...")
    files = glob.glob(os.path.join(path, '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*', '*.npz'))
    return sorted(files, key=get_subject_id)

def count_epochs_fast(indices, cfg, files=None):
    if files is None:
        files = _get_npz_files(cfg.data_path)
    total = 0
    for i in indices:
        if i < 0 or i >= len(files):
            continue  # Skip invalid indices
        with np.load(files[i], allow_pickle=True) as npz_file:
            total += npz_file['y'].shape[0]
    return total

seq_len = 12    #sequence length used for the seqLearner

cnn_acc = []
seq_acc = []

# ======= MAIN FUNCS =======
def train_fold(cfg: Configuration, fold: int, data_path: str, results_path: str, resume: bool = False, files: list = None) -> dict:
    """
    Train one fold
    
    Args:
        cfg (Configuration): The configuration object containing all necessary parameters.
        fold (int): Fold number to train.
        data_path (str): Path to the dataset.
        results_path (str): Path to save the results.
        resume (bool): Whether to resume training from the last checkpoint.
        files (list): List of .npz files to use for training and evaluation.

    Returns:
        dict: A dictionary containing the training results: Accuracy, MF1, Kohens Kappa, per-class F1 scores, all for CNN and Seq models

    Raises:
        Exception: if something is invalid

    """ 
    try:
        t = cfg.training
        hp = cfg.hparams
        epoch_duration = t.get("epoch_duration",30)
        blocks = t.get("num_blocks", 3) # baseline if not set in config
        width = t.get("width_mult", 1) # baseline if not set in config

        # callbacks + checkpoint creation
        best_model_file_cnn = os.path.join(results_path, f'{cfg.name}_best_cnn_fold{fold}.h5')
        best_model_file_seq = os.path.join(results_path, f'{cfg.name}_best_seq_fold{fold}.h5')
        tflite_path = os.path.join(results_path, f"cnn_full_int_fold{fold}.tflite")
        test_ind_path = os.path.join(results_path, f'test_ind_dodh_fold{fold}.npy')
        fold_done = os.path.isfile(best_model_file_seq) and os.path.isfile(tflite_path) and os.path.isfile(test_ind_path) # is fold complete? cnn + seq -> nothing to train for this fold
        cnn_done = os.path.isfile(best_model_file_cnn) and os.path.isfile(tflite_path) # is only cnn for this fold complete? -> train seq

        if resume and fold_done:
            print(f"Fold {fold} already completed. Skipping this fold.")
            return None 

        # Define callbacks
        checkpoint_callback_cnn = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file_cnn, 
                                                        monitor='val_loss', 
                                                        mode = 'min',
                                                        save_best_only=True,
                                                        save_freq="epoch")
        
        checkpoint_callback_seq = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file_seq, 
                                                        monitor='val_loss', 
                                                        mode = 'min',
                                                        save_best_only=True,
                                                        save_freq="epoch")
        
        # create TensorBoard callback
        tensorboard_callback_cnn = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(results_path, f'tb_cnn_fold{fold}'), histogram_freq=1)
        tensorboard_callback_seq = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(results_path, f'tb_seq_fold{fold}'), histogram_freq=1)

        # resource callbacks
        cnn_logger = ResourceLogger(log_path=os.path.join(results_path, f'cnn_resource_log_fold{fold}.json'))
        seq_logger = ResourceLogger(log_path=os.path.join(results_path, f'seq_resource_log_fold{fold}.json'))

        tb_log_dir = "/tmp/profile_local"  
        profiler_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            profile_batch='10,20'  # profiles steps 10 through 20
        )

        # ====== data splits ======
        train_inds, test_inds, val_inds = get_fold_indices(fold, t.get("folds", 25), seed=100, cv_type=t.get("cv_type", "kfold"), mat_path=t.get("mat_path", None))
        np.save(os.path.join(results_path, f'train_ind_fold{fold}.npy'), train_inds)
        np.save(os.path.join(results_path, f'test_ind_fold{fold}.npy'), test_inds)
        np.save(os.path.join(results_path, f'val_ind_fold{fold}.npy'), val_inds)
        print(f"Fold {fold}: train: {len(train_inds)} | val: {len(val_inds)} | test: {len(test_inds)}")

        if not (resume and cnn_done):
            # ===== CNN =====
            optimizer = tf.keras.optimizers.Adam(learning_rate = hp.get("cnn_lr", 0.001))
            model = separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=blocks, width_mult=width)
            model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
            # handle physio separately due to dataset size, avoid memory issues
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                n_train = count_epochs_fast(train_inds, cfg, files=files)
                n_val = count_epochs_fast(val_inds, cfg, files=files)
                steps = math.ceil(n_train / hp.get("cnn_batch_size", 512))
                val_steps = math.ceil(n_val / hp.get("cnn_batch_size", 512))
                print(f"Training with {n_train} epochs, {steps} steps per epoch | Validation with {n_val} epochs, {val_steps} steps per epoch")
                model.fit(data_generator(train_inds, cfg, files=files), steps_per_epoch=steps,
                            validation_data=data_generator(val_inds, cfg, files=files), validation_steps=val_steps,
                            callbacks=[checkpoint_callback_cnn, tensorboard_callback_cnn, cnn_logger, profiler_callback], epochs=hp.get("cnn_epochs", 10))
            else:
                # create train, val, test sets
                x_train, y_train = create_set(train_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
                x_val, y_val = create_set(val_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
                x_train, y_train = shuffle(x_train, y_train)
                model.fit(x_train, to_categorical(y_train), batch_size=hp.get("cnn_batch_size", 512), epochs=hp.get("cnn_epochs", 10), 
                    validation_data = (x_val, to_categorical(y_val)), callbacks = [checkpoint_callback_cnn, tensorboard_callback_cnn, cnn_logger])
            
            model = tf.keras.models.load_model(best_model_file_cnn)

            # ===== PTQ =====
            saved_model_dir = os.path.join(results_path, f'saved_model_fold_{fold}')
            model.export(saved_model_dir)

            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                rep_samples = []
                for i in train_inds:
                    epochs, _ = load_subject(i, cfg, files=files)
                    rep_samples.extend(epochs)
                rep_samples = np.array(rep_samples)
                x_rep = rep_samples[np.random.randint(0, len(rep_samples), 500)]
            else:
                x_rep = x_train[np.random.randint(0,len(x_train),500)]

            def representative_dataset():
                for data in x_rep:
                    yield [data.astype(np.float32).reshape((1,1,epoch_duration*100,1))]
                
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  
            converter.inference_output_type = tf.int8  

            tflite_path = os.path.join(results_path, f"cnn_full_int_fold{fold}.tflite")
            with open(tflite_path, "wb") as f:
                f.write(converter.convert())

            # cnn predictions 
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                y_pred_cnn, y_test = [], []
                for i in test_inds:
                    epochs, labels = load_subject(i, cfg, files=files)
                    if epochs is None or labels is None:
                        continue  # Skip if data extraction failed
                    preds = np.argmax(model.predict(np.array(epochs), verbose=0), axis=1)
                    y_pred_cnn.extend(preds)
                    y_test.extend(labels)
                y_pred_cnn, y_test = np.array(y_pred_cnn), np.array(y_test)
            else:
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
                y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)
            print(confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4]))
        else:
            print(f"Fold {fold} CNN already completed. Skipping CNN training.")
            model = tf.keras.models.load_model(best_model_file_cnn)
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                y_pred_cnn, y_test = [], []
                for i in test_inds:
                    epochs, labels = load_subject(i, cfg)
                    if epochs is None or labels is None:
                        continue
                    preds = np.argmax(model.predict(np.array(epochs), verbose=0), axis=1)
                    y_pred_cnn.extend(preds)
                    y_test.extend(labels)
                y_pred_cnn, y_test = np.array(y_pred_cnn), np.array(y_test)
            else:
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
                y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        # ===== SEQ =====
        # sets
        x_train_seq, y_train_seq = create_seq_sets(train_inds, interpreter, cfg)
        x_val_seq, y_val_seq = create_seq_sets(val_inds, interpreter, cfg)
        x_test_seq, y_test_seq = create_seq_sets(test_inds, interpreter, cfg)

        x_train_seq = np.reshape(np.array(x_train_seq),(len(x_train_seq),int(seq_len*5)))
        x_val_seq = np.reshape(np.array(x_val_seq),(len(x_val_seq),int(seq_len*5)))
        x_test_seq = np.reshape(np.array(x_test_seq),(len(x_test_seq),int(seq_len*5)))

        optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)
        seq_learner = seq_model(int(seq_len*5))
        seq_learner.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        seq_learner.fit(x_train_seq, to_categorical(y_train_seq), batch_size=hp["seq_batch_size"], epochs=hp["seq_epochs"], 
                validation_data = (x_val_seq, to_categorical(y_val_seq)), callbacks = [checkpoint_callback_seq, tensorboard_callback_seq, seq_logger])
        seq_learner = tf.keras.models.load_model(best_model_file_seq)
            
        y_pred_seq = np.argmax(seq_learner.predict(x_test_seq, verbose=0), axis=1)
        print(confusion_matrix(y_test_seq, y_pred_seq, labels=[0, 1, 2, 3, 4]))  # rows=true, cols=pred
        print(classification_report(y_test_seq, y_pred_seq, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))

        return {
            "cnn_acc":       float(np.mean(y_pred_cnn == y_test.flatten())),
            "cnn_mf1":       float(f1_score(y_test, y_pred_cnn, average="macro")),
            "cnn_kappa":     float(cohen_kappa_score(y_test.flatten(), y_pred_cnn)),
            "cnn_per_class": f1_score(y_test, y_pred_cnn, average=None, labels=[0,1,2,3,4]).tolist(),
            "seq_acc":       float(np.mean(np.array(y_pred_seq) == np.array(y_test_seq))),
            "seq_mf1":       float(f1_score(y_test_seq, y_pred_seq, average="macro")),
            "seq_kappa":     float(cohen_kappa_score(y_test_seq, y_pred_seq)),
            "seq_per_class": f1_score(y_test_seq, y_pred_seq, average=None, labels=[0,1,2,3,4]).tolist(),
        }
    except Exception as e:
        print(f"Error in fold {fold}: {e}")
        notify(f"Training failed for {cfg.name} fold {fold} | Error: {e}")
        raise e

# ====== Aggregate fold results ======
def aggregate_and_save(cfg: Configuration, fold_results: list) -> dict:
    """
    Aggregate results from all folds and save to one JSON file
    
    Args:
        cfg (Configuration): The configuration object containing all necessary parameters.
        fold_results (list): List of dictionaries containing results from each fold.

    Returns:
        dict: A dictionary containing the training results: Accuracy, MF1, Kohens Kappa, per-class F1 scores, all for CNN and Seq models

    Raises:
        Exception: if something is invalid
    """ 
    class_names = ["Wake", "N1", "N2", "N3", "REM"]

    def means(key):  return [r[key] for r in fold_results] # extract values for a specific key across all folds
 
    results = {
        "dataset": cfg.name,
        "cnn": {
            "accuracy_mean":   float(np.mean(means("cnn_acc"))),
            "mf1_mean":        float(np.mean(means("cnn_mf1"))),
            "kappa_mean":      float(np.mean(means("cnn_kappa"))),
            "per_class_f1":    {n: float(np.mean([r["cnn_per_class"][i] for r in fold_results]))
                                for i, n in enumerate(class_names)},
            "accuracy_per_fold": means("cnn_acc"),
            "mf1_per_fold":      means("cnn_mf1"),
        },
        "seq": {
            "accuracy_mean":   float(np.mean(means("seq_acc"))),
            "mf1_mean":        float(np.mean(means("seq_mf1"))),
            "kappa_mean":      float(np.mean(means("seq_kappa"))),
            "per_class_f1":    {n: float(np.mean([r["seq_per_class"][i] for r in fold_results]))
                                for i, n in enumerate(class_names)},
            "accuracy_per_fold": means("seq_acc"),
            "mf1_per_fold":      means("seq_mf1"),
        },
    }
 
    out = os.path.join(cfg.save_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
 
    print(f"\nCNN  MF1: {results['cnn']['mf1_mean']:.4f}")
    print(f"Seq  MF1: {results['seq']['mf1_mean']:.4f}")
    print(f"Results → {out}")
    return results

def small_train(cfg, fold, files):
    """
    Helper function to train on small subset (first 20 subjects) for debugging.

    Args:
        cfg (Configuration): The configuration object containing all necessary parameters.
        fold (int): Fold number to train.
        files (list): List of .npz files to use for training and evaluation.
    Returns:
        None
    """
    t = cfg.training
    train_inds, _, _ = get_fold_indices(fold, t.get("folds", 25), seed=100, cv_type=t.get("cv_type", "kfold"), mat_path=t.get("mat_path", None))

    subset_inds = train_inds[:20]
    x_list, y_list = [], []
    for i in subset_inds:
        epochs, labels = load_subject(i, cfg, files=files)
        if epochs is not None:
            x_list.append(epochs)
            y_list.append(labels)

    # check for NaN or Inf in the epochs
    print(np.isnan(epochs).any(), np.isinf(epochs).any())
    print(epochs.min(), epochs.max(), epochs.mean(), epochs.std())
    print(np.var(epochs, axis=0).mean(), np.var(epochs, axis=0).std())
    x_subset = np.concatenate(x_list, axis=0)
    y_subset = np.concatenate(y_list, axis=0).flatten()
    x_subset, y_subset = shuffle(x_subset, y_subset)

    print(x_subset.shape, y_subset.shape)
    print(np.unique(y_subset, return_counts=True))

    # instantiate and compile the model
    model = separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=3, width_mult=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    print(y_subset.min(), y_subset.max(), y_subset.dtype)
    cat = to_categorical(y_subset, 5)
    print(np.isnan(cat).any(), cat.shape)
    print(cat[:5])  # spot check a few rows manually
    print(cat.sum(axis=1)[:20])
    for i in range(10):
        preds = model.predict(x_subset[i:i+1], verbose=10)
        if np.isnan(preds).any():
            print(f"epoch {i} produces NaN, max_abs input value: {np.abs(x_subset[i]).max():.2f}")

    print("NaN in preds:", np.isnan(preds).any())
    print("Inf in preds:", np.isinf(preds).any())
    print(preds[:3])
    model.fit(x_subset, to_categorical(y_subset, 5), batch_size=128, epochs=10, validation_split=0.1)

    x_subset_clipped = np.clip(x_subset, -10, 10)

    model2 = separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=3, width_mult=1.0)
    model2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    model2.fit(x_subset_clipped, to_categorical(y_subset, 5), batch_size=128, epochs=10, validation_split=0.1)
    

def run(base_config_path, override_config_path=None, debug=False, resume=False):
    """
    Main function to run the training process.
    Args:
        base_config_path (str): Path to the base configuration file.
        override_config_path (str, optional): Path to the override configuration file. Defaults to None.
        debug (bool, optional): Flag to indicate if the run is in debug mode. Defaults to False.
        resume (bool, optional): Flag to indicate if training should resume from the last checkpoint. Defaults to False.
    Raises:
        Exception: if something is invalid
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)])  # limit to 3GB
        # tf.config.experimental.set_memory_growth(gpu, True)

    cfg = Configuration(base_config_path, override_config_path)
    os.makedirs(cfg.save_dir, exist_ok=True)

    results_file_path = os.path.join(cfg.save_dir, "results.json")
    # if os.path.isfile(results_file_path) and not debug:
    #     print(f"Results already exist for {cfg.name} → {results_file_path}")
    #     return

    print(f"\n{'='*80}")
    print(f"Config: {cfg.name}  |  epoch: {cfg.training.get("epoch_duration")}s  |  folds: {cfg.training.get("folds")} | Blocks: {cfg.training.get("num_blocks", 3)} | Width multiplier: {cfg.training.get("width_mult", 1.0)}")
    print(f"{'='*80}")
    
    fold_results = []
    npz_files = _get_npz_files(cfg.data_path, channel=cfg.dataset.get("eeg_channel"))
    if npz_files is None or len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {cfg.data_path} for channel {cfg.dataset.get('eeg_channel')}")
    
    try:
        for fold in range(cfg.training.get("folds", 25)):
            result = train_fold(cfg, fold, cfg.data_path, cfg.save_dir, resume=resume, files=npz_files)
            if result is not None:
                fold_results.append(result)
    except Exception as e:
        print(f"Error during training: {e}")
        notify(f"Training failed for {cfg.name} | Error: {e}")
        raise e
    aggregate_and_save(cfg, fold_results)
    notify(f"Training completed for {cfg.name} | MF1: CNN={np.mean([r['cnn_mf1'] for r in fold_results]):.4f} | Seq={np.mean([r['seq_mf1'] for r in fold_results]):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='/mnt/truenas_db/user/christina', help='Path of dataset')
    parser.add_argument('--output_path', type=str, help='Path to save results')
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--debug', action='store_true', default=False, help='Set mode to debug or not')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from last checkpoint')
    args = parser.parse_args()

    run(args.config, args.override, debug=args.debug, resume=args.resume)