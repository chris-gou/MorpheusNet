import tensorflow as tf
from tqdm import tqdm
from nn_model import *
from dreemRead import *
from scipy.signal import resample
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
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
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
tf.random.set_seed(100)

# ======= HELPERS =======
def count_epochs(indices, cfg):
    total = 0
    for i  in indices:
        epochs, _ = load_subject(i, cfg)
        total += len(epochs)
    return total

def count_epochs_fast(indices, cfg, files=None):
    if files is None:
        files = get_files(cfg.data_path)
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
        if not os.path.isfile(test_ind_path):
            test_ind_path = os.path.join(results_path, f'test_ind_fold{fold}.npy')
        cnn_done = os.path.isfile(best_model_file_cnn) and os.path.isfile(tflite_path) # is only cnn for this fold complete? -> train seq
        fold_result_path = os.path.join(results_path, f'fold_{fold}_results.json')
        cm_file = os.path.join(results_path, f'fold_{fold}_cm.npz')
        fold_done = (os.path.isfile(best_model_file_seq) and os.path.isfile(tflite_path) 
                     and os.path.isfile(test_ind_path) and os.path.isfile(fold_result_path))
        print(f"Fold {fold} status: fold_done={fold_done}, cnn_done={cnn_done}, resume={resume}")

        if resume and fold_done:
            print(f"Fold {fold} already completed. Skipping this fold.")
            with open(fold_result_path, "r") as f:
                result_dict = json.load(f)
                return result_dict
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

            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
                print(f"Fold {fold}: PHYSIO2018_harmonized with physio_resnet=True, creating fp32_submodel and quant_submodel for TFLite conversion.")
                fp32_submodel, quant_submodel = physio_separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=blocks, width_mult=width)
                model = tf.keras.Model(inputs=fp32_submodel.input, outputs=quant_submodel(fp32_submodel.output), name='full_model')
            else:
                model = separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=blocks, width_mult=width)
            model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
            
            # handle physio separately due to dataset size, avoid memory issues
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                n_train = count_epochs_fast(train_inds, cfg, files=files)
                n_val = count_epochs_fast(val_inds, cfg, files=files)
                steps = math.ceil(n_train / hp.get("cnn_batch_size", 128))
                val_steps = math.ceil(n_val / hp.get("cnn_batch_size", 128))
                print(f"Training with {n_train} epochs, {steps} steps per epoch | Validation with {n_val} epochs, {val_steps} steps per epoch")
                model.fit(data_generator(train_inds, cfg, files=files), steps_per_epoch=steps,
                            validation_data=data_generator(val_inds, cfg, files=files), validation_steps=val_steps,
                            callbacks=[checkpoint_callback_cnn, tensorboard_callback_cnn, cnn_logger, profiler_callback], epochs=hp.get("cnn_epochs", 10))
            else:
                # create train, val, test sets
                x_train, y_train = create_set(train_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files)
                unique, counts = np.unique(y_train, return_counts=True)
                x_val, y_val = create_set(val_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files)
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files)
                # x_train, y_train = shuffle(x_train, y_train)
                # class_weight = compute_class_weight(
                #     class_weight='balanced',
                #     classes=np.array([0, 1, 2, 3, 4]),
                #     y=y_train.flatten()
                # )
                # class_weight = dict(enumerate(class_weight))
                model.fit(x_train, to_categorical(y_train, 5), batch_size=hp.get("cnn_batch_size", 128), epochs=hp.get("cnn_epochs", 10), 
                    validation_data = (x_val, to_categorical(y_val, 5)), callbacks = [checkpoint_callback_cnn, tensorboard_callback_cnn, cnn_logger])
            del model
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(best_model_file_cnn)
            # PHYSIO2018 
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
                print("Fold {fold}: PHYSIO2018_harmonized with physio_resnet=True, creating fp32_submodel and quant_submodel for TFLite conversion.")
                fp32_output = model.get_layer('fp32_output').output
                fp32_submodel = tf.keras.Model(inputs=model.input, outputs=fp32_output, name='fp32_submodel_reloaded')
                quant_submodel = model.get_layer('quant_body')

            # ===== PTQ =====
            saved_model_dir = os.path.join(results_path, f'saved_model_fold_{fold}')
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
                saved_model_dir_quant = os.path.join(results_path, f'saved_model_quant_fold_{fold}')
                quant_submodel.export(saved_model_dir_quant)
                convert_source_dir = saved_model_dir_quant
            else:
                model.export(saved_model_dir)
                convert_source_dir = saved_model_dir

            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                max_subjects = False
                if max_subjects:
                    rep_target = 500
                    REP_MAX_SUBJECTS = 20
                    rng = np.random.default_rng(100)
                    sample_inds = rng.choice(train_inds, size=min(REP_MAX_SUBJECTS, len(train_inds)), replace=False)
                else:
                    sample_inds = train_inds
                rep_samples = []
                for i in sample_inds:
                    epochs, _ = load_subject(i, cfg, files=files)
                    rep_samples.extend(epochs)
                    # if len(rep_samples) >= rep_target:
                    #     break
                rep_samples = np.array(rep_samples)
                # x_rep = rep_samples[np.random.randint(0, len(rep_samples), 500)]
                rng = np.random.default_rng(100)
                x_rep = rep_samples[rng.integers(0, len(rep_samples), 500)]
            else:
                x_rep = x_train[np.random.randint(0,len(x_train),500)]

            def representative_dataset():
                for data in x_rep:
                    x = data.astype(np.float32).reshape((1, 1, epoch_duration*100, 1))
                    if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
                        x = fp32_submodel.predict(x, verbose=0)  # transform raw waveform -> stem output (width 750)
                    yield [x.astype(np.float32)]
                
            converter = tf.lite.TFLiteConverter.from_saved_model(convert_source_dir)
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
                for i in tqdm(test_inds, desc=f"Fold {fold} CNN test inference"):
                    epochs, labels = load_subject(i, cfg, files=files)
                    expected_last_dim = cfg.window_length
                    if epochs is not None and epochs.shape[2] != expected_last_dim:
                        raise ValueError(f"Unexpected shape for epochs of subject {i}: expected last dimension {expected_last_dim}, got {epochs.shape[2]}")
                    if epochs is None or labels is None:
                        continue
                    preds_per_subj = np.argmax(model.predict(np.array(epochs), verbose=0), axis=1)
                    # print(f"Fold {fold}: min: {preds.min()}, max: {preds.max()}, unique: {np.unique(preds)}, mean: {preds.mean():.4f}, std: {preds.std():.4f}")
                    y_pred_cnn.extend(preds_per_subj)
                    y_test.extend(labels)
                y_pred_cnn, y_test = np.array(y_pred_cnn), np.array(y_test)
            else:
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files)
                y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)
            print(confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4]))
        else:
            print(f"Fold {fold} CNN already completed. Skipping CNN training.")
            model = tf.keras.models.load_model(best_model_file_cnn)
            # PHYSIO2018 
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
                fp32_output = model.get_layer('fp32_output').output
                fp32_submodel = tf.keras.Model(inputs=model.input, outputs=fp32_output, name='fp32_submodel_reloaded')
                quant_submodel = model.get_layer('quant_body')
                
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
                y_pred_cnn, y_test = [], []
                for i in tqdm(test_inds, desc=f"Fold {fold} CNN test inference"):
                    epochs, labels = load_subject(i, cfg, files=files)
                    expected_last_dim = cfg.window_length
                    if epochs is not None and epochs.shape[2] != expected_last_dim:
                        raise ValueError(f"Unexpected shape for epochs of subject {i}: expected last dimension {expected_last_dim}, got {epochs.shape[2]}")
                    if epochs is None or labels is None:
                        continue
                    preds_per_subj = np.argmax(model.predict(np.array(epochs), verbose=0), axis=1)
                    # print(f"Fold {fold}: min: {preds.min()}, max: {preds.max()}, unique: {np.unique(preds)}, mean: {preds.mean():.4f}, std: {preds.std():.4f}")
                    y_pred_cnn.extend(preds_per_subj)
                    y_test.extend(labels)
                y_pred_cnn, y_test = np.array(y_pred_cnn), np.array(y_test)
            else:
                x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files)
                y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # ===== SEQ =====
        # sets
        print(f"Fold {fold}: creating sequence sets with seq_len={seq_len}")
        if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t["physio_resnet"]=="true":
            x_train_seq, y_train_seq = create_seq_sets(train_inds, interpreter, cfg, fp32_submodel=fp32_submodel, files=files)
            print(f"Fold {fold}: created train sequence set with {len(x_train_seq)} sequences")
            x_val_seq, y_val_seq = create_seq_sets(val_inds, interpreter, cfg, fp32_submodel=fp32_submodel, files=files)
            print(f"Fold {fold}: created val sequence set with {len(x_val_seq)} sequences")
            x_test_seq, y_test_seq = create_seq_sets(test_inds, interpreter, cfg, fp32_submodel=fp32_submodel, files=files)
            print(f"Fold {fold}: created test sequence set with {len(x_test_seq)} sequences")
        else:
            x_train_seq, y_train_seq = create_seq_sets(train_inds, interpreter, cfg, files=files)
            x_val_seq, y_val_seq = create_seq_sets(val_inds, interpreter, cfg, files=files)
            x_test_seq, y_test_seq = create_seq_sets(test_inds, interpreter, cfg, files=files)

        labels, counts = np.unique(y_test_seq, return_counts=True)
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        dist = {class_names[l]: int(c) for l, c in zip(labels, counts)}
        print(f"Fold {fold} test class distribution: {dist}")
        proportions = {k: v/sum(counts) for k, v in dist.items()}
        print(f"Fold {fold} test class proportions: {proportions}")

        x_train_seq = np.reshape(np.array(x_train_seq),(len(x_train_seq),int(seq_len*5)))
        x_val_seq = np.reshape(np.array(x_val_seq),(len(x_val_seq),int(seq_len*5)))
        x_test_seq = np.reshape(np.array(x_test_seq),(len(x_test_seq),int(seq_len*5)))

        # class_weight = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.array([0, 1, 2, 3, 4]),
        #     y=y_train_seq_arr
        # )
        # class_weight = dict(enumerate(class_weight))

        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01) # gradient clipping
        seq_learner = seq_model(int(seq_len*5))
        seq_learner.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        seq_learner.fit(x_train_seq, to_categorical(y_train_seq, 5), batch_size=hp["seq_batch_size"], epochs=hp["seq_epochs"], 
                validation_data = (x_val_seq, to_categorical(y_val_seq, 5)), callbacks = [checkpoint_callback_seq, tensorboard_callback_seq, seq_logger])
        seq_learner = tf.keras.models.load_model(best_model_file_seq)
            
        y_pred_seq = np.argmax(seq_learner.predict(x_test_seq, verbose=0), axis=1)
        print(confusion_matrix(y_test_seq, y_pred_seq, labels=[0, 1, 2, 3, 4]))  # rows=true, cols=pred
        print(classification_report(y_test_seq, y_pred_seq, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], labels=[0, 1, 2, 3, 4], zero_division=0))

        # ===== save per-fold confusion matrices =====
        cm_cnn = confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4])
        cm_seq = confusion_matrix(y_test_seq, y_pred_seq, labels=[0, 1, 2, 3, 4])
        cm_path = os.path.join(results_path, f'fold_{fold}_cm.npz')
        np.savez(cm_path, cnn_cm=cm_cnn, seq_cm=cm_seq)

        fold_result_path = os.path.join(results_path, f'fold_{fold}_results.json')
        result_dict = {
            "cnn_acc":       float(np.mean(y_pred_cnn == y_test.flatten())),
            "cnn_mf1":       float(f1_score(y_test, y_pred_cnn, average="macro")),
            "cnn_kappa":     float(cohen_kappa_score(y_test.flatten(), y_pred_cnn)),
            "cnn_per_class": f1_score(y_test, y_pred_cnn, average=None, labels=[0,1,2,3,4]).tolist(),
            "seq_acc":       float(np.mean(np.array(y_pred_seq) == np.array(y_test_seq))),
            "seq_mf1":       float(f1_score(y_test_seq, y_pred_seq, average="macro")),
            "seq_kappa":     float(cohen_kappa_score(y_test_seq, y_pred_seq)),
            "seq_per_class": f1_score(y_test_seq, y_pred_seq, average=None, labels=[0,1,2,3,4]).tolist(),
        }
        with open(fold_result_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        return result_dict
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
        "dataset path": cfg.dataset["path"],
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

def run(base_config_path, override_config_path=None, debug=False, resume=False, name=None):
    """
    Main function to run the training process.
    Args:
        base_config_path (str): Path to the base configuration file.
        override_config_path (str, optional): Path to the override configuration file. Defaults to None.
        debug (bool, optional): Flag to indicate if the run is in debug mode. Defaults to False.
        resume (bool, optional): Flag to indicate if training should resume from the last checkpoint. Defaults to False.
        name (str, optional): Name of the experiment. Defaults to None.
    Raises:
        Exception: if something is invalid
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024*5)])  # limit to 5GB
        # tf.config.experimental.set_memory_growth(gpu, True)

    cfg = Configuration(base_config_path, override_config_path, name=name)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Config: {cfg.name} | results path: {cfg.save_dir}  |  epoch: {cfg.training.get("epoch_duration")}s  |  folds: {cfg.training.get("folds")} | Blocks: {cfg.training.get("num_blocks", 3)} | Width multiplier: {cfg.training.get("width_mult", 1.0)}")
    print(f"{'='*80}")
    
    fold_results = []
    files = get_files(cfg.data_path, channel=cfg.dataset.get("eeg_channel"))

    try:
        for fold in range(cfg.training.get("folds", 25)):
        # for fold in range(1):
            result = train_fold(cfg, fold, cfg.data_path, cfg.save_dir, resume=resume, files=files)
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
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--debug', action='store_true', default=False, help='Set mode to debug or not')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from last checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    args = parser.parse_args()

    run(args.config, args.override, debug=args.debug, resume=args.resume, name=args.name)
