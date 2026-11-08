import os
import math
import argparse
import json

from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score

from utils import *
from nn_model import separable_resnet, physio_separable_resnet, seq_model
from configuration import Configuration
from logger import ResourceLogger

from pyutils.helpers import notify # own function for sending notifications via ntfy.sh

seq_len = 12    #sequence length used for the seqLearner
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
tf.random.set_seed(100)

# ======= HELPERS =======
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

def find_existing_test_indices_path(paths: dict[str, Path]) -> Path:
    canonical = paths["test_indices"]

    if canonical.exists():
        return canonical

    legacy = canonical.with_name(
        canonical.name.replace("test_ind_", "test_ind_dodh_")
    )

    if legacy.exists():
        return legacy

    return canonical

# fold and path helpers
def prepare_fold_paths(cfg: Configuration, fold:int, results_path: str | Path) -> dict[str, Path]:
    results_path = Path(results_path)
    test_indices = results_path / f"test_ind_fold{fold}.npy"
    if not test_indices.exists():
        test_indices = results_path / f"test_ind_dodh_fold{fold}.npy"  # legacy path
    
    return {
        "best_model_file_cnn": (results_path / f'{cfg.name}_best_cnn_fold{fold}.h5'),
        "best_model_file_seq": (results_path / f'{cfg.name}_best_seq_fold{fold}.h5'),
        "tflite_path": (results_path / f"cnn_full_int_fold{fold}.tflite"),
        "fold_result_path": (results_path / f'fold_{fold}_results.json'),
        "cm_file": (results_path / f'fold_{fold}_cm.npz'),
        "train_indices": (results_path / f"train_ind_fold{fold}.npy"),
        "val_indices": (
            results_path / f"val_ind_fold{fold}.npy"
        ),
        "test_indices": (test_indices),
        "cnn_tensorboard": (
            results_path / f"tb_cnn_fold{fold}"
        ),
        "seq_tensorboard": (
            results_path / f"tb_seq_fold{fold}"
        ),
        "cnn_resource_log": (
            results_path / f"cnn_resource_log_fold{fold}.json"
        ),
        "seq_resource_log": (
            results_path / f"seq_resource_log_fold{fold}.json"
        ),
        "saved_model": (
            results_path / f"saved_model_fold_{fold}"
        ),
        "quantized_saved_model": (
            results_path / f"saved_model_quant_fold_{fold}"
        ),
    }
    
def prepare_fold_data(cfg: Configuration, fold: int, files: list[str]) -> tuple[list[int], list[int], list[int]]:
    training = cfg.training
    train_inds, val_inds, test_inds = get_fold_indices(fold, 
                                    training.get("folds", 25), seed=100, 
                                    cv_type=training.get("cv_type", "kfold"), 
                                    mat_path=training.get("mat_path", None), 
                                    data_path=cfg.data_path)

    if cfg.training.get("mat_path"):  # SEDF: recordings share subject IDs
        sid = lambda i: os.path.basename(files[i])[3:5]
        subj = [{sid(i) for i in s} for s in (train_inds, val_inds, test_inds)]
        assert not (subj[0] & subj[2]) and not (subj[1] & subj[2]), f"fold {fold}: subject leakage"
    return train_inds, val_inds, test_inds

# model helpers
def build_cnn(cfg: Configuration, fold: int):
    t = cfg.training
    hp = cfg.hparams
    blocks = t.get("num_blocks", 3) # baseline if not set in config
    width = t.get("width_mult", 1) # baseline if not set in config
    optimizer = tf.keras.optimizers.Adam(learning_rate = hp.get("cnn_lr", 0.001))

    if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
        print(f"Fold {fold}: PHYSIO2018_harmonized with physio_resnet=True, creating fp32_submodel and quant_submodel for TFLite conversion.")
        fp32_submodel, quant_submodel = physio_separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=blocks, width_mult=width, sigmoid=t.get("sigmoid", True))
        model = tf.keras.Model(inputs=fp32_submodel.input, outputs=quant_submodel(fp32_submodel.output), name='full_model')
    else:
        model = separable_resnet((1, cfg.window_length, 1), 5, bias=False, blocks=blocks, width_mult=width, sigmoid=t.get("sigmoid", True))
    return model, optimizer

def compile_cnn(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer):
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model 

# train
def train_cnn(model: tf.keras.Model,
            cfg: Configuration,
            train_inds: list[int],
            val_inds: list[int],
            files: list[str] | None,
            checkpoint_callback,
            tensorboard_callback,
            profiler_callback=None,
            cnn_logger=None):
    hp = cfg.hparams
    epochs = hp.get("cnn_epochs", 10)
    
    training_callbacks = [checkpoint_callback, tensorboard_callback]
    if profiler_callback is not None:
        training_callbacks.append(profiler_callback)
    if cnn_logger is not None:
        training_callbacks.append(cnn_logger)
    if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
        n_train = count_epochs_fast(train_inds, cfg, files=files)
        n_val = count_epochs_fast(val_inds, cfg, files=files)
        steps = math.ceil(n_train / hp.get("cnn_batch_size", 128))
        val_steps = math.ceil(n_val / hp.get("cnn_batch_size", 128))
        print(f"Training with {n_train} epochs, {steps} steps per epoch | Validation with {n_val} epochs, {val_steps} steps per epoch")
        return model.fit(data_generator(train_inds, cfg, files=files), steps_per_epoch=steps,
                    validation_data=data_generator(val_inds, cfg, files=files), validation_steps=val_steps,
                    callbacks=training_callbacks, epochs=epochs)
    else:
        # create train, val, test sets
        x_train, y_train = create_set(indices=train_inds, cfg=cfg, files=files)
        x_val, y_val = create_set(indices=val_inds, cfg=cfg, files=files)
        return model.fit(x_train, to_categorical(y_train, 5), batch_size=hp.get("cnn_batch_size", 128), epochs=hp.get("cnn_epochs", 10), 
            validation_data = (x_val, to_categorical(y_val, 5)), callbacks = training_callbacks)

def prepare_representative_samples(cfg: Configuration, train_inds: list[int], files: list[str] | None):
    if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
        rng = np.random.default_rng(100)
        rep_samples = []
        seen = 0
        for i in train_inds:
            epochs, _ = load_subject(i, cfg, files=files)
            for e in epochs:
                seen += 1
                if len(rep_samples) < 500:
                    rep_samples.append(e)
                else:
                    j = rng.integers(0, seen)
                    if j < 500:
                        rep_samples[j] = e
        x_rep = np.array(rep_samples)
    else:
        x_train, _ = create_set(indices=train_inds, cfg=cfg, files=files)
        rng = np.random.default_rng(100)
        x_rep = x_train[rng.integers(0, len(x_train), 500)]
    return x_rep

def convert_cnn_to_tflite(model: tf.keras.Model, cfg: Configuration, train_inds: list[int], files: list[str] | None, tflite_path: str | Path, convert_source_dir: str | Path):
    t = cfg.training
    epoch_duration = t.get("epoch_duration",30)
    convert_source_dir = str(convert_source_dir)

    # PHYSIO2018 
    if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
        print("PHYSIO2018_harmonized with physio_resnet=True, creating fp32_submodel and quant_submodel for TFLite conversion.")
        fp32_output = model.get_layer('fp32_output').output
        fp32_submodel = tf.keras.Model(inputs=model.input, outputs=fp32_output, name='fp32_submodel_reloaded')
    
    def representative_dataset():
        for data in x_rep:
            x = data.astype(np.float32).reshape((1, 1, epoch_duration*100, 1))
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
                x = fp32_submodel.predict(x, verbose=0)  # transform raw waveform -> stem output (width 750)
            yield [x.astype(np.float32)]

    x_rep = prepare_representative_samples(cfg, train_inds, files=files)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(convert_source_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  
    converter.inference_output_type = tf.int8  
    
    return converter

def predict_cnn(cfg, model, files, test_inds):
    if cfg.dataset["name"] == 'PHYSIO2018_harmonized':
        y_pred_cnn, y_test = [], []
        for i in tqdm(test_inds, desc=f"CNN test inference"):
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
        x_test, y_test = create_set(indices=test_inds, cfg=cfg, files=files)
        y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)       
    return y_pred_cnn, y_test

def train_and_eval_seq(cfg, fold, paths, train_inds, val_inds, test_inds, interpreter, fp32_submodel, files, checkpoint_callback_seq, tensorboard_callback_seq, seq_logger):
    hp = cfg.hparams
    t = cfg.training
    kwargs = dict(fp32_submodel=fp32_submodel, files=files) if fp32_submodel is not None else dict(files=files)
    x_train_seq, y_train_seq = create_seq_sets(train_inds, interpreter, cfg, **kwargs)
    x_val_seq, y_val_seq = create_seq_sets(val_inds, interpreter, cfg, **kwargs)
    # x_test_seq, y_test_seq = create_seq_sets(test_inds, interpreter, cfg, **kwargs)
    x_test_seq, y_test_seq, test_lengths = [], [], []
    for i in test_inds:  # per recording, so predictions can be split again later
        x, y = create_seq_sets([i], interpreter, cfg, **kwargs)
        x_test_seq += x
        y_test_seq += y
        test_lengths.append(len(y))


    x_train_seq = np.reshape(np.array(x_train_seq), (len(x_train_seq), int(seq_len*5)))
    x_val_seq = np.reshape(np.array(x_val_seq), (len(x_val_seq), int(seq_len*5)))
    x_test_seq = np.reshape(np.array(x_test_seq), (len(x_test_seq), int(seq_len*5)))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01) # gradient clipping
    seq_learner = seq_model(int(seq_len*5), lstm=t.get("sigmoid", False))
    seq_learner.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    seq_learner.fit(x_train_seq, to_categorical(y_train_seq, 5), batch_size=hp["seq_batch_size"], epochs=hp["seq_epochs"], 
            validation_data = (x_val_seq, to_categorical(y_val_seq, 5)), callbacks = [checkpoint_callback_seq, tensorboard_callback_seq, seq_logger])
    seq_learner = tf.keras.models.load_model(paths["best_model_file_seq"])
        
    y_pred_seq = np.argmax(seq_learner.predict(x_test_seq, verbose=0), axis=1)
    return seq_learner,y_pred_seq, y_test_seq, test_lengths
    

# ======= MAIN FUNCS =======
def train_fold(cfg: Configuration, fold: int, results_path: str, resume: bool = False, files: list = None) -> dict:
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
        paths = prepare_fold_paths(cfg, fold, results_path)
        test_indices_path = paths["test_indices"]

        fold_done = (os.path.isfile(paths["best_model_file_seq"]) and os.path.isfile(paths["tflite_path"]) 
                     and os.path.isfile(test_indices_path) and os.path.isfile(paths["fold_result_path"]))
        cnn_done = (
            paths["best_model_file_cnn"].is_file()
            and paths["tflite_path"].is_file()
            and test_indices_path.is_file()
        )
        print(f"Files exist: CNN model: {paths['best_model_file_cnn'].is_file()} | TFLite: {paths['tflite_path'].is_file()} | Test indices: {test_indices_path.is_file()} | Fold result: {paths['fold_result_path'].is_file()}")
        print(f"Fold {fold}: CNN done: {cnn_done} | Fold done: {fold_done} | Resume: {resume}")

        if resume and fold_done:
            print(f"Fold {fold} already completed. Skipping this fold.")
            with open(paths["fold_result_path"], "r") as f:
                result_dict = json.load(f)
                return result_dict

        # Define callbacks
        checkpoint_callback_cnn = tf.keras.callbacks.ModelCheckpoint(filepath=paths["best_model_file_cnn"], 
                                                        monitor='val_loss', 
                                                        mode = 'min',
                                                        save_best_only=True,
                                                        save_freq="epoch")
        
        checkpoint_callback_seq = tf.keras.callbacks.ModelCheckpoint(filepath=paths["best_model_file_seq"], 
                                                        monitor='val_loss', 
                                                        mode = 'min',
                                                        save_best_only=True,
                                                        save_freq="epoch")
        
        # create TensorBoard callback
        tensorboard_callback_cnn = tf.keras.callbacks.TensorBoard(log_dir=paths["cnn_tensorboard"], histogram_freq=1)
        tensorboard_callback_seq = tf.keras.callbacks.TensorBoard(log_dir=paths["seq_tensorboard"], histogram_freq=1)

        # resource callbacks
        cnn_logger = ResourceLogger(log_path=paths["cnn_resource_log"])
        seq_logger = ResourceLogger(log_path=paths["seq_resource_log"])

        tb_log_dir = "/tmp/profile_local"  
        profiler_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            profile_batch='10,20'  # profiles steps 10 through 20
        )

        # ====== data splits ======
        train_inds, val_inds, test_inds = prepare_fold_data(cfg, fold, files)
        np.save(paths["train_indices"], train_inds)
        np.save(paths["val_indices"], val_inds)
        np.save(paths["test_indices"], test_inds)
        print(f"Fold {fold}: train: {len(train_inds)} | val: {len(val_inds)} | test: {len(test_inds)}")

        if not (resume and cnn_done):
            # ===== CNN =====

            model, optimizer = build_cnn(cfg, fold)
            model = compile_cnn(model, optimizer)

            # handle physio separately due to dataset size, avoid memory issues
            cnn_history = train_cnn(model=model, cfg=cfg, 
                                    train_inds=train_inds, val_inds=val_inds, 
                                    files=files, 
                                    checkpoint_callback=checkpoint_callback_cnn, tensorboard_callback=tensorboard_callback_cnn, profiler_callback=profiler_callback, 
                                    cnn_logger=cnn_logger)
            del model

            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(paths["best_model_file_cnn"])
            
            # ===== PTQ =====
            saved_model_dir = paths["saved_model"]
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
                saved_model_dir_quant = paths["quantized_saved_model"]
                # model.export(saved_model_dir_quant)
                quant_submodel = model.layers[-1]
                print("quant submodel:", quant_submodel.name, quant_submodel.input_shape)
                quant_submodel.export(saved_model_dir_quant)
                convert_source_dir = saved_model_dir_quant
            else:
                model.export(saved_model_dir)
                convert_source_dir = saved_model_dir

            representative_dataset = prepare_representative_samples(cfg, train_inds, files=files)

            tflite_path = paths["tflite_path"]
            converter = convert_cnn_to_tflite(model=model, cfg=cfg, train_inds=train_inds, files=files, tflite_path=tflite_path, convert_source_dir=convert_source_dir)
            print("model input:", model.input_shape)

            with open(tflite_path, "wb") as f:
                f.write(converter.convert())
            # cnn predictions 
            y_pred_cnn, y_test = predict_cnn(cfg=cfg, model=model, files=files, test_inds=test_inds)
            print(confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4]))
        else: # cnn done
            print(f"Fold {fold} CNN already completed. Skipping CNN training.")
            x_rep = prepare_representative_samples(cfg, train_inds, files=files)
            model = tf.keras.models.load_model(paths["best_model_file_cnn"])
            # PHYSIO2018 
            if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
                fp32_output = model.get_layer('fp32_output').output
                
            y_pred_cnn, y_test = predict_cnn(cfg=cfg, model=model, files=files, test_inds=test_inds)
        print(f"Fold {fold}: CNN test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        print(f"Fold {fold}: CNN test accuracy: {np.mean(y_pred_cnn == y_test.flatten()):.4f} | MF1: {f1_score(y_test, y_pred_cnn, average='macro'):.4f} | Kappa: {cohen_kappa_score(y_test.flatten(), y_pred_cnn):.4f}")

        interpreter = tf.lite.Interpreter(model_path=str(paths["tflite_path"]))
        interpreter.allocate_tensors()

        # ===== SEQ =====
        # sets
        if cfg.dataset["name"] == 'PHYSIO2018_harmonized' and t.get("physio_resnet")=="true":
            fp32_output = model.get_layer('fp32_output').output
            fp32_submodel = tf.keras.Model(inputs=model.input, outputs=fp32_output, name='fp32_submodel_reloaded')
            
        print(f"Fold {fold}: creating sequence sets with seq_len={seq_len}")
        seq_learner, y_pred_seq, y_test_seq, test_lengths = train_and_eval_seq(cfg, fold, paths, train_inds, val_inds, test_inds, interpreter, fp32_submodel if 'fp32_submodel' in locals() else None, files, checkpoint_callback_seq, tensorboard_callback_seq, seq_logger)
        labels, counts = np.unique(y_test_seq, return_counts=True)
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        dist = {class_names[l]: int(c) for l, c in zip(labels, counts)}
        print(f"Fold {fold} test class distribution: {dist}")
        proportions = {k: v/sum(counts) for k, v in dist.items()}
        print(f"Fold {fold} test class proportions: {proportions}")
            
        print(confusion_matrix(y_test_seq, y_pred_seq, labels=[0, 1, 2, 3, 4]))  # rows=true, cols=pred
        print(classification_report(y_test_seq, y_pred_seq, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], labels=[0, 1, 2, 3, 4], zero_division=0))

        # ===== save per-fold confusion matrices =====
        cm_cnn = confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4])
        cm_seq = confusion_matrix(y_test_seq, y_pred_seq, labels=[0, 1, 2, 3, 4])
        cm_path = paths["cm_file"]
        np.savez(cm_path, cnn_cm=cm_cnn, seq_cm=cm_seq, y_true=np.array(y_test_seq),y_pred=np.array(y_pred_seq),
            lengths=np.array(test_lengths), test_inds=np.array(test_inds),test_files=np.array([os.path.basename(files[i]) for i in test_inds]))

        fold_result_path = paths["fold_result_path"]
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
        del seq_learner, interpreter
        tf.keras.backend.clear_session()
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

def run(base_config_path, override_config_path=None, resume=False, name=None):
    """
    Main function to run the training process.
    Args:
        base_config_path (str): Path to the base configuration file.
        override_config_path (str, optional): Path to the override configuration file. Defaults to None.
        resume (bool, optional): Flag to indicate if training should resume from the last checkpoint. Defaults to False.
        name (str, optional): Name of the experiment. Defaults to None.
    Raises:
        Exception: if something is invalid
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024*6)])  # limit to 4GB
        # tf.config.experimental.set_memory_growth(gpu, True)

    cfg = Configuration(base_config_path, override_config_path, name=name)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Config: {cfg.name} | results path: {cfg.save_dir}  |  epoch: {cfg.training.get('epoch_duration', 10)}s  |  folds: {cfg.training.get('folds')} | Blocks: {cfg.training.get('num_blocks', 3)} | Width multiplier: {cfg.training.get('width_mult', 1.0)}")
    print(f"{'='*80}")
    
    fold_results = []
    files = get_files(cfg.data_path)

    try:
        for fold in range(cfg.training.get("folds", 25)):
        # for fold in range(1):
            result = train_fold(cfg=cfg, fold=fold, results_path=cfg.save_dir, resume=resume, files=files)
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
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from last checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    args = parser.parse_args()

    run(args.config, args.override, resume=args.resume, name=args.name)
