import numpy as np
from dreemRead import extract_data
import glob
import os
from utils import *
from configuration import Configuration
import tensorflow as tf
from tqdm import tqdm


mnt_path = "/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2"
pillow_path = "/home/christina/Documents/pillow/processed-datasets/PHYSIO2018_harmonized/npz/C3-RPA"

def get_subject_id(filename):
    # e.g. 'tr07-0564.npz' -> 'tr07-0564', 'C3-M2_tr08-0111.npz' -> 'tr08-0111'
    base = os.path.basename(filename).replace('.npz', '')
    if '_' in base:
        base = base.split('_')[-1] # case where channel is in front of the file name
    return base

def get_files(path, channel=None):
    print(f"Scanning for sleep files in {path} (channel={channel})...")
    files = glob.glob(os.path.join(path, '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*.npz')) + \
            glob.glob(os.path.join(path, '*', '*', '*.npz'))
    print(f"Found {len(files)} npz files in {path}, proceeding with npz")
    if files is not None:
        return sorted(files, key=get_subject_id)
    if files is None: # npz not found, check h5 (currently the only other file option)
        files = (glob.glob(os.path.join(path, '*.h5'))     +
            glob.glob(os.path.join(path, '*.hdf5'))   +
            glob.glob(os.path.join(path, '*', '*.h5')) +
            glob.glob(os.path.join(path, '*', '*.hdf5')))
    print(f"Found {len(files)} h5/hdf5 files in {path}.")
    return sorted(files, key=get_subject_id)

def transition_matrix(hyp, n_classes=5):
    hyp = np.asarray(hyp).flatten().astype(int)
    tm = np.zeros((n_classes, n_classes))
    for a, b in zip(hyp[:-1], hyp[1:]):
        tm[a, b] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # guard against unused classes
    return tm / row_sums


# files_mnt = get_files(mnt_path, channel="C3-M2")
# files_pillow = get_files(pillow_path, channel="C3-RPA")

# mnt_id_to_idx = {get_subject_id(f): i for i, f in enumerate(files_mnt)}
# pillow_id_to_idx = {get_subject_id(f): i for i, f in enumerate(files_pillow)}


# common_ids = sorted(set(mnt_id_to_idx) & set(pillow_id_to_idx))

# for idx in common_ids[:5]:
#     idx_mnt = mnt_id_to_idx[idx]
#     idx_pillow = pillow_id_to_idx[idx]
#     x_mnt, hyp_mnt = extract_data(idx_mnt, path=mnt_path, eeg_chan="C3-M2",
#                                 epoch_length=30, files=files_mnt)

#     x_pillow, hyp_pillow = extract_data(idx_pillow, path=pillow_path, eeg_chan="C3-RPA",
#                                 epoch_length=30, files=files_pillow)

#     hyp_mnt = np.asarray(hyp_mnt).flatten()
#     hyp_pillow = np.asarray(hyp_pillow).flatten()
#     n = min(len(hyp_mnt), len(hyp_pillow))

#     print(f"\nSubject {idx}: mnt len={len(hyp_mnt)}, pillow len={len(hyp_pillow)}")
#     print(f"  direct label match rate (first {n} epochs): {np.mean(hyp_mnt[:n] == hyp_pillow[:n]):.3f}")
#     print(f"  mnt transition matrix:\n{transition_matrix(hyp_mnt).round(2)}")
#     print(f"  pillow transition matrix:\n{transition_matrix(hyp_pillow).round(2)}")


# reuse run_tflite from utils.py, but score its output directly (no seq windowing)
from sklearn.metrics import f1_score
import numpy as np
fold = 2
_, test_inds, _ = get_fold_indices(fold, 5, seed=100, cv_type="kfold", mat_path=None)

base_config_path = "configs/PHYSIO.json"
override_config_path = "configs/ablations/PHYSIO_30_3_new_data_old_resnet.json"
name = "seq_set_files"
cfg = Configuration(base_config_path, override_config_path, name=None)
print(cfg.dataset)
files = get_files(cfg.dataset["path"], channel=cfg.dataset["eeg_channel"])


tflite_path = os.path.join(cfg.save_dir, f"cnn_full_int_fold{fold}.tflite")
print(f"Loading TFLite model from {tflite_path}...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
model = tf.keras.models.load_model(os.path.join(cfg.save_dir, f"{cfg.name}_best_cnn_fold{fold}.h5"))
# fp32_output = model.get_layer('fp32_output').output
# fp32_submodel = tf.keras.Model(inputs=model.input, outputs=fp32_output, name='fp32_submodel_reloaded')

y_pred_tflite, y_true = [], []
# for i in tqdm(test_inds, desc="Scoring TFLite model"):
#     epochs, hyp = load_subject(i, cfg, files=files)
#     if epochs is None:
#         continue
#     if cfg.training["physio_resnet"]=="true":
#         preds = run_tflite(interpreter, epochs, cfg, fp32_submodel=fp32_submodel)
#     else:
#         preds = run_tflite(interpreter, epochs, cfg)  # fp32_submodel=None since physio_resnet=false
#     preds_cls = [np.argmax(p) for p in preds]
#     y_pred_tflite.extend(preds_cls)
#     y_true.extend(hyp.flatten())

# print("TFLite MF1:", f1_score(y_true, y_pred_tflite, average="macro"))

import time
x_dummy = np.random.randn(128, 1, cfg.window_length, 1).astype(np.float32)
y_dummy = np.random.randint(0, 5, size=(128,))
y_dummy = to_categorical(y_dummy, 5)

# warm up (first call includes graph tracing overhead)
model.train_on_batch(x_dummy, y_dummy)

t0 = time.perf_counter()
for _ in range(20):
    model.train_on_batch(x_dummy, y_dummy)
t1 = time.perf_counter()
print(f"Mean train_on_batch time: {(t1-t0)/20*1000:.1f} ms")