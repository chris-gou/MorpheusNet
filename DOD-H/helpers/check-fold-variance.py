import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
from configuration import Configuration
# import tensorflow as tf
# from sklearn.metrics import confusion_matrix
import os
import h5py
import glob
from collections import Counter
import tensorflow as tf
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    override_config_path = "configs/ablations/PHYSIO_30_3_new_data_new_resnet.json"
    base_config_path = "configs/PHYSIO.json"
    cfg = Configuration(base_config_path, override_config_path)
    cfg.name = f"{cfg.name}_seq_set_files"

    results_path = os.path.join("results", cfg.name)
    t = cfg.training
    data_path = cfg.dataset['path']

    # build fold indices for problematic fold
    fold = 3
    path = cfg.dataset['path']
    files = get_files(path)
    files_sorted = sorted(files, key=get_subject_id)
    raw_subject_path = files_sorted[fold]

    train_inds, test_inds, val_inds = get_fold_indices(fold, t.get("folds", 25), seed=100, cv_type=t.get("cv_type", "kfold"), mat_path=t.get("mat_path", None))
    best_model_file_cnn = os.path.join(results_path, f'{cfg.name}_best_cnn_fold{fold}.h5')
    model = tf.keras.models.load_model(best_model_file_cnn)
    # print class distribution for the fold
    x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t, cfg=cfg, files=files_sorted)
    y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4]))

    print(f"Fold {fold}: using file {raw_subject_path}")

    with h5py.File(raw_subject_path, 'r') as f:
        hypnogram = f['hypnogram'][:]  


    counts = Counter(hypnogram)
    total = sum(counts.values())
    for stage, n in sorted(counts.items()):
        print(f"{stage}: {n} epochs ({100*n/total:.1f}%)")