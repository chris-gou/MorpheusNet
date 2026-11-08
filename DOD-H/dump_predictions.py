import numpy as np
import tensorflow as tf
from pathlib import Path

from utils import *
from configuration import Configuration

SEQ_LEN = 12
CONFIG, OVERRIDE = "configs/SEDF20.json", "configs/ablations/DOD-H_10_3.json"

OVERRIDES=["configs/ablations/SEDF20_5_2.json"]

for override in OVERRIDES:
    cfg = Configuration(CONFIG, override)
    cfg.save_dir = f"{cfg.save_dir}_subj"

    files = get_files(cfg.data_path)
    res = Path(cfg.save_dir)
    out = res / "per_subject_preds"
    out.mkdir(exist_ok=True)
    print(cfg.data_path)

    for fold in range(cfg.training.get("folds", 25)):
        ind_path = res / f"test_ind_fold{fold}.npy"
        if not ind_path.exists():
            ind_path = res / f"test_ind_dodh_fold{fold}.npy"  # legacy name
        test_inds = np.load(ind_path)

        interpreter = tf.lite.Interpreter(model_path=str(res / f"cnn_full_int_fold{fold}.tflite"))
        interpreter.allocate_tensors()
        
        seq_model = tf.keras.models.load_model(res / f"{cfg.name}_subj_best_seq_fold{fold}.h5")

        for subj in test_inds:
            x, y = create_seq_sets([subj], interpreter, cfg, files=files)
            x = np.reshape(np.array(x), (len(x), SEQ_LEN * 5))
            y_pred = np.argmax(seq_model.predict(x, verbose=0), axis=1)
            np.savez(out / f"fold{fold:02d}_subj{subj}.npz", y_true=np.array(y), y_pred=y_pred)

        tf.keras.backend.clear_session()
        print(f"fold {fold+1} done")