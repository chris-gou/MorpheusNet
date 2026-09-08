import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from configuration import Configuration
from utils import get_files, load_subject
import os

def compare_float_vs_quant(cfg, fold, results_path, files):
    """
    Compare float CNN vs TFLite INT8 CNN predictions on the same test set.
    Returns metrics for both plus their agreement rate.
    """
    # ---- load both models ----
    float_model_path = os.path.join(results_path, f'{cfg.name}_best_cnn_fold{fold}.h5')
    tflite_path = os.path.join(results_path, f"cnn_full_int_fold{fold}.tflite")

    float_model = tf.keras.models.load_model(float_model_path)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # ---- get test data (same indices used in train_fold) ----
    test_ind_path = os.path.join(results_path, f'test_ind_fold{fold}.npy')
    if test_ind_path is None or not os.path.exists(test_ind_path):
        test_ind_path = os.path.join(results_path, f'test_ind_dodh_fold{fold}.npy')
    test_inds = np.load(test_ind_path)

    y_true, y_pred_float, y_pred_quant = [], [], []

    for i in test_inds:
        epochs, labels = load_subject(i, cfg, files=files)
        if epochs is None or labels is None:
            continue
        epochs = np.array(epochs)

        # float predictions (batched, fast)
        preds_float = np.argmax(float_model.predict(epochs, verbose=0), axis=1)

        # quantized predictions (TFLite interpreter is single-sample only)
        preds_quant = []
        scale, zero_point = input_details['quantization']
        for e in epochs:
            e_reshaped = e.reshape(input_details['shape']).astype(np.float32)
            e_int8 = (e_reshaped / scale + zero_point).astype(np.int8)
            interpreter.set_tensor(input_details['index'], e_int8)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])
            preds_quant.append(np.argmax(out, axis=1)[0])

        y_true.extend(labels)
        y_pred_float.extend(preds_float)
        y_pred_quant.extend(preds_quant)

    y_true = np.array(y_true)
    y_pred_float = np.array(y_pred_float)
    y_pred_quant = np.array(y_pred_quant)

    agreement = np.mean(y_pred_float == y_pred_quant)

    print(f"--- Fold {fold} ---")
    print(f"Float  | Acc: {accuracy_score(y_true, y_pred_float):.4f} | MF1: {f1_score(y_true, y_pred_float, average='macro'):.4f}")
    print(f"Quant  | Acc: {accuracy_score(y_true, y_pred_quant):.4f} | MF1: {f1_score(y_true, y_pred_quant, average='macro'):.4f}")
    print(f"Float vs Quant agreement: {agreement:.4f}")

    return {
        "fold": fold,
        "float_acc": accuracy_score(y_true, y_pred_float),
        "float_mf1": f1_score(y_true, y_pred_float, average='macro'),
        "quant_acc": accuracy_score(y_true, y_pred_quant),
        "quant_mf1": f1_score(y_true, y_pred_quant, average='macro'),
        "agreement": agreement,
    }

config_file = 'configs/PHYSIO.json'
override = 'configs/ablations/PHYSIO_5_3.json'
cfg = Configuration(config_file, override_config_path=override)
files = get_files(cfg.data_path, channel=cfg.dataset.get("eeg_channel"))


r2 = compare_float_vs_quant(cfg, fold=2, results_path=cfg.save_dir, files=files)
r4 = compare_float_vs_quant(cfg, fold=4, results_path=cfg.save_dir, files=files)