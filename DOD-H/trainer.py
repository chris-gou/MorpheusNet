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

tf.random.set_seed(100)
SEQ_LEN=12
DB_PATH = "/mnt/truenas_db/user/christina/morpheus"

TRAIN_PARAMS = {
    "gh_original": {
        "cnn_batch_size": 32,
        "cnn_epochs": 5,
        "cnn_lr": 10e-3,
        "seq_batch_size": 32,
        "seq_epochs": 15,
    },
    "paper_original": {
        "cnn_batch_size": 128,
        "cnn_epochs": 10,
        "cnn_lr": 0.001,
        "seq_batch_size": 32,
        "seq_epochs": 15,
    }
}
class Configuration:
    """
    Class for MorpheusNet training configuration

    Attributes:

    Methods:
    """

    def __init__(self, base_config_path, override_config_path=None):
        """
        Initializes the Configuration object.

        Args:
            base_config_path (str): Path to the base configuration file.
            override_config_path (str): Path to the override configuration file (optional).

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file is not in a valid format.
        """

        self.config = self._load_config(base_config_path, override_config_path)
        self.dataset = self.config.get("dataset", {})
        self.training = self.config.get("training_params", {})
        self.name = self.config.get("name", os.path.basename(override_config_path).replace(".json", "") if override_config_path else os.path.basename(base_config_path).replace(".json", ""))

        # default to paper-specified hyperparams since they give the best results
        self.run_type = self.config.get("run_type", "paper_original")
        self.hparams = TRAIN_PARAMS.get(self.run_type, "paper_original")
        self.data_path = self.dataset.get("path", "")

        self.window_length = self.dataset.get("epoch_duration", 30)  * 100
        self.save_dir = os.path.join(DB_PATH, self.name)

    def _load_config(self, base_config_path, override_config_path):
        with open(base_config_path) as base_config_file:
            base_config = json.load(base_config_file)
        # override if necessary
        if override_config_path:
            with open(override_config_path) as override_config_file:
                override_config = json.load(override_config_file)
            
            # update any values if they are found in the override, else keep the ones from base
            end_config = self._update_config(base_config, override_config)
        return end_config if override_config_path else base_config
    
    def _update_config(self, base_config, override_config):
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
        return base_config

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



seq_len = 12    #sequence length used for the seqLearner

cnn_acc = []
seq_acc = []

def train_fold(cfg: Configuration, fold: int, data_path: str, results_path: str) -> dict:
    """
    Train one fold
    
    Args:
        configuration (Configuration): The configuration object containing all necessary parameters.
        fold (int): Fold number to train.
        data_path (str): Path to the dataset.
        results_path (str): Path to save the results.

    Returns:
        dict: A dictionary containing the training results.

    Raises:
        Exception: if something is invalid

    """ 
    t = cfg.training
    hp = cfg.hparams
    epoch_duration = t.get("epoch_duration",30)

    # create checkpoints
    best_model_file_cnn = os.path.join(results_path, f'{cfg.name}_best_cnn_fold{fold}.h5')
    best_model_file_seq = os.path.join(results_path, f'{cfg.name}_best_seq_fold{fold}.h5')
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

    # ====== data splits ======
    train_inds, test_inds, val_inds = get_fold_indices(fold, t.get("folds", 25), seed=100, cv_type=t.get("cv_type", "kfold"))
    np.save(os.path.join(results_path, f'train_ind_dooh_fold{fold}.npy'), train_inds)
    np.save(os.path.join(results_path, f'test_ind_dodh_fold{fold}.npy'), test_inds)
    np.save(os.path.join(results_path, f'val_ind_dodh_fold{fold}.npy'), val_inds)

    # create train, val, test sets
    x_train, y_train = create_set(train_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
    x_val, y_val = create_set(val_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
    x_test, y_test = create_set(test_inds, cfg.dataset["eeg_channel"], data_path, training_params=t)
    x_train, y_train = shuffle(x_train, y_train)

    # ===== CNN =====
    optimizer = tf.keras.optimizers.Adam(learning_rate = hp.get("cnn_lr", 0.001))
    model = separable_resnet((1,cfg.window_length,1), 5, y_train = y_train, bias = False)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    model.fit(x_train, to_categorical(y_train), batch_size=hp.get("cnn_batch_size", 512), epochs=hp.get("cnn_epochs", 10), 
              validation_data = (x_val, to_categorical(y_val)), callbacks = [checkpoint_callback_cnn])
    
    model = tf.keras.models.load_model(best_model_file_cnn)

    y_pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2, 3, 4]))

    # ===== PQT =====
    saved_model_dir = os.path.join(results_path, f'saved_model_fold_{fold}')
    model.export(saved_model_dir)
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
            validation_data = (x_val_seq, to_categorical(y_val_seq)), callbacks = [checkpoint_callback_seq])
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

# ====== Aggregate fold results ======
def aggregate_and_save(cfg: Configuration, fold_results: list):
    class_names = ["Wake", "N1", "N2", "N3", "REM"]

    def means(key):  return [r[key] for r in fold_results]
 
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

def run(base_config_path, override_config_path=None):
    cfg = Configuration(base_config_path, override_config_path)
    
    os.makedirs(cfg.save_dir, exist_ok=True)

    results_file_path = os.path.join(cfg.save_dir, "results.json")
    if os.path.isfile(results_file_path):
        print(f"Results already exist for {cfg.name} → {results_file_path}")
        return

    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}  |  type: {cfg.get("run_type")}  |  epoch: {cfg.training.get("epoch_duration")}s  |  folds: {cfg.training.get("folds")}")
    print(f"{'='*60}")

    fold_results = []
    for fold in range(cfg.training.get("folds", 25)):
        fold_results.append(train_fold(cfg, fold, cfg.data_path, cfg.save_dir))

    aggregate_and_save(cfg, fold_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='/mnt/truenas_db/user/christina', help='Path of dataset')
    parser.add_argument('--output_path', type=str, help='Path to save results')
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--debug', action='store_true', default=False, help='Set mode to debug or not')
    parser.add_argument('--override', type=str, help='config override path')

    args = parser.parse_args()

    run(args.config, args.override)