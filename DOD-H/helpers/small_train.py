import tensorflow as tf
from tqdm import tqdm
from nn_model import *
from dreemRead import *
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

from utils import *
from configuration import Configuration
import math
from logger import ResourceLogger
import functools

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
    
