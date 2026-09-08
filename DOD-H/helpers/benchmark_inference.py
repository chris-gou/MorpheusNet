from collections import deque
import time
import pynvml
import threading
import numpy as np
import psutil
from contextlib import nullcontext
import tensorflow as tf
import argparse
from utils import load_subject, get_fold_indices
from monitoring import GPUSampler, CPUSampler
from dreemRead import *
from configuration import Configuration


class FIFOContextBuffer():
    """
    FIFO buffer to store context for inference. Holds context window needed for one prediction (according to each model).
    """
    def __init__(self, context_size: int, stack_fn=None):
        self.buffer = deque(maxlen=context_size)
        self.context_size = context_size
        self.stack_fn = stack_fn or (lambda items: np.stack(items, axis=0)) 

    def push(self, epoch):
        """
        Add an epoch into the buffer
        """
        self.buffer.append(epoch)
        if len(self.buffer) < self.context_size:
            return None
        return self.stack_fn(list(self.buffer))

def run_streaming_benchmark_tf(predict_fn, epochs, context_size, n_warmup=10,
                               use_gpu=False, n_iters=200, monitor=True, stack_fn=None):
    """
    TensorFlow / TFLite counterpart of run_streaming_benchmark. Same FIFO +
    back-to-back-call design; only the timing and monitoring backend differ.
 
    predict_fn:    callable taking one window (np.ndarray, shape
                   [context_size, ...], batch dim already added) and running
                   one forward pass. Wrap either a TFLite Interpreter
                   (set_tensor/invoke/get_tensor) or a Keras model call in a
                   small function with this signature, e.g.:
 
                       def predict_fn(x):
                           interpreter.set_tensor(input_idx, x.astype(np.float32))
                           interpreter.invoke()
                           return interpreter.get_tensor(output_idx)
 
    epochs:        iterable of np.ndarray epochs, in arrival order.
    context_size:  trailing epochs consumed per call (1 for the MorpheusNet
                   CNN, k for the sequence learner).
    use_gpu:       True if predict_fn runs on GPU (monitored via NVML);
                   False for CPU, e.g. TFLite (monitored via psutil).
    """

    fifo = FIFOContextBuffer(context_size=context_size, stack_fn=stack_fn)
    epoch_iter = iter(epochs)

    def next_window():
        window = None
        while window is None:
            try:
                epoch = next(epoch_iter)
            except StopIteration:
                return None
            window = fifo.push(epoch)
        return np.expand_dims(window, axis=0) # batch size 1

    # warmup 
    for _ in range(n_warmup):
        x = next_window()
        if x is None:
            raise RuntimeError("epochs generator exhausted during warmup; "
                                "provide more epochs or lower n_warmup.")
        predict_fn(x)

    if monitor:
        if use_gpu:
            monitor = GPUSampler()
        else:
            monitor = CPUSampler()
    else:
        monitor = None

    latencies_ms = []
    
    if monitor is not None:
        for _ in range(n_iters):
            x = next_window()
            if x is None:
                print(f"epochs generator exhausted after {len(latencies_ms)} "
                      f"iterations (requested {n_iters}); stopping early.")
                break
            t0 = time.perf_counter()
            predict_fn(x)
            latencies_ms.append((time.perf_counter() - t0) * 1000) # convert to ms

    latencies_ms = np.array(latencies_ms)
    results = {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_std_ms": float(np.std(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_max_ms": float(np.max(latencies_ms)),
    }

    if monitor is not None:
        results.update(monitor.stats())
    return results

# ==== geenerators ======
# yield epoch one at a time 
def raw_epoch_stream(indices, cfg):
    epoch_duration = cfg.training.get('epoch_duration', 30)
    fs = cfg.training.get('fs', 100)
    for index in indices:
        epochs, hyp = load_subject(index, cfg)
        if epochs is None or hyp is None:
            continue
        for epoch, label in zip(epochs, hyp):
            yield np.asarray(epoch, dtype=np.float32).reshape(epoch_duration * fs, 1), label

def cnn_output_stream(indices, cfg, interpreter, input_detail, output_detail):
    epoch_duration = cfg.training.get('epoch_duration', 30)
    fs = cfg.training.get('fs', 100)
    print(f"Streaming benchmark: yielding CNN outputs one at a time, epoch_duration={epoch_duration}s, fs={fs}Hz")

    input_scale, input_zero_point = input_detail['quantization']
    output_scale, output_zero_point = output_detail['quantization']
    for epoch in raw_epoch_stream(indices, cfg):
        x = epoch.reshape(1, 1, fs*epoch_duration, 1) # (1, 1, 3000, 1)
        x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_detail['index'], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(output_detail['index'])
        y = output_scale * (y_q.astype(np.float32) - output_zero_point)
        yield y[0]

def reshape_seq_window(items):
    """items: seq_len arrays of shape (5,), oldest first -> (seq_len*5, 1)."""
    return np.stack(items, axis=0).reshape(-1, 1)
 
def make_combined_predict_fn(cnn_predict_fn, seq_predict_fn, seq_len):
    """Builds one predict_fn that runs CNN -> seq FIFO -> seq model as a
    single unit, for end-to-end streaming latency. The seq FIFO persists
    across calls (it's a rolling window, not reset per call)."""
    seq_fifo = FIFOContextBuffer(context_size=seq_len, stack_fn=reshape_seq_window)
 
    def combined_predict_fn(x_raw):
        # x_raw: (1, 1, 3000, 1) raw epoch
        cnn_out = cnn_predict_fn(x_raw)          # (1, 5)
        window = seq_fifo.push(cnn_out[0])       # push (5,) vector
        if window is None:
            return None                          # buffer still filling (priming only)
        seq_x = np.expand_dims(window, axis=0)   # (1, 60, 1)
        return seq_predict_fn(seq_x)
 
    return combined_predict_fn, seq_fifo
 
def measure_e2e_streaming_latency(combined_predict_fn, test_inds, cfg, n_warmup=10):
    epoch_stream = raw_epoch_stream(test_inds, cfg) # get (epoch, label) pairs
    epoch_iter = iter(epoch_stream) # iterator object of the pairs
    # exclude warmup from latency measurements
    for _ in range(n_warmup):
        x_raw, _ = next(epoch_iter)
        combined_predict_fn(x_raw)
 
    latencies_ms = []
    y_true, y_pred = [], []
 
    for x_raw, label in epoch_iter:
        start = time.perf_counter()
        out = combined_predict_fn(x_raw)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        if out is not None:
            y_true.append(label)
            y_pred.append(int(np.argmax(out)))
 
    latencies_ms = np.array(latencies_ms)
    stats = {
        'mean_latency_ms': float(np.mean(latencies_ms)),
        'std_latency_ms': float(np.std(latencies_ms)),
        'p50_latency_ms': float(np.percentile(latencies_ms, 50)),
        'p95_latency_ms': float(np.percentile(latencies_ms, 95)),
        'p99_latency_ms': float(np.percentile(latencies_ms, 99)),
        'raw_latencies_ms': latencies_ms,
    }
    return stats, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run streaming benchmark for MorpheusNet models.")
    parser.add_argument("--model", type=str, help="Which model to choose from.")
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    args = parser.parse_args()

    cfg = Configuration(args.config, args.override, name=args.name)
    results_dir = cfg.save_dir
    total_folds = cfg.training.get("folds", 25)
    t = cfg.training
    
    for fold in range(total_folds):
        _, test_inds, _ = get_fold_indices(fold, total_folds, 
                                    seed=100, cv_type=t.get("cv_type", "kfold"), mat_path=t.get("mat_path", None))
    
        model_path = f"{results_dir}/cnn_full_int_fold{fold}.tflite" # tflite path

        # Morpheus TFLite
        interpreter = tf.lite.Interpreter(model_path=model_path) # tflite path
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']

        # ensure that input length matches the expected epoch length
        expected_len = t['fs']*t['epoch_duration']
        model_input_len = input_details['shape'][2]
        if expected_len != model_input_len:
            raise ValueError(f"Model input length {model_input_len} does not match expected epoch length {expected_len}. Check configuration and model.")
        print("==================================")
        print(f"Fold {fold}: epoch_duration={t['epoch_duration']}s, fs={t['fs']}Hz, "
              f"input_len={model_input_len}, model={model_path}")
        
        seq_len = 12
        best_model_file_seq = os.path.join(results_dir, f'{cfg.name}_best_seq_fold{fold}.h5')

        seq_learner= tf.keras.models.load_model(best_model_file_seq)

        def cnn_predict_fn(x):
            x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)
            interpreter.set_tensor(input_details['index'], x_q)
            interpreter.invoke()
            y_q = interpreter.get_tensor(output_details['index'])
            return (y_q.astype(np.float32) - output_zero_point) * output_scale

        def predict_seq_fn(x): # x has shape (1, seq_len*num_classes, 1) -> (1, 60, 1)
            out = seq_learner(x, training=False)
            return int(np.argmax(out.numpy()[0])) # predicted index

        def reshape_window(x):
            return np.stack(x, axis=0).reshape(-1, 1) # (seq_len, 1)

        combined_predict_fn, seq_fifo = make_combined_predict_fn(
                cnn_predict_fn, predict_seq_fn, seq_len
            )
        epoch_source = raw_epoch_stream(test_inds, cfg)

        for _ in range(seq_len-1):
            x_raw = next(epoch_source).reshape(1, 1, t['fs']*t['epoch_duration'], 1)
            combined_predict_fn(x_raw) # prime the seq FIFO


        result = run_streaming_benchmark_tf(
            predict_fn=combined_predict_fn,
            epochs=epoch_source,
            context_size=1,        # k for the seq learner instead
            use_gpu=False,
            n_iters=200,
        )
        print(f"Fold {fold} streaming benchmark results: {result}")
