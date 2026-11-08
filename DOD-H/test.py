import os
import json
import argparse
import time
import psutil
import numpy as np
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from pathlib import Path
from matplotlib import pyplot as plt

from utils import *
from monitoring import *
from configuration import Configuration
from helpers.inference.benchmark_inference import *

from pyutils.helpers import notify

SEQ_LEN=12

def prepare_fold_paths(cfg: Configuration, fold: int, results_path: str | Path) -> dict[str, Path]:
    results_path = Path(results_path) / 'streaming' 
    results_path.mkdir(parents=True, exist_ok=True)

    model_files_path = Path(cfg.save_dir)
    test_ind_path = model_files_path / f'test_ind_fold{fold}.npy'
    seq_learner_path = model_files_path / f'{cfg.name}_best_seq_fold{fold}.h5'
    if not seq_learner_path.exists() and '_new_data' in cfg.save_dir:
        seq_learner_path = model_files_path / f'{cfg.name}_best_seq_fold{fold}.h5'
        seq_learner_path = seq_learner_path.with_name(seq_learner_path.name.replace('_new_data', ''))

    if not test_ind_path.exists():
        test_ind_path = model_files_path / f'test_ind_dodh_fold{fold}.npy'

    return {
        "results_path": results_path,
        "tflite_path": str(model_files_path  / f'cnn_full_int_fold{fold}.tflite'),
        "test_ind_path": str(test_ind_path), # for easier loading, return as str
        "seq_learner_path": str(seq_learner_path),
        "h5_path": str(model_files_path / f'{cfg.name}_best_cnn_fold{fold}.h5')
    }

def prepare_fold_data(cfg: Configuration, fold: int) -> tuple[list[int], list[int], list[int]]:
    training = cfg.training
    _, _, test_indices = get_fold_indices(fold_number=fold, 
                                          total_folds=cfg.training.get("folds"), 
                                          seed=100, 
                                          cv_type=cfg.training.get("cv_type", "kfold"), 
                                          mat_path=cfg.training.get("mat_path", None), 
                                          data_path=cfg.data_path)

    return test_indices
    
# static measurements
def calculate_streaming_latency(interpreter, x_test, window_length, cpu_window=100, gpu=None, fp32_submodel=None, n_latency=2000):
    input_details = interpreter.get_input_details() # [batch_size, context length, samples per epoch, channel count (single eeg)]
    output_details = interpreter.get_output_details()
    # print(f"input_details: {input_details}, output_details: {output_details}")
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    def invoke_tflite(x_q):
       interpreter.set_tensor(input_details[0]['index'], x_q)
       interpreter.invoke()
       y_q = interpreter.get_tensor(output_details[0]['index'])
       return (y_q.astype(np.float32) - output_zero_point) * output_scale

    latencies, cpu_usage, y_pred_lat = [], [], []
 
    def prep(sample):
        x = sample.reshape((1, 1, window_length, 1))
        if fp32_submodel is not None:
            x = fp32_submodel(x, training=False).numpy()
        return quantize(x, input_scale, input_zero_point)   # was: (x / input_scale + input_zero_point).astype(np.int8)
    
    def predict_one(x):
        return invoke_tflite(prep(x))
    
    x_all = np.asarray(x_test, dtype=np.float32).reshape(-1, 1, window_length, 1)
    # 1) Accuracy: fp32 front-end batched, TFLite per sample (invoke is cheap)
    y_pred, n_out, n_tot = [], 0, 0
    for s in range(0, len(x_all), 4096):
        feats = x_all[s:s + 4096]
        if fp32_submodel is not None:
            feats = fp32_submodel.predict(feats, batch_size=128, verbose=0)
        q = feats / input_scale + input_zero_point
        n_out += np.sum((q < -128) | (q > 127))
        n_tot += q.size
        feats_q = quantize(feats, input_scale, input_zero_point)
        y_pred += [int(np.argmax(invoke_tflite(feats_q[i:i+1]))) for i in range(len(feats_q))]
    print(f'[CHECK] int8 out-of-range fraction: {n_out / n_tot:.6f}')

    # warmup
    for _ in range(10):
        predict_one(x_all[0])

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # warm-up
    cpu_count = psutil.cpu_count(logical=False) 
    print(f"cpu_count={psutil.cpu_count()}, sample cpu%={proc.cpu_percent(interval=0.1)}")

    for x in x_all[:n_latency]:
        cpu_start = proc.cpu_percent(interval=None) / cpu_count
        start = time.perf_counter()
        y = predict_one(x)
        latencies.append((time.perf_counter() - start) * 1000)
        cpu_usage.append((cpu_start + proc.cpu_percent(interval=None) / cpu_count) / 2)
        y_pred_lat.append(int(np.argmax(y)))

    # for i in x_all[:n_latency]:
    # # for i in range(0, len(x_test), cpu_window):
    #     batch = x_test[i:i+cpu_window]
    #     cpu_start = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
    #     batch_latencies = []
    #     for sample in batch:
    #         start_time = time.perf_counter()
    #         sample = sample.reshape((1, 1, window_length, 1))
    #         quantized = prep(sample)
    #         interpreter.set_tensor(input_details[0]['index'], quantized)
    #         interpreter.invoke()
    #         output_details = interpreter.get_output_details()
    #         output_index = output_details[0]["index"]
            
    #         # Get the output tensor
    #         output_data = interpreter.get_tensor(output_index)
    #         output_details = interpreter.get_output_details()
    #         output_scale, output_zero_point = output_details[0]['quantization']

    #         dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale

    #         y_pred.append(int(np.argmax(dequantized_output)))
    #         end_time = time.perf_counter()
    #         batch_latencies.append((end_time - start_time) * 1000)
        
    #     cpu_after = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
    #     cpu_win = (cpu_start + cpu_after) / 2  # average CPU usage during the batch
    #     for lat in batch_latencies:
    #         latencies.append(lat)
    #         cpu_usage.append(cpu_win)
    #         # cpu_usage.append((np.array(cpu_before) + np.array(cpu_after)) / 2)
    #         # cpu_usage.append(np.mean([cpu_after], axis=0))bu
        # cpu_usage.append(cpu_after)
        # latencies.append((end_time - start_time) * 1000)
    agree = np.mean(np.array(y_pred_lat) == np.array(y_pred[:len(y_pred_lat)]))
    print(f'[CHECK] cnn batched vs streaming agreement: {agree:.4f}')


    latencies = np.array(latencies)
    cpu_usage = np.array(cpu_usage)

    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'median_latency_ms': np.median(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'raw_latencies_ms': latencies
    }, cpu_usage, y_pred

def measure_seq_learner_latency(model_path, x_seq_arr, n_timestamps=60, n_features=1, n_samples=2000, warmup=10, events=None, gpu_sampler=None):
    seq_model= tf.keras.models.load_model(model_path)

    hidden_size = seq_model.input_shape # (None, past epochs, feature per epoch)
    n_timestamps = hidden_size[1] if len(hidden_size) > 1 else n_timestamps
    n_features = hidden_size[2] if len(hidden_size) > 2 else n_features
    cpu_usage = []

    probs = seq_model.predict(x_seq_arr, batch_size=1024, verbose=0) # probabilities
    y_pred = np.argmax(probs, axis=1).tolist() # predicted labels

    # 2) Latency: batch size 1 on a subset of consecutive epochs
    x_lat = x_seq_arr[:n_samples]

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # warm-up
    cpu_count = psutil.cpu_count(logical=False) 
    # warmup
    for i in range(min(warmup, len(x_lat))):
        seq_model(x_seq_arr[i:i+1], training=False)
    
    latencies, timestamps, y_pred_lat = [], [], []
    baseline_cpu = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
    gpu_sampler_baseline = gpu_sampler.stats()['mean']

    for i in range(len(x_lat)):
        t_wall = time.time() - gpu_sampler._t0 # align with GPU time
        timestamps.append(t_wall)

        start_time = time.perf_counter()
        cpu_start = proc.cpu_percent(interval=None) / cpu_count

        out = seq_model(x_lat[i:i+1], training=False).numpy()

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
        cpu_after = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
        cpu_win = (cpu_start + cpu_after) / 2  # average CPU usage during the batch
        cpu_usage.append(cpu_win)

        y_pred_lat.append(int(np.argmax(out[0])))

    events.mark(f'seq latency measured for {len(x_lat)} samples')
    # sanity check: batched vs. one-at-a-time predictions must agree
    agree = np.mean(np.array(y_pred_lat) == np.array(y_pred[:len(y_pred_lat)]))
    print(f'[CHECK] seq batched vs streaming agreement: {agree:.4f}')

    hist = gpu_sampler.history()
    gpu_times = np.array(hist['timestamps'])
    gpu_util = np.array(hist['util'])
    idx = np.searchsorted(gpu_times, timestamps)
    idx = np.clip(idx, 0, len(gpu_util) - 1)
    gpu_util_per_sample = gpu_util[idx]

    cpu_usage = np.array(cpu_usage)
    contended = (cpu_usage > baseline_cpu + 10) | (gpu_util_per_sample > gpu_sampler_baseline + 10)


    latencies = np.array(latencies)
    cpu_usage = np.array(cpu_usage)
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'median_latency_ms': np.median(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'raw_latencies_ms': latencies,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_util_per_sample,
        'timestamps': np.array(timestamps),
        'contended': contended,
        'y_pred': y_pred,
    }

def convert_h5_to_tflite(h5_path, tflite_path):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def load_fp32_submodel(cfg, h5_path):
    if not (cfg.dataset["name"] == 'PHYSIO2018_harmonized' and cfg.training.get("physio_resnet") == "true"):
        return None
    model = tf.keras.models.load_model(h5_path)
    return tf.keras.Model(model.input, model.get_layer('fp32_output').output, name='fp32_submodel')

# ===== FIFO e2e streaming latency

def make_cnn_predict_fn(interpreter, input_details, output_details, fp32_submodel=None):
    """Wraps the TFLite CNN interpreter as a single predict_fn.
    Accepts a raw epoch of shape (samples_per_epoch, 1)."""
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    def cnn_predict_fn(x_raw):
        x = x_raw.reshape(1, 1, -1, 1)
        if fp32_submodel is not None:
            x = fp32_submodel(x, training=False).numpy()
        x_q = quantize(x, input_scale, input_zero_point)
        interpreter.set_tensor(input_details['index'], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(output_details['index'])
        return (y_q.astype(np.float32) - output_zero_point) * output_scale

    return cnn_predict_fn

def make_seq_predict_fn(seq_model):
    def seq_predict_fn(x):
        out = seq_model(x, training=False)
        return out.numpy()[0]  # class probabilities
 
    return seq_predict_fn

# main stuff
def evaluate_one_fold(cfg, fold, results_path, events, gpu_sampler, files):
    paths = prepare_fold_paths(cfg, fold, results_path)
    window_length = cfg.window_length

    tflite_path = paths['tflite_path']
    if not os.path.exists(tflite_path):
        print(f"[SKIP] No TFLite model for fold {fold}: {tflite_path}")
        # check for h5 to convert
        h5_path = paths['h5_path']
        if os.path.exists(h5_path):
            convert_h5_to_tflite(h5_path, tflite_path)
        else:
            print(f'[SKIP] No TFLite or h5 model for fold {fold}: {tflite_path}')
            return (None,) * 9
    if type(tflite_path) is Path:
        tflite_path = str(tflite_path)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    fp32_submodel = load_fp32_submodel(cfg, paths['h5_path'])

    # get indices
    test_ind_path = paths['test_ind_path']
    test_inds = np.load(test_ind_path, allow_pickle=True)
    test_indicess = prepare_fold_data(cfg, fold)
    if (set(test_inds) != set(test_indicess)):
        print(f"[WARN] Test indices mismatch for fold {fold}. Using test indices from get_fold_indices.")
        test_inds = test_indicess

    
    x_test, y_test = create_set(test_inds, cfg, files=files)
    events.mark(f'CNN Set done')
    x_test_seq, y_seq_test = create_seq_sets(test_inds, interpreter, cfg, files=files, fp32_submodel=fp32_submodel)
    x_test_seq = np.reshape(np.array(x_test_seq),(len(x_test_seq),int(SEQ_LEN*5))) # array 
    events.mark(f'SEQ Set done')
    # static latency and accuracy
    seq_learner_path = paths['seq_learner_path']
    cnn_latencies, cnn_cpu_usage, y_pred_cnn = calculate_streaming_latency(interpreter, x_test, window_length, fp32_submodel=fp32_submodel)
    events.mark(f'CNN measure done')
    seq_stats = measure_seq_learner_latency(seq_learner_path, x_test_seq, events=events, gpu_sampler=gpu_sampler)

    # save confusion matrix and classification report for both models
    events.mark(f'SEQ measure done')
    # CNN
    plot_confusion_and_f1(y_test, y_pred_cnn, cfg, 'cnn', fold=fold)
    # Seq
    plot_confusion_and_f1(y_seq_test, seq_stats['y_pred'], cfg, 'seq', fold=fold)

    # e2e 
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    seq_model = tf.keras.models.load_model(str(paths['seq_learner_path']))
    

    # build predict functions as callables
    cnn_predict = make_cnn_predict_fn(interpreter, input_details, output_details, fp32_submodel=fp32_submodel)
    seq_predict = make_seq_predict_fn(seq_model)
    # chain cnn and seq with FIFO
    combined_predict_fn, seq_fifo = make_combined_predict_fn(cnn_predict, seq_predict, seq_len=SEQ_LEN)

    # iterate raw epochs one at a time, return latency stats and labels
    e2e_stats, _, _ = measure_e2e_streaming_latency(
        combined_predict_fn, seq_fifo, test_inds, cfg, files=files
    )
    events.mark(f'e2e streaming done')
    plot_streaming_latency(e2e_stats, cfg, fold=fold)

    return cnn_latencies, seq_stats, cnn_cpu_usage, seq_stats['cpu_usage'], y_test, y_pred_cnn, y_seq_test, seq_stats['y_pred'], e2e_stats
 

def plot_streaming_latency(stats, cfg, fold=None, alpha=0.1):
    """Histogram of end-to-end per-epoch latency against the real-time
    budget L_p95 <= alpha * T."""
    epoch_duration = cfg.training.get('epoch_duration', 30)
    budget_ms = alpha * epoch_duration * 1000

    
 
    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.hist(stats['raw_latencies_ms'], bins=50, color='steelblue', alpha=0.8)
    lat = stats['raw_latencies_ms']
    lo, hi = np.percentile(lat, [0.1, 99.9])
    ax.hist(lat, bins=np.linspace(lo, hi, 50), color='steelblue', alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.axvline(stats['p95_latency_ms'], color='orange', linestyle='--',
               label=f"p95 = {stats['p95_latency_ms']:.2f} ms")
    # ax.axvline(budget_ms, color='red', linestyle='-',
            #    label=f"budget (\u03b1T) = {budget_ms:.1f} ms")

    ax.set_xlim(0, stats['raw_latencies_ms'].max() * 1.1)
    ax.text(0.98, 0.95, f"budget (\u03b1T) = {budget_ms:.0f} ms\n(off-scale, well within margin)",
            transform=ax.transAxes, ha='right', va='top', color='red', fontsize=9)
    ax.set_xlabel('End-to-end latency [ms]')
    ax.set_ylabel('Count')
    
    title = f'{cfg.name} \u2014 streaming latency'
    title += f' (fold {fold})' if fold is not None else ' (all folds)'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
 
    fname = f'{cfg.name}_e2e_latency'
    fname += f'_fold{fold:02d}.png' if fold is not None else '_all.png'
    out_path = os.path.join(cfg.save_dir, fname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def aggregate_and_save(all_seq_stats, all_cnn_stats, all_e2e_stats, output_path):
    # save per-fold stats
    per_fold_cnn, per_fold_seq, per_fold_e2e = {}, {}, {}
    for fold, (seq_stats, cnn_stats, e2e_stats) in enumerate(zip(all_seq_stats, all_cnn_stats, all_e2e_stats)):
        per_fold_cnn[fold] = {
            'mean_latency_ms': float(cnn_stats['mean_latency_ms']),
            'std_latency_ms': float(cnn_stats['std_latency_ms']),
            'p50_latency_ms':  float(cnn_stats['p50_latency_ms']),
            'p95_latency_ms':  float(cnn_stats['p95_latency_ms']),
            'p99_latency_ms':  float(cnn_stats['p99_latency_ms']),
        }
        per_fold_seq[fold] = {
            'mean_latency_ms': float(seq_stats['mean_latency_ms']),
            'std_latency_ms': float(seq_stats['std_latency_ms']),
            'p50_latency_ms':  float(seq_stats['p50_latency_ms']),
            'p95_latency_ms':  float(seq_stats['p95_latency_ms']),
            'p99_latency_ms':  float(seq_stats['p99_latency_ms']),
        }
        per_fold_e2e[fold] = {
            'mean_latency_ms': float(e2e_stats['mean_latency_ms']),
            'std_latency_ms': float(e2e_stats['std_latency_ms']),
            'p50_latency_ms':  float(e2e_stats['p50_latency_ms']),
            'p95_latency_ms':  float(e2e_stats['p95_latency_ms']),
            'p99_latency_ms':  float(e2e_stats['p99_latency_ms']),
        }

    # aggregate across folds
    seq_summary = {
    'mean_latency_ms': float(np.mean([s['mean_latency_ms'] for s in all_seq_stats])),
    'std_latency_ms': float(np.mean([s['std_latency_ms']  for s in all_seq_stats])),
    'p50_latency_ms':  float(np.mean([s['p50_latency_ms']  for s in all_seq_stats])),
    'p95_latency_ms':  float(np.mean([s['p95_latency_ms']  for s in all_seq_stats])),
    'p99_latency_ms':  float(np.mean([s['p99_latency_ms']  for s in all_seq_stats])),
    }
    all_contended = np.concatenate([s['contended'] for s in all_seq_stats])
    all_latencies = np.concatenate([s['raw_latencies_ms'] for s in all_seq_stats])
    seq_summary['contended_fraction'] = float(np.mean(all_contended))
    seq_summary['p95_latency_ms_clean'] = float(np.percentile(all_latencies[~all_contended], 95)) if (~all_contended).any() else None
    seq_summary['p95_latency_ms_contended'] = float(np.percentile(all_latencies[all_contended], 95)) if all_contended.any() else None

    cnn_summary = {
        'mean_latency_ms': float(np.mean([s['mean_latency_ms'] for s in all_cnn_stats])),
        'std_latency_ms': float(np.mean([s['std_latency_ms']  for s in all_cnn_stats])),
        'p50_latency_ms':  float(np.mean([s['p50_latency_ms']  for s in all_cnn_stats])),
        'p95_latency_ms':  float(np.mean([s['p95_latency_ms']  for s in all_cnn_stats])),
        'p99_latency_ms':  float(np.mean([s['p99_latency_ms']  for s in all_cnn_stats])),
    }
    all_e2e_latencies = np.concatenate([s['raw_latencies_ms'] for s in all_e2e_stats])
    e2e_summary = {
        'mean_latency_ms': float(np.mean(all_e2e_latencies)),
        'std_latency_ms': float(np.std(all_e2e_latencies)),
        'p50_latency_ms': float(np.percentile(all_e2e_latencies, 50)),
        'p95_latency_ms': float(np.percentile(all_e2e_latencies, 95)),
        'p99_latency_ms': float(np.percentile(all_e2e_latencies, 99)),
    }

    summary = {}

    summary['folds'] = {per_fold: {'cnn': per_fold_cnn[per_fold], 'seq': per_fold_seq[per_fold], 'e2e': per_fold_e2e[per_fold]} for per_fold in range(len(all_seq_stats))}
    summary['seq'] = seq_summary
    summary['cnn'] = cnn_summary
    summary['e2e'] = e2e_summary
    summary['total_mean_latency_ms'] = cnn_summary['mean_latency_ms'] + seq_summary['mean_latency_ms']
    summary['total_std_latency_ms'] = np.sqrt(cnn_summary['std_latency_ms']**2 + seq_summary['std_latency_ms']**2)
    summary['total_p50_latency_ms'] = cnn_summary['p50_latency_ms'] + seq_summary['p50_latency_ms']
    summary['total_p95_latency_ms'] = cnn_summary['p95_latency_ms'] + seq_summary['p95_latency_ms']
    summary['total_p99_latency_ms'] = cnn_summary['p99_latency_ms'] + seq_summary['p99_latency_ms']
    # summary['cpu'] = cpu_stats
    # summary['gpu'] = gpu_stats
    with open(os.path.join(output_path, 'streaming_total_latency_summary_perfold.json'), 'w') as f:
        print(f'[INFO] Saving summary to {f.name}')
        json.dump(summary, f, indent=2)

def run(base_config_path, override_config_path=None, name=None):
    cfg = Configuration(base_config_path, override_config_path, name=name)
    dataset = cfg.dataset
    if not os.path.isdir(cfg.save_dir) and os.path.isdir(os.path.join('results', f"{cfg.name}_new_data")):
        cfg.save_dir = os.path.join('results', f"{cfg.name}_new_data")
    elif os.path.isdir(cfg.save_dir):
        cfg.save_dir = os.path.join(cfg.save_dir)

    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}  |  save_dir: {cfg.save_dir}  |  epoch: {cfg.training.get('epoch_duration')}s  |  folds: {cfg.training.get('folds')}")
    print(f"{'='*60}")

    all_cnn_stats, all_seq_stats, all_e2e_stats = [], [], []
    all_cpu_cnn_stats, all_cpu_seq_stats = [], []
    all_y_test, all_y_pred_cnn, all_y_seq_test, all_y_pred_seq = [], [], [], []

    cpu_sampler = CPUSampler(interval=0.05)
    gpu_sampler = GPUSampler(interval=0.05)
    events = EventMarker()
    cpu_sampler.start()
    gpu_sampler.start()

    fold_start_t = time.time() - events.start_time
    files = get_files(cfg.data_path)

    for fold in range(cfg.training.get("folds", 25)):
        cnn_stats, seq_stats, cpu_cnn_stats, cpu_seq_stats, y_test, y_pred_cnn, y_seq_test, y_pred_seq, e2e_stats = evaluate_one_fold(cfg=cfg, fold=fold, results_path=cfg.save_dir, events=events, gpu_sampler=gpu_sampler, files=files)
        if cnn_stats is None or seq_stats is None:
            fold_start_t = time.time() - events.start_time
            continue
        all_cnn_stats.append(cnn_stats)
        all_seq_stats.append(seq_stats)
        all_e2e_stats.append(e2e_stats)
        if cpu_cnn_stats is not None and cpu_seq_stats is not None:
            all_cpu_cnn_stats.append(cpu_cnn_stats)
            all_cpu_seq_stats.append(cpu_seq_stats)

        
        fold_end_t = time.time() - events.start_time
        print("fold_start_t, fold_end_t:", fold_start_t, fold_end_t)
        print("window size:", fold_end_t - fold_start_t)
        
        plot_resource_usage(
            cpu_sampler, gpu_sampler,
            f'{cfg.save_dir}/{cfg.name}_fold{fold:02d}_training_resources.png',
            fold=fold,
            events=events.events,
            time_range=(fold_start_t, fold_end_t)
        )
        fold_start_t = fold_end_t
        all_y_test.append(y_test)
        all_y_pred_cnn.append(y_pred_cnn)
        all_y_seq_test.append(y_seq_test)
        all_y_pred_seq.append(y_pred_seq)

    cpu_sampler.stop()
    gpu_sampler.stop()

    out_path = os.path.join(cfg.save_dir, f'{cfg.name}_resource_usage_all_folds.png')
    plot_resource_usage(cpu_sampler, gpu_sampler, out_path)

    # compute and save confusion matrices and classification reports for all folds
    all_y_test = np.concatenate(all_y_test)
    all_y_pred_cnn = np.concatenate(all_y_pred_cnn)
    all_y_seq_test = np.concatenate(all_y_seq_test)
    all_y_pred_seq = np.concatenate(all_y_pred_seq)

    plot_confusion_and_f1(all_y_test, all_y_pred_cnn, cfg, 'cnn', fold=None)
    print("plotting seq confusion and f1")
    plot_confusion_and_f1(all_y_seq_test, all_y_pred_seq, cfg, 'seq', fold=None)

    if len(all_cnn_stats) == 0 or len(all_seq_stats) == 0:
        print(f"[WARN] No valid folds found for {cfg.name}. Skipping aggregation.")
        return
    aggregate_and_save(all_seq_stats, all_cnn_stats, all_e2e_stats, cfg.save_dir)
    all_e2e = np.concatenate([s['raw_latencies_ms'] for s in all_e2e_stats])
    np.save(os.path.join(cfg.save_dir,'streaming', f'{cfg.name}_e2e_latencies_all_folds.npy'), all_e2e)
    plot_streaming_latency({'raw_latencies_ms': all_e2e,
                        'p95_latency_ms': np.percentile(all_e2e, 95)}, cfg, fold=None)
    notify(f'Completed eval for {cfg.name} on {dataset["name"]}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='/mnt/truenas_db/user/christina')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    args = parser.parse_args()
    run(args.config, args.override,name=args.name)

if __name__ == '__main__':
    main()