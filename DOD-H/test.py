import os
import json
import argparse
import numpy as np
import tensorflow as tf
from utils import *
from monitoring import *
import time
from pyutils.helpers import notify
import psutil
from configuration import Configuration
from sklearn.metrics import confusion_matrix, classification_report
import subprocess
from benchmark_inference import *
from matplotlib import pyplot as plt

SEQ_LEN=12

# static measurements
def calculate_streaming_latency(interpreter, x_test, window_length, cpu_window=100, gpu=None):
    input_details = interpreter.get_input_details() # [batch_size, context length, samples per epoch, channel count (single eeg)]
    output_details = interpreter.get_output_details()
    print(f"input_details: {input_details}, output_details: {output_details}")
    input_scale, input_zero_point = input_details[0]['quantization']

    latencies = []
    cpu_usage = []
    y_pred = []

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # warm-up
    cpu_count = psutil.cpu_count(logical=False) 
    print(f"cpu_count={psutil.cpu_count()}, sample cpu%={proc.cpu_percent(interval=0.1)}")

    # warmup
    for _ in range(10):
        sample = x_test[0].reshape((1, 1, window_length, 1))
        quantized = (sample / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized)
        interpreter.invoke()


    for i in range(0, len(x_test), cpu_window):
        batch = x_test[i:i+cpu_window]
        cpu_start = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
        batch_latencies = []
        for sample in batch:
            sample = sample.reshape((1, 1, window_length, 1))
            quantized = (sample / input_scale + input_zero_point).astype(np.int8)

            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], quantized)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_index = output_details[0]["index"]
            
            # Get the output tensor
            output_data = interpreter.get_tensor(output_index)
            output_details = interpreter.get_output_details()
            output_scale, output_zero_point = output_details[0]['quantization']

            dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale

            y_pred.append(int(np.argmax(dequantized_output)))
            end_time = time.perf_counter()
            batch_latencies.append((end_time - start_time) * 1000)
        
        cpu_after = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
        cpu_win = (cpu_start + cpu_after) / 2  # average CPU usage during the batch
        for lat in batch_latencies:
            latencies.append(lat)
            cpu_usage.append(cpu_win)
            # cpu_usage.append((np.array(cpu_before) + np.array(cpu_after)) / 2)
            # cpu_usage.append(np.mean([cpu_after], axis=0))bu
        # cpu_usage.append(cpu_after)
        # latencies.append((end_time - start_time) * 1000)

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

def measure_seq_learner_latency(model_path, x_seq_arr, n_timestamps=60, n_features=1, n_samples=1000, warmup=10, events=None, gpu_sampler=None):
    seq_model= tf.keras.models.load_model(model_path)
    hidden_size = seq_model.input_shape # (None, past epochs, feature per epoch)
    n_timestamps = hidden_size[1] if len(hidden_size) > 1 else n_timestamps
    n_features = hidden_size[2] if len(hidden_size) > 2 else n_features
    cpu_usage = []
    y_pred = []

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # warm-up
    cpu_count = psutil.cpu_count(logical=False) 
    # warmup
    for i in range(min(warmup, len(x_seq_arr))):
        seq_model(x_seq_arr[i:i+1], training=False)
    
    latencies, timestamps = [], []
    baseline_cpu = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
    gpu_sampler_baseline = gpu_sampler.stats()['mean']

    for i in range(len(x_seq_arr)):
        t_wall = time.time() - gpu_sampler._t0 # align with GPU time
        start_time = time.perf_counter()
        cpu_start = proc.cpu_percent(interval=None) / cpu_count

        out = seq_model(x_seq_arr[i:i+1], training=False)

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        timestamps.append(t_wall)
        cpu_after = proc.cpu_percent(interval=None) / cpu_count  # get average CPU usage across all cores
        cpu_win = (cpu_start + cpu_after) / 2  # average CPU usage during the batch
        cpu_usage.append(cpu_win)

        y_pred.append(int(np.argmax(out.numpy()[0])))

    events.mark(f'seq latency measured for {len(x_seq_arr)} samples')
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

# ===== FIFO e2e streaming latency

def make_cnn_predict_fn(interpreter, input_details, output_details):
    """Wraps the TFLite CNN interpreter as a single predict_fn.
    Accepts a raw epoch of shape (samples_per_epoch, 1)."""
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    def cnn_predict_fn(x_raw):
        x = x_raw.reshape(1, 1, -1, 1)
        x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)
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
def evaluate_one_fold(cfg, fold, model_files_path, results_path, events, gpu_sampler):
    dataset = cfg.dataset
    training_params = cfg.training
    data_path = cfg.data_path
    window_length = cfg.window_length

    tflite_path = os.path.join('results', cfg.name+"_new_data", f'cnn_full_int_fold{fold}.tflite')
    output_path = os.path.join('results', cfg.name+"_new_data")
    if not os.path.exists(tflite_path):
        print(f"[SKIP] No TFLite model for fold {fold}: {tflite_path}")
        # check for h5 to convert
        h5_path = os.path.join(output_path, f'{cfg.name}_best_cnn_fold{fold}.h5')
        h5_path = os.path.join(model_files_path, f'ase_model_fold{fold}.h5')
        if os.path.exists(h5_path):
            convert_h5_to_tflite(h5_path, tflite_path)
        else:
            print(f'[SKIP] No TFLite or h5 model for fold {fold}: {tflite_path}')
            return None, None, None, None, None, None, None, None
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # get indices
    test_ind_path = os.path.join(model_files_path, f'test_ind_fold{fold}.npy')
    if test_ind_path is None or not os.path.exists(test_ind_path):
        test_ind_path = os.path.join(model_files_path, f'test_ind_dodh_fold{fold}.npy')
    test_inds = np.load(test_ind_path, allow_pickle=True)
    _, test_idx, _ = get_fold_indices(fold, cfg.training.get("folds"), seed=100, cv_type=cfg.training.get("cv_type", "kfold"), mat_path=cfg.training.get("mat_path", None))
    if (set(test_inds) != set(test_idx)):
        print(f"[WARN] Test indices mismatch for fold {fold}. Using test indices from get_fold_indices.")
        test_inds = test_idx

    events.mark(f'cnn set')
    x_test, y_test = create_set(test_inds, dataset['eeg_channel'], data_path,
                            training_params, cfg)
    events.mark(f'seq set')
    x_test_seq, y_seq_test = create_seq_sets(test_inds, interpreter, cfg)
    x_test_seq = np.reshape(np.array(x_test_seq),(len(x_test_seq),int(SEQ_LEN*5))) # array 

    # static latency and accuracy
    seq_learner_path = os.path.join(model_files_path, f'{cfg.name}_best_seq_fold{fold}.h5')
    events.mark(f'cnn')
    cnn_latencies, cnn_cpu_usage, y_pred_cnn = calculate_streaming_latency(interpreter, x_test, window_length)
    events.mark(f'seq')
    seq_stats = measure_seq_learner_latency(seq_learner_path, x_test_seq, events=events, gpu_sampler=gpu_sampler)

    # save confusion matrix and classification report for both models
    events.mark(f'confusion')
    # CNN
    plot_confusion_and_f1(y_test, y_pred_cnn, cfg, 'cnn', fold=fold)
    # Seq
    plot_confusion_and_f1(y_seq_test, seq_stats['y_pred'], cfg, 'seq', fold=fold)

    # e2e 
    events.mark(f'e2e streaming')
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    seq_model = tf.keras.models.load_model(os.path.join(model_files_path, f'{cfg.name}_best_seq_fold{fold}.h5'))

    # build predict functions as callables
    cnn_predict = make_cnn_predict_fn(interpreter, input_details, output_details)
    seq_predict = make_seq_predict_fn(seq_model)
    # chain cnn and seq with FIFO
    combined_predict_fn, seq_fifo = make_combined_predict_fn(cnn_predict, seq_predict, seq_len=SEQ_LEN)

    # iterate raw epochs one at a time, return latency stats and labels
    e2e_stats, e2e_y_true, e2e_y_pred = measure_e2e_streaming_latency(
        combined_predict_fn, test_inds, cfg
    )
    plot_streaming_latency(e2e_stats, cfg, fold=fold)

    return cnn_latencies, seq_stats, cnn_cpu_usage, seq_stats['cpu_usage'], y_test, y_pred_cnn, y_seq_test, seq_stats['y_pred'], e2e_stats
 

def plot_streaming_latency(stats, cfg, fold=None, alpha=0.1):
    """Histogram of end-to-end per-epoch latency against the real-time
    budget L_p95 <= alpha * T."""
    epoch_duration = cfg.training.get('epoch_duration', 30)
    budget_ms = alpha * epoch_duration * 1000
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(stats['raw_latencies_ms'], bins=50, color='steelblue', alpha=0.8)
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

def run(base_config_path, override_config_path=None):
    cfg = Configuration(base_config_path, override_config_path)
    dataset = cfg.dataset
    cfg.save_dir = os.path.join('results', f"{cfg.name}_new_data", 'streaming')
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}  |  type: {cfg.get("run_type")}  |  epoch: {cfg.training.get("epoch_duration")}s  |  folds: {cfg.training.get("folds")}")
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

    for fold in range(cfg.training.get("folds", 25)):
        cnn_stats, seq_stats, cpu_cnn_stats, cpu_seq_stats, y_test, y_pred_cnn, y_seq_test, y_pred_seq, e2e_stats = evaluate_one_fold(cfg, fold, os.path.join('results', cfg.name), cfg.save_dir, events, gpu_sampler)
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
    notify(f'Completed eval for {cfg.name} on {dataset["name"]}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='/mnt/truenas_db/user/christina')
    args = parser.parse_args()
    run(args.config, args.override)

if __name__ == '__main__':
    main()