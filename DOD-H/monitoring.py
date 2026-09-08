import psutil
import threading
import time
import numpy as np
import os
import pynvml
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import threading
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter
from sklearn.metrics import confusion_matrix, classification_report
import threading

class EventMarker:
    def __init__(self, start_time=None):
        self.start_time = start_time or time.time()
        self.events = []  # list of (elapsed_seconds, label)

    def mark(self, label):
        elapsed = time.time() - self.start_time
        self.events.append((elapsed, label))
        print(f"[EVENT] {label} @ {elapsed:.1f}s")
        
class PeriodicPlotter:
    def __init__(self, cpu_sampler, gpu_sampler, out_path, interval_sec=2 * 3600):
        self.cpu_sampler = cpu_sampler
        self.gpu_sampler = gpu_sampler
        self.out_path = out_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.wait(self.interval_sec):
            plot_resource_usage(self.cpu_sampler, self.gpu_sampler, self.out_path)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

class GPUSampler:
    def __init__(self, interval=0.05):
        self.lock = threading.Lock()
        self.interval = interval
        self._util_samples = []
        self._mem_samples = []
        self._timestamps = []
        self._stop = threading.Event()
        self._thread = None
    
    def _run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # single GPU
        start_time = time.time()
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            with self.lock:
                self._util_samples.append(util)
                self._mem_samples.append(mem.used / (1024 ** 2))
                self._timestamps.append(time.time() - start_time)
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def start(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] GPU sampler started with interval: {self.interval} seconds")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def stats(self):
        if not self._util_samples:
            return {'mean': None, 'p95': None, 'max': None,
                    'mem_mean': None, 'mem_max': None}
        import numpy as np
        return {
            'mean': float(np.mean(self._util_samples)),
            'p95': float(np.percentile(self._util_samples, 95)),
            'max': float(np.max(self._util_samples)),
            'mem_mean': float(np.mean(self._mem_samples)),
            'mem_max': float(np.max(self._mem_samples)),
        }

    def history(self):
        """Raw traces for plotting."""
        with self.lock:
            return {
                'timestamps': self._timestamps,
                'util': self._util_samples,
                'mem_mb': self._mem_samples,
            }

class CPUSampler:
    def __init__(self, interval=0.05):
        self.lock = threading.Lock()
        self.interval = interval
        self.samples = []
        self.timestamps = []
        self._stop = threading.Event()

    def start(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] CPU sampler started with interval: {self.interval} seconds")

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            with self.lock:
                self.samples.append(psutil.cpu_percent(interval=None))
                self.timestamps.append(t)
            time.sleep(self.interval)

    def stats(self):
        if not self.samples:
            return {}
        return {
            'mean': float(np.mean(self.samples)),
            'p95':  float(np.percentile(self.samples, 95)),
            'max':  float(np.max(self.samples)),
        }

    def history(self):
        """Raw traces for plotting."""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'util': list(self.samples),
                'mem_mb': [0] * len(self.timestamps),  
            }
    
def plot_cpu_usage_per_fold(cpu_samplers, config_name, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(len(cpu_samplers), 1,
                             figsize=(12, 3 * len(cpu_samplers)),
                             sharex=False)
    if len(cpu_samplers) == 1:
        axes = [axes]

    for fold_idx, sampler in enumerate(cpu_samplers):
        times   = np.array(sampler.timestamps)
        samples = np.array(sampler.samples)
        stats   = sampler.stats()

        ax = axes[fold_idx]
        ax.plot(times, samples, linewidth=0.8, color='steelblue')
        ax.axhline(stats['mean'], color='red',    linestyle='--', linewidth=1, label=f"mean={stats['mean']:.1f}%")
        ax.axhline(stats['p95'],  color='orange', linestyle=':',  linewidth=1, label=f"p95={stats['p95']:.1f}%")
        ax.axhline(stats['max'],  color='black',  linestyle=':',  linewidth=1, label=f"max={stats['max']:.1f}%")
        ax.set_ylabel('CPU %')
        ax.set_ylim(0, 100)
        ax.set_title(f'Fold {fold_idx + 1}')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'CPU Usage During Streaming Evaluation\n{config_name}', fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'{config_name}_cpu_per_fold.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'[INFO] CPU plot saved to {out_path}')

def plot_resource_usage(cpu_sampler, gpu_sampler, out_path, fold=None, events=None, time_range=None, relative=True):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    offset = time_range[0] if (relative and time_range is not None) else 0.0
    if time_range is not None:
        start_time, end_time = time_range
        xlim = (0, end_time - start_time) if relative else (start_time, end_time)
        for ax in axes:
            ax.set_xlim(*xlim)
        if events:
            events = [(t - offset, label) for t, label in events if start_time <= t <= end_time]

    def _aligned(hist, *keys):
        n = min(len(hist[k]) for k in keys)
        return [np.array(hist[k][:n]) for k in keys]
    
    if cpu_sampler is not None:
        cpu_hist = cpu_sampler.history()  
        t, util = _aligned(cpu_hist, 'timestamps', 'util')
        t -= offset
        axes[0].plot(t, util)
        axes[0].set_ylabel('CPU (%)')

    if gpu_sampler is not None:
        gpu_hist = gpu_sampler.history()
        t, util, mem = _aligned(gpu_hist, 'timestamps', 'util', 'mem_mb')
        t -= offset
        axes[1].plot(t, util)
        axes[1].set_ylabel('GPU util (%)')
        axes[2].plot(t, mem)
        axes[2].set_ylabel('GPU mem (MB)')

    if events:
        for ax in axes:
            for t, label in events:
                ax.axvline(t, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        # label only on top subplot to avoid clutter
        for t, label in events:
            axes[0].text(t, axes[0].get_ylim()[1], label,
                        rotation=90, va='top', ha='right', fontsize=7, color='red')


    axes[-1].set_xlabel('Time (s)' + ' — relative to fold start' if relative else '')
    fig.tight_layout()
    fig.suptitle(f'Resource Usage fold {fold}')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def get_confusion_matrix_and_report(y_true, y_pred, fold, type, cfg, num_classes=5):
    if type != 'seq' and type != 'cnn':
        raise ValueError(f"Invalid type: {type}. Must be 'seq' or 'cnn'.")
    results_path = os.path.join('results', cfg.name)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True)
    np.save(os.path.join(results_path, f'{type}_confusion_fold{fold}.npy'), cm)
    np.save(os.path.join(results_path, f'{type}_classification_report_fold{fold}.npy'), report)

    # plot and save cm
    save_path_prefix = os.path.join(cfg.save_dir, f'confusion_fold{fold}')
    cm_color = cm.astype(float).copy()
    cm_color[cm_color == 0] = 0.5 
    annot_labels = cm.astype(str) 
    class_names=['Wake', 'N1', 'N2', 'N3', 'REM']
    blocks = cfg.training['num_blocks'] if 'num_blocks' in cfg.training else '3'
    width = cfg.training['width_mult'] if 'width' in cfg.training else '1.0'
    run = f"{cfg.dataset['name']} | Epoch duration: {cfg.training['epoch_duration']}s | Blocks: {blocks} | Width: {width} | {fold}"
    

    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sns.heatmap(cm_color, annot=annot_labels, fmt='', cmap='BuPu', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=ax, norm=LogNorm(vmin=0.5, vmax=cm.max()))
    ax.set_xlabel('predicted labels')
    ax.set_ylabel('true labels')
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
    cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    fig.suptitle(f'{type.upper()} Confusion Matrix | {run}', fontsize=11, y=0.98)
    fig.savefig(f'{save_path_prefix}_{type}_confusion.png')
    plt.close(fig)

    return cm, report

def plot_confusion_and_f1(y_true, y_pred, cfg, type,fold=None):
    if type != 'seq' and type != 'cnn':
        raise ValueError(f"Invalid type: {type}. Must be 'seq' or 'cnn'.")

    save_path_prefix = os.path.join(cfg.save_dir, f'confusion_fold{'all' if fold is None else fold}')
    class_names=['Wake', 'N1', 'N2', 'N3', 'REM']
    # fold = f"Fold {fold}" if fold is not None else "All Folds"
    blocks = cfg.training['num_blocks'] if 'num_blocks' in cfg.training else '3'
    width = cfg.training['width_mult'] if 'width' in cfg.training else '1.0'
    run = f"{cfg.dataset['name']} | Epoch duration: {cfg.training['epoch_duration']}s | Blocks: {blocks} | Width: {width} | {fold}"

    # compute confusion matrices
    _, report = get_confusion_matrix_and_report(y_true, y_pred, fold, type, cfg, num_classes=5)
    f1s = [report[str(i)]['f1-score'] for i in range(5)]
    fig, ax = plt.subplots()
    normalized_f1 = [f1 * 100 for f1 in f1s]
    ax.bar(class_names, normalized_f1)
    ax.set_ylabel('F1')
    ax.set_ylim(0, 100)
    fig.suptitle(f'{type.upper()} Per-Class F1 | {run}', fontsize=11, y=0.98)
    fig.savefig(f'{save_path_prefix}_{type}_per_class_f1.png')
    plt.close(fig)


if __name__ == "__main__":
    config_name = "configs/ablations_"


