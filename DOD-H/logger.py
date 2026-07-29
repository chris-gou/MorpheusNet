import tensorflow as tf
import pynvml
import time
import psutil
import json

class ResourceLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_path, gpu=0):
        super().__init__()
        self.log_path = log_path
        self.records = []
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.records.append({
            "epoch": epoch,
            "epoch_time_s": time.perf_counter() - self._t0,
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_util_percent": gpu_util.gpu,
            "gpu_mem_used_mb": gpu_mem.used / 1e6,
        })

    def on_train_end(self, logs=None):
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)
