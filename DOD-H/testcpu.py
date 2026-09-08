import time
import tensorflow as tf

seq_model= tf.keras.models.load_model('results/PHYSIO_5_1/PHYSIO_5_1_seq_model.h5')

for i in range(5):
    t0 = time.perf_counter()
    _ = seq_model(x_seq_arr[i:i+1], training=False)
    print(f'call {i}: {(time.perf_counter()-t0)*1000:.1f}ms')