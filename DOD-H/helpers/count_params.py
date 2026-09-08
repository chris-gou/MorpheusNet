from nn_model import *
import numpy as np


print("\n=== seq_model ===")
seq_len = 12  
m = seq_model(int(seq_len * 5))
seq_params = m.count_params()
print(f"  params: {m.count_params():,}")
seq_bytes_analytical = seq_params * 4

window_length = 3000 
# params invariant to window length
print("=== separable_resnet ===")
for blocks in [1, 2, 3]:
    for width_mult in [0.5, 0.75, 1.0]:
        model = separable_resnet(input_shape=(1, window_length, 1), num_classes=5, blocks=blocks, width_mult=width_mult)
        cnn_params = model.count_params()
        cnn_bytes_analytical = cnn_params * 4 
        print(f"  blocks={blocks}, width_mult={width_mult}: {model.count_params():,}")
        total_kb = (cnn_bytes_analytical + seq_bytes_analytical) / 1024
        print(f"Analytical total: {total_kb:.2f} KB")



# for window_length in [500, 1000, 3000]:  # 5s, 10s, 30s at 100Hz
#     model = separable_resnet(input_shape=(1, window_length, 1), num_classes=5, blocks=3, width_mult=1.0)
#     print(f"window={window_length}: {model.count_params():,}")