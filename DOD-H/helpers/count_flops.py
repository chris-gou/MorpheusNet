import json

import tensorflow as tf
from nn_model import *
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def get_flops(model, input_shape=None):
    if input_shape is None:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
    
    # Build a concrete function
    @tf.function
    def forward(*args):
        return model(args[0] if len(args) == 1 else list(args), training=False)
    
    batch_input = tf.TensorSpec([1] + list(input_shape[1:]), tf.float32)
    concrete = forward.get_concrete_function(batch_input) # get locked computation graph with fixed input and output
    frozen = convert_variables_to_constants_v2(concrete) # get the information
    
    graph = frozen.graph
    profile = tf.compat.v1.profiler.profile(
        graph,
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    )
    return profile.total_float_ops

def lstm_flops(units, input_dim, seq_len):
    """
    Per LSTM cell: 4 gates, each does:
      - input projection:  input_dim * units
      - hidden projection: units * units
      - bias add:          units
      - activations count as ~1 flop each (sigmoid x3, tanh x2)
    Multiply by 2 for multiply-adds, then by seq_len.
    """
    gate_flops = 4 * (2 * input_dim * units + 2 * units * units + units)
    return gate_flops * seq_len

def dense_flops(input_dim, output_dim):
    return 2 * input_dim * output_dim + output_dim  # matmul + bias

def quadratic_fit():
    x = [0.5, 0.75, 1.0, 1.25]
    x_blocks = [1,2,3,4]
    y_blocks = [6240582.416666667, 13825350.416666666,21410118.416666668,28994886.416666668 ]
    y = [7070534.416666667, 13280326.416666666, 21410118.416666668, 31459910.416666668]
    series = np.polynomial.polynomial.Polynomial.fit(x, y, 2)
    print(series.convert().coef)
    print(stats.linregress(x, y)[2])
    print(np.square(stats.linregress(x, y)[2]))

if __name__ == "__main__":
    quadratic_fit()
    exit(0)
    epoch_durations = [30]
    seq_len = 12
    y_train = [0, 1, 2, 3, 4] * 1000
    # width_multipliers = [0.5, 0.75, 1.0, 1.25]
    blocks = [1, 2, 3,4]
    lstm_units = 32
    dense1_units = 32
    n_classes = 5

    results = {}
    for epoch_duration in epoch_durations:
        for num_blocks in blocks:
            # ResNet FLOPs
            model = separable_resnet((1, epoch_duration*100, 1), 5,
                                    y_train=y_train, bias=False,
                                    blocks=num_blocks,
                                    width_mult=1.0)
            resnet_flops = get_flops(model)

            # cnn_out = model.output_shape[-1]
            gap_layer = next(l for l in model.layers if 'global_average' in l.name.lower())
            cnn_out = model.get_layer(gap_layer.name).output.shape[-1]

            # Seq learner FLOPs (LSTM input_dim = cnn_out)
            flops_lstm   = lstm_flops(lstm_units, cnn_out, seq_len)
            flops_dense1 = dense_flops(lstm_units, dense1_units)
            flops_dense2 = dense_flops(dense1_units, n_classes)
            seq_flops    = flops_lstm + flops_dense1 + flops_dense2

            amortized_total = resnet_flops + seq_flops / seq_len

            # print(f"blocks={num_blocks}, width={1.0:.2f} | "
            #       f"CNN out: {cnn_out} | "
            #       f"ResNet: {resnet_flops/1e6:.3f} MFLOPs | "
            #       f"Seq: {seq_flops/1e3:.2f} kFLOPs | "
            #       f"Amortized total: {amortized_total/1e6:.3f} MFLOPs/epoch"
            #       f" (Epoch duration: {epoch_duration}s)")
            
            results[f"blocks={num_blocks}, width={1.0:.2f}, epoch_duration={epoch_duration}"] = {
                "resnet_flops": resnet_flops,
                "seq_flops": seq_flops,
                "amortized_total": amortized_total,
                "cnn_out": cnn_out,
                "epoch_duration": epoch_duration,
            }

    # for epoch_duration in epoch_durations:
    #     num_blocks = 3 # baseline
    #     for width in width_multipliers:
    #         # ResNet FLOPs
    #         model = separable_resnet((1, epoch_duration*100, 1), 5,
    #                                 y_train=y_train, bias=False,
    #                                 blocks=num_blocks,
    #                                 width_mult=width)
    #         resnet_flops = get_flops(model)

    #         # cnn_out = model.output_shape[-1]
    #         gap_layer = next(l for l in model.layers if 'global_average' in l.name.lower())
    #         cnn_out = model.get_layer(gap_layer.name).output.shape[-1]

    #         # Seq learner FLOPs (LSTM input_dim = cnn_out)
    #         flops_lstm   = lstm_flops(lstm_units, cnn_out, seq_len)
    #         flops_dense1 = dense_flops(lstm_units, dense1_units)
    #         flops_dense2 = dense_flops(dense1_units, n_classes)
    #         seq_flops    = flops_lstm + flops_dense1 + flops_dense2

    #         amortized_total = resnet_flops + seq_flops / seq_len

    #         # print(f"blocks={num_blocks}, width={width:.2f} | "
    #         #       f"CNN out: {cnn_out} | "
    #         #       f"ResNet: {resnet_flops/1e6:.3f} MFLOPs | "
    #         #       f"Seq: {seq_flops/1e3:.2f} kFLOPs | "
    #         #       f"Amortized total: {amortized_total/1e6:.3f} MFLOPs/epoch")
    #         results[f"blocks={num_blocks}, width={width:.2f}, epoch_duration={epoch_duration}"] = {
    #             "resnet_flops": resnet_flops,
    #             "seq_flops": seq_flops,
    #             "amortized_total": amortized_total,
    #             "cnn_out": cnn_out,
    #             "epoch_duration": epoch_duration,
    #         }

    with open("new_flops_results.json", "w") as f:
        json.dump(results, f, indent=4)
