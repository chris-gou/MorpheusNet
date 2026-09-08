import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))  # should print: []

# # load h5
# with tf.device('/CPU:0'):
#     model=tf.keras.models.load_model("/mnt/truenas_db/user/christina/morpheus/SEDF78_paper_original/dreem_seq_model_fold2.h5")
# tflite_converter=tf.lite.TFLiteConverter.from_keras_model(model)

# tflite_converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]
# tflite_converter._experimental_lower_tensor_list_ops = False



# tflite_model = tflite_converter.convert()
# open("/mnt/truenas_db/user/christina/morpheus/SEDF78_paper_original/dreem_seq_model_fold2.tflite", "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_path="/mnt/truenas_db/user/christina/morpheus/SEDF78_paper_original/dreem_seq_model_fold2.tflite")
interpreter.allocate_tensors()
total = 0
for t in interpreter.get_tensor_details():
    detail = interpreter.get_tensor(t['index'])
    size = detail.nbytes
    total += size
    print(f"{t['name']:40s} shape={t['shape']} dtype={t['dtype']} bytes={size}")
print(f"\nSum of tensor bytes: {total} ({total/1024:.2f} KB)")
print(f"File size on disk: {os.path.getsize('/mnt/truenas_db/user/christina/morpheus/SEDF78_paper_original/dreem_seq_model_fold2.tflite')/1024:.2f} KB")