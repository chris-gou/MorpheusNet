from sklearn.metrics import confusion_matrix
import os
import numpy as np

def get_confusion_matrix(npz_file):
    if not os.path.exists(npz_file):
        print(f"File {npz_file} does not exist.")
        return None

    data = np.load(npz_file)
    cnn_cm = data['cnn_cm']
    seq_cm = data['seq_cm']
    return cnn_cm, seq_cm

file1 = '../results/PHYSIO_30_3_new_data_new_resnet_seq_set_files/'
file2 = '../results/PHYSIO_30_3_new_data_new_resnet_no_bandpass/'

for fold in range(5):
    if os.path.exists(os.path.join(file1, f'fold_{fold}_cm.npz')):
        cnn_cm1, seq_cm1 = get_confusion_matrix(os.path.join(file1, f'fold_{fold}_cm.npz'))
    else:
        cnn_cm1, seq_cm1 = None, None
    if os.path.exists(os.path.join(file2, f'fold_{fold}_cm.npz')):
        cnn_cm2, seq_cm2 = get_confusion_matrix(os.path.join(file2, f'fold_{fold}_cm.npz'))
    else:   
        cnn_cm2, seq_cm2 = None, None
    print(f"Confusion matrix cnn for fold {fold} in {file1} and {file2}:")
    print(cnn_cm1)
    print(cnn_cm2)

    print(f"Confusion matrix seq for fold {fold} in {file1} and {file2}:")

    print(seq_cm1)
    print(seq_cm2)