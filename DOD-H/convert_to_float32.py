import numpy as np
import os

harm_data_path = '/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30'

for file in os.listdir(harm_data_path):
    if file.endswith('.npz'):
        file_path = os.path.join(harm_data_path, file)
        with np.load(file_path, allow_pickle=True) as npz_file:
            data = {k: npz_file[k] for k in npz_file.files}
        data['x'] = data['x'].astype(np.float32)

        np.savez(file_path, **data)
