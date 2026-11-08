from yasa import Hypnogram, spindles_detect
import numpy as np
import matplotlib.pyplot as plt
# tr03-0005 
file = np.load('/home/christina/Documents/MorpheusNet/data/npz/C3-M2/C3-M2_tr03-0005.npz')
hyp = file['y']
raw = file['x']
sf = file['fs']
hyp = Hypnogram.from_integers(hyp)
print(hyp.hypno)
include = ["N1", "N2", "N3"]
print(np.isin(hyp.hypno, include).any())
fig, ax = plt.subplots(figsize=(12, 4))
hyp.plot_hypnogram(ax=ax)
# fig.savefig('/home/christina/Documents/MorpheusNet/data/npz/C3-M2/C3-M2_tr03-0005_hypnogram.png', dpi=300)
fig.tight_layout()
sf = float(file['fs'])            # npz stores fs as a 0-d array
data = raw.reshape(-1)            # (790, 3000) -> (2_370_000,)
hypno_up = np.repeat(hyp.as_int(), raw.shape[1])  # one label per sample

assert data.size == hypno_up.size

print(f"Data shape: {data.shape}, Hypnogram shape: {hypno_up.shape}")

# assert raw.shape == hyp.as_int().size, f"Length of raw data and hypnogram do not match: {(raw.shape)} vs {hyp.as_int().size}"
sp = spindles_detect(data, sf=sf, hypno=hypno_up)
print(sp.summary())