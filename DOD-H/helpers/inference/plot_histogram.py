import numpy as np
import matplotlib.pyplot as plt
ALPHA = 0.1
EPOCH_SEC = 30
OUT = "morpheusnet_latency_hist.png"
lat = np.load("../../results/PHYSIO_10_2_og_data_new_resnet_relu/streaming/PHYSIO_10_2_og_data_new_resnet_relu_e2e_latencies_all_folds.npy")
lo, hi = np.percentile(lat, [0.1, 99.9])
print(f"latency range: {lo:.2f} ms - {hi:.2f} ms")

p50, p95, p99 = np.percentile(lat, [50, 95, 99])
budget = ALPHA * EPOCH_SEC * 1000
 
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(lat, bins=np.linspace(lo, hi, 60), color="tab:blue", alpha=0.8)
ax.axvline(p50, color="k", ls="--", lw=1, label=f"$L_{{p50}}$ = {p50:.2f} ms")
ax.axvline(p95, color="tab:red", ls="--", lw=1, label=f"$L_{{p95}}$ = {p95:.2f} ms")
ax.axvline(p99, color="tab:orange", ls="--", lw=1, label=f"$L_{{p99}}$ = {p99:.2f} ms")
# only draw the budget line if it is near the data, otherwise it squashes the plot
if budget < 3 * lat.max():
    ax.axvline(budget, color="tab:green", lw=1.5, label=rf"$\alpha T$ = {budget:.0f} ms")
 
ax.set_xlabel("Inference latency per epoch (ms)")
ax.set_ylabel("Count")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"n={len(lat)}  p50={p50:.2f} ms  p95={p95:.2f} ms  budget={budget:.0f} ms")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(lat, ',', alpha=0.4)
ax.set_xlabel('Epoch index (folds concatenated)')
ax.set_ylabel('Latency (ms)')
fig.savefig('latency_vs_index.png', dpi=150)