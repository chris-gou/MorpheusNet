import numpy as np, glob, os

def scan_subject_stats(path, pattern="*.npz", n=50):
    files = sorted(glob.glob(os.path.join(path, pattern)))[:n]
    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            x = d['x']
            rows.append((os.path.basename(f), x.mean(), x.std(), np.abs(x).max()))
    return rows

local_stats = scan_subject_stats("/home/christina/Documents/MorpheusNet/harm/PHYSIO2018_harmonized/npz/30")
mnt_stats   = scan_subject_stats("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2", "C3-M2_*.npz")

for tag, stats in [("local", local_stats), ("mnt", mnt_stats)]:
    stds = [s[2] for s in stats]
    maxs = [s[3] for s in stats]
    print(f"{tag}: std range [{min(stds):.2f}, {max(stds):.2f}], max-abs range [{min(maxs):.1f}, {max(maxs):.1f}]")
    # flag outliers
    outliers = [s for s in stats if s[3] > 5 * np.median(maxs)]
    print(f"{tag}: {len(outliers)} outlier subjects (max-abs > 5x median):", [o[0] for o in outliers[:5]])

def full_stats(stats):
    stds = np.array([s[2] for s in stats])
    maxs = np.array([s[3] for s in stats])
    print(f"  std:  p10={np.percentile(stds,10):.2f} p50={np.percentile(stds,50):.2f} p90={np.percentile(stds,90):.2f}")
    print(f"  maxabs: p10={np.percentile(maxs,10):.1f} p50={np.percentile(maxs,50):.1f} p90={np.percentile(maxs,90):.1f}")

print("local:"); full_stats(local_stats)
print("mnt:");   full_stats(mnt_stats)


for f in sorted(glob.glob("/mnt/truenas_db/user/christina/PHYSIO2018_harmonized/100Hz_filt/npz/C3-M2/*.npz"))[:10]:
    with np.load(f, allow_pickle=True) as d:
        print(os.path.basename(f), d['unit'] if 'unit' in d.files else 'N/A')


