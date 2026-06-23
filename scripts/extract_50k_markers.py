"""Extract 50K markers from VCF → wide-format genotype matrix (efficient)."""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import write_table

VCF = ROOT / "data/raw/g2f/genotype/inbreds_G2F_2014-2023_437k.vcf"
TRAIT = ROOT / "data/raw/g2f/competition_2024/1_Training_Trait_Data_2014_2023.csv"
OUT = ROOT / "data/processed/g2f"
N_MARKERS = 50000

print(f"Extracting {N_MARKERS} markers from VCF...")
t0 = time.time()

# Step 1: Read VCF header
skip = 0
inbred_names = []
with open(VCF, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("##"):
            skip += 1
        elif line.startswith("#CHROM"):
            inbred_names = line.strip().split("\t")[9:]
            skip += 1
            break
print(f"  Inbreds: {len(inbred_names)}")

# Step 2: Count markers & select
total_markers = sum(1 for _ in open(VCF, "r", encoding="utf-8")) - skip
step = max(1, total_markers // N_MARKERS)
selected = set(range(0, total_markers, step))
if len(selected) > N_MARKERS:
    rng = np.random.default_rng(42)
    selected = set(rng.choice(list(selected), N_MARKERS, replace=False))
selected_sorted = sorted(selected)
print(f"  Markers: {total_markers} total → {len(selected_sorted)} selected (step={step})")

# Step 3: Parse selected markers into numpy
marker_ids = []
G_inbred = []  # will be [n_markers, n_inbreds]

with open(VCF, "r", encoding="utf-8") as f:
    for _ in range(skip): next(f)
    for li, line in enumerate(f):
        if li not in selected_sorted: continue
        parts = line.strip().split("\t")
        if len(parts) < 10: continue
        marker_ids.append(parts[2])
        dosages = np.full(len(inbred_names), np.nan, dtype=np.float32)
        for j, gt_field in enumerate(parts[9:]):
            gt = gt_field.split(":")[0]
            if gt in ("./.", ".", ""): continue
            try:
                dosages[j] = sum(int(a) for a in gt.split("/") if a != ".")
            except ValueError: pass
        G_inbred.append(dosages)
        if len(marker_ids) % 10000 == 0:
            print(f"    {len(marker_ids)} markers...")

G_inbred = np.array(G_inbred, dtype=np.float32)  # [n_markers, n_inbreds]
# Fill NaN with column mean
col_means = np.nanmean(G_inbred, axis=0)
nan_mask = np.isnan(G_inbred)
G_inbred[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
print(f"  Inbred matrix: {G_inbred.shape} ({time.time()-t0:.1f}s)")

# Step 4: Load parent info
trait = pd.read_csv(TRAIT)
hybrid_info = trait[["Hybrid", "Hybrid_Parent1", "Hybrid_Parent2"]].drop_duplicates(subset=["Hybrid"])
hybrid_info = hybrid_info.dropna(subset=["Hybrid_Parent1", "Hybrid_Parent2"])
hybrid_names = hybrid_info["Hybrid"].tolist()
inbred_to_idx = {name: i for i, name in enumerate(inbred_names)}
print(f"  Hybrids: {len(hybrid_names)}")

# Step 5: Build hybrid genotype matrix
G_hybrid = np.zeros((len(hybrid_names), len(marker_ids)), dtype=np.float32)
missing = 0
for i, (_, row) in enumerate(hybrid_info.iterrows()):
    i1 = inbred_to_idx.get(row["Hybrid_Parent1"])
    i2 = inbred_to_idx.get(row["Hybrid_Parent2"])
    if i1 is not None and i2 is not None:
        G_hybrid[i] = (G_inbred[:, i1] + G_inbred[:, i2]) / 2.0
    elif i1 is not None:
        G_hybrid[i] = G_inbred[:, i1]
    elif i2 is not None:
        G_hybrid[i] = G_inbred[:, i2]
    else:
        missing += 1
print(f"  Hybrid matrix: {G_hybrid.shape} ({time.time()-t0:.1f}s, missing={missing})")

# Step 6: Save as wide parquet (much faster!)
df_wide = pd.DataFrame(G_hybrid, index=hybrid_names, columns=marker_ids)
df_wide.index.name = "genotype_id"
df_wide = df_wide.reset_index()
write_table(df_wide, OUT / "genotype_50k_wide.parquet")

n_m = len(marker_ids)
print(f"\nDone! Saved {len(hybrid_names)} hybrids × {n_m} markers to genotype_50k_wide.parquet")
print(f"Total time: {time.time()-t0:.1f}s")
