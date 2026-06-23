"""Genomic embeddings: PCA on 50K markers → dense embedding vector.

Phase 1: PCA embedding (fast, practical)
Phase 2: NTV3 foundation model embedding (requires reference genome)
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table, write_table

OUT = ROOT / "data/processed/g2f"

# Load 50K wide matrix
print("Loading 50K genotype matrix...")
t0 = time.time()
geno_50k = pd.read_parquet(OUT / "genotype_50k_wide.parquet")
marker_cols = [c for c in geno_50k.columns if c != "genotype_id"]
geno_mat = geno_50k[marker_cols].fillna(0.0).to_numpy(dtype=np.float32)
genotype_ids = geno_50k["genotype_id"].tolist()
print(f"  {geno_mat.shape[0]} genotypes × {geno_mat.shape[1]} markers ({time.time()-t0:.1f}s)")

# PCA to reduce 50K → 512 dimensions
print("Running PCA (50K → 512)...")
t0 = time.time()
pca = PCA(n_components=512, random_state=42)
embeddings = pca.fit_transform(geno_mat)
var_explained = pca.explained_variance_ratio_.sum()
print(f"  512 PCs explain {var_explained:.1%} of variance ({time.time()-t0:.1f}s)")
print(f"  Embeddings shape: {embeddings.shape}")

# Save embeddings as parquet
emb_df = pd.DataFrame(embeddings, index=genotype_ids,
                       columns=[f"emb_{i}" for i in range(512)])
emb_df.index.name = "genotype_id"
emb_df = emb_df.reset_index()
OUT_EMB = OUT / "genotype_embeddings_pca512.parquet"
write_table(emb_df, OUT_EMB)
print(f"  Saved: {OUT_EMB}")

# Also test: does embedding similarity reflect genetic relationships?
from sklearn.metrics.pairwise import cosine_similarity
# Take first 100 genotypes
subset = embeddings[:100]
sim = cosine_similarity(subset)
# Check if self-similarity is highest
diag_mean = np.diag(sim).mean()
off_diag_mean = (sim.sum() - np.diag(sim).sum()) / (100 * 99)
print(f"  Self-similarity: {diag_mean:.3f}, Cross-similarity: {off_diag_mean:.3f}")

# Also check: embedding similarity vs original marker similarity
# Top 10 most similar pairs from embeddings
flat_indices = np.triu_indices_from(sim, k=1)
top_n = 10
top_indices = np.argsort(sim[flat_indices])[-top_n:][::-1]
print(f"\n  Top {top_n} most similar genotype pairs (by PCA embedding):")
for idx in top_indices[:5]:
    i, j = flat_indices[0][idx], flat_indices[1][idx]
    # Original marker similarity
    orig_sim = np.dot(geno_mat[i], geno_mat[j]) / (np.linalg.norm(geno_mat[i]) * np.linalg.norm(geno_mat[j]))
    print(f"    {genotype_ids[i]:25s} - {genotype_ids[j]:25s}: emb_sim={sim[i,j]:.3f}  orig_sim={orig_sim:.3f}")

print("\nDone. PCA embeddings ready for SMGP pipeline.")
