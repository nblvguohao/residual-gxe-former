"""NTV3 genomic foundation model embeddings for maize hybrids.

Pipeline:
  1. Extract ±500bp DNA sequences around each SNP from B73 reference genome
  2. Tokenize with NTV3 tokenizer
  3. Run NTV3 v2 50M multi-species to get per-sequence embeddings
  4. For each hybrid, average parental inbred embeddings

Requires:
  - B73 reference genome (FASTA)
  - transformers + accelerate
  - pyfaidx
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REF_GENOME = ROOT / "data/reference/Zm-B73-REFERENCE-NAM-5.0.fa.gz"
VCF_PATH = ROOT / "data/raw/g2f/genotype/inbreds_G2F_2014-2023_437k.vcf"
TRAIT = ROOT / "data/raw/g2f/competition_2024/1_Training_Trait_Data_2014_2023.csv"
OUT_EMB = ROOT / "data/processed/g2f/genotype_embeddings_ntv3.parquet"

FLANK = 500  # bp on each side of SNP
N_MARKERS = 2000  # start with 2000 markers for speed test
EMB_DIM = 512

print("=" * 60)
print("NTV3 Genomic Embeddings for Maize Hybrids")
print("=" * 60)

# ---- 1. Check reference genome ----
if not REF_GENOME.exists():
    print(f"ERROR: Reference genome not found at {REF_GENOME}")
    print("Download from: https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/")
    sys.exit(1)

# ---- 2. Load VCF markers + select subset ----
print("\n[1/5] Selecting markers from VCF...")
t0 = time.time()

# Read VCF header
vcf_samples = []
skip = 0
with open(VCF_PATH, "r") if VCF_PATH.suffix == ".gz" else open(VCF_PATH) as f:
    for line in f:
        if line.startswith("##"): skip += 1
        elif line.startswith("#CHROM"):
            vcf_samples = line.strip().split("\t")[9:]
            skip += 1
            break

# Count markers
with open(VCF_PATH, "r") if VCF_PATH.suffix == ".gz" else open(VCF_PATH) as f:
    for _ in range(skip): next(f)
    total_markers = sum(1 for _ in f)

rng = np.random.default_rng(42)
selected_idx = set(rng.choice(total_markers, min(N_MARKERS, total_markers), replace=False))

print(f"  VCF: {len(vcf_samples)} inbreds, {total_markers} total markers")
print(f"  Selected {len(selected_idx)} markers for NTV3 embedding")

# ---- 3. Extract DNA sequences ----
print("\n[2/5] Extracting DNA sequences from reference genome...")
t0 = time.time()

# Simple fasta reader (no external dependency)
def open_fasta(path):
    """Return dict of {chrom: sequence} from FASTA file (supports .gz)."""
    if str(path).endswith('.gz'):
        import gzip
        f = gzip.open(path, 'rt')
    else:
        f = open(path, 'r')
    seqs = {}
    current_chrom = None
    current_seq = []
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_chrom:
                seqs[current_chrom] = ''.join(current_seq).upper()
            current_chrom = line[1:].split()[0]  # first word after >
            current_seq = []
        elif current_chrom:
            current_seq.append(line)
    if current_chrom:
        seqs[current_chrom] = ''.join(current_seq).upper()
    f.close()
    return seqs

print("  Loading reference genome...")
genome = open_fasta(REF_GENOME)
print(f"  Loaded {len(genome)} chromosomes")
# Show first few chrom names
for i, name in enumerate(sorted(genome.keys())[:5]):
    print(f"    {name}: {len(genome[name]):,} bp")

marker_info = []  # [(chrom, pos, ref, alt, marker_id)]

with open(VCF_PATH, "r") if VCF_PATH.suffix == ".gz" else open(VCF_PATH) as f:
    for _ in range(skip): next(f)
    for li, line in enumerate(f):
        if li not in selected_idx: continue
        parts = line.strip().split("\t")
        if len(parts) < 5: continue
        chrom, pos, mid, ref, alt = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
        marker_info.append((chrom, pos, ref, alt, mid))

print(f"  Extracted {len(marker_info)} marker positions ({time.time()-t0:.1f}s)")

# Extract sequences
sequences = []
valid_markers = []
for chrom, pos, ref, alt, mid in marker_info:
    # Match chromosome name
    chrom_key = None
    for candidate in [chrom, f"chr{chrom}", chrom.replace("S", "chr").replace("chrhr", "chr")]:
        if candidate in genome:
            chrom_key = candidate
            break
    # Also try partial match
    if chrom_key is None:
        for gname in genome:
            if str(chrom) in gname or gname.endswith(str(chrom)):
                chrom_key = gname
                break
    if chrom_key is None:
        continue
    try:
        seq = genome[chrom_key]
        start = max(0, pos - FLANK - 1)
        end = min(len(seq), pos + len(ref) + FLANK - 1)
        context = seq[start:end]
        # Verify REF at expected position
        ref_pos = min(FLANK, pos - start - 1)
        if ref_pos + len(ref) <= len(context) and context[ref_pos:ref_pos+len(ref)].upper() == ref.upper():
            # Replace REF with ALT
            mutated = context[:ref_pos] + alt + context[ref_pos+len(ref):]
            seq_out = mutated[:FLANK*2]
            if len(seq_out) >= 100:  # minimum length
                sequences.append(seq_out)
                valid_markers.append(mid)
    except Exception:
        pass

print(f"  Valid sequences: {len(sequences)}/{len(marker_info)} ({time.time()-t0:.1f}s)")
print(f"  Example sequence length: {len(sequences[0]) if sequences else 'N/A'}")

if len(sequences) < 100:
    print("ERROR: Too few valid sequences. Check chromosome naming.")
    sys.exit(1)

# ---- 4. NTV3 inference ----
print(f"\n[3/5] Running NTV3 inference on {len(sequences)} sequences...")
t0 = time.time()

from transformers import AutoTokenizer, AutoModel
model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

print(f"  Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Process in batches
BATCH = 16
all_embeddings = []
for i in range(0, len(sequences), BATCH):
    batch = sequences[i:i+BATCH]
    tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1000)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    # Mean pool the last hidden state
    hidden = outputs.hidden_states[-1]  # [B, L, D]
    mask = tokens["attention_mask"].unsqueeze(-1)
    mean_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    all_embeddings.append(mean_emb.cpu().numpy())
    if (i//BATCH + 1) % 10 == 0:
        print(f"    {min(i+BATCH, len(sequences))}/{len(sequences)} sequences processed")

embeddings = np.concatenate(all_embeddings, axis=0)
print(f"  Embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

# ---- 5. Build hybrid embeddings ----
print("\n[5/5] Building hybrid embeddings from parental inbreds...")
t0 = time.time()

# Map marker_id → embedding index
marker_to_emb = {mid: i for i, mid in enumerate(valid_markers)}

# Load inbred genotypes for these markers from VCF
# For each inbred, get which allele they have at each marker
# Then compute weighted embedding for each inbred
inbred_ids = {}
inbred_emb = {}

# Parse VCF genotypes for each inbred
with open(VCF_PATH, "r") if VCF_PATH.suffix == ".gz" else open(VCF_PATH) as f:
    for _ in range(skip): next(f)
    for li, line in enumerate(f):
        if li not in selected_idx: continue
        parts = line.strip().split("\t")
        if len(parts) < 10: continue
        mid = parts[2]
        if mid not in marker_to_emb: continue
        emb_idx = marker_to_emb[mid]

        for j, gt_field in enumerate(parts[9:]):
            name = vcf_samples[j]
            gt = gt_field.split(":")[0]
            if gt in ("./.", ".", ""):
                dosage = 0.5  # missing = mean
            else:
                alleles = gt.split("/")
                try:
                    dosage = sum(int(a) for a in alleles if a != ".") / 2.0
                except:
                    dosage = 0.5

            if name not in inbred_emb:
                inbred_emb[name] = np.zeros(len(valid_markers), dtype=np.float32)
                inbred_ids[name] = 0
            # Accumulate embedding weighted by dosage (for haploid ref/alt)
            # Embedding is for reference allele context
            # dosage 0 = homozygous ref → weight 1 for ref emb
            # dosage 1 = homozygous alt → weight 0 for ref emb
            # dosage 0.5 = heterozygous → weight 0.5
            ref_weight = 1.0 - dosage
            inbred_emb[name] += ref_weight * embeddings[emb_idx]

# Normalize by count
for name in inbred_emb:
    inbred_emb[name] /= len(valid_markers)

print(f"  Inbred embeddings: {len(inbred_emb)} inbreds ({time.time()-t0:.1f}s)")

# Load hybrid parent info
trait = pd.read_csv(TRAIT)
hybrid_info = trait[["Hybrid", "Hybrid_Parent1", "Hybrid_Parent2"]].drop_duplicates(subset=["Hybrid"])
hybrid_info = hybrid_info.dropna(subset=["Hybrid_Parent1", "Hybrid_Parent2"])

hybrid_names = []
hybrid_emb = []

for _, row in hybrid_info.iterrows():
    p1, p2 = row["Hybrid_Parent1"], row["Hybrid_Parent2"]
    e1 = inbred_emb.get(p1)
    e2 = inbred_emb.get(p2)
    if e1 is not None and e2 is not None:
        hybrid_emb.append((e1 + e2) / 2.0)
    elif e1 is not None:
        hybrid_emb.append(e1)
    elif e2 is not None:
        hybrid_emb.append(e2)
    else:
        hybrid_emb.append(np.zeros(len(valid_markers), dtype=np.float32))
    hybrid_names.append(row["Hybrid"])

hybrid_emb = np.array(hybrid_emb)
print(f"  Hybrid embeddings: {hybrid_emb.shape}")

# Save
emb_df = pd.DataFrame(hybrid_emb, index=hybrid_names,
                       columns=[f"ntv3_{i}" for i in range(hybrid_emb.shape[1])])
emb_df.index.name = "genotype_id"
emb_df = emb_df.reset_index()

from residual_gxe.data.loaders import write_table
write_table(emb_df, OUT_EMB)
print(f"\nSaved: {OUT_EMB}")
print(f"Total time: {time.time()-t0:.1f}s")
