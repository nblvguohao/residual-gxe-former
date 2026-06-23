"""Convert G2F 2024 Competition data to standard processed parquet format."""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import write_table

COMP = ROOT / "data" / "raw" / "g2f" / "competition_2024"
OUT = ROOT / "data" / "processed" / "g2f"
OUT.mkdir(parents=True, exist_ok=True)

print("Converting G2F Competition 2024 data to parquet...")

# ============================================================
# 1. GENOTYPE: numerical matrix -> long format parquet
# ============================================================
print("\n[1/4] Converting genotype...")
t0 = time.time()

geno_rows = []
with open(COMP / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt", "r") as f:
    f.readline()  # skip <Numeric>
    header = f.readline().strip().split("\t")
    marker_ids = header[1:]  # S1_1007742, ...

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        hybrid = parts[0]
        dosages = parts[1:]
        for mid, dos in zip(marker_ids, dosages):
            if dos == "NA" or dos == "":
                dosage = np.nan
            else:
                try:
                    dosage = float(dos)
                except ValueError:
                    dosage = np.nan
            geno_rows.append({
                "genotype_id": hybrid,
                "marker_id": mid,
                "chromosome": mid.split("_")[0],  # S1, S2, ..., S10
                "position": 0,
                "allele_dosage": dosage,
            })

df_geno = pd.DataFrame(geno_rows)
# Fill NaN with per-marker mean
marker_means = df_geno.groupby("marker_id")["allele_dosage"].transform("mean")
df_geno["allele_dosage"] = df_geno["allele_dosage"].fillna(marker_means).fillna(0.0)
df_geno["allele_dosage"] = df_geno["allele_dosage"].astype(np.float32)

n_geno = df_geno["genotype_id"].nunique()
n_markers = df_geno["marker_id"].nunique()
print(f"  {n_geno} hybrids x {n_markers} markers, {len(df_geno):,} rows ({time.time()-t0:.1f}s)")
write_table(df_geno, OUT / "genotype.parquet")

# ============================================================
# 2. PHENOTYPE: trait data -> phenotype.parquet
# ============================================================
print("\n[2/4] Converting phenotype...")
t0 = time.time()

trait = pd.read_csv(COMP / "1_Training_Trait_Data_2014_2023.csv")

pheno = pd.DataFrame()
pheno["sample_id"] = trait.apply(
    lambda r: f"{r['Env']}_{r['Hybrid']}_{r['Replicate']}_{r['Plot']}", axis=1
)
pheno["genotype_id"] = trait["Hybrid"]
pheno["environment_id"] = trait["Env"]
pheno["year"] = trait["Year"]
pheno["location_id"] = trait["Field_Location"]
pheno["trait_id"] = "yield"
pheno["trait_name"] = "grain_yield"
pheno["trait_family"] = "yield"
pheno["phenotype_value"] = trait["Yield_Mg_ha"].astype(np.float32)
pheno["phenotype_unit"] = "Mg/ha"
pheno["replicate_id"] = trait["Replicate"]
pheno["block_id"] = trait["Block"]
pheno["plot_id"] = trait["Plot"]
pheno["source_dataset"] = "g2f_competition_2024"
pheno["grain_moisture"] = trait.get("Grain_Moisture", np.nan)

# Drop samples with missing yield
pheno = pheno.dropna(subset=["phenotype_value"])

# Uniqueness check
dupes = pheno["sample_id"].duplicated()
if dupes.any():
    print(f"  Dropping {dupes.sum()} duplicate sample_ids")
    pheno = pheno.drop_duplicates(subset=["sample_id"], keep="first")

print(f"  {len(pheno)} samples, {pheno['environment_id'].nunique()} envs, {pheno['genotype_id'].nunique()} genotypes")
print(f"  Years: {sorted(pheno['year'].unique())}")
print(f"  Yield: {pheno['phenotype_value'].min():.2f} - {pheno['phenotype_value'].max():.2f} Mg/ha (mean={pheno['phenotype_value'].mean():.2f})")
write_table(pheno, OUT / "phenotype.parquet")
print(f"  Wrote: phenotype.parquet ({time.time()-t0:.1f}s)")

# ============================================================
# 3. ENVIRONMENT
# ============================================================
print("\n[3/4] Building environment...")
t0 = time.time()

meta = pd.read_csv(COMP / "2_Training_Meta_Data_2014_2023.csv")
ec_data = pd.read_csv(COMP / "6_Training_EC_Data_2014_2023.csv")

env = pd.DataFrame()
env["environment_id"] = meta["Env"]
env["year"] = meta["Year"]
env["location_id"] = meta["Field"] if "Field" in meta.columns else meta["Env"]
env["source_dataset"] = "g2f_competition_2024"

# Add metadata columns
for col in meta.columns:
    if col not in ("Env", "Year", "Field"):
        env[col.lower().replace(" ", "_").replace("(", "").replace(")", "")] = meta[col]

# Merge EC data
if ec_data is not None and len(ec_data) > 0:
    env = env.merge(ec_data, left_on="environment_id", right_on="Env", how="left", suffixes=("", "_ec"))

print(f"  {len(env)} environments ({time.time()-t0:.1f}s)")
write_table(env, OUT / "environment.parquet")

# ============================================================
# 4. WEATHER: daily data -> weather_daily.parquet
# ============================================================
print("\n[4/4] Converting weather...")
t0 = time.time()

weather = pd.read_csv(COMP / "4_Training_Weather_Data_2014_2023_full_year.csv")

# Parse date
weather["date"] = pd.to_datetime(weather["Date"], errors="coerce")
weather = weather.dropna(subset=["date"])

wd = pd.DataFrame()
wd["environment_id"] = weather["Env"]
wd["date"] = weather["date"]
wd["tmax"] = pd.to_numeric(weather.get("T2M_MAX", 0), errors="coerce").fillna(0)
wd["tmin"] = pd.to_numeric(weather.get("T2M_MIN", 0), errors="coerce").fillna(0)
wd["tmean"] = pd.to_numeric(weather.get("T2M", 0), errors="coerce").fillna(0)
wd["precipitation"] = pd.to_numeric(weather.get("PRECTOTCORR", 0), errors="coerce").fillna(0)
wd["solar_radiation"] = pd.to_numeric(weather.get("ALLSKY_SFC_SW_DWN", 0), errors="coerce").fillna(0)
wd["relative_humidity"] = pd.to_numeric(weather.get("RH2M", 0), errors="coerce").fillna(0)

wd = wd.drop_duplicates(subset=["environment_id", "date"], keep="first")

print(f"  {len(wd)} daily records, {wd['environment_id'].nunique()} envs ({time.time()-t0:.1f}s)")
write_table(wd, OUT / "weather_daily.parquet")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"Pipeline complete!")
print(f"  Phenotype:   {len(pheno):,} samples ({len(pheno['year'].unique())} years)")
print(f"  Genotype:    {n_geno} hybrids x {n_markers} markers")
print(f"  Environment: {len(env)} envs")
print(f"  Weather:     {len(wd):,} daily records")
print(f"  Output:      {OUT}")
print(f"{'='*60}")
