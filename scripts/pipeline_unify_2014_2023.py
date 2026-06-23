"""Pipeline: unify 2014-2023 G2F raw CSVs into standard processed parquet format."""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import write_table

RAW = ROOT / "data" / "raw" / "g2f"
OUT = ROOT / "data" / "processed" / "g2f"
OUT.mkdir(parents=True, exist_ok=True)

MAX_MARKERS = 50000  # 50K markers from VCF
TRAIT = "grain_yield"
TRAIT_COL = "Grain Yield [bu/A]"

print(f"Pipeline: 2014-2023 G2F -> {OUT}")
print(f"Max markers: {MAX_MARKERS}")

# ============================================================
# 1. GENOTYPE: Parse VCF (50K markers)
# ============================================================
print("\n[1/5] Parsing genotype VCF...")
t0 = time.time()

VCF = RAW / "genotype" / "inbreds_G2F_2014-2023_437k.vcf"
if not VCF.exists():
    raise FileNotFoundError(f"VCF not found: {VCF}")

import gzip

# Read VCF header (skip ## lines, read #CHROM header)
skip = 0
samples = []
with open(VCF, "r", encoding="utf-8") as fi:
    for line in fi:
        if line.startswith("##"):
            skip += 1
        elif line.startswith("#CHROM"):
            parts = line.strip().split("\t")
            samples = parts[9:]
            skip += 1
            break

print(f"  {len(samples)} samples, skipping {skip} header lines")

# Read markers: subsample with regular spacing for speed
# First count total markers via quick line count
total_lines = sum(1 for _ in open(VCF, "r", encoding="utf-8")) - skip
step = max(1, total_lines // MAX_MARKERS)
print(f"  Total markers: {total_lines}, step: {step} (target ~{MAX_MARKERS})")

marker_data = []
rng = np.random.default_rng(42)
chunk_size = 50000

# Use random subsampling for representative coverage
marker_indices = set(rng.choice(total_lines, size=min(MAX_MARKERS, total_lines), replace=False))
marker_indices_sorted = sorted(marker_indices)

with open(VCF, "r", encoding="utf-8") as fi:
    # Skip header
    for _ in range(skip):
        next(fi)

    for line_idx, line in enumerate(fi):
        if line_idx in marker_indices_sorted:
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            chrom = parts[0]
            pos = int(parts[1])
            marker_id = parts[2]
            # Parse GT field (first element of first colon-separated field)
            genotypes = parts[9:]
            for sample_name, gt_field in zip(samples, genotypes):
                gt = gt_field.split(":")[0]
                if gt in ("./.", ".", ""):
                    dosage = np.nan
                else:
                    alleles = gt.split("/")
                    try:
                        dosage = sum(int(a) for a in alleles if a != ".")
                    except ValueError:
                        dosage = np.nan
                marker_data.append({
                    "genotype_id": sample_name,
                    "marker_id": marker_id,
                    "chromosome": chrom,
                    "position": pos,
                    "allele_dosage": dosage,
                })

    if len(marker_data) > 10000000:
        # Switch to chunked processing if too large
        pass

df_geno = pd.DataFrame(marker_data)
n_markers = df_geno["marker_id"].nunique()
n_genos = df_geno["genotype_id"].nunique()
print(f"  Parsed: {len(df_geno):,} rows, {n_genos} genotypes x {n_markers} markers ({time.time()-t0:.1f}s)")

write_table(df_geno, OUT / "genotype.parquet")
print(f"  Wrote: genotype.parquet")

# ============================================================
# 2. PHENOTYPE: Unify all years
# ============================================================
print("\n[2/5] Unifying phenotype...")
t0 = time.time()

pheno_cols_map = {
    "Year": "year",
    "Field-Location": "location_id",
    "Pedigree": "genotype_id",
    "Grain Yield [bu/A]": "phenotype_value",
    "Grain Moisture [%]": "grain_moisture",
    "Test Weight [lbs]": "test_weight",
    "Replicate": "replicate_id",
    "Block": "block_id",
    "Plot": "plot_id",
    "Date Plot Planted": "planting_date",
    "Date Plot Harvested": "harvest_date",
}
extra_cols = ["State", "City", "Row spacing (in inches)", "Rows per plot", "Plot area (ft2)", "Family", "Tester", "Source"]

all_pheno = []
for yr_dir in sorted(RAW.iterdir()):
    yr_name = yr_dir.name
    if not yr_name.isdigit():
        continue
    year_val = int(yr_name)
    csv_found = None
    for f in sorted(yr_dir.iterdir()):
        fn = f.name.lower()
        if f.suffix == ".csv" and ("pheno" in fn or "hybrid" in fn) and f.stat().st_size > 10000:
            csv_found = f
            break
    if csv_found is None:
        print(f"  {year_val}: SKIP (no phenotype CSV)")
        continue

    try:
        df = pd.read_csv(csv_found, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_found, encoding="latin-1", low_memory=False)

    # Unify columns
    out_cols = {}
    for src, dst in pheno_cols_map.items():
        if src in df.columns:
            out_cols[dst] = df[src]
        elif src.replace("[", "[").replace("]", "]") in [c for c in df.columns]:
            out_cols[dst] = df[src]

    if "phenotype_value" not in out_cols:
        print(f"  {year_val}: SKIP (no yield column)")
        continue

    out = pd.DataFrame(out_cols)
    out["year"] = out.get("year", year_val)
    out["trait_id"] = TRAIT
    out["trait_name"] = TRAIT
    out["source_dataset"] = f"g2f_{year_val}"

    # Build environment_id = year_location
    if "location_id" in out.columns:
        out["environment_id"] = out["year"].astype(str) + "_" + out["location_id"].astype(str)
    else:
        out["environment_id"] = str(year_val) + "_unknown"

    # Build sample_id
    out["sample_id"] = out.apply(
        lambda r: f"{r['environment_id']}_{r['genotype_id']}_{r.get('replicate_id', r.name)}", axis=1
    )

    # Track extra columns
    for c in extra_cols:
        if c in df.columns:
            out[c.lower().replace(" ", "_").replace("(", "").replace(")", "")] = df[c]

    all_pheno.append(out)
    print(f"  {year_val}: {len(out)} samples, {out['environment_id'].nunique()} envs, {out['genotype_id'].nunique()} genotypes")

pheno = pd.concat(all_pheno, ignore_index=True)
pheno["trait_family"] = "yield"
pheno = pheno.drop_duplicates(subset=["sample_id"], keep="first")

print(f"  Total: {len(pheno)} samples, {pheno['environment_id'].nunique()} envs, {pheno['genotype_id'].nunique()} genotypes")
print(f"  Years: {sorted(pheno['year'].unique())}")
print(f"  Yield range: {pheno['phenotype_value'].min():.1f} - {pheno['phenotype_value'].max():.1f} bu/ac")

write_table(pheno, OUT / "phenotype.parquet")
print(f"  Wrote: phenotype.parquet ({time.time()-t0:.1f}s)")

# ============================================================
# 3. ENVIRONMENT: Build from phenotype aggregations
# ============================================================
print("\n[3/5] Building environment table...")
t0 = time.time()

env_rows = []
for env_id, grp in pheno.groupby("environment_id"):
    yr = grp["year"].iloc[0]
    # Extract location from env_id (year_location -> location)
    loc = env_id.replace(f"{yr}_", "", 1) if str(yr) in env_id else env_id
    env_rows.append({
        "environment_id": env_id,
        "year": yr,
        "location_id": str(loc),
        "source_dataset": grp["source_dataset"].iloc[0],
    })

env = pd.DataFrame(env_rows)
write_table(env, OUT / "environment.parquet")
print(f"  {len(env)} environments ({time.time()-t0:.1f}s)")

# ============================================================
# 4. WEATHER: Unify from yearly CSVs
# ============================================================
print("\n[4/5] Unifying weather...")
t0 = time.time()

weather_cols_std = {
    "tmax": ["Tmax", "tmax", "TMAX", "Maximum Temperature", "T2M_MAX"],
    "tmin": ["Tmin", "tmin", "TMIN", "Minimum Temperature", "T2M_MIN"],
    "tmean": ["Tmean", "tmean", "TMEAN", "Mean Temperature"],
    "precipitation": ["Precipitation", "precipitation", "PRECIP", "Rain", "rain", "PRECTOTCORR"],
    "solar_radiation": ["Solar Radiation", "solar_radiation", "Solar", "solar", "ALLSKY_SFC_SW_DWN"],
    "relative_humidity": ["Relative Humidity", "relative_humidity", "RH", "rh", "RH2M"],
}
weather_date_cols = ["Date", "date", "DATE", "Day", "day", "Date_key", "Date_Key"]
weather_loc_cols = ["Field-Location", "Field_Location", "Location", "location", "environment_id"]

all_weather = []
for yr_dir in sorted(RAW.iterdir()):
    yr_name = yr_dir.name
    if not yr_name.isdigit():
        continue
    year_val = int(yr_name)
    csv_found = None
    for f in sorted(yr_dir.iterdir()):
        fn = f.name.lower()
        if "weather" in fn and f.suffix == ".csv" and f.stat().st_size > 50000:
            csv_found = f
            break
    if csv_found is None:
        print(f"  {year_val}: SKIP (no weather)")
        continue

    try:
        df = pd.read_csv(csv_found, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_found, encoding="latin-1", low_memory=False)

    # Find date column
    date_col = None
    for c in weather_date_cols:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        print(f"  {year_val}: SKIP (no date column, cols={list(df.columns)[:10]})")
        continue

    # Find location column
    loc_col = None
    for c in weather_loc_cols:
        if c in df.columns:
            loc_col = c
            break

    # Parse date
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        print(f"  {year_val}: SKIP (can't parse dates)")
        continue

    out = pd.DataFrame()
    out["date"] = dates.dt.date if hasattr(dates, "dt") else dates
    out["year"] = year_val

    # Build environment_id
    if loc_col:
        out["location_id"] = df[loc_col].astype(str)
        out["environment_id"] = str(year_val) + "_" + out["location_id"]
    else:
        out["environment_id"] = str(year_val) + "_unknown"

    # Map weather columns
    for std_name, candidates in weather_cols_std.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found:
            out[std_name] = pd.to_numeric(df[found], errors="coerce")
        else:
            out[std_name] = 0.0

    out = out.dropna(subset=["date"])
    all_weather.append(out)
    print(f"  {year_val}: {len(out)} daily records, cols: {[c for c in weather_cols_std if c in out.columns]}")

weather = pd.concat(all_weather, ignore_index=True)
weather = weather.drop_duplicates(subset=["environment_id", "date"], keep="first")
print(f"  Total: {len(weather)} daily records, {weather['environment_id'].nunique()} envs ({time.time()-t0:.1f}s)")

write_table(weather, OUT / "weather_daily.parquet")
print(f"  Wrote: weather_daily.parquet")

# ============================================================
# 5. Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Pipeline complete!")
print(f"  Phenotype:   {len(pheno):,} samples")
print(f"  Genotype:    {n_genos} genotypes x {n_markers} markers")
print(f"  Environment: {len(env)} envs ({sorted(pheno['year'].unique())})")
print(f"  Weather:     {len(weather):,} daily records")
print(f"{'='*60}")
