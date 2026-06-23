from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from residual_gxe.data.schema import validate_phenotype_table

CATEGORIES = ["phenotype", "genotype", "environment", "weather_daily", "soil"]

# G2F native column name mappings for phenotype
YIELD_COLUMN_CANDIDATES = [
    "Grain Yield (bu/A)",
    "Yield_Mg_ha",
    "Yield_kg_ha",
    "Grain Yield [bu/A]",
    "Yield [bu/A]",
    "Yield [Mg/ha]",
]

PLANTING_DATE_CANDIDATES = ["Date Plot Planted [MM/DD/YY]", "Date Planted [MM/DD/YY]", "Planting Date"]
HARVEST_DATE_CANDIDATES = ["Date Plot Harvested [MM/DD/YY]", "Date Harvested [MM/DD/YY]", "Harvest Date"]


def _read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "latin-1", "cp1252", "ISO-8859-1"]:
        try:
            sep = _sniff_csv_delimiter(path, encoding)
            return pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Final fallback: read as latin-1 unconditionally
    sep = _sniff_csv_delimiter(path, "latin-1")
    return pd.read_csv(path, encoding="latin-1", sep=sep, low_memory=False)


def _sniff_csv_delimiter(path: Path, encoding: str) -> str:
    try:
        with path.open("r", encoding=encoding) as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def _is_native_year_layout(raw_dir: Path) -> bool:
    for child in raw_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 4:
            return True
    return False


def _parse_date_mixed(date_val) -> str | None:
    if pd.isna(date_val):
        return None
    if isinstance(date_val, (pd.Timestamp,)):
        return date_val.strftime("%Y-%m-%d")
    s = str(date_val).strip()
    for fmt in ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%m-%d-%Y", "%m-%d-%y"]:
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return s


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_phenotype_native(raw_dir: Path, target_trait: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings: list[str] = []
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for fname in sorted(year_dir.glob("*phenotypic_clean_data*.csv")):
            df = _read_csv_robust(fname)
            yield_col = _find_column(df, YIELD_COLUMN_CANDIDATES)
            if yield_col is None:
                warnings.append(f"{fname}: no recognised yield column; candidates={YIELD_COLUMN_CANDIDATES}")
                continue
            plant_col = _find_column(df, PLANTING_DATE_CANDIDATES)
            harvest_col = _find_column(df, HARVEST_DATE_CANDIDATES)

            pedigree_col = "Pedigree" if "Pedigree" in df.columns else None
            if pedigree_col is None:
                warnings.append(f"{fname}: no Pedigree column found")
                continue

            sample_id_col = "Plot_ID" if "Plot_ID" in df.columns else None
            if sample_id_col is None:
                sample_id_col = "Sample_ID" if "Sample_ID" in df.columns else None

            year_val = df["Year"].iloc[0] if "Year" in df.columns else int(year_dir.name)

            location_col = "Field-Location" if "Field-Location" in df.columns else None
            if location_col is None:
                location_col = "Field_Location" if "Field_Location" in df.columns else None
            if location_col is None:
                warnings.append(f"{fname}: no location column found")
                continue

            state_col = "State" if "State" in df.columns else None
            city_col = "City" if "City" in df.columns else None

            rep_col = "Replicate" if "Replicate" in df.columns else None
            block_col = "Block" if "Block" in df.columns else None

            for _, row in df.iterrows():
                planting_date = _parse_date_mixed(row[plant_col]) if plant_col else None
                harvest_date = _parse_date_mixed(row[harvest_col]) if harvest_col else None
                env_id = f"{year_val}_{row[location_col]}"

                rows.append({
                    "sample_id": str(row[sample_id_col]) if sample_id_col else f"{year_val}_{row[location_col]}_{row[pedigree_col]}_{row.get(rep_col, '')}",
                    "genotype_id": str(row[pedigree_col]),
                    "environment_id": env_id,
                    "year": int(year_val),
                    "location_id": str(row[location_col]),
                    "trait_id": target_trait,
                    "trait_name": target_trait,
                    "trait_family": "yield",
                    "phenotype_value": float(row[yield_col]) if pd.notna(row[yield_col]) else None,
                    "phenotype_unit": "bu/ac",
                    "replicate_id": str(int(row[rep_col])) if rep_col and pd.notna(row.get(rep_col)) else None,
                    "block_id": str(int(row[block_col])) if block_col and pd.notna(row.get(block_col)) else None,
                    "trial_id": str(row.get("Experiment", "")) if "Experiment" in df.columns else None,
                    "source_dataset": "g2f",
                    "planting_date": planting_date,
                    "harvest_date": harvest_date,
                    "state": str(row[state_col]) if state_col and pd.notna(row.get(state_col)) else None,
                    "city": str(row[city_col]) if city_col and pd.notna(row.get(city_col)) else None,
                })

    pheno = pd.DataFrame(rows)
    pheno = pheno.dropna(subset=["sample_id", "genotype_id", "environment_id", "phenotype_value"])
    return pheno.reset_index(drop=True), warnings


def _make_environment_native(raw_dir: Path) -> pd.DataFrame:
    env_rows: dict[str, dict] = {}
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year_val = int(year_dir.name)
        field_meta_files = list(year_dir.glob("*field_metadata*.csv"))
        meta = None
        if field_meta_files:
            meta = _read_csv_robust(field_meta_files[0])

        for fname in sorted(year_dir.glob("*phenotypic_clean_data*.csv")):
            df = _read_csv_robust(fname)
            location_col = "Field-Location" if "Field-Location" in df.columns else "Field_Location"
            if location_col not in df.columns:
                continue
            locs = df[location_col].dropna().unique()
            for loc in locs:
                env_id = f"{year_val}_{loc}"
                if env_id in env_rows:
                    continue

                state = None
                city = None
                lat = None
                lon = None
                altitude = None
                if meta is not None:
                    meta_loc_col = None
                    for mc in ["Experiment_Code", "Field_Location", "Field-Location", "Location"]:
                        if mc in meta.columns:
                            meta_loc_col = mc
                            break
                    if meta_loc_col:
                        meta_match = meta[meta[meta_loc_col].astype(str).str.strip() == str(loc).strip()]
                        if len(meta_match) > 0:
                            mr = meta_match.iloc[0]
                            for lat_col in ["Weather_Station_Latitude (in decimal numbers NOT DMS)", "Latitude", "lat"]:
                                if lat_col in meta.columns:
                                    lat = float(mr[lat_col]) if pd.notna(mr[lat_col]) else None
                                    break
                            for lon_col in ["Weather_Station_Longitude (in decimal numbers NOT DMS)", "Longitude", "lon"]:
                                if lon_col in meta.columns:
                                    lon = float(mr[lon_col]) if pd.notna(mr[lon_col]) else None
                                    break
                            for alt_col in ["Elevation (ft above sea level)", "Elevation", "altitude"]:
                                if alt_col in meta.columns:
                                    altitude = float(mr[alt_col]) if pd.notna(mr[alt_col]) else None
                                    break

                if "State" in df.columns:
                    state_vals = df.loc[df[location_col] == loc, "State"].dropna()
                    if len(state_vals) > 0:
                        state = str(state_vals.iloc[0])
                if "City" in df.columns:
                    city_vals = df.loc[df[location_col] == loc, "City"].dropna()
                    if len(city_vals) > 0:
                        city = str(city_vals.iloc[0])

                plant_col = _find_column(df, PLANTING_DATE_CANDIDATES)
                plant_date = None
                if plant_col:
                    plant_vals = df.loc[df[location_col] == loc, plant_col].dropna()
                    if len(plant_vals) > 0:
                        plant_date = _parse_date_mixed(plant_vals.iloc[0])

                harvest_col = _find_column(df, HARVEST_DATE_CANDIDATES)
                harv_date = None
                if harvest_col:
                    harv_vals = df.loc[df[location_col] == loc, harvest_col].dropna()
                    if len(harv_vals) > 0:
                        harv_date = _parse_date_mixed(harv_vals.iloc[0])

                notes_parts = []
                if state:
                    notes_parts.append(f"State={state}")
                if city:
                    notes_parts.append(f"City={city}")

                env_rows[env_id] = {
                    "environment_id": env_id,
                    "year": year_val,
                    "location_id": str(loc),
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": altitude,
                    "planting_date": plant_date,
                    "harvest_date": harv_date,
                    "management_notes": "; ".join(notes_parts) if notes_parts else "",
                    "source_dataset": "g2f",
                }
    return pd.DataFrame(list(env_rows.values()))


def _make_weather_native(raw_dir: Path) -> pd.DataFrame:
    weather_rows = []
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for fname in sorted(year_dir.glob("*weather_cleaned*.csv")):
            df = _read_csv_robust(fname)
            loc_col = "Field Location" if "Field Location" in df.columns else None
            if loc_col is None:
                for c in ["Field_Location", "Field-Location", "Location"]:
                    if c in df.columns:
                        loc_col = c
                        break
            if loc_col is None:
                continue

            date_col = "Date_key" if "Date_key" in df.columns else None
            if date_col is None:
                for c in ["Date", "date"]:
                    if c in df.columns:
                        date_col = c
                        break
            if date_col is None:
                continue

            temp_col = "Temperature [C]" if "Temperature [C]" in df.columns else None
            rain_col = "Rainfall [mm]" if "Rainfall [mm]" in df.columns else None
            solar_col = "Solar Radiation [W/m2]" if "Solar Radiation [W/m2]" in df.columns else None
            humid_col = "Relative Humidity [%]" if "Relative Humidity [%]" in df.columns else None
            wind_col = "Wind Speed [m/s]" if "Wind Speed [m/s]" in df.columns else None
            dew_col = "Dew Point [C]" if "Dew Point [C]" in df.columns else None

            year_val = df["Year"].iloc[0] if "Year" in df.columns else int(year_dir.name)

            # Extract date-only key from datetime string for daily grouping
            date_only_col = "_date_only"
            try:
                df[date_only_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            except Exception:
                df[date_only_col] = df[date_col].astype(str).str.split(" ").str[0]

            for env_name in df[loc_col].dropna().unique():
                env_id = f"{year_val}_{env_name}"
                env_df = df[df[loc_col] == env_name].copy()
                daily = env_df.groupby(date_only_col, dropna=False).agg({
                    temp_col: "max" if temp_col else None,
                } if temp_col else {})
                if not daily.empty and temp_col:
                    # Build daily aggregations in steps
                    agg_map = {}
                    if temp_col:
                        agg_map[temp_col] = ["max", "min", "mean"]
                    if rain_col:
                        agg_map[rain_col] = "sum"
                    if solar_col:
                        agg_map[solar_col] = "mean"
                    if humid_col:
                        agg_map[humid_col] = "mean"
                    if wind_col:
                        agg_map[wind_col] = "mean"
                    if dew_col:
                        agg_map[dew_col] = "mean"

                    daily_agg = env_df.groupby(date_only_col, dropna=False).agg(agg_map)
                    daily_agg.columns = ["_".join(c).strip() if isinstance(c, tuple) else c for c in daily_agg.columns]

                    for _, day_row in daily_agg.iterrows():
                        date_str = str(day_row.name)
                        parsed_date = _parse_date_mixed(date_str)

                        tmax = None
                        tmin = None
                        tmean = None
                        if temp_col:
                            tmax_col = f"{temp_col}_max"
                            tmin_col = f"{temp_col}_min"
                            tmean_col = f"{temp_col}_mean"
                            tmax = float(day_row[tmax_col]) if pd.notna(day_row.get(tmax_col)) else None
                            tmin = float(day_row[tmin_col]) if pd.notna(day_row.get(tmin_col)) else None
                            tmean = float(day_row[tmean_col]) if pd.notna(day_row.get(tmean_col)) else None

                        precip = float(day_row[f"{rain_col}_sum"]) if rain_col and f"{rain_col}_sum" in daily_agg.columns and pd.notna(day_row.get(f"{rain_col}_sum")) else None
                        solar = float(day_row[f"{solar_col}_mean"]) if solar_col and f"{solar_col}_mean" in daily_agg.columns and pd.notna(day_row.get(f"{solar_col}_mean")) else None
                        rh = float(day_row[f"{humid_col}_mean"]) if humid_col and f"{humid_col}_mean" in daily_agg.columns and pd.notna(day_row.get(f"{humid_col}_mean")) else None
                        ws = float(day_row[f"{wind_col}_mean"]) if wind_col and f"{wind_col}_mean" in daily_agg.columns and pd.notna(day_row.get(f"{wind_col}_mean")) else None
                        vpd_val = None
                        if tmean is not None and rh is not None:
                            es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
                            ea = es * rh / 100.0
                            vpd_val = round(es - ea, 4)

                        weather_rows.append({
                            "environment_id": env_id,
                            "date": parsed_date if parsed_date else date_str,
                            "day_after_planting": None,
                            "tmax": tmax,
                            "tmin": tmin,
                            "tmean": tmean if tmean is not None else (round((tmax + tmin) / 2, 2) if tmax is not None and tmin is not None else None),
                            "precipitation": precip,
                            "solar_radiation": solar,
                            "relative_humidity": rh,
                            "wind_speed": ws,
                            "vpd": vpd_val,
                            "gdd": round(max(0, (tmax or 0) + (tmin or 0)) / 2 - 10, 2) if tmax is not None and tmin is not None else None,
                        })
    return pd.DataFrame(weather_rows)


def _make_soil_native(raw_dir: Path) -> pd.DataFrame:
    soil_rows = []
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year_val = int(year_dir.name)
        for fname in sorted(year_dir.glob("*soil_data*.csv")):
            df = _read_csv_robust(fname)
            loc_col = "Location" if "Location" in df.columns else None
            if loc_col is None:
                for c in ["Field_Location", "Field-Location", "Field Location"]:
                    if c in df.columns:
                        loc_col = c
                        break
            if loc_col is None:
                continue

            for _, row in df.iterrows():
                env_id = f"{year_val}_{row[loc_col]}"
                ph = float(row["1:1 Soil pH"]) if "1:1 Soil pH" in df.columns and pd.notna(row.get("1:1 Soil pH")) else None
                om = float(row["Organic Matter LOI %"]) if "Organic Matter LOI %" in df.columns and pd.notna(row.get("Organic Matter LOI %")) else None
                sand = float(row["% Sand"]) if "% Sand" in df.columns and pd.notna(row.get("% Sand")) else None
                silt = float(row["% Silt"]) if "% Silt" in df.columns and pd.notna(row.get("% Silt")) else None
                clay = float(row["% Clay"]) if "% Clay" in df.columns and pd.notna(row.get("% Clay")) else None
                cec_val = float(row["CEC/Sum of Cations me/100g"]) if "CEC/Sum of Cations me/100g" in df.columns and pd.notna(row.get("CEC/Sum of Cations me/100g")) else None

                soil_rows.append({
                    "environment_id": env_id,
                    "ph": ph,
                    "organic_matter": om,
                    "sand": sand,
                    "silt": silt,
                    "clay": clay,
                    "cec": cec_val,
                    "water_holding_capacity": None,
                    "soil_depth": None,
                })
    return pd.DataFrame(soil_rows)


def _parse_vcf_genotype(raw_dir: Path, max_markers: int = 10000) -> pd.DataFrame | None:
    vcf_files = list(raw_dir.rglob("*.vcf"))
    if not vcf_files:
        return None
    vcf_path = vcf_files[0]

    # Find the CHROM header line and count skip lines
    skip_lines = 0
    samples = []
    with vcf_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##"):
                skip_lines += 1
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                samples = parts[9:]
                skip_lines += 1
                break

    if not samples:
        return None

    # Use pandas to read the data part efficiently
    vcf_data = pd.read_csv(
        vcf_path, sep="\t", skiprows=skip_lines, header=None,
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples,
        low_memory=False,
    )

    # Subsample markers if too many
    if len(vcf_data) > max_markers:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(len(vcf_data), size=max_markers, replace=False))
        vcf_data = vcf_data.iloc[indices].reset_index(drop=True)

    rows = []
    for _, row in vcf_data.iterrows():
        marker_id = str(row["ID"])
        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        for sample in samples:
            gt_field = str(row[sample]).split(":")[0]
            if gt_field in ("./.", ".", "./.", "", "nan"):
                dosage = np.nan
            else:
                alleles = gt_field.replace("|", "/").split("/")
                try:
                    dosage = sum(int(a) for a in alleles if a != ".")
                except ValueError:
                    dosage = np.nan
            rows.append({
                "genotype_id": sample,
                "marker_id": marker_id,
                "chromosome": chrom,
                "position": pos,
                "allele_dosage": dosage,
            })
    return pd.DataFrame(rows)


def _genotype_wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    marker_cols = [c for c in wide.columns if c != "genotype_id"]
    rows = []
    for _, row in wide.iterrows():
        gid = row["genotype_id"]
        for m in marker_cols:
            rows.append({"genotype_id": gid, "marker_id": m, "chromosome": None, "position": None, "allele_dosage": row[m]})
    return pd.DataFrame(rows)


@dataclass
class G2fPrepareResult:
    ok: bool
    missing_categories: list[str] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)
    audit_path: str = ""


def prepare_g2f(
    raw_dir: str | Path,
    out_dir: str | Path,
    audit_path: str | Path,
    target_trait: str = "grain_yield",
) -> G2fPrepareResult:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    audit_path = Path(audit_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    missing: list[str] = []

    # Detect mode: unified (pre-formatted CSVs) or native (year folders)
    is_native = _is_native_year_layout(raw_dir)

    if is_native:
        # --- NATIVE G2F layout ---
        audit_lines = [
            "# G2F 数据审计报告",
            "",
            f"审计范围：`{raw_dir}`",
            "",
            "模式：native year-folder layout",
            "",
        ]

        # Phenotype
        phenotype, pheno_warnings = _make_phenotype_native(raw_dir, target_trait)
        if len(phenotype) == 0:
            missing.append("phenotype")
            audit_lines.append("- phenotype: 未找到可解析的表型数据")
        else:
            validation = validate_phenotype_table(phenotype)
            if not validation.ok:
                missing.append("phenotype")
                audit_lines.append(f"- phenotype: schema 校验失败 — {validation.missing_columns} {validation.extra_message}")
            else:
                phenotype.to_parquet(out_dir / "phenotype.parquet", index=False)
                output_paths["phenotype"] = str(out_dir / "phenotype.parquet")
                audit_lines.append(f"- phenotype: {len(phenotype)} samples, {phenotype['genotype_id'].nunique()} genotypes, {phenotype['environment_id'].nunique()} environments")
        if pheno_warnings:
            audit_lines.append(f"- phenotype warnings: {len(pheno_warnings)}")
            for w in pheno_warnings[:10]:
                audit_lines.append(f"  - {w}")

        # Environment
        environment = _make_environment_native(raw_dir)
        if len(environment) == 0:
            missing.append("environment")
            audit_lines.append("- environment: 未找到环境元数据")
        else:
            environment.to_parquet(out_dir / "environment.parquet", index=False)
            output_paths["environment"] = str(out_dir / "environment.parquet")
            audit_lines.append(f"- environment: {len(environment)} environments")

        # Weather
        weather = _make_weather_native(raw_dir)
        if len(weather) == 0:
            missing.append("weather_daily")
            audit_lines.append("- weather_daily: 未找到天气数据")
        else:
            weather.to_parquet(out_dir / "weather_daily.parquet", index=False)
            output_paths["weather_daily"] = str(out_dir / "weather_daily.parquet")
            audit_lines.append(f"- weather_daily: {len(weather)} daily records")

        # Soil
        soil = _make_soil_native(raw_dir)
        if len(soil) == 0:
            missing.append("soil")
            audit_lines.append("- soil: 未找到土壤数据")
        else:
            soil.to_parquet(out_dir / "soil.parquet", index=False)
            output_paths["soil"] = str(out_dir / "soil.parquet")
            audit_lines.append(f"- soil: {len(soil)} soil records")

        # Genotype — look for VCF in genotype/ subdirectory
        geno_vcf_dir = raw_dir / "genotype"
        genotype = None
        if geno_vcf_dir.exists():
            genotype = _parse_vcf_genotype(geno_vcf_dir)
        if genotype is None or len(genotype) == 0:
            # Try parent directory
            genotype = _parse_vcf_genotype(raw_dir)
        if genotype is not None and len(genotype) > 0:
            genotype.to_parquet(out_dir / "genotype.parquet", index=False)
            output_paths["genotype"] = str(out_dir / "genotype.parquet")
            manifest_data = {
                "n_markers": int(genotype["marker_id"].nunique()),
                "n_genotypes": int(genotype["genotype_id"].nunique()),
                "chromosomes": sorted([str(x) for x in genotype["chromosome"].dropna().unique()]) if "chromosome" in genotype.columns else [],
            }
            (out_dir / "genotype_manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            audit_lines.append(f"- genotype: {len(genotype)} records, {genotype['genotype_id'].nunique()} genotypes, {genotype['marker_id'].nunique()} markers")
        else:
            missing.append("genotype")
            audit_lines.append("- genotype: 未找到 VCF 基因型文件")

    else:
        # --- UNIFIED layout (pre-formatted CSVs) ---
        audit_lines = [
            "# G2F 数据审计报告",
            "",
            f"审计范围：`{raw_dir}`",
            "",
            "模式：unified pre-formatted CSV layout",
            "",
        ]

        file_map = {
            "phenotype": "phenotype.csv",
            "genotype": "genotype.csv",
            "environment": "environment.csv",
            "weather_daily": "weather_daily.csv",
            "soil": "soil.csv",
        }

        for category, fname in file_map.items():
            fpath = raw_dir / fname
            if not fpath.exists():
                missing.append(category)
                audit_lines.append(f"- {category}: 缺失 (`{fname}` 未找到)")
                continue
            try:
                df = _read_csv_robust(fpath)
            except Exception as e:
                missing.append(category)
                audit_lines.append(f"- {category}: 读取失败 — {e}")
                continue

            if category == "phenotype":
                validation = validate_phenotype_table(df)
                if not validation.ok:
                    raise ValueError(
                        f"phenotype schema validation failed: missing columns={validation.missing_columns}, "
                        f"message={validation.extra_message}"
                    )
                if target_trait not in df["trait_id"].values:
                    # Allow through but warn
                    pass

            out_path = out_dir / f"{category}.parquet"
            df.to_parquet(out_path, index=False)
            output_paths[category] = str(out_path)
            audit_lines.append(f"- {category}: {len(df)} rows, {len(df.columns)} cols")

        # If genotype is wide-format, convert to long; also save a marker manifest
        genotype_path = out_dir / "genotype.parquet"
        if genotype_path.exists() and "genotype" not in missing:
            geno = pd.read_parquet(genotype_path)
            marker_cols = [c for c in geno.columns if c != "genotype_id"]
            if marker_cols and "marker_id" not in geno.columns:
                geno_long = _genotype_wide_to_long(geno)
                geno_long.to_parquet(genotype_path, index=False)

            # Write genotype manifest
            manifest_path = out_dir / "genotype_manifest.json"
            final_geno = pd.read_parquet(genotype_path)
            manifest_data = {
                "n_markers": int(final_geno["marker_id"].nunique()),
                "n_genotypes": int(final_geno["genotype_id"].nunique()),
                "chromosomes": sorted([str(x) for x in final_geno["chromosome"].dropna().unique()]) if "chromosome" in final_geno.columns and not final_geno["chromosome"].isna().all() else [],
            }
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Metadata
    ok = len(missing) == 0
    metadata: dict[str, Any] = {
        "dataset": "g2f",
        "target_trait": target_trait,
        "mode": "native" if is_native else "unified",
        "ok": ok,
        "missing_categories": missing,
        "output_paths": output_paths,
    }
    if "phenotype" in output_paths:
        pheno = pd.read_parquet(output_paths["phenotype"])
        metadata.update({
            "n_samples": int(len(pheno)),
            "n_genotypes": int(pheno["genotype_id"].nunique()),
            "n_environments": int(pheno["environment_id"].nunique()),
            "n_years": int(pheno["year"].nunique()) if "year" in pheno.columns else None,
        })

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths["metadata"] = str(out_dir / "metadata.json")

    manifest = {
        "dataset": "g2f",
        "raw_data_path": str(raw_dir),
        "target_trait": target_trait,
        "n_samples": metadata.get("n_samples"),
        "n_genotypes": metadata.get("n_genotypes"),
        "n_environments": metadata.get("n_environments"),
        "n_years": metadata.get("n_years"),
        "traits": [target_trait],
        "missingness": {
            "genotype": None,
            "weather": None,
            "soil": None,
        },
        "notes": [] if ok else [f"Missing: {missing}"],
    }
    (out_dir / "data_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    output_paths["manifest"] = str(out_dir / "data_manifest.yaml")

    if not ok:
        audit_lines.append("")
        audit_lines.append("## 结论")
        audit_lines.append(f"缺失类型：`{missing}`")
        audit_lines.append("脚本会停止，避免生成不完整数据。")

    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    return G2fPrepareResult(
        ok=ok,
        missing_categories=missing,
        output_paths=output_paths,
        audit_path=str(audit_path),
    )
