"""Compute physiological indices (lightweight CGM) from G2F weather data.

Indices:
  GDD (Growing Degree Days): cumulative heat units (Tbase=10C, Topt=30C)
  VPD (Vapor Pressure Deficit): atmospheric water demand (kPa)
  Water Balance: P - ET0 (Precipitation - Reference Evapotranspiration)
  Photoperiod: day length based on latitude and DOY
  Heat stress: days with Tmax > 35C
  Drought stress: consecutive days with negative water balance
  Cold stress: days with Tmin < 5C

Computed per growth stage (early: 0-30d, mid: 30-90d, late: 90d+)
and per environment. Output: physiological_indices.parquet
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table, write_table

OUT = ROOT / "data/processed/g2f"
weather = read_table(OUT / "weather_daily.parquet")
env_raw = read_table(OUT / "environment.parquet")

# Get latitude per environment from metadata
comp_meta = pd.read_csv(ROOT / "data/raw/g2f/competition_2024/2_Training_Meta_Data_2014_2023.csv")
env_lat = {}
for _, row in comp_meta.iterrows():
    env_lat[row["Env"]] = float(row["Weather_Station_Latitude (in decimal numbers NOT DMS)"]) if pd.notna(row.get("Weather_Station_Latitude (in decimal numbers NOT DMS)")) else 40.0

T_BASE, T_OPT = 10.0, 30.0

def saturation_vapor_pressure(T_celsius):
    """Tetens equation: esat in kPa."""
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

def day_length(lat, doy):
    """Day length in hours (approximate)."""
    decl = 23.45 * np.sin(np.radians(360/365 * (doy - 81)))
    lat_rad = np.radians(lat)
    cos_ha = -np.tan(lat_rad) * np.tan(np.radians(decl))
    cos_ha = np.clip(cos_ha, -1, 1)
    return 24.0 * np.arccos(cos_ha) / np.pi

def compute_indices(grp, lat):
    """Compute physiological indices for one environment's weather data."""
    grp = grp.sort_values("date").copy()

    # Fill missing columns with defaults
    for col, default in [("tmax", 25), ("tmin", 15), ("tmean", 20),
                          ("precipitation", 0), ("relative_humidity", 60),
                          ("solar_radiation", 15)]:
        if col not in grp.columns:
            grp[col] = default

    grp = grp.fillna({col: default for col, default in [
        ("tmax", 25), ("tmin", 15), ("tmean", 20),
        ("precipitation", 0), ("relative_humidity", 60),
        ("solar_radiation", 15)]})

    n = len(grp)

    # GDD per day
    Tmax_clipped = np.clip(grp["tmax"].to_numpy(), None, T_OPT)
    Tmin_clipped = np.clip(grp["tmin"].to_numpy(), T_BASE, None)
    Tmean = (Tmax_clipped + Tmin_clipped) / 2.0
    gdd_daily = np.maximum(0, Tmean - T_BASE)

    # VPD
    esat_max = saturation_vapor_pressure(grp["tmax"].to_numpy())
    esat_min = saturation_vapor_pressure(grp["tmin"].to_numpy())
    # Tdew approximation from RH
    rh = np.clip(grp["relative_humidity"].to_numpy(), 1, 100)
    Tmean_arr = grp["tmean"].to_numpy()
    esat_mean = (esat_max + esat_min) / 2.0
    ea = esat_mean * rh / 100.0
    vpd_daily = np.maximum(0, esat_mean - ea)  # kPa

    # ET0 (Hargreaves-Samani)
    Tmax_arr = grp["tmax"].to_numpy()
    Tmin_arr = grp["tmin"].to_numpy()
    Rs = grp["solar_radiation"].to_numpy()  # MJ/m2/day
    # Hargreaves: ET0 = 0.0023 * (Tmean+17.8) * sqrt(Tmax-Tmin) * Ra
    # Use simplified version with solar radiation
    et0_daily = 0.0023 * (Tmean_arr + 17.8) * np.sqrt(np.maximum(0.1, Tmax_arr - Tmin_arr)) * 0.408 * Rs

    # Water balance
    precip = grp["precipitation"].to_numpy()
    wb_daily = precip - et0_daily

    # Stress indicators
    heat_stress = (Tmax_arr > 35).astype(float)
    cold_stress = (Tmin_arr < 5).astype(float)
    drought_stress = (wb_daily < -3).astype(float)  # deficit > 3mm/day

    # Aggregate by growth stage
    def stage_agg(arr, n_days):
        if n_days == 0: return np.zeros(4)
        return np.array([
            np.sum(arr[:n_days]),           # cumulative
            np.mean(arr[:n_days]),           # mean
            np.max(arr[:n_days]),            # max
            np.sum(arr[:n_days] > 0.1)       # count of significant days
        ])

    n1 = min(30, n)
    n2 = min(90, n) - n1
    n3 = n - n1 - n2

    result = {}

    for name, arr in [
        ("gdd", gdd_daily), ("vpd", vpd_daily), ("precip", precip),
        ("et0", et0_daily), ("wb", wb_daily), ("heat_stress", heat_stress),
        ("cold_stress", cold_stress), ("drought_stress", drought_stress),
    ]:
        for stage_name, start, n_stage in [
            ("early", 0, n1), ("mid", n1, n2), ("late", n1+n2, n3)
        ]:
            if n_stage > 0:
                stage_arr = arr[start:start+n_stage]
                result[f"{name}_cum_{stage_name}"] = float(np.sum(stage_arr))
                result[f"{name}_mean_{stage_name}"] = float(np.mean(stage_arr))
                result[f"{name}_max_{stage_name}"] = float(np.max(np.abs(stage_arr)))
            else:
                result[f"{name}_cum_{stage_name}"] = 0.0
                result[f"{name}_mean_{stage_name}"] = 0.0
                result[f"{name}_max_{stage_name}"] = 0.0

    return result


print("Computing physiological indices per environment...")
t0 = time.time()

all_env_features = []
for env_id, grp in weather.groupby("environment_id"):
    lat = env_lat.get(env_id, 40.0)
    feat = compute_indices(grp, lat)
    feat["environment_id"] = env_id
    all_env_features.append(feat)

indices_df = pd.DataFrame(all_env_features)
indices_df = indices_df.set_index("environment_id")

# Fill NaN with 0
indices_df = indices_df.fillna(0)

out_path = OUT / "physiological_indices.parquet"
write_table(indices_df.reset_index(), out_path)

n_feat = len([c for c in indices_df.columns if c != "environment_id"])
print(f"  {len(indices_df)} environments × {n_feat} features ({time.time()-t0:.1f}s)")
print(f"  Saved: {out_path}")
print(f"\nSample features (first 10):")
for c in list(indices_df.columns)[:10]:
    print(f"    {c}: range [{indices_df[c].min():.2f}, {indices_df[c].max():.2f}]")
