from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from residual_gxe.data.loaders import read_table


WEATHER_COLUMN_CANDIDATES = {
    "environment_id": ["environment_id", "yearsite_uid", "env_id", "trial_env", "location_year"],
    "date": ["date", "Date", "DATE", "timestamp", "datetime", "Date_key"],
    "tmax": ["tmax", "TMAX", "max_temperature", "temperature_max", "Maximum temperature", "Temperature max"],
    "tmin": ["tmin", "TMIN", "min_temperature", "temperature_min", "Minimum temperature", "Temperature min"],
    "tmean": ["tmean", "TMEAN", "mean_temperature", "temperature_mean", "Temperature [C]", "temp"],
    "precipitation": ["precipitation", "precip", "rainfall", "Rainfall [mm]", "rain_mm"],
    "solar_radiation": ["solar_radiation", "solar", "radiation", "Solar Radiation [W/m2]"],
    "relative_humidity": ["relative_humidity", "rh", "humidity", "Relative Humidity [%]"],
    "wind_speed": ["wind_speed", "wind", "Wind Speed [m/s]"],
    "vpd": ["vpd", "VPD"],
    "gdd": ["gdd", "GDD"],
}


@dataclass(frozen=True)
class WeatherPrepareResult:
    weather: pd.DataFrame
    manifest: dict[str, Any]


def _first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    lower_map = {c.lower(): c for c in columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def _normalise_weather_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = set(raw.columns)
    mapping: dict[str, str] = {}
    for target, candidates in WEATHER_COLUMN_CANDIDATES.items():
        found = _first_existing(columns, candidates)
        if found is not None:
            mapping[target] = found
    if "environment_id" not in mapping:
        raise ValueError("Weather table must contain an environment_id/yearsite_uid/env_id-like column")
    if "date" not in mapping:
        raise ValueError("Weather table must contain a date-like column")

    out = pd.DataFrame()
    out["environment_id"] = raw[mapping["environment_id"]].astype(str)
    out["date"] = pd.to_datetime(raw[mapping["date"]], errors="coerce").dt.date.astype(str)
    for col in ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity", "wind_speed", "vpd", "gdd"]:
        if col in mapping:
            out[col] = pd.to_numeric(raw[mapping[col]], errors="coerce")
        else:
            out[col] = np.nan
    return out, mapping


def _derive_weather_variables(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    if out["tmean"].isna().all() and not out["tmax"].isna().all() and not out["tmin"].isna().all():
        out["tmean"] = (out["tmax"] + out["tmin"]) / 2.0

    if out["gdd"].isna().all():
        temp_for_gdd = out["tmean"].copy()
        if temp_for_gdd.isna().all() and not out["tmax"].isna().all() and not out["tmin"].isna().all():
            temp_for_gdd = (out["tmax"] + out["tmin"]) / 2.0
        out["gdd"] = np.maximum(0.0, temp_for_gdd - 10.0)

    if out["vpd"].isna().all() and not out["tmean"].isna().all() and not out["relative_humidity"].isna().all():
        es = 0.6108 * np.exp(17.27 * out["tmean"] / (out["tmean"] + 237.3))
        ea = es * out["relative_humidity"] / 100.0
        out["vpd"] = es - ea
    return out


def _daily_aggregate(weather: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "tmax": "max",
        "tmin": "min",
        "tmean": "mean",
        "precipitation": "sum",
        "solar_radiation": "mean",
        "relative_humidity": "mean",
        "wind_speed": "mean",
        "vpd": "mean",
        "gdd": "mean",
    }
    grouped = weather.groupby(["environment_id", "date"], dropna=False).agg(agg).reset_index()
    return grouped


def _add_day_after_planting(weather: pd.DataFrame, environment: pd.DataFrame | None) -> pd.DataFrame:
    out = weather.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if environment is None or "planting_date" not in environment.columns:
        out = out.sort_values(["environment_id", "date"])
        out["day_after_planting"] = out.groupby("environment_id").cumcount()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out

    env = environment[["environment_id", "planting_date"]].copy()
    env["environment_id"] = env["environment_id"].astype(str)
    env["planting_date"] = pd.to_datetime(env["planting_date"], errors="coerce")
    out = out.merge(env, on="environment_id", how="left")
    out["day_after_planting"] = (out["date"] - out["planting_date"]).dt.days
    missing = out["day_after_planting"].isna()
    if missing.any():
        fallback = out.loc[missing].sort_values(["environment_id", "date"]).groupby("environment_id").cumcount()
        out.loc[missing, "day_after_planting"] = fallback.to_numpy()
    out = out.drop(columns=["planting_date"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def prepare_weather_daily(
    raw_weather: pd.DataFrame,
    environment: pd.DataFrame | None = None,
    source_dataset: str = "unknown",
) -> WeatherPrepareResult:
    normalized, mapping = _normalise_weather_columns(raw_weather)
    normalized = normalized.dropna(subset=["environment_id", "date"])
    daily = _daily_aggregate(normalized)
    daily = _derive_weather_variables(daily)
    daily = _add_day_after_planting(daily, environment)
    daily["stage_label"] = pd.NA
    daily = daily[
        [
            "environment_id",
            "date",
            "day_after_planting",
            "stage_label",
            "tmax",
            "tmin",
            "tmean",
            "precipitation",
            "solar_radiation",
            "relative_humidity",
            "wind_speed",
            "vpd",
            "gdd",
        ]
    ].sort_values(["environment_id", "date"]).reset_index(drop=True)

    missingness = {
        col: float(daily[col].isna().mean())
        for col in ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity", "wind_speed", "vpd", "gdd"]
    }
    manifest = {
        "source_dataset": source_dataset,
        "n_raw_rows": int(len(raw_weather)),
        "n_daily_rows": int(len(daily)),
        "n_environments": int(daily["environment_id"].nunique()),
        "column_mapping": mapping,
        "missingness": missingness,
        "notes": [
            "Weather normalization uses user-provided local files only.",
            "day_after_planting is computed from environment.planting_date when available; otherwise environment-wise row order is used.",
        ],
    }
    return WeatherPrepareResult(weather=daily, manifest=manifest)


def read_weather_input(path: str | Path) -> pd.DataFrame:
    return read_table(path)

