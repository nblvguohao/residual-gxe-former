from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from residual_gxe.data.preprocess import select_markers_by_strategy


WEATHER_FEATURE_CANDIDATES = [
    "tmax",
    "tmin",
    "tmean",
    "precipitation",
    "solar_radiation",
    "relative_humidity",
    "vpd",
    "gdd",
]

DEFAULT_STAGE_WINDOWS = [
    ("early", 0, 30),
    ("vegetative", 31, 60),
    ("flowering", 61, 90),
    ("grain_fill", 91, 130),
    ("late", 131, 180),
]
SUMMARY_STATS = ["mean", "min", "max", "sum", "std"]


def _is_numeric_dtype(dtype: Any) -> bool:
    return getattr(dtype, "kind", "") in {"b", "i", "u", "f"}


def _recover_yyyymmdd_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    """Recover dates accidentally parsed as nanoseconds after Unix epoch."""
    if pd.isna(value):
        return pd.NaT
    try:
        ns_value = int(pd.Timestamp(value).value)
    except Exception:
        return pd.NaT
    text = str(ns_value)
    if len(text) != 8:
        return pd.NaT
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def normalize_weather_dates(weather: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a fold-safe, sortable `_weather_date` column.

    Some source files encode dates as YYYYMMDD integers. If such integers were
    parsed by pandas as datetime nanoseconds, they appear as 1970 timestamps.
    This helper recovers the original calendar date without using target data.
    """
    out = weather.copy()
    if "date" not in out.columns:
        out["_weather_date"] = pd.NaT
        return out

    parsed = pd.to_datetime(out["date"], errors="coerce")
    if len(parsed) == 0:
        out["_weather_date"] = parsed
        return out

    years = parsed.dt.year
    suspicious = years.eq(1970) & parsed.notna()
    if suspicious.any():
        recovered = out.loc[suspicious, "date"].map(_recover_yyyymmdd_timestamp)
        parsed.loc[suspicious] = recovered

    if parsed.isna().any():
        numeric = pd.to_numeric(out.loc[parsed.isna(), "date"], errors="coerce")
        numeric_text = numeric.dropna().astype("Int64").astype(str)
        recovered_numeric = pd.to_datetime(numeric_text, format="%Y%m%d", errors="coerce")
        parsed.loc[numeric_text.index] = recovered_numeric

    out["_weather_date"] = parsed
    return out


def build_genotype_wide(genotype: pd.DataFrame | None, max_markers: int | None = None) -> pd.DataFrame | None:
    """Normalize supported genotype formats to one row per genotype."""
    if genotype is None or len(genotype) == 0:
        return None

    if "genotype_id" not in genotype.columns:
        raise ValueError("Genotype table must contain genotype_id")

    if "marker_biallelic_codes" in genotype.columns:
        arrays: list[np.ndarray] = []
        gids: list[str] = []
        for _, row in genotype.iterrows():
            codes = row["marker_biallelic_codes"]
            if codes is None or (isinstance(codes, float) and np.isnan(codes)):
                continue
            arr = np.asarray(codes, dtype=np.float32)
            if len(arr) == 0:
                continue
            arrays.append(arr)
            gids.append(str(row["genotype_id"]))
        if not arrays:
            return None
        n_markers = min(len(arrays[0]), max_markers or len(arrays[0]))
        wide = pd.DataFrame(np.vstack([arr[:n_markers] for arr in arrays]), columns=[f"m{i:05d}" for i in range(n_markers)])
        wide.insert(0, "genotype_id", gids)
        return wide

    if "marker_id" in genotype.columns and "allele_dosage" in genotype.columns:
        geno = genotype.copy()
        marker_ids = list(pd.unique(geno["marker_id"]))
        if max_markers is not None and len(marker_ids) > max_markers:
            rng = np.random.default_rng(42)
            marker_ids = list(rng.choice(marker_ids, size=max_markers, replace=False))
            geno = geno[geno["marker_id"].isin(marker_ids)]
        wide = geno.pivot_table(
            index="genotype_id",
            columns="marker_id",
            values="allele_dosage",
            aggfunc="first",
        )
        wide = wide.dropna(axis=1, how="all").reset_index()
        wide["genotype_id"] = wide["genotype_id"].astype(str)
        return wide

    marker_cols = [c for c in genotype.columns if c != "genotype_id"]
    if max_markers is not None and len(marker_cols) > max_markers:
        rng = np.random.default_rng(42)
        marker_cols = list(rng.choice(marker_cols, size=max_markers, replace=False))
    out = genotype[["genotype_id"] + marker_cols].copy()
    out["genotype_id"] = out["genotype_id"].astype(str)
    return out


@dataclass
class FoldPreprocessor:
    """Fold-fitted feature builder.

    All learned quantities are fitted from the training phenotype rows only.
    The same fitted columns and fill values are then applied to validation/test.
    """

    max_markers: int = 5000
    marker_strategy: str = "random"
    weather_seq_len: int = 30
    weather_feat_dim: int = len(WEATHER_FEATURE_CANDIDATES)
    weather_mode: str = "daily"
    weather_standardize: bool = False
    stage_windows: list[tuple[str, int, int]] = field(default_factory=lambda: list(DEFAULT_STAGE_WINDOWS))
    marker_cols: list[str] = field(default_factory=list)
    marker_fill_values: dict[str, float] = field(default_factory=dict)
    env_cols: list[str] = field(default_factory=list)
    env_fill_values: dict[str, float] = field(default_factory=dict)
    weather_cols: list[str] = field(default_factory=list)
    weather_fill_values: dict[str, float] = field(default_factory=dict)
    weather_feature_names: list[str] = field(default_factory=list)
    weather_center: list[float] = field(default_factory=list)
    weather_scale: list[float] = field(default_factory=list)
    n_train_samples: int = 0
    n_train_genotypes: int = 0
    n_train_environments: int = 0

    def fit(
        self,
        train_pheno: pd.DataFrame,
        geno_wide: pd.DataFrame,
        env_feats: pd.DataFrame | None = None,
        weather_data: pd.DataFrame | None = None,
    ) -> "FoldPreprocessor":
        if "genotype_id" not in geno_wide.columns:
            raise ValueError("geno_wide must contain genotype_id")

        self.n_train_samples = int(len(train_pheno))
        train_genotypes = set(train_pheno["genotype_id"].astype(str))
        train_envs = set(train_pheno["environment_id"].astype(str))
        self.n_train_genotypes = len(train_genotypes)
        self.n_train_environments = len(train_envs)

        all_marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]
        train_geno = geno_wide[geno_wide["genotype_id"].astype(str).isin(train_genotypes)]
        if len(all_marker_cols) > self.max_markers:
            self.marker_cols = select_markers_by_strategy(
                all_marker_cols,
                self.max_markers,
                strategy=self.marker_strategy,
                geno_wide=train_geno[["genotype_id"] + all_marker_cols],
            )
        else:
            self.marker_cols = list(all_marker_cols)
        marker_means = train_geno[self.marker_cols].mean(axis=0, skipna=True)
        self.marker_fill_values = {
            col: float(0.0 if pd.isna(marker_means.get(col)) else marker_means.get(col))
            for col in self.marker_cols
        }

        self.env_cols = []
        self.env_fill_values = {}
        if env_feats is not None and len(env_feats) > 0:
            env = env_feats.copy()
            candidate_cols = [
                c for c in env.columns
                if c != "environment_id" and _is_numeric_dtype(env[c].dtype)
            ][:20]
            self.env_cols = list(candidate_cols)
            train_env = env[env["environment_id"].astype(str).isin(train_envs)]
            env_means = train_env[self.env_cols].mean(axis=0, skipna=True) if self.env_cols else pd.Series(dtype=float)
            self.env_fill_values = {
                col: float(0.0 if pd.isna(env_means.get(col)) else env_means.get(col))
                for col in self.env_cols
            }

        self.weather_cols = []
        self.weather_fill_values = {}
        if weather_data is not None and len(weather_data) > 0:
            self.weather_cols = [
                c for c in WEATHER_FEATURE_CANDIDATES
                if c in weather_data.columns
            ][:self.weather_feat_dim]
            if self.weather_mode == "stage_summary":
                self.weather_seq_len = len(self.stage_windows)
                self.weather_feature_names = [
                    f"{col}_{stat}"
                    for col in self.weather_cols
                    for stat in SUMMARY_STATS
                ]
                self.weather_feature_names.extend(["heat_days_tmax_gt30", "rain_days", "dry_days"])
                self.weather_feat_dim = len(self.weather_feature_names)
            else:
                self.weather_feature_names = list(self.weather_cols)
                self.weather_feat_dim = max(self.weather_feat_dim, len(self.weather_cols))
            train_weather = weather_data[weather_data["environment_id"].astype(str).isin(train_envs)]
            weather_means = train_weather[self.weather_cols].mean(axis=0, skipna=True) if self.weather_cols else pd.Series(dtype=float)
            self.weather_fill_values = {
                col: float(0.0 if pd.isna(weather_means.get(col)) else weather_means.get(col))
                for col in self.weather_cols
            }

        return self

    def fit_transform(
        self,
        train_pheno: pd.DataFrame,
        geno_wide: pd.DataFrame,
        env_feats: pd.DataFrame | None = None,
        weather_data: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(train_pheno, geno_wide, env_feats, weather_data)
        X_geno, X_weather, X_env = self.transform(train_pheno, geno_wide, env_feats, weather_data)
        if self.weather_standardize and X_weather.size:
            flat = X_weather.reshape(-1, X_weather.shape[-1])
            center = np.nanmean(flat, axis=0)
            scale = np.nanstd(flat, axis=0)
            scale = np.where((scale == 0) | np.isnan(scale), 1.0, scale)
            center = np.where(np.isnan(center), 0.0, center)
            self.weather_center = center.astype(float).tolist()
            self.weather_scale = scale.astype(float).tolist()
            X_weather = self._standardize_weather(X_weather)
        return X_geno, X_weather, X_env

    def _standardize_weather(self, X_weather: np.ndarray) -> np.ndarray:
        if not self.weather_center or not self.weather_scale:
            return X_weather
        center = np.asarray(self.weather_center, dtype=np.float32).reshape(1, 1, -1)
        scale = np.asarray(self.weather_scale, dtype=np.float32).reshape(1, 1, -1)
        return ((X_weather - center) / scale).astype(np.float32)

    def _weather_by_environment(self, weather_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        weather = normalize_weather_dates(weather_data)
        weather["environment_id"] = weather["environment_id"].astype(str)
        sort_cols = []
        if "day_after_planting" in weather.columns:
            sort_cols.append("day_after_planting")
        if "_weather_date" in weather.columns:
            sort_cols.append("_weather_date")
        if not sort_cols:
            sort_cols = list(weather.columns[:1])
        return {
            str(env_id): group.sort_values(sort_cols)
            for env_id, group in weather.groupby("environment_id", sort=False)
        }

    def _build_daily_weather_sequence(self, group: pd.DataFrame) -> np.ndarray:
        seq = np.zeros((self.weather_seq_len, self.weather_feat_dim), dtype=np.float32)
        if group is None or len(group) == 0 or not self.weather_cols:
            return seq
        source = group
        if "day_after_planting" in source.columns:
            dap = pd.to_numeric(source["day_after_planting"], errors="coerce")
            source = source[(dap >= 0) & (dap < self.weather_seq_len)].copy()
            source["_dap"] = dap.loc[source.index].astype(int)
            if len(source) == 0:
                return seq
            for _, row in source.iterrows():
                day = int(row["_dap"])
                values = row[self.weather_cols].fillna(self.weather_fill_values).to_numpy(dtype=np.float32)
                seq[day, : len(values)] = values
            return seq

        arr = (
            source[self.weather_cols]
            .fillna(self.weather_fill_values)
            .to_numpy(dtype=np.float32)
        )
        actual_len = min(len(arr), self.weather_seq_len)
        if actual_len > 0:
            arr = arr[:actual_len, :self.weather_feat_dim]
            seq[:actual_len, :arr.shape[1]] = arr
        return seq

    def _build_stage_weather_sequence(self, group: pd.DataFrame) -> np.ndarray:
        seq = np.zeros((len(self.stage_windows), self.weather_feat_dim), dtype=np.float32)
        if group is None or len(group) == 0 or not self.weather_cols:
            return seq

        source = group.copy()
        if "day_after_planting" in source.columns:
            source["_dap"] = pd.to_numeric(source["day_after_planting"], errors="coerce")
        else:
            source["_dap"] = np.arange(len(source), dtype=float)

        for stage_idx, (_name, start, end) in enumerate(self.stage_windows):
            stage = source[(source["_dap"] >= start) & (source["_dap"] <= end)]
            values: list[float] = []
            for col in self.weather_cols:
                series = pd.to_numeric(stage[col], errors="coerce") if col in stage.columns else pd.Series(dtype=float)
                if len(series) == 0 or series.dropna().empty:
                    fill = float(self.weather_fill_values.get(col, 0.0))
                    stats = [fill, fill, fill, fill, 0.0]
                else:
                    stats = [
                        float(series.mean()),
                        float(series.min()),
                        float(series.max()),
                        float(series.sum()),
                        float(series.std(ddof=0) if len(series.dropna()) > 1 else 0.0),
                    ]
                values.extend(stats)
            tmax = pd.to_numeric(stage["tmax"], errors="coerce") if "tmax" in stage.columns else pd.Series(dtype=float)
            precip = pd.to_numeric(stage["precipitation"], errors="coerce") if "precipitation" in stage.columns else pd.Series(dtype=float)
            values.extend([
                float((tmax > 30.0).sum()) if len(tmax) else 0.0,
                float((precip > 0.0).sum()) if len(precip) else 0.0,
                float((precip <= 0.0).sum()) if len(precip) else 0.0,
            ])
            seq[stage_idx, : min(len(values), self.weather_feat_dim)] = np.asarray(values[:self.weather_feat_dim], dtype=np.float32)
        return seq

    def transform(
        self,
        pheno: pd.DataFrame,
        geno_wide: pd.DataFrame,
        env_feats: pd.DataFrame | None = None,
        weather_data: pd.DataFrame | None = None,
        main_effects: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(pheno)

        geno_table = geno_wide[["genotype_id"] + self.marker_cols].copy()
        geno_table["genotype_id"] = geno_table["genotype_id"].astype(str)
        geno_indexed = pheno[["genotype_id"]].astype(str).merge(
            geno_table,
            on="genotype_id",
            how="left",
        )
        X_geno = (
            geno_indexed[self.marker_cols]
            .fillna(self.marker_fill_values)
            .to_numpy(dtype=np.float32)
        )

        X_weather = np.zeros((n, self.weather_seq_len, self.weather_feat_dim), dtype=np.float32)
        if weather_data is not None and self.weather_cols:
            weather_by_env = self._weather_by_environment(weather_data)
            env_ids = pheno["environment_id"].astype(str).to_numpy()
            sequence_by_env: dict[str, np.ndarray] = {}
            for env_id in pd.unique(env_ids):
                group = weather_by_env.get(str(env_id))
                if self.weather_mode == "stage_summary":
                    sequence_by_env[str(env_id)] = self._build_stage_weather_sequence(group)
                else:
                    sequence_by_env[str(env_id)] = self._build_daily_weather_sequence(group)
            for env_id, seq in sequence_by_env.items():
                X_weather[env_ids == env_id] = seq
            X_weather = self._standardize_weather(X_weather)

        if env_feats is not None and self.env_cols:
            env_table = env_feats[["environment_id"] + self.env_cols].copy()
            env_table["environment_id"] = env_table["environment_id"].astype(str)
            env_indexed = pheno[["environment_id"]].astype(str).merge(
                env_table,
                on="environment_id",
                how="left",
            )
            X_env = (
                env_indexed[self.env_cols]
                .fillna(self.env_fill_values)
                .to_numpy(dtype=np.float32)
            )
        else:
            X_env = np.zeros((n, 4), dtype=np.float32)

        if main_effects is not None:
            X_env = np.column_stack([X_env, np.asarray(main_effects, dtype=np.float32).reshape(-1, 1)])

        return X_geno, X_weather, X_env.astype(np.float32)

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = 1
        data["fit_scope"] = "training_fold_only"
        return data

    def write_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_manifest(), sort_keys=False), encoding="utf-8")
