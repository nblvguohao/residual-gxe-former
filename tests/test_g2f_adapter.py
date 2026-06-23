from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from residual_gxe.data.g2f import prepare_g2f
from residual_gxe.data.schema import validate_phenotype_table


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_minimal_g2f_raw(raw: Path) -> None:
    _write_csv(
        raw / "phenotype.csv",
        [
            {
                "sample_id": "s1",
                "genotype_id": "g1",
                "environment_id": "e1",
                "year": 2021,
                "location_id": "loc1",
                "trait_id": "grain_yield",
                "trait_name": "Grain yield",
                "trait_family": "yield",
                "phenotype_value": 10.5,
                "phenotype_unit": "bu/ac",
                "replicate_id": "r1",
                "block_id": "b1",
                "trial_id": "t1",
                "source_dataset": "g2f",
            },
            {
                "sample_id": "s2",
                "genotype_id": "g2",
                "environment_id": "e2",
                "year": 2022,
                "location_id": "loc2",
                "trait_id": "grain_yield",
                "trait_name": "Grain yield",
                "trait_family": "yield",
                "phenotype_value": 12.0,
                "phenotype_unit": "bu/ac",
                "replicate_id": "r1",
                "block_id": "b1",
                "trial_id": "t2",
                "source_dataset": "g2f",
            },
        ],
    )
    _write_csv(raw / "genotype.csv", [{"genotype_id": "g1", "m1": 0, "m2": 1}, {"genotype_id": "g2", "m1": 2, "m2": 1}])
    _write_csv(
        raw / "environment.csv",
        [
            {
                "environment_id": "e1",
                "year": 2021,
                "location_id": "loc1",
                "latitude": 40.0,
                "longitude": -90.0,
                "altitude": 100,
                "planting_date": "2021-04-15",
                "harvest_date": "2021-09-20",
                "management_notes": "",
                "source_dataset": "g2f",
            },
            {
                "environment_id": "e2",
                "year": 2022,
                "location_id": "loc2",
                "latitude": 41.0,
                "longitude": -91.0,
                "altitude": 120,
                "planting_date": "2022-04-16",
                "harvest_date": "2022-09-21",
                "management_notes": "",
                "source_dataset": "g2f",
            },
        ],
    )
    _write_csv(
        raw / "weather_daily.csv",
        [
            {
                "environment_id": "e1",
                "date": "2021-04-15",
                "day_after_planting": 0,
                "tmax": 25.0,
                "tmin": 12.0,
                "tmean": 18.5,
                "precipitation": 0.0,
                "solar_radiation": 20.0,
                "relative_humidity": 65.0,
                "wind_speed": 2.0,
                "vpd": 1.0,
                "gdd": 8.5,
            }
        ],
    )
    _write_csv(raw / "soil.csv", [{"environment_id": "e1", "ph": 6.5, "organic_matter": 3.2}])


def test_prepare_g2f_converts_minimal_unified_raw(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    audit = tmp_path / "audit.md"
    _write_minimal_g2f_raw(raw)

    result = prepare_g2f(raw_dir=raw, out_dir=out, audit_path=audit, target_trait="grain_yield")

    phenotype = pd.read_parquet(out / "phenotype.parquet")

    assert result.ok is True
    assert validate_phenotype_table(phenotype).ok
    assert phenotype["sample_id"].tolist() == ["s1", "s2"]
    assert (out / "genotype.parquet").exists()
    assert (out / "environment.parquet").exists()
    assert (out / "weather_daily.parquet").exists()
    assert (out / "soil.parquet").exists()
    assert (out / "metadata.json").exists()
    assert (out / "data_manifest.yaml").exists()
    assert audit.exists()


def test_prepare_g2f_empty_raw_writes_audit_and_no_processed_tables(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    audit = tmp_path / "audit.md"
    raw.mkdir()

    result = prepare_g2f(raw_dir=raw, out_dir=out, audit_path=audit, target_trait="grain_yield")

    assert result.ok is False
    assert "phenotype" in result.missing_categories
    assert audit.exists()
    assert not (out / "phenotype.parquet").exists()


def test_prepare_g2f_script_runs_from_repo_root(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _write_minimal_g2f_raw(raw)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_prepare_g2f.py",
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "phenotype.parquet").exists()


def test_prepare_g2f_raises_for_invalid_phenotype(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    audit = tmp_path / "audit.md"
    _write_minimal_g2f_raw(raw)
    pd.DataFrame([{"sample_id": "s1"}]).to_csv(raw / "phenotype.csv", index=False)

    with pytest.raises(ValueError, match="phenotype"):
        prepare_g2f(raw_dir=raw, out_dir=out, audit_path=audit, target_trait="grain_yield")


def test_prepare_g2f_converts_native_year_folders(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    audit = tmp_path / "audit.md"
    _write_csv(
        raw / "2023" / "g2f_2023_phenotypic_clean_data.csv",
        [
            {
                "Year": 2023,
                "Field-Location": "IAH1",
                "State": "IA",
                "City": "Ames",
                "Experiment": "HIP_Hybrid",
                "Pedigree": "B73/PHN82",
                "Replicate": 1,
                "Block": 2,
                "Plot": 3,
                "Plot_ID": "G2F230001",
                "Date Plot Planted [MM/DD/YY]": "5/1/2023",
                "Date Plot Harvested [MM/DD/YY]": "10/1/2023",
                "Grain Yield (bu/A)": 201.5,
            },
            {
                "Year": 2023,
                "Field-Location": "IAH1",
                "State": "IA",
                "City": "Ames",
                "Experiment": "HIP_Hybrid",
                "Pedigree": "LH244/MO17",
                "Replicate": 1,
                "Block": 2,
                "Plot": 4,
                "Plot_ID": "G2F230002",
                "Date Plot Planted [MM/DD/YY]": "5/1/2023",
                "Date Plot Harvested [MM/DD/YY]": "10/1/2023",
                "Grain Yield (bu/A)": 198.0,
            },
        ],
    )
    _write_csv(
        raw / "2023" / "g2f_2023_weather_cleaned.csv",
        [
            {
                "Field Location": "IAH1",
                "Date_key": "5/1/2023 11:00",
                "Year": 2023,
                "Temperature [C]": 20.0,
                "Rainfall [mm]": 1.5,
                "Solar Radiation [W/m2]": 800.0,
                "Relative Humidity [%]": 70.0,
                "Wind Speed [m/s]": 2.0,
            },
            {
                "Field Location": "IAH1",
                "Date_key": "5/1/2023 12:00",
                "Year": 2023,
                "Temperature [C]": 24.0,
                "Rainfall [mm]": 0.5,
                "Solar Radiation [W/m2]": 900.0,
                "Relative Humidity [%]": 60.0,
                "Wind Speed [m/s]": 3.0,
            },
        ],
    )
    _write_csv(
        raw / "2023" / "g2f_2023_soil_data.csv",
        [{"Location": "IAH1", "1:1 Soil pH": 6.7, "Organic Matter LOI %": 3.1, "% Sand": 40, "% Silt": 35, "% Clay": 25}],
    )
    _write_csv(
        raw / "2023" / "g2f_2023_field_metadata.csv",
        [
            {
                "Experiment_Code": "IAH1",
                "Weather_Station_Latitude (in decimal numbers NOT DMS)": 42.0,
                "Weather_Station_Longitude (in decimal numbers NOT DMS)": -93.0,
            }
        ],
    )
    (raw / "genotype").mkdir(parents=True)
    (raw / "genotype" / "key_inbreds_G2F_2014-2023.txt").write_text(
        "Cultivar\tDataset\tSourceName\tBioproject\tBioSample\tAlternative name\tComments\n"
        "B73\tAssembly\tNAM\tP1\tS1\t\t\n"
        "PHN82\tImputed\tG2F\tP2\tS2\t\t\n"
        "LH244\tAssembly\tNAM\tP3\tS3\t\t\n"
        "MO17\tAssembly\tNAM\tP4\tS4\t\t\n",
        encoding="utf-8",
    )
    (raw / "genotype" / "inbreds_G2F_2014-2023_437k.vcf").write_text(
        "##fileformat=VCFv4.0\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tB73\tPHN82\tLH244\tMO17\n"
        "1\t10\tS1_10\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/0\n",
        encoding="utf-8",
    )

    result = prepare_g2f(raw_dir=raw, out_dir=out, audit_path=audit, target_trait="grain_yield")

    phenotype = pd.read_parquet(out / "phenotype.parquet")
    weather = pd.read_parquet(out / "weather_daily.parquet")
    soil = pd.read_parquet(out / "soil.parquet")
    genotype = pd.read_parquet(out / "genotype.parquet")

    assert result.ok is True
    assert validate_phenotype_table(phenotype).ok
    assert phenotype["sample_id"].tolist() == ["G2F230001", "G2F230002"]
    assert phenotype["genotype_id"].tolist() == ["B73/PHN82", "LH244/MO17"]
    assert phenotype["environment_id"].tolist() == ["2023_IAH1", "2023_IAH1"]
    assert weather.loc[0, "environment_id"] == "2023_IAH1"
    assert weather.loc[0, "tmax"] == 24.0
    assert weather.loc[0, "tmin"] == 20.0
    assert weather.loc[0, "precipitation"] == 2.0
    assert soil.loc[0, "environment_id"] == "2023_IAH1"
    assert set(genotype["genotype_id"]) == {"B73", "PHN82", "LH244", "MO17"}
    assert (out / "genotype_manifest.json").exists()
