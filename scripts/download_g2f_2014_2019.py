#!/usr/bin/env python
"""Download G2F 2014-2019 data from CyVerse Data Commons."""
from __future__ import annotations

import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "g2f"

BASE_URL = "https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/curated"

# (year, cyverse_path, phenotype_subdir, weather_subdir, soil_subdir, genotype_subdir)
DATASETS: list[tuple[str, str, str, str, str | None, str | None]] = [
    (
        "2014",
        "GenomesToFields_2014_2017_v1/G2F_Planting_Season_2014_v4",
        "a._2014_hybrid_phenotypic_data",
        "b._2014_weather_data",
        None,
        None,
    ),
    (
        "2015",
        "G2F_Planting_Season_2015_v2",
        "a._2015_hybrid_phenotypic_data",
        "b._2015_weather_data",
        "d._2015_soil_data",
        None,
    ),
    (
        "2016",
        "GenomesToFields_2014_2017_v1/G2F_Planting_Season_2016_v2",
        "a._2016_hybrid_phenotypic_data",
        "b._2016_weather_data",
        "c._2016_soil_data",
        None,
    ),
    (
        "2017",
        "GenomesToFields_2014_2017_v1/G2F_Planting_Season_2017_v1",
        "a._2017_hybrid_phenotypic_data",
        "b._2017_weather_data",
        "c._2017_soil_data",
        None,
    ),
    (
        "2018",
        "GenomesToFields_G2F_Data_2018",
        "a._2018_hybrid_phenotypic_data",
        "b._2018_weather_data",
        "c._2018_soil_data",
        "d._2018_genotypic_data",
    ),
    (
        "2019",
        "GenomesToFields_data_2019",
        "a._2019_phenotypic_data",
        "b._2019_weather_data",
        "c._2019_soil_data",
        None,
    ),
]

# Shared genotype VCF for 2014-2017 (in bundle root or top-level)
GENOTYPE_2014_2017 = "GenomesToFields_2014_2017_v1"


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sz = dest.stat().st_size
        if sz > 1000:
            print(f"  SKIP (exists, {sz/1e6:.1f}MB): {dest.name}")
            return True
    print(f"  DOWNLOAD: {dest.name} ...", end=" ", flush=True)
    try:
        subprocess.run(
            ["curl", "-L", "--retry", "3", "-o", str(dest), url],
            check=True, capture_output=True, timeout=600,
        )
        sz = dest.stat().st_size
        print(f"OK ({sz/1e6:.1f}MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading G2F 2014-2019 Data from CyVerse")
    print("=" * 60)

    # 1. Download year-level phenotype and weather data
    for year, path, pheno_dir, weather_dir, soil_dir, geno_dir in DATASETS:
        print(f"\n--- {year} ---")
        year_dir = RAW / year
        year_dir.mkdir(parents=True, exist_ok=True)

        # Phenotype
        url = f"{BASE_URL}/{path}/{pheno_dir}/g2f_{year}_hybrid_data_clean.csv"
        download(url, year_dir / f"g2f_{year}_hybrid_data_clean.csv")

        # Weather
        url = f"{BASE_URL}/{path}/{weather_dir}/g2f_{year}_weather.csv"
        download(url, year_dir / f"g2f_{year}_weather.csv")

        # Soil
        if soil_dir:
            url = f"{BASE_URL}/{path}/{soil_dir}/g2f_{year}_soil.csv"
            download(url, year_dir / f"g2f_{year}_soil.csv")

        # Genotype (year-specific)
        if geno_dir:
            # CyVerse doesn't easily list files — try common VCF names
            for geno_name in [
                f"g2f_{year}_genotype.vcf",
                f"g2f_{year}_genotype.hmp.txt",
                f"G2F_{year}_genotype.vcf",
            ]:
                url = f"{BASE_URL}/{path}/{geno_dir}/{geno_name}"
                if download(url, year_dir / geno_name):
                    break

    # 2. Try to get shared 2014-2017 genotype data
    print(f"\n--- Genotype 2014-2017 (shared) ---")
    geno_url = f"{BASE_URL}/{GENOTYPE_2014_2017}"
    # Try common genotype filenames
    for geno_name in [
        "g2f_2014_2017_genotype.hmp.txt",
        "G2F_2014_2017_genotype.hmp.txt",
        "all_genotypes.vcf",
        "genotype_data/",
    ]:
        url = f"{geno_url}/{geno_name}"
        dest = RAW / "genotype_2014_2017" / geno_name
        if "/" not in geno_name.rstrip("/"):
            download(url, dest)
        # else it's a directory, skip

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Data saved to: {RAW}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
