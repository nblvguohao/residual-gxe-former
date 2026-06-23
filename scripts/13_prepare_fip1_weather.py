from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.weather import prepare_weather_daily, read_weather_input


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare user-provided FIP1 weather data into unified weather_daily.parquet.")
    parser.add_argument("--weather-file", type=Path, required=True, help="Local CSV/TSV/Parquet weather file. No download is performed.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Processed FIP1 directory containing environment.parquet.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _write_gap_report(weather_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# FIP1 Weather Preparation Gap Report",
        "",
        f"Requested weather file: `{weather_file}`",
        "",
        "The file was not found. No weather_daily.parquet was generated.",
        "",
        "Provide a local CSV/TSV/Parquet file with at least:",
        "",
        "- environment_id or yearsite_uid",
        "- date",
        "- one or more weather variables such as tmax, tmin, tmean, precipitation, solar_radiation, relative_humidity",
        "",
        "The adapter does not download external data.",
    ]
    (out_dir / "fip1_weather_gap_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "ok": False,
        "weather_file": str(weather_file),
        "output": None,
        "reason": "weather file not found",
    }
    (out_dir / "weather_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.weather_file.exists():
        _write_gap_report(args.weather_file, args.out_dir)
        print(args.out_dir / "fip1_weather_gap_report.md")
        return

    env_path = args.data_dir / "environment.parquet"
    environment = read_table(env_path) if env_path.exists() else None
    raw_weather = read_weather_input(args.weather_file)
    result = prepare_weather_daily(raw_weather, environment, source_dataset="fip1")

    weather_path = args.out_dir / "weather_daily.parquet"
    write_table(result.weather, weather_path)
    manifest = {
        "ok": True,
        "weather_file": str(args.weather_file),
        "output": str(weather_path),
        **result.manifest,
    }
    (args.out_dir / "weather_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(weather_path)


if __name__ == "__main__":
    main()

