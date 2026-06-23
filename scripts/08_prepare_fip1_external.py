from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.fip1 import prepare_fip1_external


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare FIP1 external validation data.')
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=False, default=None)
    parser.add_argument("--split-dir", type=Path, required=False, default=None)
    parser.add_argument("--residual-dir", type=Path, required=False, default=None)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main():
    args = parse_args()
    config = _load_config(args.config)
    data_config = config.get("data", {})
    raw_dir = args.raw_dir or Path(data_config.get("raw_dir", "data/raw/fip1"))
    target_trait = data_config.get("target_trait", "yield_adjusted")
    result = prepare_fip1_external(
        raw_dir=raw_dir,
        out_dir=args.out_dir,
        target_trait=target_trait,
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
