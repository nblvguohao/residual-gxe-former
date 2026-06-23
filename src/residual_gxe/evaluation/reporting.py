from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metrics_table(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def weighted_score(row: dict, weights: dict[str, float] | None = None) -> float:
    weights = weights or {
        "leave_environment": 0.40,
        "leave_year": 0.35,
        "leave_genotype": 0.15,
        "random": 0.10,
    }
    score = 0.0
    total = 0.0
    for key, weight in weights.items():
        value = row.get(key)
        if value is not None and value == value:
            score += weight * float(value)
            total += weight
    return score / total if total > 0 else float("nan")
