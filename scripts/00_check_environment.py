from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

REPORT = Path("outputs/reports/phase0_environment_check.md")

PACKAGE_MAP = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "torch": "torch",
    "yaml": "PyYAML",
}


def check_pkg(import_name: str, dist_name: str) -> str:
    """Return package version without importing heavy packages.

    Importing frameworks such as torch in a tiny environment check can create
    background runtime state that interferes with later test collection in some
    CI containers, so prefer package metadata.
    """
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main():
    lines = [
        "# Phase 0 Environment Check",
        "",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        "## Packages",
        "",
    ]
    for import_name, dist_name in PACKAGE_MAP.items():
        lines.append(f"- {import_name}: {check_pkg(import_name, dist_name)}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
