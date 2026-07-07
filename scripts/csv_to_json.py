#!/usr/bin/env python3
"""Convert all summary.csv files under Output/experiments/ to JSON.

For every summary.csv found via recursive glob under Output/experiments/,
this script reads the CSV with csv.DictReader (columns are discovered
dynamically, never hardcoded), infers a Python type per value
(int -> float -> str, with empty strings becoming None), and writes one
JSON file per CSV into docs/json_output/.

A summary index file docs/json_output/index.json is also written, listing
every converted file along with its row and column counts.

Only the Python standard library is used (csv, json, pathlib, datetime).
"""

import csv
import json
from datetime import datetime
from pathlib import Path

# Resolve paths relative to the project root (parent of this scripts/ dir)
# so the script works regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "Output" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "json_output"


def coerce(value):
    """Infer a Python type from a raw CSV string value.

    - None or empty string  -> None
    - parses as int         -> int
    - parses as float       -> float
    - otherwise             -> original string

    int() is tried before float() so that "100" stays an int while
    "1.724e+06" (rejected by int) becomes a float.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


def convert_csv(csv_path):
    """Read a single summary.csv and return a JSON-serializable dict.

    Handles three robustness concerns:
    - Empty CSV (header only, no data rows): rows == [], row_count == 0.
    - DictReader produces a None key for values beyond the header when a
      row has trailing commas; those entries are filtered out.
    - Missing values for named columns are filled with None by
      DictReader (restval default) and preserved as None by coerce().
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        # Fieldnames come from the header row. Drop any spurious None
        # (defensive; the header itself should never yield None).
        columns = [c for c in (reader.fieldnames or []) if c is not None]

        rows = []
        for raw_row in reader:
            # Filter out the None key that DictReader creates for
            # trailing-comma extra columns before coercing values.
            row = {k: coerce(v) for k, v in raw_row.items() if k is not None}
            rows.append(row)

    # Derive experiment / subdir from the path relative to experiments dir.
    # Path layout: experiments/<experiment>/<subdir...>/summary.csv
    rel = csv_path.relative_to(EXPERIMENTS_DIR)
    parts = rel.parts  # e.g. ("exp_0615", "bend_aware", "summary.csv")
    experiment = parts[0]
    subdir = "/".join(parts[1:-1])
    source_csv = str(csv_path.relative_to(PROJECT_ROOT))

    return {
        "experiment": experiment,
        "subdir": subdir,
        "source_csv": source_csv,
        "columns": columns,
        "row_count": len(rows),
        "rows": rows,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # rglob("summary.csv") naturally ignores .log and .md files.
    # Sort for deterministic output / stable index ordering.
    csv_files = sorted(EXPERIMENTS_DIR.rglob("summary.csv"))

    index_entries = []
    for csv_path in csv_files:
        data = convert_csv(csv_path)

        # Output filename: {experiment}__{subdir}.json (double underscore).
        # Replace any path separators in subdir (defensive; current data is
        # single-level so this is a no-op) to keep the filename flat.
        subdir_safe = data["subdir"].replace("/", "_").replace("\\", "_")
        json_filename = f"{data['experiment']}__{subdir_safe}.json"
        json_path = OUTPUT_DIR / json_filename

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

        index_entries.append({
            "experiment": data["experiment"],
            "subdir": data["subdir"],
            "json_file": json_filename,
            "row_count": data["row_count"],
            "column_count": len(data["columns"]),
        })

    index = {
        "generated_at": datetime.now().isoformat(),
        "total_files": len(index_entries),
        "experiments": index_entries,
    }
    with open(OUTPUT_DIR / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Converted {len(index_entries)} CSV files into {OUTPUT_DIR}")
    print(f"Index written to {OUTPUT_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
