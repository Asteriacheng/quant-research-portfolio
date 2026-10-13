"""Build the study's point-in-time-ready daily panel from the pinned GitHub snapshot.

Source data is preserved verbatim under data/raw/astock; this script only filters and
normalizes the five complete calendar years 2021-2025.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "astock"
OUT = ROOT / "data" / "processed"
START, END = "2021-01-01", "2025-12-31"
COMMIT = "0babe4cf6c1a175c6b84add10803b4b1f9c6000c"
FILES = ["kline_000.parquet", "kline_002.parquet", "kline_300.parquet",
         "kline_600.parquet", "kline_688.parquet", "kline_other.parquet",
         "stock_list.parquet", "index_sh.parquet"]


def fetch_snapshot() -> None:
    """Download the pinned public snapshot; data files are not redistributed by this repo."""
    RAW.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = RAW / filename
        if target.exists():
            continue
        url = f"https://raw.githubusercontent.com/newbiestring-lang/astock/{COMMIT}/{filename}"
        print(f"Downloading {filename} from pinned source commit ...", flush=True)
        urllib.request.urlretrieve(url, target)


def main() -> None:
    fetch_snapshot()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in sorted(RAW.glob("kline_*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame["code"].str.fullmatch(r"\d{6}")]
        frame = frame[frame["date"].between(START, END)]
        if not frame.empty:
            parts.append(frame)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    panel.to_parquet(OUT / "daily_2021_2025.parquet", index=False, compression="zstd")
    manifest = {
        "source": "https://github.com/newbiestring-lang/astock",
        "source_commit": COMMIT,
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "study_start": START,
        "study_end": END,
        "rows": len(panel),
        "codes": int(panel.code.nunique()),
        "sha256": hashlib.sha256((OUT / "daily_2021_2025.parquet").read_bytes()).hexdigest(),
        "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("*.parquet"))},
    }
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
