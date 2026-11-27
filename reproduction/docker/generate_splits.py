#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = [
    "user_id",
    "video_id",
    "is_hate",
    "fraction_play_time",
    "is_hate_y",
    "fraction_play_time_y",
]
MANIFEST_NAME = ".split-manifest-False-0.3.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_dataframe_write(data: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        data.to_csv(temporary, index=False, sep=" ", header=False)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, destination)


def _required_paths(output_dir: Path, runs: int) -> list[Path]:
    return [
        output_dir / f"{prefix}_{run_id}_False_0.3.txt"
        for run_id in range(runs)
        for prefix in ("train", "test_calibration")
    ]


def generate_splits(input_path: Path, output_dir: Path, runs: int) -> list[Path]:
    if runs <= 0:
        raise ValueError("runs must be positive")

    data = pd.read_csv(input_path, compression="gzip")[REQUIRED_COLUMNS]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for run_id in range(runs):
        train, test_calibration = train_test_split(
            data,
            test_size=0.3,
            stratify=data["is_hate"],
            random_state=run_id,
        )
        train_path = output_dir / f"train_{run_id}_False_0.3.txt"
        test_path = output_dir / f"test_calibration_{run_id}_False_0.3.txt"
        _atomic_dataframe_write(train, train_path)
        _atomic_dataframe_write(test_calibration, test_path)
        generated.extend([train_path, test_path])

    manifest = {
        "source_sha256": _sha256(input_path),
        "files": {path.name: _sha256(path) for path in generated},
    }
    manifest_path = output_dir / MANIFEST_NAME
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_dir, prefix=f".{MANIFEST_NAME}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(manifest, temporary, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, manifest_path)
    return generated


def validate_splits(input_path: Path, output_dir: Path, runs: int) -> bool:
    if runs <= 0:
        raise ValueError("runs must be positive")
    manifest_path = output_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if manifest.get("source_sha256") != _sha256(input_path):
        return False
    recorded = manifest.get("files", {})
    for path in _required_paths(output_dir, runs):
        if not path.is_file() or recorded.get(path.name) != _sha256(path):
            return False
    return True


def ensure_splits(input_path: Path, output_dir: Path, runs: int) -> list[Path]:
    if validate_splits(input_path, output_dir, runs):
        return _required_paths(output_dir, runs)
    return generate_splits(input_path, output_dir, runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    already_valid = validate_splits(args.input, args.output_dir, args.runs)
    paths = ensure_splits(args.input, args.output_dir, args.runs)
    action = "Validated" if already_valid else "Generated"
    print(f"{action} {len(paths)} split files in {args.output_dir}")


if __name__ == "__main__":
    main()
