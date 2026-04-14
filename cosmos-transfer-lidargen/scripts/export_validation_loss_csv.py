#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path


ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
VAL_LOSS_PATTERN = re.compile(
    r"\[(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\|[A-Z]+[^\]]*)\].*?Validation loss \(iteration (?P<iteration>\d+)\): (?P<loss>[-+eE0-9.]+)"
)
VAL_DEPTH_PATTERN = re.compile(
    r"\[(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\|[A-Z]+[^\]]*)\].*?Validation depth metrics \(iteration (?P<iteration>\d+)\): "
    r"mae=(?P<mae>[-+eE0-9.]+), rmse=(?P<rmse>[-+eE0-9.]+), rel=(?P<rel>[-+eE0-9.]+)"
)
VAL_BREAKDOWN_PATTERN = re.compile(
    r"\[(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\|[A-Z]+[^\]]*)\].*?Validation loss breakdown \(iteration (?P<iteration>\d+)\): "
    r"(?P<breakdown>.+)"
)
BREAKDOWN_ITEM_PATTERN = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[-+eE0-9.]+)")


def sanitize_log_line(line: str) -> str:
    line = line.replace("\r", "")
    return ANSI_ESCAPE_PATTERN.sub("", line)


def parse_validation_losses(log_path: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    seen_iterations: set[int] = set()

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = sanitize_log_line(line)
            match = VAL_LOSS_PATTERN.search(line)
            if match is None:
                continue
            iteration = int(match.group("iteration"))
            if iteration in seen_iterations:
                continue
            seen_iterations.add(iteration)
            timestamp = match.group("timestamp").split("|", 1)[0]
            rows.append(
                (
                    timestamp,
                    iteration,
                    float(match.group("loss")),
                )
            )

    rows.sort(key=lambda item: item[1])
    return rows


def parse_validation_depth_metrics(log_path: Path) -> list[tuple[str, int, float, float, float]]:
    rows: list[tuple[str, int, float, float, float]] = []
    seen_iterations: set[int] = set()

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = sanitize_log_line(line)
            match = VAL_DEPTH_PATTERN.search(line)
            if match is None:
                continue
            iteration = int(match.group("iteration"))
            if iteration in seen_iterations:
                continue
            seen_iterations.add(iteration)
            timestamp = match.group("timestamp").split("|", 1)[0]
            rows.append(
                (
                    timestamp,
                    iteration,
                    float(match.group("mae")),
                    float(match.group("rmse")),
                    float(match.group("rel")),
                )
            )

    rows.sort(key=lambda item: item[1])
    return rows


def parse_validation_loss_breakdown(log_path: Path) -> list[tuple[str, int, dict[str, float]]]:
    rows: list[tuple[str, int, dict[str, float]]] = []
    seen_iterations: set[int] = set()

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = sanitize_log_line(line)
            match = VAL_BREAKDOWN_PATTERN.search(line)
            if match is None:
                continue
            iteration = int(match.group("iteration"))
            if iteration in seen_iterations:
                continue
            seen_iterations.add(iteration)
            timestamp = match.group("timestamp").split("|", 1)[0]
            breakdown: dict[str, float] = {}
            for item in BREAKDOWN_ITEM_PATTERN.finditer(match.group("breakdown")):
                breakdown[item.group("key")] = float(item.group("value"))
            rows.append((timestamp, iteration, breakdown))

    rows.sort(key=lambda item: item[1])
    return rows


def write_csv(rows: list[tuple[str, int, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "iteration", "validation_loss"])
        for timestamp, iteration, loss in rows:
            writer.writerow([timestamp, iteration, f"{loss:.6f}"])


def write_depth_csv(rows: list[tuple[str, int, float, float, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "iteration", "depth_mae", "depth_rmse", "depth_relative_error"])
        for timestamp, iteration, mae, rmse, rel in rows:
            writer.writerow([timestamp, iteration, f"{mae:.6f}", f"{rmse:.6f}", f"{rel:.6f}"])


def write_merged_csv(
    loss_rows: list[tuple[str, int, float]],
    breakdown_rows: list[tuple[str, int, dict[str, float]]],
    depth_rows: list[tuple[str, int, float, float, float]],
    output_path: Path,
) -> None:
    merged_rows: dict[int, dict[str, object]] = {}

    for timestamp, iteration, loss in loss_rows:
        merged_rows.setdefault(iteration, {"timestamp": timestamp, "iteration": iteration})
        merged_rows[iteration]["timestamp"] = timestamp
        merged_rows[iteration]["validation_loss"] = loss

    for timestamp, iteration, breakdown in breakdown_rows:
        merged_rows.setdefault(iteration, {"timestamp": timestamp, "iteration": iteration})
        merged_rows[iteration]["timestamp"] = timestamp
        for key, value in breakdown.items():
            merged_rows[iteration][key] = value

    for timestamp, iteration, mae, rmse, rel in depth_rows:
        merged_rows.setdefault(iteration, {"timestamp": timestamp, "iteration": iteration})
        merged_rows[iteration]["timestamp"] = timestamp
        merged_rows[iteration]["depth_mae"] = mae
        merged_rows[iteration]["depth_rmse"] = rmse
        merged_rows[iteration]["depth_relative_error"] = rel

    dynamic_keys: list[str] = sorted(
        {
            key
            for row in merged_rows.values()
            for key in row.keys()
            if key not in {"timestamp", "iteration", "validation_loss", "depth_mae", "depth_rmse", "depth_relative_error"}
        }
    )
    fieldnames = [
        "timestamp",
        "iteration",
        "validation_loss",
        *dynamic_keys,
        "depth_mae",
        "depth_rmse",
        "depth_relative_error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in sorted(merged_rows):
            row = merged_rows[iteration]
            formatted_row = {key: row.get(key, "") for key in fieldnames}
            for key, value in list(formatted_row.items()):
                if isinstance(value, float):
                    formatted_row[key] = f"{value:.6f}"
            writer.writerow(formatted_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation loss history from tokenizer stdout.log.")
    parser.add_argument("--log_path", type=Path, required=True, help="Path to stdout.log")
    parser.add_argument("--output_csv", type=Path, required=True, help="Path to merged validation CSV")
    parser.add_argument(
        "--output_depth_csv",
        type=Path,
        default=None,
        help="Optional path to legacy depth-only CSV",
    )
    args = parser.parse_args()

    loss_rows = parse_validation_losses(args.log_path)
    breakdown_rows = parse_validation_loss_breakdown(args.log_path)
    depth_rows = parse_validation_depth_metrics(args.log_path)
    write_merged_csv(loss_rows, breakdown_rows, depth_rows, args.output_csv)
    print(f"Exported {len(loss_rows)} validation points to {args.output_csv}")
    if args.output_depth_csv is not None:
        write_depth_csv(depth_rows, args.output_depth_csv)
        print(f"Exported {len(depth_rows)} validation depth points to {args.output_depth_csv}")


if __name__ == "__main__":
    main()
