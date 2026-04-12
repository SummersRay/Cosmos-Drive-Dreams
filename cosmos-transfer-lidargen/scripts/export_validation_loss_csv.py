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


def write_csv(rows: list[tuple[str, int, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "iteration", "validation_loss"])
        for timestamp, iteration, loss in rows:
            writer.writerow([timestamp, iteration, f"{loss:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation loss history from tokenizer stdout.log.")
    parser.add_argument("--log_path", type=Path, required=True, help="Path to stdout.log")
    parser.add_argument("--output_csv", type=Path, required=True, help="Path to output CSV")
    args = parser.parse_args()

    rows = parse_validation_losses(args.log_path)
    write_csv(rows, args.output_csv)
    print(f"Exported {len(rows)} validation points to {args.output_csv}")


if __name__ == "__main__":
    main()
