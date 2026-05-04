#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PilotNet session metadata.csv files into teacher path CSVs."
    )
    parser.add_argument("--data-root", default="/home/yuwei/pilotnet_data_manual")
    parser.add_argument("--metadata-glob", default="session_*/metadata.csv")
    parser.add_argument("--output-dir", default="/home/yuwei/teacher_paths/from_manual_sessions")
    parser.add_argument("--min-sample-distance", type=float, default=0.25)
    parser.add_argument("--min-sample-heading", type=float, default=0.08)
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.05,
        help="Skip stopped frames before extracting path points.",
    )
    return parser.parse_args()


def should_keep(samples, x, y, yaw, min_distance, min_heading):
    if not samples:
        return True

    prev = samples[-1]
    distance = math.hypot(x - prev["x"], y - prev["y"])
    heading_delta = abs(normalize_angle(yaw - prev["yaw"]))
    return distance >= min_distance or heading_delta >= min_heading


def convert_metadata(metadata_path, output_path, args):
    samples = []
    cumulative_s = 0.0
    last_kept = None

    with open(metadata_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            speed = float(row["speed"])
            if abs(speed) < args.min_speed:
                continue

            x = float(row["x"])
            y = float(row["y"])
            yaw = float(row["yaw"])
            if not should_keep(
                samples,
                x,
                y,
                yaw,
                args.min_sample_distance,
                args.min_sample_heading,
            ):
                continue

            if last_kept is not None:
                cumulative_s += math.hypot(x - last_kept["x"], y - last_kept["y"])

            sample = {
                "index": len(samples),
                "s": cumulative_s,
                "x": x,
                "y": y,
                "yaw": yaw,
            }
            samples.append(sample)
            last_kept = sample

    if len(samples) < 2:
        raise RuntimeError(f"Not enough moving path points in {metadata_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "s", "x", "y", "yaw"])
        for sample in samples:
            writer.writerow(
                [
                    sample["index"],
                    f"{sample['s']:.9f}",
                    f"{sample['x']:.9f}",
                    f"{sample['y']:.9f}",
                    f"{sample['yaw']:.9f}",
                ]
            )

    return samples


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_paths = sorted(data_root.glob(args.metadata_glob))
    if not metadata_paths:
        raise RuntimeError(f"No metadata files found under {data_root}")

    for metadata_path in metadata_paths:
        session_name = metadata_path.parent.name
        output_path = output_dir / f"{session_name}_teacher_path.csv"
        samples = convert_metadata(metadata_path, output_path, args)
        print(
            f"{metadata_path} -> {output_path} "
            f"points={len(samples)} length={samples[-1]['s']:.2f}m"
        )


if __name__ == "__main__":
    main()
