#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect converted real PilotNet datasets.")
    parser.add_argument("--data-root", default="real_pilotnet_data")
    parser.add_argument("--preview-dir", default="dataset_previews")
    parser.add_argument("--samples-per-session", type=int, default=3)
    return parser.parse_args()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def describe(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def read_metadata(metadata_path):
    rows = []
    with open(metadata_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def collect_sessions(data_root):
    # Each converted bag is one session under train/valid/test.
    sessions = []
    for split_name in ("train", "valid", "test"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        for metadata_path in sorted(split_dir.glob("*/metadata.csv")):
            session_dir = metadata_path.parent
            rows = read_metadata(metadata_path)
            sessions.append(
                {
                    "split": split_name,
                    "session": session_dir.name,
                    "session_dir": session_dir,
                    "metadata_path": metadata_path,
                    "rows": rows,
                }
            )
    return sessions


def validate_images(session):
    # Catch broken conversions before training: missing paths, corrupt JPEGs,
    # or unexpected image sizes.
    missing = 0
    unreadable = 0
    first_size = None
    for row in session["rows"]:
        image_path = session["session_dir"] / row["image_path"]
        if not image_path.exists():
            missing += 1
            continue
        try:
            with Image.open(image_path) as image:
                if first_size is None:
                    first_size = image.size
        except Exception:
            unreadable += 1
    return missing, unreadable, first_size


def make_preview_for_split(split_name, sessions, preview_dir, samples_per_session):
    # Build a contact sheet so image color, scene content, labels, and split
    # membership can be checked quickly by eye.
    split_sessions = [session for session in sessions if session["split"] == split_name]
    if not split_sessions:
        return None

    thumbs = []
    for session in split_sessions:
        rows = session["rows"]
        if not rows:
            continue
        if len(rows) == 1:
            indices = [0]
        else:
            indices = sorted(
                set(
                    int(round((len(rows) - 1) * index / max(samples_per_session - 1, 1)))
                    for index in range(samples_per_session)
                )
            )
        for index in indices:
            row = rows[index]
            image_path = session["session_dir"] / row["image_path"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            image.thumbnail((240, 150))
            thumbs.append(
                (
                    f"{session['session']} #{row['frame_id']}",
                    float(row["steering"]),
                    float(row["speed"]),
                    image.copy(),
                )
            )

    if not thumbs:
        return None

    cell_w = 240
    cell_h = 185
    cols = samples_per_session
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    for item_index, (label, steering, speed, image) in enumerate(thumbs):
        x = (item_index % cols) * cell_w
        y = (item_index // cols) * cell_h
        draw.text((x + 5, y + 4), label[:34], fill=(0, 0, 0))
        draw.text((x + 5, y + 18), f"steer={steering:.3f} speed={speed:.3f}", fill=(0, 0, 0))
        sheet.paste(image, (x, y + 35))

    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / f"preview_{split_name}.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def session_stats(session):
    # Summarize the label, speed, and synchronization distributions for one run.
    rows = session["rows"]
    steering = [float(row["steering"]) for row in rows]
    speed = [float(row["speed"]) for row in rows]
    steering_dt = [float(row.get("steering_dt_ms", 0.0)) for row in rows]
    speed_dt = [float(row.get("speed_dt_ms", 0.0)) for row in rows]
    missing, unreadable, image_size = validate_images(session)
    return {
        "split": session["split"],
        "session": session["session"],
        "count": len(rows),
        "image_size": image_size,
        "missing_images": missing,
        "unreadable_images": unreadable,
        "steering": describe(steering),
        "speed": describe(speed),
        "steering_dt_ms": describe(steering_dt),
        "speed_dt_ms": describe(speed_dt),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    preview_dir = Path(args.preview_dir).expanduser().resolve()
    sessions = collect_sessions(data_root)
    if not sessions:
        raise RuntimeError(f"No converted metadata.csv files found under {data_root}")

    stats = [session_stats(session) for session in sessions]
    split_totals = {}
    for item in stats:
        split_totals[item["split"]] = split_totals.get(item["split"], 0) + item["count"]

    for item in stats:
        print(
            f"{item['split']}/{item['session']}: count={item['count']} "
            f"image_size={item['image_size']} "
            f"steer_mean={item['steering'].get('mean', 0.0):.3f} "
            f"steer_p95={item['steering'].get('p95', 0.0):.3f} "
            f"speed_mean={item['speed'].get('mean', 0.0):.3f} "
            f"missing={item['missing_images']} unreadable={item['unreadable_images']}"
        )

    print("split totals:", split_totals)
    for split_name in ("train", "valid", "test"):
        path = make_preview_for_split(split_name, sessions, preview_dir, args.samples_per_session)
        if path:
            print(f"wrote {path}")

    output_path = data_root / "inspection_summary.json"
    with open(output_path, "w") as handle:
        json.dump({"split_totals": split_totals, "sessions": stats}, handle, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
