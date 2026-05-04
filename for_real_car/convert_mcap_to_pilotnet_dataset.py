#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
from bisect import bisect_left
from pathlib import Path

from rosbags.highlevel import AnyReader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert real ROS 2 MCAP driving bags into PilotNet image/CSV sessions."
    )
    parser.add_argument("--splits", default="real_data_splits.json")
    parser.add_argument("--output-root", default="real_pilotnet_data")
    parser.add_argument("--image-topic", default="/oak/rgb/image_raw/compressed")
    parser.add_argument("--steering-topic", default="/pacmod/steering_rpt")
    parser.add_argument("--speed-topic", default="/pacmod/vehicle_speed_rpt")
    parser.add_argument(
        "--steering-field",
        choices=["output", "manual_input", "command"],
        default="output",
        help="Field from pacmod2_msgs/SystemRptFloat to use as the label.",
    )
    parser.add_argument("--min-speed", type=float, default=0.05)
    parser.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames-per-session", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def load_splits(path):
    with open(path) as handle:
        splits = json.load(handle)
    for split_name in ("train", "valid", "test"):
        if split_name not in splits:
            raise RuntimeError(f"Missing split {split_name!r} in {path}")
    return splits


def nearest_record(records, record_times, target_time):
    """Return the label record closest in time to one camera frame.

    Camera, steering, and speed topics are recorded at different rates. For
    each image timestamp we use nearest-neighbor synchronization instead of
    requiring exact timestamp matches.
    """
    if not records:
        return None, None
    index = bisect_left(record_times, target_time)
    candidates = []
    if index < len(records):
        candidates.append(records[index])
    if index > 0:
        candidates.append(records[index - 1])
    best = min(candidates, key=lambda item: abs(item["stamp_ns"] - target_time))
    return best, abs(best["stamp_ns"] - target_time)


def collect_label_streams(bag_dir, args):
    steering = []
    speeds = []
    wanted_topics = {args.steering_topic, args.speed_topic}

    with AnyReader([bag_dir]) as reader:
        connections = [conn for conn in reader.connections if conn.topic in wanted_topics]
        missing = wanted_topics - {conn.topic for conn in connections}
        if missing:
            raise RuntimeError(f"{bag_dir} is missing required topics: {sorted(missing)}")

        for conn, bag_time, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            msg_stamp_ns = stamp_ns(msg.header)
            if conn.topic == args.steering_topic:
                steering.append(
                    {
                        "stamp_ns": msg_stamp_ns,
                        "bag_time_ns": int(bag_time),
                        "value": float(getattr(msg, args.steering_field)),
                    }
                )
            elif conn.topic == args.speed_topic:
                speeds.append(
                    {
                        "stamp_ns": msg_stamp_ns,
                        "bag_time_ns": int(bag_time),
                        "value": float(msg.vehicle_speed),
                        "valid": bool(msg.vehicle_speed_valid),
                    }
                )

    steering.sort(key=lambda item: item["stamp_ns"])
    speeds.sort(key=lambda item: item["stamp_ns"])
    return steering, speeds


def open_metadata(path):
    handle = open(path, "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(
        [
            "frame_id",
            "timestamp",
            "image_path",
            "steering",
            "speed",
            "x",
            "y",
            "yaw",
            "source_bag",
            "steering_source",
            "speed_source",
            "steering_dt_ms",
            "speed_dt_ms",
        ]
    )
    return handle, writer


def save_image_bytes(path, data):
    with open(path, "wb") as handle:
        handle.write(bytes(data))


def convert_bag(bag_dir, split_name, output_root, args):
    bag_dir = Path(bag_dir).expanduser().resolve()
    session_name = bag_dir.name
    session_dir = output_root / split_name / session_name
    images_dir = session_dir / "images"

    if session_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output session already exists: {session_dir}. Use --overwrite.")
        shutil.rmtree(session_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    steering, speeds = collect_label_streams(bag_dir, args)
    steering_times = [item["stamp_ns"] for item in steering]
    speed_times = [item["stamp_ns"] for item in speeds]

    metadata_handle, metadata_writer = open_metadata(session_dir / "metadata.csv")
    kept = 0
    skipped_low_speed = 0
    skipped_sync = 0
    skipped_missing = 0
    seen_images = 0

    try:
        with AnyReader([bag_dir]) as reader:
            image_connections = [conn for conn in reader.connections if conn.topic == args.image_topic]
            if not image_connections:
                raise RuntimeError(f"{bag_dir} is missing image topic {args.image_topic}")

            for conn, bag_time, raw in reader.messages(connections=image_connections):
                msg = reader.deserialize(raw, conn.msgtype)
                image_stamp_ns = stamp_ns(msg.header)
                seen_images += 1

                # Step 3: synchronize each camera frame to the nearest steering
                # and speed reports. These reports are usually higher/lower rate
                # than the camera stream, so exact stamp equality is not expected.
                steering_record, steering_delta_ns = nearest_record(
                    steering, steering_times, image_stamp_ns
                )
                speed_record, speed_delta_ns = nearest_record(speeds, speed_times, image_stamp_ns)
                if steering_record is None or speed_record is None:
                    skipped_missing += 1
                    continue

                # Keep the sync deltas in metadata for debugging, and drop frames
                # whose nearest label is too far away to trust as the human action
                # corresponding to this image.
                steering_dt_ms = steering_delta_ns / 1_000_000.0
                speed_dt_ms = speed_delta_ns / 1_000_000.0
                if steering_dt_ms > args.max_sync_delta_ms or speed_dt_ms > args.max_sync_delta_ms:
                    skipped_sync += 1
                    continue

                # Step 4: filter out stopped/invalid samples. Low-speed frames are
                # often startup, braking, or paused segments and add weak labels
                # for a lane-following steering policy.
                speed = speed_record["value"]
                if (not speed_record["valid"]) or abs(speed) < args.min_speed:
                    skipped_low_speed += 1
                    continue

                kept += 1
                frame_id = f"{kept:06d}"
                image_filename = f"{frame_id}.jpg"
                image_path = images_dir / image_filename
                save_image_bytes(image_path, msg.data)

                metadata_writer.writerow(
                    [
                        frame_id,
                        f"{image_stamp_ns / 1_000_000_000.0:.9f}",
                        f"images/{image_filename}",
                        f"{steering_record['value']:.9f}",
                        f"{speed:.9f}",
                        "0.0",
                        "0.0",
                        "0.0",
                        session_name,
                        args.steering_field,
                        "vehicle_speed",
                        f"{steering_dt_ms:.6f}",
                        f"{speed_dt_ms:.6f}",
                    ]
                )

                if args.max_frames_per_session > 0 and kept >= args.max_frames_per_session:
                    break
    finally:
        metadata_handle.close()

    return {
        "split": split_name,
        "session": session_name,
        "source": str(bag_dir),
        "seen_images": seen_images,
        "kept": kept,
        "skipped_low_speed": skipped_low_speed,
        "skipped_sync": skipped_sync,
        "skipped_missing": skipped_missing,
    }


def main():
    args = parse_args()
    splits = load_splits(args.splits)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for split_name, bag_paths in splits.items():
        for bag_path in bag_paths:
            result = convert_bag(Path(bag_path), split_name, output_root, args)
            summary.append(result)
            print(
                f"{result['split']}/{result['session']}: "
                f"kept={result['kept']} seen_images={result['seen_images']} "
                f"low_speed={result['skipped_low_speed']} sync={result['skipped_sync']}"
            )

    with open(output_root / "conversion_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {output_root / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
