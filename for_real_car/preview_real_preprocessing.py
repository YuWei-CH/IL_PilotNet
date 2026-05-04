#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

IMAGE_MODES = ("rgb", "gray", "gray_autocontrast", "gray_contrast_sharp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the real-data preprocessing that PilotNet will see."
    )
    parser.add_argument("--metadata", required=True, help="Path to one converted metadata.csv")
    parser.add_argument("--output", default="real_preprocessing_preview.jpg")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--crop-top-ratio", type=float, default=0.35)
    parser.add_argument("--crop-bottom-ratio", type=float, default=0.10)
    parser.add_argument("--crop-left-ratio", type=float, default=0.0)
    parser.add_argument("--crop-right-ratio", type=float, default=0.0)
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default="rgb")
    return parser.parse_args()


def read_rows(metadata_path):
    with open(metadata_path, newline="") as handle:
        return list(csv.DictReader(handle))


def crop_image(image, args):
    width, height = image.size
    left = int(width * args.crop_left_ratio)
    right = int(width * (1.0 - args.crop_right_ratio))
    top = int(height * args.crop_top_ratio)
    bottom = int(height * (1.0 - args.crop_bottom_ratio))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return image.crop((left, top, right, bottom))


def apply_image_mode(image, image_mode):
    if image_mode == "rgb":
        return image

    gray = image.convert("L")
    if image_mode == "gray":
        return gray.convert("RGB")
    if image_mode == "gray_autocontrast":
        return ImageOps.autocontrast(gray, cutoff=1).convert("RGB")
    if image_mode == "gray_contrast_sharp":
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = ImageEnhance.Contrast(gray).enhance(1.6)
        gray = ImageEnhance.Sharpness(gray).enhance(1.4)
        return gray.convert("RGB")
    raise RuntimeError(f"Unsupported image mode: {image_mode}")


def sample_indices(count, samples):
    if count <= 0:
        return []
    if samples <= 1:
        return [0]
    return sorted(
        set(int(round((count - 1) * index / (samples - 1))) for index in range(samples))
    )


def main():
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    session_dir = metadata_path.parent
    rows = read_rows(metadata_path)
    if not rows:
        raise RuntimeError(f"No rows in {metadata_path}")

    items = []
    for index in sample_indices(len(rows), args.samples):
        row = rows[index]
        raw = Image.open(session_dir / row["image_path"]).convert("RGB")
        cropped = crop_image(raw, args)
        model_input = cropped.resize((args.image_width, args.image_height), Image.BILINEAR)
        model_input = apply_image_mode(model_input, args.image_mode)

        raw_preview = raw.copy()
        raw_preview.thumbnail((240, 150))
        cropped_preview = cropped.copy()
        cropped_preview.thumbnail((240, 150))
        model_preview = model_input.resize((240, 80), Image.NEAREST)
        items.append((row, raw_preview, cropped_preview, model_preview))

    cell_w = 250
    cell_h = 285
    sheet = Image.new("RGB", (cell_w * len(items), cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (row, raw, cropped, model_input) in enumerate(items):
        x = col * cell_w
        draw.text((x + 5, 4), f"#{row['frame_id']} steer={float(row['steering']):.2f}", fill=(0, 0, 0))
        draw.text((x + 5, 18), "raw", fill=(0, 0, 0))
        sheet.paste(raw, (x + 5, 35))
        draw.text((x + 5, 190), "crop", fill=(0, 0, 0))
        sheet.paste(cropped, (x + 5, 207))
        draw.text((x + 5, 260), "model input", fill=(0, 0, 0))
        sheet.paste(model_input, (x + 5, 275 - model_input.height))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(output_path)


if __name__ == "__main__":
    main()
