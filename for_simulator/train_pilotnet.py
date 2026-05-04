#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pilotnet_model import PilotNet


@dataclass
class Sample:
    image_path: str
    steering: float
    speed: float
    x: float
    y: float
    yaw: float
    session: str


class PilotNetDataset(Dataset):
    def __init__(
        self,
        samples,
        image_width,
        image_height,
        crop_top_ratio,
        crop_bottom_ratio,
        crop_left_ratio,
        crop_right_ratio,
        label_scale,
        preload_images=False,
        preload_workers=0,
    ):
        self.samples = samples
        self.image_width = image_width
        self.image_height = image_height
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_left_ratio = crop_left_ratio
        self.crop_right_ratio = crop_right_ratio
        self.label_scale = label_scale
        self.preload_workers = max(0, int(preload_workers))
        self.preloaded_images = None
        if preload_images:
            self.preloaded_images = self.preload_images()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.preloaded_images is None:
            image = Image.open(sample.image_path).convert("RGB")
            image = self.preprocess_image(image)
        else:
            image = torch.from_numpy(self.preloaded_images[index]).float().div(255.0)
        steering = torch.tensor(sample.steering / self.label_scale, dtype=torch.float32)
        return image, steering

    def preprocess_image_uint8(self, image):
        width, height = image.size
        left = int(width * self.crop_left_ratio)
        right = int(width * (1.0 - self.crop_right_ratio))
        top = int(height * self.crop_top_ratio)
        bottom = int(height * (1.0 - self.crop_bottom_ratio))
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = np.asarray(image, dtype=np.uint8)
        image = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(image)

    def preprocess_image(self, image):
        image = self.preprocess_image_uint8(image).astype(np.float32) / 255.0
        return torch.from_numpy(image)

    def preload_one_image(self, sample):
        with Image.open(sample.image_path) as image:
            return self.preprocess_image_uint8(image.convert("RGB"))

    def preload_images(self):
        images = []
        total = len(self.samples)
        if self.preload_workers <= 1:
            for index, sample in enumerate(self.samples, start=1):
                images.append(self.preload_one_image(sample))
                if index % 2000 == 0:
                    print(f"preloaded {index}/{total} images")
            return images

        print(f"preloading {total} images with {self.preload_workers} workers")
        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            for index, image in enumerate(executor.map(self.preload_one_image, self.samples), start=1):
                images.append(image)
                if index % 2000 == 0:
                    print(f"preloaded {index}/{total} images")
        return images


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PilotNet-style steering regressor")
    parser.add_argument(
        "--data-root",
        default=os.path.expanduser("~/pilotnet_data"),
        help="Root containing session directories with metadata.csv and images/",
    )
    parser.add_argument(
        "--metadata-glob",
        default="**/metadata.csv",
        help="Glob under --data-root used to discover metadata files",
    )
    parser.add_argument("--output-dir", default="pilotnet_runs/run_001", help="Training output directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["auto", "session", "per-session-frame"],
        default="auto",
        help=(
            "auto keeps the previous behavior: session split for multiple sessions "
            "and frame split for one session. per-session-frame splits every session "
            "by time so each session contributes to train and validation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--crop-top-ratio", type=float, default=0.35)
    parser.add_argument("--crop-bottom-ratio", type=float, default=0.10)
    parser.add_argument("--crop-left-ratio", type=float, default=0.0)
    parser.add_argument("--crop-right-ratio", type=float, default=0.0)
    parser.add_argument(
        "--label-scale",
        type=float,
        default=1.0,
        help="Divide steering labels by this value during training. Use 10.0 for real GEM PACMod steering output.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--image-cache-policy",
        choices=["auto", "ram", "off"],
        default="auto",
        help=(
            "Image caching strategy. auto preloads processed uint8 images into CPU RAM "
            "when the estimated cache fits the configured RAM budget; ram always "
            "preloads; off streams images from disk through the DataLoader."
        ),
    )
    parser.add_argument(
        "--image-cache-max-gb",
        type=float,
        default=24.0,
        help="Maximum estimated CPU RAM cache size allowed by --image-cache-policy auto.",
    )
    parser.add_argument(
        "--image-cache-ram-fraction",
        type=float,
        default=0.75,
        help="Maximum fraction of currently available RAM allowed by --image-cache-policy auto.",
    )
    parser.add_argument(
        "--preload-images",
        action="store_true",
        help=(
            "Deprecated alias for --image-cache-policy ram. Preload cropped/resized "
            "images into CPU RAM as uint8 tensors before training."
        ),
    )
    parser.add_argument(
        "--preload-workers",
        type=int,
        default=0,
        help=(
            "Number of threads for --preload-images. Use 0 or 1 for sequential preload; "
            "4-8 is usually enough for JPEG/PIL preprocessing."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument(
        "--amp",
        choices=["off", "fp16", "bf16", "auto"],
        default="off",
        help="Mixed precision mode for CUDA training. auto prefers bf16 when supported.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use NHWC/channels-last memory format on CUDA for convolution speed.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model with torch.compile. This can help long runs but adds startup overhead.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode used when --torch-compile is enabled.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable cuDNN benchmarking. Leave enabled for fixed-size images and faster convolutions.",
    )
    parser.add_argument(
        "--min-abs-speed",
        type=float,
        default=0.0,
        help="Ignore samples below this absolute speed",
    )
    parser.add_argument(
        "--max-samples-per-session",
        type=int,
        default=0,
        help="If >0, cap samples per session for quick experiments",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_metadata_files(data_root, metadata_glob):
    pattern = os.path.join(data_root, metadata_glob)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise RuntimeError(f"No metadata files found under {data_root} with glob {metadata_glob}")
    return files


def load_samples(metadata_files, min_abs_speed=0.0, max_samples_per_session=0):
    sessions = {}
    for metadata_path in metadata_files:
        session_dir = os.path.dirname(metadata_path)
        session_name = os.path.basename(session_dir)
        session_samples = []
        with open(metadata_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                speed = float(row["speed"])
                if abs(speed) < min_abs_speed:
                    continue
                image_path = os.path.join(session_dir, row["image_path"])
                if not os.path.exists(image_path):
                    continue
                session_samples.append(
                    Sample(
                        image_path=image_path,
                        steering=float(row["steering"]),
                        speed=speed,
                        x=float(row["x"]),
                        y=float(row["y"]),
                        yaw=float(row["yaw"]),
                        session=session_name,
                    )
                )
        if max_samples_per_session > 0:
            session_samples = session_samples[:max_samples_per_session]
        if session_samples:
            sessions[session_name] = session_samples

    if not sessions:
        raise RuntimeError("No usable samples found in discovered metadata files")
    return sessions


def split_single_session(samples, val_fraction):
    total = len(samples)
    if total < 2:
        raise RuntimeError("Need at least 2 samples to split a single-session dataset")

    val_count = max(1, int(math.ceil(total * val_fraction)))
    if val_count >= total:
        val_count = total - 1

    split_index = total - val_count
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]
    return train_samples, val_samples


def split_sessions_per_session_frame(session_samples, val_fraction):
    train_samples = []
    val_samples = []
    split_details = {}

    for session_name in sorted(session_samples.keys()):
        session_train, session_val = split_single_session(session_samples[session_name], val_fraction)
        train_samples.extend(session_train)
        val_samples.extend(session_val)
        split_details[session_name] = {
            "train": len(session_train),
            "val": len(session_val),
        }

    return (
        train_samples,
        val_samples,
        sorted(session_samples.keys()),
        sorted(session_samples.keys()),
        "per_session_frame_split",
        split_details,
    )


def split_sessions(session_samples, val_fraction, seed, split_mode):
    session_names = sorted(session_samples.keys())
    if split_mode == "per-session-frame":
        return split_sessions_per_session_frame(session_samples, val_fraction)

    if split_mode == "session" and len(session_names) == 1:
        raise RuntimeError("--split-mode session requires at least two sessions")

    if split_mode == "auto" and len(session_names) == 1:
        session_name = session_names[0]
        train_samples, val_samples = split_single_session(session_samples[session_name], val_fraction)
        split_details = {
            session_name: {
                "train": len(train_samples),
                "val": len(val_samples),
            }
        }
        return (
            train_samples,
            val_samples,
            [session_name],
            [session_name],
            "single_session_frame_split",
            split_details,
        )

    rng = random.Random(seed)
    rng.shuffle(session_names)
    val_count = max(1, int(math.ceil(len(session_names) * val_fraction)))
    if val_count >= len(session_names) and len(session_names) > 1:
        val_count = len(session_names) - 1

    val_sessions = set(session_names[:val_count])
    train_sessions = [name for name in session_names if name not in val_sessions]
    if not train_sessions:
        raise RuntimeError("Need at least one training session after split")

    train_samples = []
    val_samples = []
    split_details = {}
    for session_name, samples in session_samples.items():
        if session_name in val_sessions:
            val_samples.extend(samples)
            split_details[session_name] = {"train": 0, "val": len(samples)}
        else:
            train_samples.extend(samples)
            split_details[session_name] = {"train": len(samples), "val": 0}
    return train_samples, val_samples, train_sessions, sorted(val_sessions), "session_split", split_details


def build_loaders(args, train_samples, val_samples):
    train_dataset = PilotNetDataset(
        train_samples,
        args.image_width,
        args.image_height,
        args.crop_top_ratio,
        args.crop_bottom_ratio,
        args.crop_left_ratio,
        args.crop_right_ratio,
        args.label_scale,
        preload_images=args.preload_images,
        preload_workers=args.preload_workers,
    )
    val_dataset = PilotNetDataset(
        val_samples,
        args.image_width,
        args.image_height,
        args.crop_top_ratio,
        args.crop_bottom_ratio,
        args.crop_left_ratio,
        args.crop_right_ratio,
        args.label_scale,
        preload_images=args.preload_images,
        preload_workers=args.preload_workers,
    )
    prefetch_factor = 2 if args.num_workers > 0 else None
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def compute_loss(predictions, targets, loss_name):
    if loss_name == "mse":
        return F.mse_loss(predictions, targets)
    return F.smooth_l1_loss(predictions, targets)


def resolve_amp_dtype(amp_mode, device):
    if device.type != "cuda" or amp_mode == "off":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_epoch(model, loader, optimizer, device, loss_name, training, amp_dtype=None, scaler=None, channels_last=False):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_mae = 0.0
    total = 0

    for images, steering in loader:
        images = images.to(device, non_blocking=True)
        steering = steering.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                predictions = model(images)
                loss = compute_loss(predictions, steering, loss_name)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_mae += float(torch.abs(predictions - steering).mean().item()) * batch_size
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "mae": running_mae / max(total, 1),
    }


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def validate_args(args):
    for name in ("crop_top_ratio", "crop_bottom_ratio", "crop_left_ratio", "crop_right_ratio"):
        value = getattr(args, name)
        if value < 0.0 or value >= 1.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be in [0.0, 1.0)")
    if args.crop_top_ratio + args.crop_bottom_ratio >= 1.0:
        raise RuntimeError("--crop-top-ratio + --crop-bottom-ratio must be < 1.0")
    if args.crop_left_ratio + args.crop_right_ratio >= 1.0:
        raise RuntimeError("--crop-left-ratio + --crop-right-ratio must be < 1.0")
    if args.label_scale <= 0.0:
        raise RuntimeError("--label-scale must be positive")


def get_available_ram_bytes():
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * available_pages)
        except (OSError, ValueError):
            pass
    return None


def estimate_image_cache_bytes(num_samples, image_width, image_height):
    return int(num_samples) * int(image_width) * int(image_height) * 3


def format_gb(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f}GB"


def configure_image_cache(args, num_samples):
    if args.preload_images:
        args.image_cache_policy = "ram"

    estimated_bytes = estimate_image_cache_bytes(num_samples, args.image_width, args.image_height)
    available_bytes = get_available_ram_bytes()
    max_bytes = int(max(args.image_cache_max_gb, 0.0) * (1024 ** 3))
    fraction_bytes = None
    if available_bytes is not None:
        fraction_bytes = int(max(args.image_cache_ram_fraction, 0.0) * available_bytes)

    should_cache = False
    reason = ""
    if args.image_cache_policy == "ram":
        should_cache = True
        reason = "forced by --image-cache-policy ram"
    elif args.image_cache_policy == "off":
        should_cache = False
        reason = "disabled by --image-cache-policy off"
    else:
        budget_bytes = max_bytes
        if fraction_bytes is not None:
            budget_bytes = min(budget_bytes, fraction_bytes)
        should_cache = estimated_bytes <= budget_bytes
        if should_cache:
            reason = f"estimated cache {format_gb(estimated_bytes)} fits budget {format_gb(budget_bytes)}"
        else:
            reason = f"estimated cache {format_gb(estimated_bytes)} exceeds budget {format_gb(budget_bytes)}"

    available_text = format_gb(available_bytes) if available_bytes is not None else "unknown"
    print(
        "image cache: "
        f"policy={args.image_cache_policy} enabled={should_cache} "
        f"estimated={format_gb(estimated_bytes)} available_ram={available_text} "
        f"max={args.image_cache_max_gb:.2f}GB ram_fraction={args.image_cache_ram_fraction:.2f} "
        f"reason={reason}"
    )

    args.preload_images = should_cache
    if args.preload_images and args.num_workers > 0:
        print("warning: RAM image cache uses --num-workers 0 to avoid duplicating the cache in workers")
        args.num_workers = 0


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    metadata_files = discover_metadata_files(args.data_root, args.metadata_glob)
    session_samples = load_samples(
        metadata_files,
        min_abs_speed=args.min_abs_speed,
        max_samples_per_session=args.max_samples_per_session,
    )
    train_samples, val_samples, train_sessions, val_sessions, split_mode, split_details = split_sessions(
        session_samples,
        args.val_fraction,
        args.seed,
        args.split_mode,
    )
    configure_image_cache(args, len(train_samples) + len(val_samples))

    os.makedirs(args.output_dir, exist_ok=True)
    train_loader, val_loader = build_loaders(args, train_samples, val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        torch.set_float32_matmul_precision("high")

    amp_dtype = resolve_amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    model = PilotNet(dropout=args.dropout).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.torch_compile:
        model = torch.compile(model, mode=args.compile_mode)

    print(
        "torch optimization: "
        f"device={device} amp={args.amp} amp_dtype={amp_dtype} "
        f"channels_last={args.channels_last and device.type == 'cuda'} "
        f"torch_compile={args.torch_compile} "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark if device.type == 'cuda' else False}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics = {
        "split_mode": split_mode,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "split_details": split_details,
        "epochs": [],
    }

    if split_mode != "session_split":
        print(
            "warning: validation uses frame-level splits within session(s), so it "
            "may be optimistic compared with a held-out session"
        )

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    latest_ckpt_path = os.path.join(args.output_dir, "latest_model.pt")
    save_json(os.path.join(args.output_dir, "train_args.json"), vars(args))
    save_json(os.path.join(args.output_dir, "metrics.json"), metrics)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss,
            training=True,
            amp_dtype=amp_dtype,
            scaler=scaler,
            channels_last=args.channels_last and device.type == "cuda",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            args.loss,
            training=False,
            amp_dtype=amp_dtype,
            scaler=None,
            channels_last=args.channels_last and device.type == "cuda",
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        metrics["epochs"].append(epoch_metrics)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, latest_ckpt_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, best_ckpt_path)
        save_json(os.path.join(args.output_dir, "metrics.json"), metrics)
        save_json(os.path.join(args.output_dir, "train_args.json"), vars(args))

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

    print(f"saved best checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
