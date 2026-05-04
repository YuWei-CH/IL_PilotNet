#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pilotnet_model import PilotNet  # noqa: E402

IMAGE_MODES = ("rgb", "gray", "gray_autocontrast", "gray_contrast_sharp")


@dataclass
class Sample:
    image_path: Path
    steering: float
    speed: float
    session: str


class RealPilotNetDataset(Dataset):
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
        image_mode,
        augment=False,
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
        self.image_mode = image_mode
        self.augment = augment
        self.preload_workers = max(0, int(preload_workers))
        self.preloaded_images = None
        if preload_images:
            if self.augment:
                raise RuntimeError("Cannot preload images while augmentation is enabled")
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

        raw_target = sample.steering
        steering = torch.tensor(raw_target / self.label_scale, dtype=torch.float32)
        raw_target = torch.tensor(raw_target, dtype=torch.float32)
        return image, steering, raw_target

    def apply_image_mode(self, image):
        if self.image_mode == "rgb":
            return image

        gray = image.convert("L")
        if self.image_mode == "gray":
            return gray.convert("RGB")
        if self.image_mode == "gray_autocontrast":
            return ImageOps.autocontrast(gray, cutoff=1).convert("RGB")
        if self.image_mode == "gray_contrast_sharp":
            gray = ImageOps.autocontrast(gray, cutoff=1)
            gray = ImageEnhance.Contrast(gray).enhance(1.6)
            gray = ImageEnhance.Sharpness(gray).enhance(1.4)
            return gray.convert("RGB")
        raise RuntimeError(f"Unsupported image mode: {self.image_mode}")

    def preprocess_image_uint8(self, image):
        # Crop irrelevant upper/background regions, then resize to the shared
        # PilotNet input size.
        width, height = image.size
        left = int(width * self.crop_left_ratio)
        right = int(width * (1.0 - self.crop_right_ratio))
        top = int(height * self.crop_top_ratio)
        bottom = int(height * (1.0 - self.crop_bottom_ratio))
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        image = image.crop((left, top, right, bottom))

        if self.augment:
            # Real data has stronger lighting variation than simulation, so use
            # conservative photometric augmentation only on the training split.
            if random.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.8:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.25))
            if random.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = self.apply_image_mode(image)
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
    parser = argparse.ArgumentParser(description="Train PilotNet on converted real driving data.")
    parser.add_argument("--data-root", default="real_pilotnet_data")
    parser.add_argument("--output-dir", default="real_pilotnet_runs/run_001")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--crop-top-ratio", type=float, default=0.35)
    parser.add_argument("--crop-bottom-ratio", type=float, default=0.10)
    parser.add_argument("--crop-left-ratio", type=float, default=0.0)
    parser.add_argument("--crop-right-ratio", type=float, default=0.0)
    parser.add_argument(
        "--image-mode",
        choices=IMAGE_MODES,
        default="rgb",
        help=(
            "Final image preprocessing mode. Non-rgb modes are also stored in "
            "the checkpoint so real-car inference can apply the same transform."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--label-scale",
        type=float,
        default=None,
        help=(
            "Normalize target labels by this value during training. Defaults "
            "to 10.0 for real-vehicle PACMod steering output."
        ),
    )
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument(
        "--image-cache-policy",
        choices=["auto", "ram", "off"],
        default="auto",
        help=(
            "Image caching strategy. auto preloads processed uint8 images into CPU RAM "
            "when the estimated cache fits the configured RAM budget. Training images "
            "are cached only when augmentation is disabled."
        ),
    )
    parser.add_argument("--image-cache-max-gb", type=float, default=24.0)
    parser.add_argument("--image-cache-ram-fraction", type=float, default=0.75)
    parser.add_argument(
        "--preload-images",
        action="store_true",
        help="Deprecated alias for --image-cache-policy ram.",
    )
    parser.add_argument("--preload-workers", type=int, default=0)
    parser.add_argument(
        "--amp",
        choices=["off", "fp16", "bf16", "auto"],
        default="off",
        help="Mixed precision mode for CUDA training. auto prefers bf16 when supported.",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finalize_args(args):
    for name in ("crop_top_ratio", "crop_bottom_ratio", "crop_left_ratio", "crop_right_ratio"):
        value = getattr(args, name)
        if value < 0.0 or value >= 1.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be in [0.0, 1.0)")
    if args.crop_top_ratio + args.crop_bottom_ratio >= 1.0:
        raise RuntimeError("--crop-top-ratio + --crop-bottom-ratio must be < 1.0")
    if args.crop_left_ratio + args.crop_right_ratio >= 1.0:
        raise RuntimeError("--crop-left-ratio + --crop-right-ratio must be < 1.0")
    if args.label_scale is None:
        args.label_scale = 10.0
    if args.label_scale <= 0.0:
        raise RuntimeError("--label-scale must be positive")
    args.label_unit = "pacmod steering_rpt.output"
    return args


def load_split_samples(data_root, split_name):
    # Read every converted session under one fixed split. We keep the session
    # name so test metrics can be reported per run.
    split_dir = data_root / split_name
    if not split_dir.exists():
        raise RuntimeError(f"Missing split directory: {split_dir}")

    samples = []
    for metadata_path in sorted(split_dir.glob("*/metadata.csv")):
        session_dir = metadata_path.parent
        session_name = session_dir.name
        with open(metadata_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = session_dir / row["image_path"]
                if not image_path.exists():
                    continue
                samples.append(
                    Sample(
                        image_path=image_path,
                        steering=float(row["steering"]),
                        speed=float(row["speed"]),
                        session=session_name,
                    )
                )
    if not samples:
        raise RuntimeError(f"No samples found in split {split_name!r} under {data_root}")
    return samples


def load_optional_split_samples(data_root, split_name):
    split_dir = data_root / split_name
    if not split_dir.exists():
        return []
    return load_split_samples(data_root, split_name)


def build_loader(samples, args, training, preload_images):
    # Augmentation and shuffling are enabled only for training.
    dataset = RealPilotNetDataset(
        samples=samples,
        image_width=args.image_width,
        image_height=args.image_height,
        crop_top_ratio=args.crop_top_ratio,
        crop_bottom_ratio=args.crop_bottom_ratio,
        crop_left_ratio=args.crop_left_ratio,
        crop_right_ratio=args.crop_right_ratio,
        label_scale=args.label_scale,
        image_mode=args.image_mode,
        augment=training and not args.no_augment,
        preload_images=preload_images,
        preload_workers=args.preload_workers,
    )
    prefetch_factor = 2 if args.num_workers > 0 else None
    persistent_workers = args.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


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


def run_epoch(model, loader, optimizer, device, args, training, amp_dtype=None, scaler=None, channels_last=False):
    # Shared train/eval loop. When training=False, gradients and optimizer steps
    # are disabled but the same metrics are accumulated.
    model.train(training)
    total = 0
    running_loss = 0.0
    running_mae = 0.0
    running_rmse_sum = 0.0

    for images, steering, raw_target in loader:
        images = images.to(device, non_blocking=True)
        steering = steering.to(device, non_blocking=True)
        raw_target = raw_target.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                predictions = model(images)
                loss = compute_loss(predictions, steering, args.loss)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Convert predictions back to PACMod steering output units for metrics.
        raw_predictions = predictions * args.label_scale
        raw_error = raw_predictions - raw_target
        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_mae += float(torch.abs(raw_error).mean().item()) * batch_size
        running_rmse_sum += float(torch.square(raw_error).sum().item())
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "mae": running_mae / max(total, 1),
        "rmse": (running_rmse_sum / max(total, 1)) ** 0.5,
    }


def evaluate_by_session(model, samples, args, device, amp_dtype=None, channels_last=False):
    # Test data is split by run to expose failures hidden by an overall average.
    session_metrics = {}
    sessions = sorted({sample.session for sample in samples})
    for session in sessions:
        session_samples = [sample for sample in samples if sample.session == session]
        loader = build_loader(session_samples, args, training=False, preload_images=args.preload_images_eval)
        session_metrics[session] = run_epoch(
            model,
            loader,
            None,
            device,
            args,
            training=False,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        session_metrics[session]["num_samples"] = len(session_samples)
    return session_metrics


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


def configure_image_cache(args, train_count, valid_count, test_count):
    if args.preload_images:
        args.image_cache_policy = "ram"

    train_cache_allowed = args.no_augment
    cache_count = valid_count + test_count + (train_count if train_cache_allowed else 0)
    estimated_bytes = estimate_image_cache_bytes(cache_count, args.image_width, args.image_height)
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
        f"train_cache_allowed={train_cache_allowed} estimated={format_gb(estimated_bytes)} "
        f"available_ram={available_text} max={args.image_cache_max_gb:.2f}GB "
        f"ram_fraction={args.image_cache_ram_fraction:.2f} reason={reason}"
    )

    args.preload_images_train = should_cache and train_cache_allowed
    args.preload_images_eval = should_cache
    if should_cache and args.num_workers > 0:
        print("warning: RAM image cache uses --num-workers 0 to avoid duplicating the cache in workers")
        args.num_workers = 0


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_pretrained_if_requested(model, checkpoint_path, device):
    # Optional warm start from another compatible PilotNet checkpoint.
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint_path


def get_model_state_dict(model):
    # torch.compile wraps the original module and prefixes state_dict keys with
    # "_orig_mod.". Save the underlying module so inference scripts can load the
    # checkpoint into a normal PilotNet.
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)


def main():
    args = parse_args()
    args = finalize_args(args)
    set_seed(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_split_samples(data_root, "train")
    valid_samples = load_split_samples(data_root, "valid")
    test_samples = load_optional_split_samples(data_root, "test")
    configure_image_cache(args, len(train_samples), len(valid_samples), len(test_samples))

    train_loader = build_loader(train_samples, args, training=True, preload_images=args.preload_images_train)
    valid_loader = build_loader(valid_samples, args, training=False, preload_images=args.preload_images_eval)
    test_loader = (
        build_loader(test_samples, args, training=False, preload_images=args.preload_images_eval)
        if test_samples
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        torch.set_float32_matmul_precision("high")

    amp_dtype = resolve_amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    model = PilotNet(dropout=args.dropout).to(device)
    loaded_checkpoint = load_pretrained_if_requested(model, args.pretrained_checkpoint, device)
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
        "num_train_samples": len(train_samples),
        "num_valid_samples": len(valid_samples),
        "num_test_samples": len(test_samples),
        "label_scale": args.label_scale,
        "label_unit": args.label_unit,
        "image_mode": args.image_mode,
        "pretrained_checkpoint": loaded_checkpoint,
        "epochs": [],
    }

    best_valid_loss = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"
    latest_ckpt_path = output_dir / "latest_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            training=True,
            amp_dtype=amp_dtype,
            scaler=scaler,
            channels_last=args.channels_last and device.type == "cuda",
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            None,
            device,
            args,
            training=False,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last and device.type == "cuda",
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "valid_loss": valid_metrics["loss"],
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
        }
        metrics["epochs"].append(epoch_metrics)

        # Save latest every epoch, and keep best_model.pt selected by validation
        # loss. Test metrics are computed only after training is finished.
        checkpoint = {
            "model_state_dict": get_model_state_dict(model),
            "args": vars(args),
            "epoch": epoch,
            "metrics": epoch_metrics,
            "label_unit": args.label_unit,
            "image_mode": args.image_mode,
        }
        torch.save(checkpoint, latest_ckpt_path)
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            torch.save(checkpoint, best_ckpt_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.4f} "
            f"valid_loss={valid_metrics['loss']:.6f} valid_mae={valid_metrics['mae']:.4f}"
        )

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "train_args.json", vars(args))

    print(f"saved best checkpoint to {best_ckpt_path}")

    if test_loader is not None:
        best_checkpoint = torch.load(best_ckpt_path, map_location=device)
        load_model_state_dict(model, best_checkpoint["model_state_dict"])
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            device,
            args,
            training=False,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last and device.type == "cuda",
        )
        test_report = {
            "overall": test_metrics,
            "by_session": evaluate_by_session(
                model,
                test_samples,
                args,
                device,
                amp_dtype=amp_dtype,
                channels_last=args.channels_last and device.type == "cuda",
            ),
        }
        save_json(output_dir / "test_metrics.json", test_report)
        print(
            f"test_loss={test_metrics['loss']:.6f} "
            f"test_mae={test_metrics['mae']:.4f} test_rmse={test_metrics['rmse']:.4f}"
        )
    else:
        print("no test split found; skipped test evaluation")


if __name__ == "__main__":
    main()
