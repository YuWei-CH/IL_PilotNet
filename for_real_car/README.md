# Real GEM4 PilotNet Workflow

This folder contains the real-car side of the PilotNet project:

- convert ROS2 MCAP bags into image/metadata datasets;
- inspect and preview the converted data;
- train a PilotNet model that predicts real GEM4 PACMod steering output;
- run an experimental ROS2 PACMod2 control node.

The shared model definition remains at:

```text
../pilotnet_model.py
```

## Files

- `convert_mcap_to_pilotnet_dataset.py`: convert real ROS2 bags into PilotNet image/CSV sessions.
- `inspect_real_dataset.py`: validate converted sessions and generate split previews.
- `preview_real_preprocessing.py`: visualize crop/resize/image-mode preprocessing.
- `train_real_pilotnet.py`: train the real-car PilotNet model.
- `real_data_splits.json`: selected train/valid/test bag split.
- `il_pilotnet_inference.py`: experimental ROS2 PACMod2 control node for GEM4.

Generated datasets, previews, and debug dumps are ignored by git. Put all model
runs in the repository-level `../pilotnet_runs/` directory so simulator and
real-car checkpoints share one place.

## 1. Convert MCAP Bags

Run commands from this folder:

```bash
cd for_real_car
```

The Python environment is managed by `uv` from the repository root. The commands
below use `uv run`, which will use the project `.venv` created by `uv sync`.

Smoke conversion:

```bash
uv run python convert_mcap_to_pilotnet_dataset.py \
  --splits real_data_splits.json \
  --output-root real_pilotnet_data_smoke \
  --max-frames-per-session 20 \
  --overwrite
```

Full conversion:

```bash
uv run python convert_mcap_to_pilotnet_dataset.py \
  --splits real_data_splits.json \
  --output-root real_pilotnet_data \
  --overwrite
```

The steering label is:

```text
/pacmod/steering_rpt.output
```

## 2. Inspect Dataset

```bash
uv run python inspect_real_dataset.py \
  --data-root real_pilotnet_data \
  --preview-dir dataset_previews
```

This checks image readability, steering/speed distributions, synchronization
deltas, and writes preview contact sheets.

## 3. Preview Preprocessing

```bash
uv run python preview_real_preprocessing.py \
  --metadata real_pilotnet_data/train/lane1_2026-04-20_18-45-24/metadata.csv \
  --output dataset_previews/real_preprocessing_preview.jpg \
  --crop-top-ratio 0.65 \
  --crop-bottom-ratio 0.05 \
  --crop-left-ratio 0.08 \
  --crop-right-ratio 0.08 \
  --image-mode rgb
```

Supported image modes:

```text
rgb
gray
gray_autocontrast
gray_contrast_sharp
```

Use the same crop and image mode during training and inference.

## 4. Train

Basic run:

```bash
uv run python train_real_pilotnet.py \
  --data-root real_pilotnet_data \
  --output-dir ../pilotnet_runs/run_real_vehicle \
  --epochs 25 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --num-workers 4
```

Optimized CUDA run:

```bash
uv run python train_real_pilotnet.py \
  --data-root real_pilotnet_data \
  --output-dir ../pilotnet_runs/run_real_gem4_full_optimized \
  --epochs 25 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --crop-top-ratio 0.65 \
  --crop-bottom-ratio 0.05 \
  --crop-left-ratio 0.08 \
  --crop-right-ratio 0.08 \
  --image-mode rgb \
  --num-workers 4 \
  --image-cache-policy auto \
  --image-cache-max-gb 24 \
  --image-cache-ram-fraction 0.75 \
  --preload-workers 8 \
  --amp auto \
  --channels-last \
  --torch-compile \
  --compile-mode reduce-overhead
```

Optimization notes:

- `--image-cache-policy auto` caches processed uint8 images in CPU RAM when the
  cache fits the configured memory budget. Validation/test can be cached by
  default; training images are cached only when augmentation is disabled with
  `--no-augment`.
- `--preload-workers` parallelizes one-time image preload.
- `--amp auto` enables CUDA mixed precision when supported.
- `--channels-last` uses a convolution-friendly CUDA memory layout.
- `--torch-compile` compiles the model. It adds startup cost but can help
  longer runs.

When RAM cache is enabled, the script sets DataLoader `num_workers` to `0` to
avoid duplicating cached images across worker processes.

Outputs:

- `../pilotnet_runs/<run_name>/best_model.pt`
- `../pilotnet_runs/<run_name>/latest_model.pt`
- `../pilotnet_runs/<run_name>/metrics.json`
- `../pilotnet_runs/<run_name>/train_args.json`
- `../pilotnet_runs/<run_name>/test_metrics.json` if a test split exists

The model predicts normalized steering during training:

```text
network_output = pacmod_steering_output / label_scale
```

Default `label_scale` is `10.0`.

## 5. Real-Car PACMod2 Control

`il_pilotnet_inference.py` is an experimental ROS2 node for the GEM4 PACMod2 stack.
It subscribes to the compressed OAK camera image and publishes PACMod steering,
gear, accel, brake, turn, and global commands.

Example direct run after sourcing the vehicle ROS2 workspace:

```bash
python3 il_pilotnet_inference.py \
  --ros-args \
  -p model_path:=/path/to/best_model.pt \
  -p desired_speed:=0.3 \
  -p speed_control_mode:=accel \
  -p max_acceleration:=0.2 \
  -p max_steering_wheel_rad:=2.5 \
  -p steering_scale:=0.5 \
  -p steer_smoothing_alpha:=0.3
```

Before driving, verify the active topic types on the vehicle:

```bash
source install/setup.bash
ros2 topic info /oak/rgb/image_raw/compressed
ros2 topic info /pacmod/steering_cmd
ros2 topic info /pacmod/accel_cmd
ros2 topic info /pacmod/brake_cmd
ros2 topic info /pacmod/enabled
```

The driver loads the shared `../pilotnet_model.py` architecture. For
checkpoints produced by `train_real_pilotnet.py`, it also reads preprocessing
metadata from `checkpoint["args"]` and applies the same image width/height,
crop ratios, image mode, and `label_scale` used during training. If an older
checkpoint has no metadata, it falls back to the ROS parameters.

## Real-Data-To-Simulator Experiments

We tried using real GEM4 highbay data to train a model for simulator driving.
Those experiments included crop alignment, grayscale/autocontrast
preprocessing, steering sign checks, and estimated PACMod-to-Ackermann steering
ratios. The results were poor because of visual domain shift and steering unit
mismatch.

Simulator code lives in this repository under:

```text
../for_simulator
```
