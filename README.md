# IL PilotNet

This project uses imitation learning (IL) to train a vehicle to perform lane
following autonomously from camera images and human/teacher steering labels.
The model architecture is based on NVIDIA PilotNet, adapted here for both the
POLARIS GEM simulator and the real GEM4 vehicle workflow.

## Layout

- `pilotnet_model.py`: shared PilotNet model definition.
- `for_simulator/`: simulator training, teacher-student data collection notes,
  and inference commands.
- `for_real_car/`: real GEM4 data conversion, training, preprocessing previews,
  and experimental ROS2 PACMod2 inference.

Start with:

```text
for_simulator/README.md
for_real_car/README.md
```

Large datasets, local virtual environments, and debug image dumps are
intentionally ignored by git. The current best simulator checkpoint is kept in
the repository for convenience.

## Python Environment

Create the local Python environment with `uv`:

```bash
uv sync
```

Run project scripts with `uv run python ...`; `uv` will use the project `.venv`.
You can also activate it manually with `source .venv/bin/activate` if preferred.

This installs the training/data-processing dependencies, including PyTorch. For
CUDA training, verify PyTorch sees the GPU:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

The real-car ROS2 inference script also needs the vehicle workspace sourced so
`rclpy`, `sensor_msgs`, and `pacmod2_msgs` are available:

```bash
source install/setup.bash
```
