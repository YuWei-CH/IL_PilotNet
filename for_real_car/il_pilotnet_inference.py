#!/usr/bin/env python3
# ================================================================
# File name: il_pilotnet_inference.py
# Description:
#   Real GEM vehicle IL driver using PilotNet + PACMod.
#
#   Camera image -> PilotNet -> PACMod steering command
#
#   Speed / gear / brake / PACMod enable logic follows the existing
#   pure_pursuit.py structure.
#
# Safety:
#   - Joystick required
#   - LB + RB: enable vehicle
#   - LB only: disable vehicle
#   - Image timeout: brake and stop
#   - Steering clamp
#   - Steering smoothing
#   - Low default speed
# ================================================================



'''
how to run:

ros2 run your_package il_pilotnet_inference.py \
  --ros-args \
  -p model_path:=/path/to/models/best_model.pt \
  -p desired_speed:=0.5 \
  -p speed_control_mode:=accel \
  -p max_acceleration:=0.15 \
  -p max_steering_wheel_rad:=2.5 \
  -p steering_scale:=0.5 \
  -p steer_smoothing_alpha:=0.25
'''


import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.signal as signal
import pygame
from PIL import Image, ImageEnhance, ImageOps

import torch

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from pacmod2_msgs.msg import (
    PositionWithSpeed,
    VehicleSpeedRpt,
    GlobalCmd,
    SystemCmdFloat,
    SystemCmdInt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pilotnet_model import PilotNet  # noqa: E402

IMAGE_MODES = ("rgb", "gray", "gray_autocontrast", "gray_contrast_sharp")


# ================================================================
# PID Controller
# Same structure as pure_pursuit.py
# ================================================================
class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.wg = wg
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def reset(self):
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = 0.0
            de = 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0

        self.iterm += e * dt

        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)

        self.last_e = e
        self.last_t = t

        return self.kp * e + self.ki * self.iterm + self.kd * de


# ================================================================
# Speed low-pass filter
# Same structure as pure_pursuit.py
# ================================================================
class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        self.b, self.a = signal.butter(
            order,
            normal_cutoff,
            btype='low',
            analog=False
        )
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filtered, self.z = signal.lfilter(
            self.b,
            self.a,
            [data],
            zi=self.z
        )
        return filtered[0]


# ================================================================
# Main IL PACMod Driver
# ================================================================
class ILPacmodDriver(Node):
    def __init__(self):
        super().__init__('il_pilotnet_inference')

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('rate_hz', 20)

        # Camera topic
        self.declare_parameter(
            'camera_topic',
            '/oak/rgb/image_raw/compressed'
        )

        # Model path
        self.declare_parameter(
            'model_path',
            'models/best_model.pt'
        )

        # Driving parameters
        # For first real-car IL test, keep speed low.
        self.declare_parameter('desired_speed', 0.3)
        self.declare_parameter('speed_control_mode', 'accel')
        self.declare_parameter('max_acceleration', 0.2)
        self.declare_parameter('speed_cmd_topic', '/pacmod/vehicle_speed_cmd')

        # Steering safety. The real-data label is /pacmod/steering_rpt.output,
        # matching pure_pursuit.py's final /pacmod/steering_cmd angular_position:
        # steering wheel / PACMod motor angle in radians.
        self.declare_parameter('label_scale', 1.0)
        self.declare_parameter('steering_scale', 0.5)
        self.declare_parameter('max_steering_wheel_rad', 2.5)

        # Steering smoothing:
        # smoothed = alpha * current + (1 - alpha) * previous
        self.declare_parameter('steer_smoothing_alpha', 0.3)

        # If no image arrives within this time, stop the vehicle.
        self.declare_parameter('image_timeout_sec', 0.5)

        # PACMod steering wheel angular velocity limit
        self.declare_parameter('steering_velocity_limit', 4.0)

        # PID parameters
        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10.0)

        # Speed filter
        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30)
        self.declare_parameter('filter/order', 4)

        # Image preprocessing
        self.declare_parameter('input_width', 200)
        self.declare_parameter('input_height', 66)
        self.declare_parameter('crop_top_ratio', 0.35)
        self.declare_parameter('crop_bottom_ratio', 0.10)
        self.declare_parameter('crop_left_ratio', 0.0)
        self.declare_parameter('crop_right_ratio', 0.0)
        self.declare_parameter('image_mode', 'rgb')

        # ----------------------------
        # Load parameters
        # ----------------------------
        self.rate_hz = self.get_parameter('rate_hz').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.model_path = self.get_parameter('model_path').value

        self.desired_speed = min(
            1.0,
            float(self.get_parameter('desired_speed').value)
        )
        self.speed_control_mode = self.get_parameter('speed_control_mode').value
        if self.speed_control_mode not in ('accel', 'speed_cmd'):
            raise RuntimeError("speed_control_mode must be 'accel' or 'speed_cmd'")
        self.max_accel = min(
            0.5,
            float(self.get_parameter('max_acceleration').value)
        )
        self.speed_cmd_topic = self.get_parameter('speed_cmd_topic').value

        self.label_scale = float(self.get_parameter('label_scale').value)
        self.steering_scale = float(self.get_parameter('steering_scale').value)
        self.max_steering_wheel_rad = float(
            self.get_parameter('max_steering_wheel_rad').value
        )
        self.steer_alpha = float(
            self.get_parameter('steer_smoothing_alpha').value
        )
        self.image_timeout_sec = float(
            self.get_parameter('image_timeout_sec').value
        )
        self.steering_velocity_limit = float(
            self.get_parameter('steering_velocity_limit').value
        )

        self.input_width = int(self.get_parameter('input_width').value)
        self.input_height = int(self.get_parameter('input_height').value)
        self.crop_top_ratio = float(self.get_parameter('crop_top_ratio').value)
        self.crop_bottom_ratio = float(self.get_parameter('crop_bottom_ratio').value)
        self.crop_left_ratio = float(self.get_parameter('crop_left_ratio').value)
        self.crop_right_ratio = float(self.get_parameter('crop_right_ratio').value)
        self.image_mode = self.get_parameter('image_mode').value
        if self.image_mode not in IMAGE_MODES:
            raise RuntimeError(f"image_mode must be one of: {IMAGE_MODES}")

        # ----------------------------
        # Device and model
        # ----------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        checkpoint_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
        self.apply_checkpoint_preprocessing(checkpoint_args)

        # Support both raw state_dict and checkpoint dict.
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dict = self.normalize_state_dict(state_dict)

        self.model = PilotNet(dropout=float(checkpoint_args.get('dropout', 0.1))).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.get_logger().info(f"Loaded model from: {self.model_path}")
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(
            "IL command config: speed_control_mode=%s speed_cmd_topic=%s desired_speed=%.2f max_accel=%.2f "
            "steering_unit=pacmod_steering_wheel_rad label_scale=%.2f steering_scale=%.2f "
            "max_steering_wheel_rad=%.2f preprocessing=%dx%d crop=(top=%.2f,bottom=%.2f,left=%.2f,right=%.2f) image_mode=%s"
            % (
                self.speed_control_mode,
                self.speed_cmd_topic,
                self.desired_speed,
                self.max_accel,
                self.label_scale,
                self.steering_scale,
                self.max_steering_wheel_rad,
                self.input_width,
                self.input_height,
                self.crop_top_ratio,
                self.crop_bottom_ratio,
                self.crop_left_ratio,
                self.crop_right_ratio,
                self.image_mode,
            )
        )

        # ----------------------------
        # Joystick init
        # ----------------------------
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick connected. Real-car test requires joystick safety control.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.get_logger().info("Joystick connected.")
        self.get_logger().warn("Safety control: LB + RB enable, LB only disable.")

        # ----------------------------
        # Subscriptions
        # ----------------------------
        self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10
        )

        self.create_subscription(
            Bool,
            '/pacmod/enabled',
            self.enable_callback,
            10
        )

        self.create_subscription(
            VehicleSpeedRpt,
            '/pacmod/vehicle_speed_rpt',
            self.speed_callback,
            10
        )

        # ----------------------------
        # PACMod Publishers
        # ----------------------------
        self.global_pub = self.create_publisher(
            GlobalCmd,
            '/pacmod/global_cmd',
            10
        )

        self.gear_pub = self.create_publisher(
            SystemCmdInt,
            '/pacmod/shift_cmd',
            10
        )

        self.brake_pub = self.create_publisher(
            SystemCmdFloat,
            '/pacmod/brake_cmd',
            10
        )

        self.accel_pub = self.create_publisher(
            SystemCmdFloat,
            '/pacmod/accel_cmd',
            10
        )

        self.speed_cmd_pub = None
        if self.speed_control_mode == 'speed_cmd':
            self.speed_cmd_pub = self.create_publisher(
                SystemCmdFloat,
                self.speed_cmd_topic,
                10
            )

        self.turn_pub = self.create_publisher(
            SystemCmdInt,
            '/pacmod/turn_cmd',
            10
        )

        self.steer_pub = self.create_publisher(
            PositionWithSpeed,
            '/pacmod/steering_cmd',
            10
        )

        # ----------------------------
        # PACMod command messages
        # ----------------------------
        self.global_cmd = GlobalCmd(
            enable=False,
            clear_override=True
        )

        # PACMod gear command:
        # 2: NEUTRAL in your pure_pursuit.py
        # 3: FORWARD in your pure_pursuit.py
        self.gear_cmd = SystemCmdInt(command=2)

        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.speed_cmd = SystemCmdFloat(command=0.0)

        # 1: no signal in your pure_pursuit.py
        self.turn_cmd = SystemCmdInt(command=1)

        self.steer_cmd = PositionWithSpeed(
            angular_position=0.0,
            angular_velocity_limit=self.steering_velocity_limit
        )

        # ----------------------------
        # Runtime states
        # ----------------------------
        self.pacmod_enable = False
        self.speed = 0.0

        self.latest_steering_wheel_rad = 0.0
        self.smoothed_steering_wheel_rad = 0.0

        self.last_image_time = None
        self.received_image = False

        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value,
        )

        self.pid_speed = PID(
            kp=self.get_parameter('pid/kp').value,
            ki=self.get_parameter('pid/ki').value,
            kd=self.get_parameter('pid/kd').value,
            wg=self.get_parameter('pid/wg').value,
        )

        # ----------------------------
        # Timer
        # ----------------------------
        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.control_loop
        )

        self.get_logger().info("IL PACMod driver initialized.")

    def apply_checkpoint_preprocessing(self, checkpoint_args):
        """
        Training stores the preprocessing contract in checkpoint["args"]. When
        it is present, use it as the source of truth so deployment cannot
        accidentally run a different crop, image mode, resize, or label scale.
        """
        if not checkpoint_args:
            self.get_logger().warn(
                "Checkpoint has no args metadata. Falling back to ROS preprocessing parameters."
            )
            self.validate_preprocessing_config()
            return

        self.input_width = int(checkpoint_args.get('image_width', self.input_width))
        self.input_height = int(checkpoint_args.get('image_height', self.input_height))
        self.crop_top_ratio = float(checkpoint_args.get('crop_top_ratio', self.crop_top_ratio))
        self.crop_bottom_ratio = float(checkpoint_args.get('crop_bottom_ratio', self.crop_bottom_ratio))
        self.crop_left_ratio = float(checkpoint_args.get('crop_left_ratio', self.crop_left_ratio))
        self.crop_right_ratio = float(checkpoint_args.get('crop_right_ratio', self.crop_right_ratio))
        self.image_mode = checkpoint_args.get('image_mode', self.image_mode)
        self.label_scale = float(checkpoint_args.get('label_scale', self.label_scale))

        label_unit = checkpoint_args.get('label_unit', '')
        if label_unit and 'pacmod' not in label_unit.lower():
            self.get_logger().warn(
                f"Checkpoint label_unit is '{label_unit}', but this node expects PACMod steering radians."
            )

        self.validate_preprocessing_config()

    def validate_preprocessing_config(self):
        if self.input_width <= 0 or self.input_height <= 0:
            raise RuntimeError("input_width and input_height must be positive")
        if self.image_mode not in IMAGE_MODES:
            raise RuntimeError(f"image_mode must be one of: {IMAGE_MODES}")
        for name in (
            'crop_top_ratio',
            'crop_bottom_ratio',
            'crop_left_ratio',
            'crop_right_ratio',
        ):
            value = getattr(self, name)
            if value < 0.0 or value >= 1.0:
                raise RuntimeError(f"{name} must be in [0.0, 1.0)")
        if self.crop_top_ratio + self.crop_bottom_ratio >= 1.0:
            raise RuntimeError("crop_top_ratio + crop_bottom_ratio must be < 1.0")
        if self.crop_left_ratio + self.crop_right_ratio >= 1.0:
            raise RuntimeError("crop_left_ratio + crop_right_ratio must be < 1.0")
        if self.label_scale <= 0.0:
            raise RuntimeError("label_scale must be positive")

    def normalize_state_dict(self, state_dict):
        # torch.compile stores parameters under "_orig_mod."; strip that so
        # inference can load optimized-training checkpoints normally.
        if not any(key.startswith('_orig_mod.') for key in state_dict):
            return state_dict
        return {
            key.replace('_orig_mod.', '', 1): value
            for key, value in state_dict.items()
        }

    def apply_image_mode(self, image):
        if self.image_mode == 'rgb':
            return image

        gray = image.convert('L')
        if self.image_mode == 'gray':
            return gray.convert('RGB')
        if self.image_mode == 'gray_autocontrast':
            return ImageOps.autocontrast(gray, cutoff=1).convert('RGB')
        if self.image_mode == 'gray_contrast_sharp':
            gray = ImageOps.autocontrast(gray, cutoff=1)
            gray = ImageEnhance.Contrast(gray).enhance(1.6)
            gray = ImageEnhance.Sharpness(gray).enhance(1.4)
            return gray.convert('RGB')
        raise RuntimeError(f"Unsupported image mode: {self.image_mode}")

    def preprocess_image(self, bgr_image):
        # OpenCV decodes compressed ROS images as BGR. Training data is loaded
        # through PIL as RGB, so convert here before applying the same crop,
        # resize, image mode, and CHW tensor layout as train_real_pilotnet.py.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)

        width, height = image.size
        left = int(width * self.crop_left_ratio)
        right = int(width * (1.0 - self.crop_right_ratio))
        top = int(height * self.crop_top_ratio)
        bottom = int(height * (1.0 - self.crop_bottom_ratio))
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        image = image.crop((left, top, right, bottom))

        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image = self.apply_image_mode(image)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    # ============================================================
    # Callbacks
    # ============================================================
    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def speed_callback(self, msg):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                self.get_logger().warn("Failed to decode image.")
                return

            x = self.preprocess_image(img)

            with torch.no_grad():
                pred = self.model(x).item()

            # The model was trained against /pacmod/steering_rpt.output.
            # That is the same PACMod steering wheel/motor angle unit that
            # pure_pursuit.py publishes to /pacmod/steering_cmd.angular_position.
            # Do not run front-wheel-to-steering-wheel conversion here.
            steering_wheel_rad = float(pred) * self.label_scale * self.steering_scale

            steering_wheel_rad = np.clip(
                steering_wheel_rad,
                -self.max_steering_wheel_rad,
                self.max_steering_wheel_rad
            )

            # Smooth steering.
            self.smoothed_steering_wheel_rad = (
                self.steer_alpha * steering_wheel_rad
                + (1.0 - self.steer_alpha) * self.smoothed_steering_wheel_rad
            )

            self.latest_steering_wheel_rad = self.smoothed_steering_wheel_rad

            self.last_image_time = time.time()
            self.received_image = True

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    # ============================================================
    # Utility functions
    # ============================================================
    def check_joystick_enable(self):
        """
        Return:
            1: enable
            0: disable
            2: keep current state
        """
        pygame.event.pump()

        try:
            # Same buttons as pure_pursuit.py
            lb = self.joystick.get_button(6)
            rb = self.joystick.get_button(7)
        except pygame.error:
            self.get_logger().warn("Joystick read failed.")
            return 2

        if lb and rb:
            return 1
        elif lb and not rb:
            return 0
        else:
            return 2

    def image_is_fresh(self):
        if not self.received_image:
            return False

        if self.last_image_time is None:
            return False

        dt = time.time() - self.last_image_time
        return dt <= self.image_timeout_sec

    def publish_speed_cmd(self, command):
        if self.speed_cmd_pub is None:
            return
        self.speed_cmd.command = float(command)
        self.speed_cmd_pub.publish(self.speed_cmd)

    def publish_stop(self):
        self.accel_cmd.command = 0.0
        self.brake_cmd.command = 0.4

        self.steer_cmd.angular_position = 0.0
        self.steer_cmd.angular_velocity_limit = self.steering_velocity_limit

        self.publish_speed_cmd(0.0)
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.steer_pub.publish(self.steer_cmd)

    # ============================================================
    # Main control loop
    # ============================================================
    def control_loop(self):
        joy_enable = self.check_joystick_enable()

        # --------------------------------------------------------
        # Case 1: joystick requests enable
        # --------------------------------------------------------
        if joy_enable == 1 and not self.pacmod_enable:
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)

            # Forward gear
            self.gear_cmd.command = 3
            self.gear_pub.publish(self.gear_cmd)

            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.publish_speed_cmd(0.0)

            # No turn signal
            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.pid_speed.reset()

            self.get_logger().warn(
                "Joystick enable requested: PACMod enable + forward gear."
            )
            return

        # --------------------------------------------------------
        # Case 2: joystick requests disable
        # --------------------------------------------------------
        if joy_enable == 0 and self.pacmod_enable:
            self.publish_stop()

            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.pid_speed.reset()

            self.get_logger().warn("Joystick disable requested: vehicle disabled.")
            return

        # --------------------------------------------------------
        # Case 3: vehicle not enabled
        # --------------------------------------------------------
        if not self.pacmod_enable:
            # Keep safe commands.
            self.accel_cmd.command = 0.0
            self.brake_cmd.command = 0.0
            self.publish_speed_cmd(0.0)
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)
            return

        # --------------------------------------------------------
        # Case 4: enabled but image missing / stale
        # --------------------------------------------------------
        if not self.image_is_fresh():
            self.publish_stop()
            self.get_logger().warn(
                "No fresh camera image. Publishing stop command."
            )
            return

        # --------------------------------------------------------
        # Case 5: normal IL driving
        # --------------------------------------------------------
        self.global_cmd.enable = True
        self.global_cmd.clear_override = True
        self.global_pub.publish(self.global_cmd)

        self.gear_cmd.command = 3
        self.gear_pub.publish(self.gear_cmd)

        self.turn_cmd.command = 1
        self.turn_pub.publish(self.turn_cmd)

        # 1. Steering control. latest_steering_wheel_rad is already in PACMod
        # steering wheel command units.
        steering_wheel_rad = self.latest_steering_wheel_rad
        steering_wheel_deg = math.degrees(steering_wheel_rad)

        self.steer_cmd.angular_position = steering_wheel_rad
        self.steer_cmd.angular_velocity_limit = self.steering_velocity_limit
        self.steer_pub.publish(self.steer_cmd)

        # 2. Speed control. Some GEM4 PACMod2 setups expose only accel/brake
        # commands, while others expose vehicle_speed_cmd. Support both.
        if self.speed_control_mode == 'speed_cmd':
            self.publish_speed_cmd(self.desired_speed)
            self.accel_cmd.command = 0.0
            self.brake_cmd.command = 0.0
            speed_control_text = f"speed_cmd={self.desired_speed:.2f}"
        else:
            now = self.get_clock().now().nanoseconds * 1e-9
            speed_error = self.desired_speed - self.speed
            if abs(speed_error) < 0.05:
                speed_error = 0.0
            accel_cmd = self.pid_speed.get_control(now, speed_error)
            accel_cmd = max(0.0, min(accel_cmd, self.max_accel))
            self.accel_cmd.command = accel_cmd
            self.brake_cmd.command = 0.0
            speed_control_text = f"accel_cmd={accel_cmd:.2f}"

        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)

        self.get_logger().info(
            f"IL driving | "
            f"speed={self.speed:.2f} m/s, "
            f"target_speed={self.desired_speed:.2f} m/s, "
            f"wheel_steer={steering_wheel_deg:.2f} deg, "
            f"{speed_control_text}, "
            f"brake_cmd={self.brake_cmd.command:.2f}, "
            f"gear_cmd={self.gear_cmd.command}, "
            f"global_enable={self.global_cmd.enable}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = ILPacmodDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt. Stopping vehicle.")
        node.publish_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
