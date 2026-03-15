"""Sensor processing for the SOS rescue gym — IMU ring buffer and camera rendering."""

import math
import numpy as np
import mujoco


class IMUBuffer:
    """Ring buffer that accumulates IMU readings at 100Hz and returns [100, 9] windows.

    9 channels: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, lin_accel_x, lin_accel_y, lin_accel_z
    """

    def __init__(self, seq_len: int = 100):
        self.seq_len = seq_len
        self.buffer = np.zeros((seq_len, 9), dtype=np.float32)
        self._idx = 0
        self._full = False

    def reset(self):
        self.buffer[:] = 0.0
        self._idx = 0
        self._full = False

    def push(self, gyro: np.ndarray, accel: np.ndarray, lin_accel: np.ndarray):
        """Push a single IMU sample (3 + 3 + 3 = 9 channels)."""
        self.buffer[self._idx] = np.concatenate([gyro, accel, lin_accel])
        self._idx = (self._idx + 1) % self.seq_len
        if self._idx == 0:
            self._full = True

    def get(self) -> np.ndarray:
        """Return the current [seq_len, 9] window in chronological order."""
        if self._full:
            return np.roll(self.buffer, -self._idx, axis=0).copy()
        return self.buffer.copy()

    @property
    def vibration_magnitude(self) -> float:
        """RMS of accelerometer magnitude deviation from gravity."""
        accel = self.buffer[:, 3:6]
        norms = np.linalg.norm(accel, axis=1)
        return float(np.std(norms))

    @property
    def tilt_angle(self) -> float:
        """Estimated tilt from accelerometer (radians from vertical)."""
        accel = self.buffer[self._idx - 1, 3:6] if self._idx > 0 else self.buffer[0, 3:6]
        norm = np.linalg.norm(accel)
        if norm < 1e-6:
            return 0.0
        cos_theta = abs(accel[2]) / norm
        return float(math.acos(min(1.0, cos_theta)))

    @property
    def impact_detected(self) -> bool:
        """Check if recent acceleration spike exceeds threshold."""
        recent = self.buffer[max(0, self._idx - 10):self._idx, 3:6] if self._idx >= 10 else self.buffer[:self._idx, 3:6]
        if len(recent) == 0:
            return False
        norms = np.linalg.norm(recent, axis=1)
        return bool(np.max(norms) > 50.0)  # ~5g impact threshold


def read_imu(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read IMU sensors from MuJoCo data. Returns (gyro[3], accel[3], lin_accel[3])."""
    gyro = data.sensor("imu-torso-angular-velocity").data.copy().astype(np.float32)
    accel = data.sensor("imu-torso-linear-acceleration").data.copy().astype(np.float32)
    # Linear acceleration = measured accel - gravity component (approx)
    # For simplicity, use pelvis sensors for the third channel
    lin_accel = data.sensor("imu-pelvis-linear-acceleration").data.copy().astype(np.float32)
    return gyro, accel, lin_accel


class CameraRenderer:
    """Renders RGB and depth images from the robot's head camera site."""

    def __init__(self, model: mujoco.MjModel, width: int = 224, height: int = 224):
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Camera tracks the head_camera site
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._model = model

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render RGB [3, H, W] and depth [3, H, W] (depth repeated to 3ch) from head camera.

        Returns float32 arrays normalized to [0, 1].
        """
        # Position camera at head_camera site
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
        if site_id >= 0:
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            forward = site_mat[:, 0]
            self.camera.lookat[:] = data.site_xpos[site_id] + forward * 2.0
            self.camera.azimuth = float(math.degrees(math.atan2(forward[1], forward[0])))
            self.camera.elevation = float(math.degrees(math.asin(np.clip(forward[2], -1, 1))))
            self.camera.distance = 2.0

        self.renderer.update_scene(data, self.camera)

        # RGB (ensure depth mode is off)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()  # [H, W, 3] uint8
        rgb_float = rgb.astype(np.float32) / 255.0
        rgb_chw = np.transpose(rgb_float, (2, 0, 1))  # [3, H, W]

        # Depth
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()  # [H, W] float32
        self.renderer.disable_depth_rendering()
        # Normalize depth to [0, 1] range
        depth_max = np.max(depth) if np.max(depth) > 0 else 1.0
        depth_norm = np.clip(depth / depth_max, 0.0, 1.0).astype(np.float32)
        depth_3ch = np.stack([depth_norm, depth_norm, depth_norm], axis=0)  # [3, H, W]

        return rgb_chw, depth_3ch

    def close(self):
        self.renderer.close()
