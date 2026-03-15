"""Episode logger compatible with the example project's EpisodeWriter format.

Saves episodes as:
    task_dir/episode_XXXX/data.json   — metadata + per-frame states/actions
    task_dir/episode_XXXX/colors/     — RGB images
    task_dir/episode_XXXX/depths/     — Depth images
"""

import os
import json
import datetime
import numpy as np
import cv2
from threading import Thread
from queue import Queue, Empty


ACTION_NAMES = [
    "navigate_to", "open_door", "clear_obstacle", "assist_stand",
    "carry_person", "communicate", "signal_team", "emergency_stop",
]


class SOSEpisodeLogger:
    """Records episodes in the same format as the example teleop EpisodeWriter."""

    def __init__(self, task_dir: str, task_goal: str = "Rescue victims in collapsed structure",
                 frequency: int = 1, image_size: tuple[int, int] = (224, 224)):
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size  # (width, height)
        self.text = {
            "goal": task_goal,
            "desc": "SOS rescue mission: navigate collapsed structure, find and extract victims",
            "steps": "step1: navigate to victim; step2: assess stability; step3: extract victim",
        }
        self.info = {
            "version": "1.0.0",
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "author": "sos-answered-sim",
            "image": {"width": image_size[0], "height": image_size[1], "fps": frequency},
            "depth": {"width": image_size[0], "height": image_size[1], "fps": frequency},
            "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
            "joint_names": {
                "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                             "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
                "left_ee": ["left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2",
                            "left_hand_middle_0", "left_hand_middle_1",
                            "left_hand_index_0", "left_hand_index_1"],
                "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                              "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
                "right_ee": ["right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2",
                             "right_hand_middle_0", "right_hand_middle_1",
                             "right_hand_index_0", "right_hand_index_1"],
                "body": ["left_hip_pitch", "left_hip_roll", "left_hip_yaw",
                         "left_knee", "left_ankle_pitch", "left_ankle_roll",
                         "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
                         "right_knee", "right_ankle_pitch", "right_ankle_roll",
                         "waist_yaw", "waist_roll", "waist_pitch"],
            },
            "sim_state": "mujoco",
        }

        os.makedirs(task_dir, exist_ok=True)
        self.episode_id = -1
        # Find existing episodes
        if os.path.exists(task_dir):
            existing = [d for d in os.listdir(task_dir) if d.startswith("episode_")]
            if existing:
                self.episode_id = max(int(d.split("_")[-1]) for d in existing)

        self._queue = Queue(-1)
        self._stop = False
        self._worker = Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        self._json_path = None
        self._first_item = True

    def create_episode(self):
        """Start a new episode."""
        self.episode_id += 1
        self._item_id = -1
        ep_dir = os.path.join(self.task_dir, f"episode_{self.episode_id:04d}")
        self._color_dir = os.path.join(ep_dir, "colors")
        self._depth_dir = os.path.join(ep_dir, "depths")
        os.makedirs(self._color_dir, exist_ok=True)
        os.makedirs(self._depth_dir, exist_ok=True)

        self._json_path = os.path.join(ep_dir, "data.json")
        with open(self._json_path, "w") as f:
            f.write("{\n")
            f.write('"info": ' + json.dumps(self.info, indent=4) + ",\n")
            f.write('"text": ' + json.dumps(self.text, indent=4) + ",\n")
            f.write('"data": [\n')
        self._first_item = True

    def add_item(self, rgb_chw: np.ndarray, depth_chw: np.ndarray,
                 joint_positions: np.ndarray, action_id: int,
                 reward: float, stability: int, robot_pos: np.ndarray,
                 imu_sample: np.ndarray):
        """Queue a frame for writing.

        Args:
            rgb_chw: [3, H, W] float32 in [0, 1]
            depth_chw: [3, H, W] float32 in [0, 1]
            joint_positions: all joint angles
            action_id: 0-7
            reward: float
            stability: 0/1/2
            robot_pos: [x, y, z]
            imu_sample: [9] single IMU reading
        """
        self._item_id += 1
        self._queue.put({
            "idx": self._item_id,
            "rgb": rgb_chw,
            "depth": depth_chw,
            "joints": joint_positions,
            "action": action_id,
            "reward": reward,
            "stability": stability,
            "robot_pos": robot_pos,
            "imu": imu_sample,
        })

    def save_episode(self):
        """Finalize the current episode."""
        self._queue.join()
        if self._json_path:
            with open(self._json_path, "a") as f:
                f.write("\n]\n}")

    def _process_queue(self):
        while not self._stop:
            try:
                item = self._queue.get(timeout=1)
            except Empty:
                continue
            try:
                self._write_item(item)
            except Exception as e:
                print(f"EpisodeLogger error: {e}")
            self._queue.task_done()

    def _write_item(self, item: dict):
        idx = item["idx"]

        # Save RGB
        rgb_hwc = (np.transpose(item["rgb"], (1, 2, 0)) * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
        color_name = f"{idx:06d}_head_cam.jpg"
        cv2.imwrite(os.path.join(self._color_dir, color_name), rgb_bgr)

        # Save depth
        depth_hw = (item["depth"][0] * 255).astype(np.uint8)
        depth_name = f"{idx:06d}_head_depth.jpg"
        cv2.imwrite(os.path.join(self._depth_dir, depth_name), depth_hw)

        # Build frame data matching example format
        frame = {
            "idx": idx,
            "colors": {"head_cam": os.path.join("colors", color_name)},
            "depths": {"head_depth": os.path.join("depths", depth_name)},
            "states": {
                "joint_positions": item["joints"].tolist(),
                "robot_position": item["robot_pos"].tolist(),
                "stability": int(item["stability"]),
                "imu": item["imu"].tolist(),
            },
            "actions": {
                "action_id": int(item["action"]),
                "action_name": ACTION_NAMES[item["action"]],
            },
            "reward": float(item["reward"]),
        }

        with open(self._json_path, "a") as f:
            if not self._first_item:
                f.write(",\n")
            f.write(json.dumps(frame, indent=4))
            self._first_item = False

    def close(self):
        self._queue.join()
        self._stop = True
        self._worker.join(timeout=5)
