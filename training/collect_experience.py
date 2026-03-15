"""Collect experience data from MuJoCo gym or MockFusion — stores to ExperienceBuffer.

Usage:
    # With MuJoCo gym (real physics simulation):
    python training/collect_experience.py --steps 5000 --mode gym

    # With mock environment (fast, no rendering):
    python training/collect_experience.py --steps 50000 --mode mock

    # With episode recording (saves images + JSON like example project):
    python training/collect_experience.py --steps 1000 --mode gym --record --task-dir episode_data/rescue
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FUSION_SEQ_LEN, FUSION_DIM, NUM_ACTIONS, IMU_SEQ_LEN, IMU_CHANNELS,
    PENALTY_TIME_STEP, PENALTY_UNSTABLE, PENALTY_IMMINENT_COLLAPSE,
    REWARD_NEW_ZONE_MAPPED, IMG_SIZE,
)
from data.experience_buffer import ExperienceBuffer


# ────────────────────────────────────────────────────────────────
# Mock environment (kept for fast data generation without MuJoCo)
# ────────────────────────────────────────────────────────────────

class MockFusion(torch.nn.Module):
    """Returns random tensors matching real Fusion output shapes."""

    def forward(self, rgb, depth, imu):
        batch = rgb.shape[0]
        fused = torch.randn(batch, FUSION_SEQ_LEN, FUSION_DIM)
        stability_logits = torch.randn(batch, 3)
        return fused, stability_logits


class MockEnvironment:
    """Simulates a rescue environment with random transitions."""

    def __init__(self):
        self.fusion = MockFusion()
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        self.step_count = 0
        return self._observe()

    def _observe(self):
        rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        depth = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        imu = torch.randn(1, IMU_SEQ_LEN, IMU_CHANNELS)
        with torch.no_grad():
            fused, stab_logits = self.fusion(rgb, depth, imu)
        stability = stab_logits.argmax(dim=-1).item()
        return fused.squeeze(0).numpy(), stability

    def step(self, action: int):
        self.step_count += 1
        s1, stability = self._observe()
        reward = PENALTY_TIME_STEP
        if stability == 1:
            reward += PENALTY_UNSTABLE
        elif stability == 2:
            reward += PENALTY_IMMINENT_COLLAPSE
        if np.random.random() < 0.05:
            reward += REWARD_NEW_ZONE_MAPPED
        done = self.step_count >= self.max_steps or stability == 2
        return s1, reward, done, stability


# ────────────────────────────────────────────────────────────────
# MuJoCo gym environment wrapper
# ────────────────────────────────────────────────────────────────

class GymEnvironment:
    """Wraps SOSRescueEnv to produce Fusion-compatible experience tuples.

    Uses MockFusion to convert raw sensor observations (RGB + depth + IMU)
    into [492, 2048] fused embeddings for the ExperienceBuffer.
    Swap MockFusion for a real Fusion model when available.
    """

    def __init__(self, render_mode=None, record=False, task_dir=None):
        from sim.env import SOSRescueEnv
        # Use headless mode (no camera rendering) unless explicitly rendering
        headless = render_mode is None
        self.env = SOSRescueEnv(render_mode=render_mode, headless=headless)
        self.fusion = MockFusion()

        self.episode_logger = None
        if record and task_dir:
            from sim.episode_logger import SOSEpisodeLogger
            self.episode_logger = SOSEpisodeLogger(task_dir=task_dir)

    def reset(self):
        obs, info = self.env.reset()
        if self.episode_logger:
            self.episode_logger.create_episode()
        return self._fuse(obs, info)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stability = info.get("stability", 0)
        done = terminated or truncated

        # Log episode data
        if self.episode_logger:
            self.episode_logger.add_item(
                rgb_chw=obs["rgb"],
                depth_chw=obs["depth"],
                joint_positions=info.get("joint_positions", np.zeros(43)),
                action_id=action,
                reward=reward,
                stability=stability,
                robot_pos=info.get("robot_pos", np.zeros(3)),
                imu_sample=obs["imu"][-1] if len(obs["imu"]) > 0 else np.zeros(9),
            )

        s1 = self._fuse(obs, info)[0]

        if done and self.episode_logger:
            self.episode_logger.save_episode()

        return s1, reward, done, stability

    def _fuse(self, obs, info):
        """Run Fusion model on observation to get [492, 2048] embedding."""
        rgb = torch.from_numpy(obs["rgb"]).unsqueeze(0)
        depth = torch.from_numpy(obs["depth"]).unsqueeze(0)
        imu = torch.from_numpy(obs["imu"]).unsqueeze(0)
        with torch.no_grad():
            fused, stab_logits = self.fusion(rgb, depth, imu)
        stability = stab_logits.argmax(dim=-1).item()
        return fused.squeeze(0).numpy(), stability

    def close(self):
        self.env.close()
        if self.episode_logger:
            self.episode_logger.close()


# ────────────────────────────────────────────────────────────────
# Collection loop
# ────────────────────────────────────────────────────────────────

def collect(num_steps: int, buffer_path: str, capacity: int = 100_000,
            mode: str = "mock", render: bool = False,
            record: bool = False, task_dir: str = None):
    """Collect experience data from specified environment.

    Args:
        num_steps: Number of environment steps to collect.
        buffer_path: Path to HDF5 experience buffer file.
        capacity: Max buffer capacity.
        mode: "mock" for fast random data, "gym" for MuJoCo simulation.
        render: Whether to render the environment (gym mode only).
        record: Whether to record episodes to disk (gym mode only).
        task_dir: Directory for episode recordings.
    """
    buf = ExperienceBuffer(buffer_path, capacity=capacity)

    if mode == "gym":
        render_mode = "human" if render else None
        env = GymEnvironment(render_mode=render_mode, record=record, task_dir=task_dir)
        print(f"  Using MuJoCo gym environment (render={render}, record={record})")
    else:
        env = MockEnvironment()
        print(f"  Using mock environment (fast mode)")

    s, stab = env.reset()
    collected = 0

    while collected < num_steps:
        action = np.random.randint(0, NUM_ACTIONS)
        s1, reward, done, stability = env.step(action)

        buf.add(s=s, a=action, s1=s1, r=reward, d=done, stab=stability)
        collected += 1

        if collected % 500 == 0:
            print(f"  collected {collected}/{num_steps} steps (buffer size: {len(buf)})")

        if done:
            s, stab = env.reset()
        else:
            s = s1

    print(f"Done. Buffer size: {len(buf)}, saved to {buffer_path}")

    if hasattr(env, "close"):
        env.close()

    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect experience data for world model training")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect")
    parser.add_argument("--output", type=str, default="experience_data/buffer.h5", help="Buffer output path")
    parser.add_argument("--capacity", type=int, default=100_000, help="Buffer capacity")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "gym"],
                        help="Environment mode: mock (fast, random) or gym (MuJoCo physics)")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer (gym mode)")
    parser.add_argument("--record", action="store_true", help="Record episodes to disk (gym mode)")
    parser.add_argument("--task-dir", type=str, default="episode_data/rescue",
                        help="Directory for episode recordings")
    args = parser.parse_args()

    collect(args.steps, args.output, args.capacity,
            mode=args.mode, render=args.render,
            record=args.record, task_dir=args.task_dir)
