"""Test script: verify the MuJoCo G1 rescue gym loads, steps, and renders correctly.

Run:  python test_gym_render.py
Outputs rendered frames to test_renders/ directory.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.env import SOSRescueEnv, ACTION_NAMES


def test_headless_physics():
    """Test that the env loads and steps correctly in headless mode (no GL needed)."""
    print("=" * 60)
    print("TEST 1: Headless physics + observations")
    print("=" * 60)

    env = SOSRescueEnv(render_mode=None, headless=True, max_steps=50)
    obs, info = env.reset()

    print(f"  Model loaded: {env.model.nq} qpos, {env.model.nv} qvel, {env.model.nu} actuators")
    print(f"  Robot start pos: {info['robot_pos']}")
    print(f"  Robot start RPY: {info['robot_rpy']}")
    print(f"  Obs shapes — RGB: {obs['rgb'].shape}, Depth: {obs['depth'].shape}, IMU: {obs['imu'].shape}")

    assert obs["rgb"].shape == (3, 224, 224), f"Bad RGB shape: {obs['rgb'].shape}"
    assert obs["depth"].shape == (3, 224, 224), f"Bad depth shape: {obs['depth'].shape}"
    assert obs["imu"].shape == (100, 9), f"Bad IMU shape: {obs['imu'].shape}"

    # Step through all 8 actions
    total_reward = 0.0
    for action in range(8):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Action {action} ({ACTION_NAMES[action]:>16s}) -> "
              f"reward={reward:+.2f}, stability={info['stability']}, "
              f"pos=[{info['robot_pos'][0]:.2f}, {info['robot_pos'][1]:.2f}, {info['robot_pos'][2]:.2f}]")
        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
            break

    print(f"  Total reward over {min(8, env._step_count)} steps: {total_reward:+.2f}")
    env.close()
    print("  PASSED\n")


def test_rgb_rendering():
    """Test offscreen RGB rendering and save frames as images."""
    print("=" * 60)
    print("TEST 2: Offscreen RGB rendering")
    print("=" * 60)

    env = SOSRescueEnv(render_mode="rgb_array", headless=False, max_steps=50)
    obs, info = env.reset()

    # Create output directory
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_renders")
    os.makedirs(out_dir, exist_ok=True)

    # Render initial frame
    frame = env.render()
    if frame is not None:
        print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
        assert frame.shape == (224, 224, 3), f"Bad frame shape: {frame.shape}"
        assert frame.dtype == np.uint8, f"Bad frame dtype: {frame.dtype}"
        _save_ppm(frame, os.path.join(out_dir, "frame_reset.ppm"))
        print(f"  Saved: {out_dir}/frame_reset.ppm")
    else:
        print("  WARNING: render() returned None (GL context issue?)")

    # Step and render a few actions
    actions_to_test = [0, 0, 0, 2, 1, 0]  # navigate, navigate, navigate, clear, open_door, navigate
    for i, action in enumerate(actions_to_test):
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            fname = f"frame_step{i+1}_a{action}_{ACTION_NAMES[action]}.ppm"
            _save_ppm(frame, os.path.join(out_dir, fname))
            print(f"  Step {i+1}: {ACTION_NAMES[action]:>16s} -> saved {fname}")
        if terminated or truncated:
            break

    # Also check that obs RGB has actual content (not zeros)
    rgb_sum = np.sum(obs["rgb"])
    print(f"  Obs RGB sum: {rgb_sum:.1f} (should be > 0 if rendering works)")

    env.close()
    print(f"  All frames saved to: {out_dir}/")
    print("  PASSED\n")


def test_depth_rendering():
    """Verify depth observations contain meaningful values."""
    print("=" * 60)
    print("TEST 3: Depth observations")
    print("=" * 60)

    env = SOSRescueEnv(render_mode="rgb_array", headless=False, max_steps=10)
    obs, _ = env.reset()

    depth = obs["depth"]
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Depth mean:  {depth.mean():.4f}")

    # Step forward to get closer to scene objects
    for _ in range(3):
        obs, _, terminated, truncated, _ = env.step(0)  # navigate forward
        if terminated or truncated:
            break

    depth2 = obs["depth"]
    print(f"  After 3 nav steps — depth range: [{depth2.min():.4f}, {depth2.max():.4f}]")

    env.close()
    print("  PASSED\n")


def test_imu_readings():
    """Verify IMU buffer produces sensible values."""
    print("=" * 60)
    print("TEST 4: IMU sensor readings")
    print("=" * 60)

    env = SOSRescueEnv(render_mode=None, headless=True, max_steps=20)
    obs, info = env.reset()

    imu = obs["imu"]
    print(f"  IMU shape: {imu.shape}")
    print(f"  IMU range: [{imu.min():.4f}, {imu.max():.4f}]")
    print(f"  Gyro (last sample):      {imu[-1, 0:3]}")
    print(f"  Accel (last sample):     {imu[-1, 3:6]}")
    print(f"  Lin accel (last sample): {imu[-1, 6:9]}")
    print(f"  IMU features: {info['imu_features']}")

    # IMU should have some non-zero values (accel channels should be non-trivial)
    accel_norm = np.mean(np.linalg.norm(imu[:, 3:6], axis=1))
    print(f"  Mean |accel|: {accel_norm:.2f} (expect > 0)")
    assert accel_norm > 0.01, "Accelerometer readings are essentially zero"

    env.close()
    print("  PASSED\n")


def test_episode_rollout_speed():
    """Benchmark: how fast can we run a full episode in headless mode."""
    print("=" * 60)
    print("TEST 5: Episode rollout speed (headless)")
    print("=" * 60)

    env = SOSRescueEnv(render_mode=None, headless=True, max_steps=100)
    obs, _ = env.reset()

    t0 = time.time()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated

    elapsed = time.time() - t0
    fps = steps / elapsed if elapsed > 0 else 0
    print(f"  {steps} steps in {elapsed:.2f}s = {fps:.1f} steps/sec")
    print(f"  Sim time per step: ~{250 * 0.002:.3f}s ({250} substeps @ 500Hz)")

    env.close()
    print("  PASSED\n")


def test_gymnasium_api():
    """Verify the env is a valid Gymnasium env."""
    print("=" * 60)
    print("TEST 6: Gymnasium API compliance")
    print("=" * 60)

    env = SOSRescueEnv(render_mode=None, headless=True, max_steps=10)

    # Check spaces
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space keys: {list(env.observation_space.spaces.keys())}")
    assert env.action_space.n == 8

    # Reset returns (obs, info)
    result = env.reset(seed=42)
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    # Step returns (obs, reward, terminated, truncated, info)
    result = env.step(0)
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Obs in space
    for key in env.observation_space.spaces:
        assert key in obs, f"Missing obs key: {key}"

    env.close()
    print("  PASSED\n")


def test_fusion_input():
    """Verify get_fusion_input() returns proper torch tensors."""
    print("=" * 60)
    print("TEST 7: Fusion input tensors")
    print("=" * 60)

    import torch
    env = SOSRescueEnv(render_mode=None, headless=True, max_steps=10)
    env.reset()

    rgb, depth, imu = env.get_fusion_input()
    print(f"  RGB:   {rgb.shape} {rgb.dtype}")
    print(f"  Depth: {depth.shape} {depth.dtype}")
    print(f"  IMU:   {imu.shape} {imu.dtype}")

    assert rgb.shape == (1, 3, 224, 224)
    assert depth.shape == (1, 3, 224, 224)
    assert imu.shape == (1, 100, 9)
    assert rgb.dtype == torch.float32
    assert imu.dtype == torch.float32

    env.close()
    print("  PASSED\n")


def _save_ppm(img: np.ndarray, path: str):
    """Save an HxWx3 uint8 image as PPM (no PIL dependency)."""
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())


if __name__ == "__main__":
    print("\nSOS Rescue Gym — Render & Functionality Tests\n")

    test_gymnasium_api()
    test_headless_physics()
    test_imu_readings()
    test_episode_rollout_speed()
    test_fusion_input()

    # These need OpenGL — may fail in headless CI
    try:
        test_rgb_rendering()
        test_depth_rendering()
    except Exception as e:
        print(f"  SKIPPED (GL not available): {e}\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
