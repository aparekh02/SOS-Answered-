"""Launch the SOS rescue gym with the MuJoCo 3D viewer.

Usage (macOS requires mjpython):
    .venv/bin/mjpython sim/view.py
    .venv/bin/mjpython sim/view.py --steps 200 --action navigate
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.env import SOSRescueEnv, ACTION_NAMES

ACTION_MAP = {name: i for i, name in enumerate(ACTION_NAMES)}


def main():
    parser = argparse.ArgumentParser(description="View SOS rescue gym in 3D")
    parser.add_argument("--steps", type=int, default=300, help="Steps to run")
    parser.add_argument("--action", type=str, default="navigate_to",
                        choices=ACTION_NAMES, help="Action to repeat")
    parser.add_argument("--random", action="store_true", help="Random actions")
    args = parser.parse_args()

    import numpy as np

    env = SOSRescueEnv(render_mode="human", headless=False)
    obs, info = env.reset()
    print(f"Robot spawned at {info['robot_pos'].round(3)}")
    print(f"Running {args.steps} steps with action: {'random' if args.random else args.action}")
    print()

    action_id = ACTION_MAP[args.action]

    for i in range(args.steps):
        if args.random:
            action_id = np.random.randint(0, len(ACTION_NAMES))

        obs, reward, terminated, truncated, info = env.step(action_id)
        env.render()

        if i % 10 == 0:
            stab = info.get("stability", 0)
            stab_name = ["STABLE", "UNSTABLE", "IMMINENT_COLLAPSE"][stab]
            pos = info["robot_pos"]
            print(f"  [{i:3d}] action={ACTION_NAMES[action_id]:16s}  "
                  f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
                  f"r={reward:+.2f}  stab={stab_name}")

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\n  Episode {reason} at step {i}. Resetting...\n")
            obs, info = env.reset()

        time.sleep(0.02)

    print("\nDone. Closing viewer.")
    env.close()


if __name__ == "__main__":
    main()
