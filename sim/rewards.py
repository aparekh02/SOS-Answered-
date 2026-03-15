"""Reward computation for the SOS rescue rubble environment.

Simplified mission: navigate rubble -> find victims -> lift debris -> rescue.
"""

import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STABLE, UNSTABLE, IMMINENT_COLLAPSE, TILT_THRESHOLD, VIBRATION_THRESHOLD

# Proximity thresholds (meters)
VICTIM_REACH_DIST = 1.5
VICTIM_RESCUE_DIST = 1.5

# Victim positions (xy only) — must match scene.xml
VICTIM_POSITIONS = [
    np.array([3.0, 0.1]),
    np.array([5.0, 3.0]),
    np.array([-2.0, -2.5]),
    np.array([-3.0, 3.0]),
]

# Debris body names associated with each victim (must match scene.xml)
VICTIM_DEBRIS = {
    0: ["victim1_debris_a", "victim1_debris_b"],
    1: ["victim2_debris_a", "victim2_debris_b"],
    2: ["victim3_debris_a", "victim3_debris_b"],
    3: ["victim4_debris_a", "victim4_debris_b"],
}

# Reward values
REWARD_VICTIM_REACHED = 10.0
REWARD_VICTIM_RESCUED = 25.0
PENALTY_TIME_STEP = -0.1
PENALTY_ROBOT_FELL = -15.0


def compute_stability(tilt_angle: float, vibration_mag: float, impact: bool,
                       robot_xy: np.ndarray) -> int:
    """Classify structural stability from IMU features.

    Returns: STABLE (0), UNSTABLE (1), or IMMINENT_COLLAPSE (2).
    """
    # Pure IMU-based stability (no danger zones in rubble environment)
    if impact or tilt_angle > 0.4 or vibration_mag > VIBRATION_THRESHOLD * 1.5:
        return IMMINENT_COLLAPSE
    if tilt_angle > TILT_THRESHOLD or vibration_mag > VIBRATION_THRESHOLD * 0.8:
        return UNSTABLE
    return STABLE


class RewardComputer:
    """Computes per-step reward for rubble rescue mission."""

    def __init__(self):
        self.victims_reached = set()
        self.victims_rescued = set()
        self.debris_cleared = {}   # victim_id -> set of cleared debris names

    def reset(self):
        self.victims_reached = set()
        self.victims_rescued = set()
        self.debris_cleared = {}

    def is_debris_cleared(self, victim_id: int) -> bool:
        """Check if all debris covering a victim has been cleared."""
        required = VICTIM_DEBRIS.get(victim_id, [])
        cleared = self.debris_cleared.get(victim_id, set())
        return len(cleared) >= len(required)

    def compute(self, robot_pos: np.ndarray, robot_height: float,
                stability: int, action_id: int) -> tuple[float, bool]:
        """Compute reward and done flag for a single timestep.

        Args:
            robot_pos: [x, y, z] robot pelvis position
            robot_height: z coordinate of pelvis
            stability: 0=STABLE, 1=UNSTABLE, 2=IMMINENT_COLLAPSE
            action_id: which action was taken (0-4)

        Returns:
            (reward, done)
        """
        reward = PENALTY_TIME_STEP
        done = False
        robot_xy = robot_pos[:2]

        # Victim proximity checks
        for i, vpos in enumerate(VICTIM_POSITIONS):
            dist = np.linalg.norm(robot_xy - vpos)

            # Reaching a victim
            if dist < VICTIM_REACH_DIST and i not in self.victims_reached:
                self.victims_reached.add(i)
                reward += REWARD_VICTIM_REACHED

            # Rescuing a victim: must be close, debris cleared, and using RESCUE_PERSON action (4)
            if (dist < VICTIM_RESCUE_DIST
                    and i not in self.victims_rescued
                    and self.is_debris_cleared(i)
                    and action_id == 4):
                self.victims_rescued.add(i)
                reward += REWARD_VICTIM_RESCUED

        # Robot fell detection
        if robot_height < 0.3:
            reward += PENALTY_ROBOT_FELL
            done = True

        # All victims rescued = success
        if len(self.victims_rescued) == len(VICTIM_POSITIONS):
            done = True

        return reward, done
