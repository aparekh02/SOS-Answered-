"""SOSRescueEnv — Gymnasium environment for Unitree G1 rubble rescue with MuJoCo.

Outputs observations compatible with the SOS-Answered world model pipeline:
    - RGB:   [3, 224, 224] float32
    - Depth: [3, 224, 224] float32
    - IMU:   [100, 9] float32

Takes discrete actions 0-4 matching the 5 rescue primitives.
Produces experience tuples (s, a, s1, r, d, stab) for ExperienceBuffer.
"""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FUSION_SEQ_LEN, FUSION_DIM, IMU_SEQ_LEN, IMU_CHANNELS, IMG_SIZE,
    STABLE, UNSTABLE, IMMINENT_COLLAPSE,
)
from sim.sensors import IMUBuffer, read_imu
from sim.rewards import RewardComputer, compute_stability, VICTIM_POSITIONS, VICTIM_DEBRIS

# 6 simplified action primitives
NUM_ACTIONS = 6
ACTION_NAMES = [
    "MOVE_FORWARD",     # 0 — walk forward
    "TURN_LEFT",        # 1 — rotate left ~15 degrees
    "TURN_RIGHT",       # 2 — rotate right ~15 degrees
    "LIFT_DEBRIS",      # 3 — crouch, grab, throw debris away
    "RESCUE_PERSON",    # 4 — rescue nearby victim (debris must be cleared)
    "STEP_OVER",        # 5 — high-step gait over small obstacles
]

_SCENE_XML = Path(__file__).parent / "scene.xml"

# G1 URDF-based model (43 actuators: 12 legs + 3 waist + 14 left arm/hand + 14 right arm/hand)
_N_ACTUATORS = 43

# Gait control parameters
_CONTROL_HZ = 50.0             # gait controller runs at 50Hz
_ACTION_DURATION = 1.5         # seconds per action (~1.8 full stride cycles)
_DEBRIS_LIFT_DIST = 4.0        # meters — max distance to grab debris
_DEBRIS_LIFT_DZ = 0.5          # meters — how high debris is lifted

# Actuator indices (43 total)
# Legs: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] x2
_LHP, _LHR, _LHY, _LK, _LAP, _LAR = 0, 1, 2, 3, 4, 5
_RHP, _RHR, _RHY, _RK, _RAP, _RAR = 6, 7, 8, 9, 10, 11
# Waist: [yaw, roll, pitch]
_WY, _WR, _WP = 12, 13, 14
# Arms: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
_LSP, _LSR, _LSY, _LE = 15, 16, 17, 18
_RSP, _RSR, _RSY, _RE = 29, 30, 31, 32

# Proven standing pose for the 43-actuator G1
_STANDING_POSE = np.zeros(_N_ACTUATORS, dtype=np.float64)
_STANDING_POSE[_LHP] = -0.1;  _STANDING_POSE[_LK] = 0.25; _STANDING_POSE[_LAP] = -0.15
_STANDING_POSE[_RHP] = -0.1;  _STANDING_POSE[_RK] = 0.25; _STANDING_POSE[_RAP] = -0.15
_STANDING_POSE[_LE] = 0.3;    _STANDING_POSE[_RE] = 0.3

# Standing height for pelvis
_STAND_HEIGHT = 0.793


class G1WalkController:
    """Discrete step-by-step bipedal walker for the URDF G1.

    Each full stride has 6 explicit phases, executed one at a time:
      1. SHIFT WEIGHT LEFT  — lean onto left foot (hip roll)
      2. LIFT RIGHT LEG     — bend right knee, raise foot off ground
      3. SWING RIGHT FWD    — flex right hip forward, extend knee to plant
      4. SHIFT WEIGHT RIGHT — lean onto right foot
      5. LIFT LEFT LEG      — bend left knee, raise foot off ground
      6. SWING LEFT FWD     — flex left hip forward, extend knee to plant

    phase 0..1 maps across all 6 sub-phases (each is 1/6 of the cycle).
    """

    def __init__(self):
        self.phase = 0.0
        self.freq = 0.5           # Hz — one full stride per 2 seconds

        # Amplitudes
        self.weight_shift = 0.03  # hip roll to shift COM over stance foot
        self.knee_lift = 0.25     # knee bend to lift foot — visible
        self.hip_reach = 0.06     # how far hip swings forward
        self.stance_push = 0.08   # stance hip extension (backward) — drives forward motion
        self.ankle_lift = 0.10    # ankle dorsiflexion during swing
        self.ankle_push = 0.08    # ankle plantarflexion during stance push
        self.arm_swing = 0.20     # arm counter-swing

    def step(self, dt: float, vx: float, vyaw: float, body_rpy: np.ndarray) -> np.ndarray:
        """Advance one control step, return 43-DOF joint targets."""
        ctrl = _STANDING_POSE.copy()

        if abs(vx) < 0.01 and abs(vyaw) < 0.01:
            return ctrl

        self.phase = (self.phase + dt * self.freq) % 1.0
        p = self.phase

        # Determine which sub-phase we're in (6 phases, each 1/6 of cycle)
        sub = int(p * 6) % 6
        t = (p * 6) % 1.0  # 0..1 progress within current sub-phase
        s = self._smooth(t)  # smoothed progress

        # ── SUB-PHASE 0: Shift weight onto LEFT leg ──
        if sub == 0:
            ctrl[_LHR] = -self.weight_shift * s
            ctrl[_RHR] = -self.weight_shift * s
            ctrl[_LAR] = self.weight_shift * 0.5 * s
            ctrl[_RAR] = self.weight_shift * 0.5 * s

        # ── SUB-PHASE 1: Lift RIGHT leg (knee bends, foot rises) ──
        elif sub == 1:
            # Keep weight on left
            ctrl[_LHR] = -self.weight_shift
            ctrl[_RHR] = -self.weight_shift
            ctrl[_LAR] = self.weight_shift * 0.5
            ctrl[_RAR] = self.weight_shift * 0.5
            # Right leg lifts
            ctrl[_RK] = _STANDING_POSE[_RK] + self.knee_lift * s
            ctrl[_RHP] = _STANDING_POSE[_RHP] + self.hip_reach * 0.3 * s
            ctrl[_RAP] = _STANDING_POSE[_RAP] + self.ankle_lift * s

        # ── SUB-PHASE 2: Swing RIGHT leg forward, plant foot ──
        elif sub == 2:
            ctrl[_LHR] = -self.weight_shift
            ctrl[_RHR] = -self.weight_shift
            ctrl[_LAR] = self.weight_shift * 0.5
            ctrl[_RAR] = self.weight_shift * 0.5
            # Right leg swings forward and extends
            ctrl[_RHP] = _STANDING_POSE[_RHP] + self.hip_reach * (0.3 + 0.7 * s)
            ctrl[_RK] = _STANDING_POSE[_RK] + self.knee_lift * (1.0 - s)  # knee straightens
            ctrl[_RAP] = _STANDING_POSE[_RAP] + self.ankle_lift * (1.0 - s)  # foot levels
            # Left (stance) leg pushes backward — hip extends + ankle plantarflexes
            ctrl[_LHP] = _STANDING_POSE[_LHP] - self.stance_push * s
            ctrl[_LAP] = _STANDING_POSE[_LAP] - self.ankle_push * s

        # ── SUB-PHASE 3: Shift weight onto RIGHT leg ──
        elif sub == 3:
            # Right leg forward, left leg pushed back after stance
            ctrl[_RHP] = _STANDING_POSE[_RHP] + self.hip_reach
            ctrl[_LHP] = _STANDING_POSE[_LHP] - self.stance_push
            ctrl[_LAP] = _STANDING_POSE[_LAP] - self.ankle_push
            # Shift weight right (transition from left-loaded to right-loaded)
            shift = self.weight_shift * (1.0 - 2.0 * s)
            ctrl[_LHR] = -shift
            ctrl[_RHR] = -shift
            ctrl[_LAR] = shift * 0.5
            ctrl[_RAR] = shift * 0.5

        # ── SUB-PHASE 4: Lift LEFT leg ──
        elif sub == 4:
            # Weight on right
            ctrl[_LHR] = self.weight_shift
            ctrl[_RHR] = self.weight_shift
            ctrl[_LAR] = -self.weight_shift * 0.5
            ctrl[_RAR] = -self.weight_shift * 0.5
            # Right stays forward, settling toward standing
            ctrl[_RHP] = _STANDING_POSE[_RHP] + self.hip_reach * (1.0 - s)
            # Left leg lifts — starts from pushed-back position
            ctrl[_LK] = _STANDING_POSE[_LK] + self.knee_lift * s
            ctrl[_LHP] = _STANDING_POSE[_LHP] - self.stance_push * (1.0 - s) + self.hip_reach * 0.3 * s
            ctrl[_LAP] = _STANDING_POSE[_LAP] - self.ankle_push * (1.0 - s) + self.ankle_lift * s

        # ── SUB-PHASE 5: Swing LEFT leg forward, plant foot ──
        else:
            ctrl[_LHR] = self.weight_shift
            ctrl[_RHR] = self.weight_shift
            ctrl[_LAR] = -self.weight_shift * 0.5
            ctrl[_RAR] = -self.weight_shift * 0.5
            # Left leg swings forward
            ctrl[_LHP] = _STANDING_POSE[_LHP] + self.hip_reach * (0.3 + 0.7 * s)
            ctrl[_LK] = _STANDING_POSE[_LK] + self.knee_lift * (1.0 - s)
            ctrl[_LAP] = _STANDING_POSE[_LAP] + self.ankle_lift * (1.0 - s)
            # Right (stance) leg pushes backward — hip extends + ankle plantarflexes
            ctrl[_RHP] = _STANDING_POSE[_RHP] - self.stance_push * s
            ctrl[_RAP] = _STANDING_POSE[_RAP] - self.ankle_push * s

        # ── ARMS — counter-swing opposite to legs ──
        # Right arm forward when left leg swings, and vice versa
        if sub in (1, 2):  # right leg swinging
            ctrl[_LSP] = -self.arm_swing * s   # left arm forward
            ctrl[_RSP] = self.arm_swing * 0.3 * s  # right arm back
        elif sub in (4, 5):  # left leg swinging
            ctrl[_RSP] = -self.arm_swing * s
            ctrl[_LSP] = self.arm_swing * 0.3 * s
        ctrl[_LE] = 0.5
        ctrl[_RE] = 0.5

        # ── HIP YAW — turning ──
        if abs(vyaw) > 0.01:
            # Yaw the swing leg outward during swing phases
            if sub in (1, 2):
                ctrl[_RHY] = vyaw * 0.06 * s
            elif sub in (4, 5):
                ctrl[_LHY] = -vyaw * 0.06 * s

        return ctrl

    @staticmethod
    def _smooth(t: float) -> float:
        """Quintic ease-in-out for smooth phase transitions."""
        t = max(0.0, min(1.0, t))
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    # Keep _swing_signal for gain switching in _execute_action
    def _swing_signal(self, phase: float) -> float:
        """Returns >0 when the leg (at this phase) is in swing."""
        sub = int((phase * 6) % 6)
        return 1.0 if sub in (1, 2) else 0.0

    def lift_pose(self, t: float) -> np.ndarray:
        """Generate a convincing crouch-grab-lift sequence.

        t: 0..1 progress through the lift motion.
        Phases:
            0.0-0.3: Crouch down, lean forward, arms reach forward/down
            0.3-0.5: Arms close (grab), fingers curl
            0.5-0.8: Stand up while holding, arms come up
            0.8-1.0: Return to standing, arms raised
        """
        ctrl = _STANDING_POSE.copy()

        # Finger joint indices (left hand / right hand)
        _L_THUMB0, _L_THUMB1, _L_MIDDLE0, _L_INDEX0 = 22, 23, 25, 27
        _R_THUMB0, _R_THUMB1, _R_MIDDLE0, _R_INDEX0 = 36, 37, 39, 41

        if t < 0.3:
            # Phase 1: Crouch down, lean torso forward, arms reach forward/down
            p = t / 0.3  # 0..1 within this phase
            crouch = p * 0.8  # deep knee bend up to 0.8 rad
            ctrl[_LHP] = _STANDING_POSE[_LHP] - crouch * 0.6
            ctrl[_RHP] = _STANDING_POSE[_RHP] - crouch * 0.6
            ctrl[_LK] = _STANDING_POSE[_LK] + crouch
            ctrl[_RK] = _STANDING_POSE[_RK] + crouch
            ctrl[_LAP] = _STANDING_POSE[_LAP] - crouch * 0.4
            ctrl[_RAP] = _STANDING_POSE[_RAP] - crouch * 0.4
            # Lean torso forward
            ctrl[_WP] = p * 0.15
            # Arms reach forward and down
            ctrl[_LSP] = -p * 1.0   # shoulder pitch forward
            ctrl[_RSP] = -p * 1.0
            ctrl[_LE] = 0.1         # elbows nearly extended
            ctrl[_RE] = 0.1

        elif t < 0.5:
            # Phase 2: Hold crouch, arms close to grab, fingers curl
            p = (t - 0.3) / 0.2  # 0..1 within this phase
            crouch = 0.8  # stay crouched
            ctrl[_LHP] = _STANDING_POSE[_LHP] - crouch * 0.6
            ctrl[_RHP] = _STANDING_POSE[_RHP] - crouch * 0.6
            ctrl[_LK] = _STANDING_POSE[_LK] + crouch
            ctrl[_RK] = _STANDING_POSE[_RK] + crouch
            ctrl[_LAP] = _STANDING_POSE[_LAP] - crouch * 0.4
            ctrl[_RAP] = _STANDING_POSE[_RAP] - crouch * 0.4
            ctrl[_WP] = 0.15
            # Arms stay forward but elbows bend to grab
            ctrl[_LSP] = -1.0
            ctrl[_RSP] = -1.0
            ctrl[_LE] = 0.1 + p * 0.5   # elbows bend to grab
            ctrl[_RE] = 0.1 + p * 0.5
            # Fingers curl to grip
            grip = p * 1.5  # curl to limit
            ctrl[_L_THUMB0] = grip;  ctrl[_L_THUMB1] = grip
            ctrl[_L_MIDDLE0] = grip; ctrl[_L_INDEX0] = grip
            ctrl[_R_THUMB0] = grip;  ctrl[_R_THUMB1] = grip
            ctrl[_R_MIDDLE0] = grip; ctrl[_R_INDEX0] = grip

        elif t < 0.8:
            # Phase 3: Stand up while holding, arms come up
            p = (t - 0.5) / 0.3  # 0..1 within this phase
            crouch = 0.8 * (1.0 - p)  # straighten legs
            ctrl[_LHP] = _STANDING_POSE[_LHP] - crouch * 0.6
            ctrl[_RHP] = _STANDING_POSE[_RHP] - crouch * 0.6
            ctrl[_LK] = _STANDING_POSE[_LK] + crouch
            ctrl[_RK] = _STANDING_POSE[_RK] + crouch
            ctrl[_LAP] = _STANDING_POSE[_LAP] - crouch * 0.4
            ctrl[_RAP] = _STANDING_POSE[_RAP] - crouch * 0.4
            # Torso straightens
            ctrl[_WP] = 0.15 * (1.0 - p)
            # Arms lift up from forward-down to forward-up
            ctrl[_LSP] = -1.0 + p * 1.3  # from -1.0 to +0.3 (raised)
            ctrl[_RSP] = -1.0 + p * 1.3
            ctrl[_LE] = 0.6 * (1.0 - p * 0.5)
            ctrl[_RE] = 0.6 * (1.0 - p * 0.5)
            # Keep gripping
            ctrl[_L_THUMB0] = 1.5;  ctrl[_L_THUMB1] = 1.5
            ctrl[_L_MIDDLE0] = 1.5; ctrl[_L_INDEX0] = 1.5
            ctrl[_R_THUMB0] = 1.5;  ctrl[_R_THUMB1] = 1.5
            ctrl[_R_MIDDLE0] = 1.5; ctrl[_R_INDEX0] = 1.5

        else:
            # Phase 4: Standing with arms raised, showing lifted object
            p = (t - 0.8) / 0.2  # 0..1 within this phase
            # Legs at standing pose (no crouch)
            # Arms raised
            ctrl[_LSP] = 0.3
            ctrl[_RSP] = 0.3
            ctrl[_LE] = 0.3
            ctrl[_RE] = 0.3
            # Gradually release grip
            grip = 1.5 * (1.0 - p)
            ctrl[_L_THUMB0] = grip;  ctrl[_L_THUMB1] = grip
            ctrl[_L_MIDDLE0] = grip; ctrl[_L_INDEX0] = grip
            ctrl[_R_THUMB0] = grip;  ctrl[_R_THUMB1] = grip
            ctrl[_R_MIDDLE0] = grip; ctrl[_R_INDEX0] = grip

        return ctrl

    def step_over_pose(self, t: float) -> np.ndarray:
        """High-step gait for stepping over small obstacles.

        t: 0..1 progress through the step-over motion.
        Phases:
            0.0-0.3: Shift weight left, lift right knee VERY high
            0.3-0.5: Swing right leg far forward with knee high, plant beyond obstacle
            0.5-0.8: Shift weight right, lift left knee high
            0.8-1.0: Swing left leg forward past obstacle, plant
        Arms held wide for balance.
        """
        ctrl = _STANDING_POSE.copy()

        # Arms out wide for balance throughout
        ctrl[_LSR] = 0.4   # left arm out to side
        ctrl[_RSR] = -0.4  # right arm out to side
        ctrl[_LE] = 0.3
        ctrl[_RE] = 0.3

        if t < 0.3:
            # Phase 1: Shift weight left, lift right knee HIGH
            p = t / 0.3
            s = self._smooth(p)
            # Weight shift left
            ctrl[_LHR] = -0.05 * s
            ctrl[_RHR] = -0.05 * s
            ctrl[_LAR] = 0.03 * s
            # Right leg lifts high
            ctrl[_RHP] = _STANDING_POSE[_RHP] + 0.3 * s   # hip flexion
            ctrl[_RK] = _STANDING_POSE[_RK] + 0.8 * s     # knee VERY high
            ctrl[_RAP] = _STANDING_POSE[_RAP] + 0.3 * s   # ankle dorsiflexion

        elif t < 0.5:
            # Phase 2: Swing right leg far forward, extend and plant
            p = (t - 0.3) / 0.2
            s = self._smooth(p)
            ctrl[_LHR] = -0.05
            ctrl[_RHR] = -0.05
            ctrl[_LAR] = 0.03
            # Right leg swings forward
            ctrl[_RHP] = _STANDING_POSE[_RHP] + 0.3 + 0.15 * s   # more hip flexion
            ctrl[_RK] = _STANDING_POSE[_RK] + 0.8 * (1.0 - 0.7 * s)  # knee extends
            ctrl[_RAP] = _STANDING_POSE[_RAP] + 0.3 * (1.0 - s)
            # Left stance leg pushes
            ctrl[_LHP] = _STANDING_POSE[_LHP] - 0.12 * s
            ctrl[_LAP] = _STANDING_POSE[_LAP] - 0.10 * s

        elif t < 0.8:
            # Phase 3: Shift weight right, lift left knee high
            p = (t - 0.5) / 0.3
            s = self._smooth(p)
            # Right leg now planted forward
            ctrl[_RHP] = _STANDING_POSE[_RHP] + 0.45 * (1.0 - 0.5 * s)
            ctrl[_RK] = _STANDING_POSE[_RK] + 0.24 * (1.0 - s)
            # Shift weight right
            ctrl[_LHR] = -0.05 + 0.10 * s  # transition to right lean
            ctrl[_RHR] = -0.05 + 0.10 * s
            ctrl[_RAR] = -0.03 * s
            # Left leg lifts high
            ctrl[_LK] = _STANDING_POSE[_LK] + 0.8 * s
            ctrl[_LHP] = _STANDING_POSE[_LHP] - 0.12 * (1.0 - s) + 0.3 * s
            ctrl[_LAP] = _STANDING_POSE[_LAP] - 0.10 * (1.0 - s) + 0.3 * s

        else:
            # Phase 4: Swing left leg forward past obstacle, plant
            p = (t - 0.8) / 0.2
            s = self._smooth(p)
            ctrl[_LHR] = 0.05
            ctrl[_RHR] = 0.05
            ctrl[_RAR] = -0.03
            # Left leg swings forward
            ctrl[_LHP] = _STANDING_POSE[_LHP] + 0.3 + 0.15 * s
            ctrl[_LK] = _STANDING_POSE[_LK] + 0.8 * (1.0 - 0.7 * s)
            ctrl[_LAP] = _STANDING_POSE[_LAP] + 0.3 * (1.0 - s)
            # Right stance pushes
            ctrl[_RHP] = _STANDING_POSE[_RHP] - 0.12 * s
            ctrl[_RAP] = _STANDING_POSE[_RAP] - 0.10 * s

        return ctrl

    def crouch_grab_throw(self, t: float) -> np.ndarray:
        """Full crouch-grab-throw manipulation sequence.

        t: 0..1 progress through the motion.
        Phases:
            0.00-0.15: Squat down deeply
            0.15-0.25: Arms reach forward and down, fingers open
            0.25-0.35: Arms close around object, fingers curl tight
            0.35-0.55: Stand back up while holding
            0.55-0.70: Wind up — arms swing back
            0.70-0.85: THROW — arms swing forward and release
            0.85-1.00: Return to standing pose
        """
        ctrl = _STANDING_POSE.copy()

        # Finger joint indices
        _L_THUMB0, _L_THUMB1, _L_MIDDLE0, _L_INDEX0 = 22, 23, 25, 27
        _R_THUMB0, _R_THUMB1, _R_MIDDLE0, _R_INDEX0 = 36, 37, 39, 41

        def _set_grip(ctrl, amount):
            ctrl[_L_THUMB0] = amount; ctrl[_L_THUMB1] = amount
            ctrl[_L_MIDDLE0] = amount; ctrl[_L_INDEX0] = amount
            ctrl[_R_THUMB0] = amount; ctrl[_R_THUMB1] = amount
            ctrl[_R_MIDDLE0] = amount; ctrl[_R_INDEX0] = amount

        # Deep knee bend amount (knees only — torso stays upright for stability)
        _CROUCH = 0.50

        if t < 0.20:
            # Phase 1: CROUCH DOWN — deep knee bend, arms at sides
            s = self._smooth(t / 0.20)
            ctrl[_LK] = _STANDING_POSE[_LK] + _CROUCH * s
            ctrl[_RK] = _STANDING_POSE[_RK] + _CROUCH * s
            ctrl[_LAP] = _STANDING_POSE[_LAP] - _CROUCH * 0.5 * s
            ctrl[_RAP] = _STANDING_POSE[_RAP] - _CROUCH * 0.5 * s

        elif t < 0.35:
            # Phase 2: REACH — stay crouched, arms reach forward, fingers open
            s = self._smooth((t - 0.20) / 0.15)
            ctrl[_LK] = _STANDING_POSE[_LK] + _CROUCH
            ctrl[_RK] = _STANDING_POSE[_RK] + _CROUCH
            ctrl[_LAP] = _STANDING_POSE[_LAP] - _CROUCH * 0.5
            ctrl[_RAP] = _STANDING_POSE[_RAP] - _CROUCH * 0.5
            ctrl[_LSP] = -s * 0.5
            ctrl[_RSP] = -s * 0.5
            ctrl[_LE] = 0.2
            ctrl[_RE] = 0.2
            _set_grip(ctrl, -0.3 * s)

        elif t < 0.45:
            # Phase 3: GRAB — fingers curl tight
            s = self._smooth((t - 0.35) / 0.10)
            ctrl[_LK] = _STANDING_POSE[_LK] + _CROUCH
            ctrl[_RK] = _STANDING_POSE[_RK] + _CROUCH
            ctrl[_LAP] = _STANDING_POSE[_LAP] - _CROUCH * 0.5
            ctrl[_RAP] = _STANDING_POSE[_RAP] - _CROUCH * 0.5
            ctrl[_LSP] = -0.5
            ctrl[_RSP] = -0.5
            ctrl[_LE] = 0.2 + s * 0.4
            ctrl[_RE] = 0.2 + s * 0.4
            _set_grip(ctrl, -0.3 + s * 1.8)

        elif t < 0.65:
            # Phase 4: STAND UP — straighten knees while holding object
            s = self._smooth((t - 0.45) / 0.20)
            crouch = _CROUCH * (1.0 - s)
            ctrl[_LK] = _STANDING_POSE[_LK] + crouch
            ctrl[_RK] = _STANDING_POSE[_RK] + crouch
            ctrl[_LAP] = _STANDING_POSE[_LAP] - crouch * 0.5
            ctrl[_RAP] = _STANDING_POSE[_RAP] - crouch * 0.5
            ctrl[_LSP] = -0.5 + s * 0.2
            ctrl[_RSP] = -0.5 + s * 0.2
            ctrl[_LE] = 0.6
            ctrl[_RE] = 0.6
            _set_grip(ctrl, 1.5)

        elif t < 0.80:
            # Phase 5: TOSS — arms swing out to sides and release
            s = self._smooth((t - 0.65) / 0.15)
            ctrl[_LSP] = -0.3
            ctrl[_RSP] = -0.3
            ctrl[_LSR] = s * 0.4        # arms spread outward
            ctrl[_RSR] = -s * 0.4
            ctrl[_LE] = 0.3
            ctrl[_RE] = 0.3
            _set_grip(ctrl, 1.5 * (1.0 - s))  # release

        else:
            # Phase 6: Return to standing
            s = self._smooth((t - 0.80) / 0.20)
            ctrl[_LSP] = -0.3 * (1.0 - s)
            ctrl[_RSP] = -0.3 * (1.0 - s)
            ctrl[_LSR] = 0.4 * (1.0 - s)
            ctrl[_RSR] = -0.4 * (1.0 - s)
            ctrl[_LE] = 0.3 * (1.0 - s)
            ctrl[_RE] = 0.3 * (1.0 - s)

        return ctrl


def _rpy(q):
    """Quaternion [w,x,y,z] to roll/pitch/yaw."""
    w, x, y, z = q
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ], dtype=np.float32)


class SOSRescueEnv(gym.Env):
    """MuJoCo Gymnasium environment for SOS rubble rescue world model data collection.

    Observation space: dict with rgb, depth, imu
    Action space: Discrete(6)

    Args:
        render_mode: "human" for viewer, "rgb_array" for offscreen rendering, None for headless.
        headless: If True, skip camera rendering entirely (fast mode for data collection).
                  RGB/depth observations will be zero tensors. IMU and physics are still real.
        max_steps: Max steps per episode before truncation.
        scene_xml: Override path to MuJoCo scene XML.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None, headless: bool = False,
                 max_steps: int = 200, scene_xml: str | None = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._headless = headless

        # Load MuJoCo model
        xml_path = scene_xml or str(_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Spaces
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(0.0, 1.0, shape=(3, IMG_SIZE, IMG_SIZE), dtype=np.float32),
            "depth": gym.spaces.Box(0.0, 1.0, shape=(3, IMG_SIZE, IMG_SIZE), dtype=np.float32),
            "imu": gym.spaces.Box(-np.inf, np.inf, shape=(IMU_SEQ_LEN, IMU_CHANNELS), dtype=np.float32),
        })

        # Sensors
        self.imu_buffer = IMUBuffer(seq_len=IMU_SEQ_LEN)
        self._camera = None  # lazy init (needs GL context)

        # Reward
        self.reward_computer = RewardComputer()

        # State
        self._step_count = 0
        self._viewer = None

        # Physics substeps per IMU sample (100Hz IMU from 500Hz physics)
        self._phys_per_imu = max(1, int(1.0 / (self.model.opt.timestep * 100)))

        # Store initial qpos for reset
        self._init_qpos = None

        # Walking controller
        self._walker = G1WalkController()
        self._phys_per_ctrl = max(1, int(1.0 / (self.model.opt.timestep * _CONTROL_HZ)))

        # Cache pelvis body id and free joint address
        self._pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        fj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._fj_qpos_adr = self.model.jnt_qposadr[fj_id]   # start of [x,y,z,qw,qx,qy,qz]
        self._fj_qvel_adr = self.model.jnt_dofadr[fj_id]     # start of [vx,vy,vz,wx,wy,wz]

        # Cache per-actuator qpos/qvel addresses
        nj = min(_N_ACTUATORS, self.model.nu)
        self._act_qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(nj)],
            dtype=int,
        )
        self._act_qvel_adr = np.array(
            [self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(nj)],
            dtype=int,
        )

        # Precompute zero image tensors for headless mode
        self._zero_rgb = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        self._zero_depth = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Cache for fusion output
        self._last_stability = STABLE

        # Cache debris body ids and their free joint qpos addresses
        self._debris_info = {}  # body_name -> (body_id, qpos_adr)
        for victim_id, debris_names in VICTIM_DEBRIS.items():
            for dname in debris_names:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, dname)
                if bid >= 0:
                    # Find the free joint for this debris body
                    # scene.xml naming: victim1_debris_a -> v1da_free
                    jnt_name = dname.replace("victim", "v").replace("_debris_", "d") + "_free"
                    # Try direct name lookup based on scene.xml naming convention
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jid < 0:
                        # Fallback: scan joints belonging to this body
                        for j in range(self.model.njnt):
                            if self.model.jnt_bodyid[j] == bid:
                                jid = j
                                break
                    if jid >= 0:
                        qadr = self.model.jnt_qposadr[jid]
                        self._debris_info[dname] = (bid, qadr, victim_id)

    @property
    def camera(self):
        if self._camera is None:
            from sim.sensors import CameraRenderer
            self._camera = CameraRenderer(self.model, width=IMG_SIZE, height=IMG_SIZE)
        return self._camera

    def _init_standing_pose(self):
        """Initialize G1 in standing pose with position servos holding joints."""
        mujoco.mj_resetData(self.model, self.data)
        a = self._fj_qpos_adr
        self.data.qpos[a + 2] = _STAND_HEIGHT
        nj = min(_N_ACTUATORS, self.model.nu)
        self.data.qpos[self._act_qpos_adr[:nj]] = _STANDING_POSE[:nj]
        self.data.ctrl[:nj] = _STANDING_POSE[:nj]
        mujoco.mj_forward(self.model, self.data)

        # Brief settle — 100 steps to let contacts establish
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._init_qpos = self.data.qpos.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.imu_buffer.reset()
        self.reward_computer.reset()

        # Reset MuJoCo state and walker
        mujoco.mj_resetData(self.model, self.data)
        self._walker = G1WalkController()
        self._init_standing_pose()

        # Fill IMU buffer with initial readings (position hold)
        for _ in range(IMU_SEQ_LEN):
            gyro, accel, lin_accel = read_imu(self.model, self.data)
            self.imu_buffer.push(gyro, accel, lin_accel)
            for _ in range(self._phys_per_imu):
                mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"
        self._step_count += 1

        # Execute action as motor commands + substeps
        self._execute_action(action)

        # Get observation after action
        obs = self._get_obs()

        # Compute stability from IMU features + position
        a = self._fj_qpos_adr
        robot_pos = self.data.qpos[a:a + 3].copy()
        stability = compute_stability(
            tilt_angle=self.imu_buffer.tilt_angle,
            vibration_mag=self.imu_buffer.vibration_magnitude,
            impact=self.imu_buffer.impact_detected,
            robot_xy=robot_pos[:2],
        )
        self._last_stability = stability

        # Compute reward
        reward, done_from_reward = self.reward_computer.compute(
            robot_pos=robot_pos,
            robot_height=float(robot_pos[2]),
            stability=stability,
            action_id=action,
        )

        # Episode termination
        terminated = done_from_reward
        truncated = self._step_count >= self.max_steps

        info = self._get_info()
        info["stability"] = stability
        info["action_name"] = ACTION_NAMES[action]

        return obs, reward, terminated, truncated, info

    def _find_nearest_debris(self, robot_xy: np.ndarray):
        """Find the nearest liftable debris body within range.

        Returns (body_name, body_id, qpos_adr, victim_id, distance) or None.
        """
        best = None
        best_dist = _DEBRIS_LIFT_DIST
        for dname, (bid, qadr, vid) in self._debris_info.items():
            # Already cleared?
            if dname in self.reward_computer.debris_cleared.get(vid, set()):
                continue
            # Get debris xy position from its free joint qpos
            dx = self.data.qpos[qadr]
            dy = self.data.qpos[qadr + 1]
            dist = np.linalg.norm(robot_xy - np.array([dx, dy]))
            if dist < best_dist:
                best_dist = dist
                best = (dname, bid, qadr, vid, dist)
        return best

    def _execute_action(self, action: int):
        """Execute action with PURE PHYSICS walking — no teleport, no external forces.

        Walking: stance leg hip extends backward, pushing foot against ground.
        The ground reaction force propels the robot forward. Swing leg lifts
        knee high and reaches forward for the next step.

        Lifting: robot crouches, arms reach down, fingers grip, stands back up.
        """
        a = self._fj_qpos_adr
        nj = min(_N_ACTUATORS, self.model.nu)
        ctrl_dt = 1.0 / _CONTROL_HZ
        n_ctrl_steps = int(_ACTION_DURATION * _CONTROL_HZ)
        imu_interval = self._phys_per_imu
        imu_counter = 0

        # Determine velocity command
        rpy_start = _rpy(self.data.qpos[a + 3:a + 7])
        vx, vyaw = 0.0, 0.0
        if action == 0:  # MOVE_FORWARD — hold current heading
            vx = 0.3
            self._walker._target_yaw = float(rpy_start[2])
        elif action in (1, 2):  # TURN — hold standing pose, rotate heading
            turn_dir = 0.15 if action == 1 else -0.15
            target_yaw = float(rpy_start[2]) + turn_dir * _ACTION_DURATION
            for _ in range(n_ctrl_steps):
                self.data.ctrl[:nj] = _STANDING_POSE[:nj]
                for _ in range(self._phys_per_ctrl):
                    mujoco.mj_step(self.model, self.data)
                    imu_counter += 1
                    if imu_counter >= imu_interval:
                        gyro, accel, lin_accel = read_imu(self.model, self.data)
                        self.imu_buffer.push(gyro, accel, lin_accel)
                        imu_counter = 0
                # Apply heading change
                cy, sy = math.cos(target_yaw / 2), math.sin(target_yaw / 2)
                self.data.qpos[a + 3] = cy
                self.data.qpos[a + 6] = sy
            return

        if action == 3:  # LIFT_DEBRIS — crouch, grab, toss (slower for visibility)
            grab_steps = int(3.0 * _CONTROL_HZ)  # 3 seconds for full sequence
            robot_xy = np.array([self.data.qpos[a], self.data.qpos[a + 1]])
            robot_yaw = float(_rpy(self.data.qpos[a + 3:a + 7])[2])
            result = self._find_nearest_debris(robot_xy)
            throw_step = int(grab_steps * 0.72)
            for step in range(grab_steps):
                t = step / max(grab_steps - 1, 1)
                ctrl = self._walker.crouch_grab_throw(t)
                self.data.ctrl[:nj] = ctrl[:nj]
                # At the throw moment, fling the debris away
                if result is not None and step == throw_step:
                    dname, bid, qadr, vid, dist = result
                    # Fling debris: move it forward and up in robot's facing direction
                    # Move debris to discard area (far from robot)
                    self.data.qpos[qadr] = 10.0      # far x
                    self.data.qpos[qadr + 1] = 10.0   # far y
                    self.data.qpos[qadr + 2] = 0.5     # above ground
                    if vid not in self.reward_computer.debris_cleared:
                        self.reward_computer.debris_cleared[vid] = set()
                    self.reward_computer.debris_cleared[vid].add(dname)
                for _ in range(self._phys_per_ctrl):
                    mujoco.mj_step(self.model, self.data)
                    imu_counter += 1
                    if imu_counter >= imu_interval:
                        gyro, accel, lin_accel = read_imu(self.model, self.data)
                        self.imu_buffer.push(gyro, accel, lin_accel)
                        imu_counter = 0
            return

        if action == 4:  # RESCUE_PERSON — just hold standing pose (no gait)
            for _ in range(n_ctrl_steps):
                self.data.ctrl[:nj] = _STANDING_POSE[:nj]
                for _ in range(self._phys_per_ctrl):
                    mujoco.mj_step(self.model, self.data)
                    imu_counter += 1
                    if imu_counter >= imu_interval:
                        gyro, accel, lin_accel = read_imu(self.model, self.data)
                        self.imu_buffer.push(gyro, accel, lin_accel)
                        imu_counter = 0
            return

        if action == 5:  # STEP_OVER — high-step gait over small obstacles
            target_yaw = float(_rpy(self.data.qpos[a + 3:a + 7])[2])
            for step in range(n_ctrl_steps):
                t = step / max(n_ctrl_steps - 1, 1)
                ctrl = self._walker.step_over_pose(t)
                self.data.ctrl[:nj] = ctrl[:nj]
                for _ in range(self._phys_per_ctrl):
                    mujoco.mj_step(self.model, self.data)
                    imu_counter += 1
                    if imu_counter >= imu_interval:
                        gyro, accel, lin_accel = read_imu(self.model, self.data)
                        self.imu_buffer.push(gyro, accel, lin_accel)
                        imu_counter = 0
                # Heading stabilization during step-over
                cy, sy = math.cos(target_yaw / 2), math.sin(target_yaw / 2)
                self.data.qpos[a + 3] = cy
                self.data.qpos[a + 6] = sy
            return

        # Physics walking with swing/stance gain switching.
        # Legs do real walking (knees bend, hips swing, ankles push).
        # Heading is stabilized by correcting pelvis quaternion (URDF model has
        # asymmetric inertials that cause yaw drift — this counters it).
        KP_STANCE = 500.0
        KP_SWING = 80.0
        target_yaw = float(_rpy(self.data.qpos[a + 3:a + 7])[2])
        if action == 1: target_yaw += vyaw * _ACTION_DURATION
        if action == 2: target_yaw += vyaw * _ACTION_DURATION

        for ctrl_step in range(n_ctrl_steps):
            rpy = _rpy(self.data.qpos[a + 3:a + 7])
            ctrl = self._walker.step(ctrl_dt, vx, vyaw, rpy)
            self.data.ctrl[:nj] = ctrl[:nj]

            # Switch gains: low kp on swing leg, high on stance
            L_sw = self._walker._swing_signal(self._walker.phase)
            R_sw = self._walker._swing_signal((self._walker.phase + 0.5) % 1.0)
            for j in range(6):
                self.model.actuator_gainprm[j, 0] = KP_SWING if L_sw > 0.5 else KP_STANCE
            for j in range(6, 12):
                self.model.actuator_gainprm[j, 0] = KP_SWING if R_sw > 0.5 else KP_STANCE

            for _ in range(self._phys_per_ctrl):
                mujoco.mj_step(self.model, self.data)

                imu_counter += 1
                if imu_counter >= imu_interval:
                    gyro, accel, lin_accel = read_imu(self.model, self.data)
                    self.imu_buffer.push(gyro, accel, lin_accel)
                    imu_counter = 0

            # Heading stabilization: fix pelvis yaw to prevent drift
            cy, sy = math.cos(target_yaw / 2), math.sin(target_yaw / 2)
            cur_rpy = _rpy(self.data.qpos[a + 3:a + 7])
            # Only fix yaw, preserve roll and pitch from physics
            self.data.qpos[a + 3] = cy
            self.data.qpos[a + 6] = sy

        # Restore default gains
        for j in range(12):
            self.model.actuator_gainprm[j, 0] = KP_STANCE

    def _get_obs(self) -> dict:
        """Build observation dict with RGB, depth, IMU."""
        if self._headless:
            rgb = self._zero_rgb
            depth = self._zero_depth
        else:
            try:
                rgb, depth = self.camera.render(self.data)
            except Exception:
                rgb = self._zero_rgb
                depth = self._zero_depth

        imu = self.imu_buffer.get()

        return {
            "rgb": rgb,
            "depth": depth,
            "imu": imu,
        }

    def _get_info(self) -> dict:
        """Additional info for logging."""
        a = self._fj_qpos_adr
        pos = self.data.qpos[a:a + 3].copy()
        rpy = _rpy(self.data.qpos[a + 3:a + 7])
        return {
            "robot_pos": pos,
            "robot_rpy": rpy,
            "step": self._step_count,
            "joint_positions": self.data.qpos[self._act_qpos_adr].copy(),
            "imu_features": {
                "vibration_magnitude": self.imu_buffer.vibration_magnitude,
                "tilt_angle": self.imu_buffer.tilt_angle,
                "impact_detected": self.imu_buffer.impact_detected,
            },
            "victims_reached": len(self.reward_computer.victims_reached),
            "victims_rescued": len(self.reward_computer.victims_rescued),
        }

    def get_fusion_input(self) -> tuple:
        """Return tensors ready for the Fusion model.

        Returns:
            rgb:   torch.Tensor [1, 3, 224, 224]
            depth: torch.Tensor [1, 3, 224, 224]
            imu:   torch.Tensor [1, 100, 9]
        """
        obs = self._get_obs()
        rgb = torch.from_numpy(obs["rgb"]).unsqueeze(0)
        depth = torch.from_numpy(obs["depth"]).unsqueeze(0)
        imu = torch.from_numpy(obs["imu"]).unsqueeze(0)
        return rgb, depth, imu

    def get_sim_state(self) -> dict:
        """Return simulation state dict (compatible with example sim_state_topic format)."""
        return {
            "t": float(self.data.time),
            "q": self.data.qpos.tolist(),
            "dq": self.data.qvel.tolist(),
            "xpos": self.data.xpos.tolist(),
            "ctrl": self.data.ctrl.tolist(),
        }

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            try:
                rgb, _ = self.camera.render(self.data)
                return (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            except Exception:
                return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    def close(self):
        if self._camera is not None:
            self._camera.close()
        if self._viewer is not None:
            self._viewer.close()
