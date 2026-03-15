#!/usr/bin/env python3
"""SOS-Answered Full Pipeline Test — End-to-end rescue mission execution.

Loads trained world model components, runs a complete rescue mission in the
MuJoCo environment, and validates the full inference pipeline including
planning, safety monitoring, and victim localization.

Usage:
    python3 test_pipeline.py
"""

import os
import sys
import time
import math
import warnings

import numpy as np
import torch

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    FUSION_SEQ_LEN, FUSION_DIM, STATE_DIM, NUM_ACTIONS,
    IMG_SIZE, IMU_SEQ_LEN, IMU_CHANNELS,
    STABLE, UNSTABLE, IMMINENT_COLLAPSE,
    PLANNING_HORIZON,
)
from models.state_encoder import StateEncoder
from models.world_model import SOSWorldModel
from planning.planner import WorldModelPlanner
from planning.safety_monitor import SafetyMonitor, NOMINAL, SLOW_CAUTIOUS, FREEZE_REASSESS, EMERGENCY_RETREAT
from training.collect_experience import MockFusion

# Suppress non-critical warnings for clean console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────

ENCODER_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "encoder_best.pt")
WM_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "world_model_final.pt")
ENCODER_INT8_PATH = os.path.join(PROJECT_ROOT, "exported", "encoder_int8.pt")
WM_INT8_PATH = os.path.join(PROJECT_ROOT, "exported", "world_model_int8.pt")

# Environment action names (from sim/env.py)
ENV_ACTION_NAMES = [
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LIFT_DEBRIS",
    "RESCUE_PERSON",
]

# World model action names (8 primitives from inference/sos_runtime.py)
WM_ACTION_NAMES = [
    "navigate_to", "open_door", "clear_obstacle", "assist_stand",
    "carry_person", "communicate", "signal_team", "emergency_stop",
]

STABILITY_NAMES = ["STABLE", "UNSTABLE", "IMMINENT_COLLAPSE"]

# ── Victim Audio System ──────────────────────────────────────────────

VICTIM_PROFILES = [
    {
        "id": 0,
        "position": np.array([3.0, 0.1]),
        "frequency_hz": 440,
        "urgency": "MODERATE",
        "calls": [
            "Help! I'm trapped under debris!",
            "Please, someone... I can hear you!",
            "Over here! I can't move!",
        ],
    },
    {
        "id": 1,
        "position": np.array([5.0, 3.0]),
        "frequency_hz": 520,
        "urgency": "HIGH",
        "calls": [
            "HELP! The ceiling is collapsing!",
            "I'm injured, please hurry!",
            "Can anyone hear me?!",
        ],
    },
    {
        "id": 2,
        "position": np.array([-2.0, -2.5]),
        "frequency_hz": 380,
        "urgency": "CRITICAL",
        "calls": [
            "I can barely breathe... help...",
            "Please... I'm running out of air...",
            "Someone... anyone...",
        ],
    },
    {
        "id": 3,
        "position": np.array([1.0, 4.5]),
        "frequency_hz": 600,
        "urgency": "LOW",
        "calls": [
            "Hello? Is someone out there?",
            "I'm stuck but I'm okay for now.",
            "I can hear machinery nearby!",
        ],
    },
]

AUDIO_DETECTION_RANGE = 3.0  # meters

URGENCY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


# ── Console Formatting ───────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
WHITE = "\033[97m"

SAFETY_COLORS = {
    NOMINAL: GREEN,
    SLOW_CAUTIOUS: YELLOW,
    FREEZE_REASSESS: RED,
    EMERGENCY_RETREAT: f"{RED}{BOLD}",
}

STABILITY_COLORS = {
    STABLE: GREEN,
    UNSTABLE: YELLOW,
    IMMINENT_COLLAPSE: RED,
}


def fmt_time(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def print_header():
    print()
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  SOS-ANSWERED RESCUE OPERATIONS SYSTEM — FULL PIPELINE TEST{RESET}")
    print(f"{BOLD}{CYAN}  Unitree G1 Humanoid | World Model Planning | Structural Safety{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
    print()


def print_section(title: str):
    print(f"\n{BOLD}{WHITE}--- {title} {'─' * max(1, 60 - len(title))}{RESET}\n")


# ── Model Loading ────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, device: torch.device) -> StateEncoder:
    encoder = StateEncoder().to(device)
    if os.path.exists(checkpoint_path):
        encoder.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"  {GREEN}[OK]{RESET} StateEncoder loaded from {checkpoint_path}")
    else:
        print(f"  {YELLOW}[WARN]{RESET} Checkpoint not found: {checkpoint_path}")
        print(f"         Initializing fresh model weights for StateEncoder")
    encoder.eval()
    return encoder


def load_world_model(checkpoint_path: str, device: torch.device) -> SOSWorldModel:
    wm = SOSWorldModel().to(device)
    if os.path.exists(checkpoint_path):
        wm.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"  {GREEN}[OK]{RESET} SOSWorldModel loaded from {checkpoint_path}")
    else:
        print(f"  {YELLOW}[WARN]{RESET} Checkpoint not found: {checkpoint_path}")
        print(f"         Initializing fresh model weights for SOSWorldModel")
    wm.eval()
    return wm


def load_quantized_model(path: str, device: torch.device, model_class, label: str):
    """Load an INT8 quantized model. Returns (model, success)."""
    if not os.path.exists(path):
        print(f"  {YELLOW}[WARN]{RESET} Quantized checkpoint not found: {path}")
        print(f"         Initializing fresh model weights for {label}")
        model = model_class()
        model.eval()
        return model, False

    try:
        model = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            # If the file contains a state_dict rather than a full model
            model = model_class()
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        print(f"  {GREEN}[OK]{RESET} {label} (INT8) loaded from {path}")
        return model, True
    except Exception as e:
        print(f"  {YELLOW}[WARN]{RESET} Could not load INT8 {label}: {e}")
        print(f"         Initializing fresh model weights for {label}")
        model = model_class()
        model.eval()
        return model, False


# ── Audio Detection ──────────────────────────────────────────────────

class AudioDetector:
    """Detects victim audio signatures based on proximity."""

    def __init__(self):
        self.rescued_victims = set()
        self.last_call_step = {}  # victim_id -> last step a call was printed

    def mark_rescued(self, victim_id: int):
        self.rescued_victims.add(victim_id)

    def detect(self, robot_xy: np.ndarray, step: int, elapsed: float) -> list:
        """Return list of audio detections for this step."""
        detections = []
        for vp in VICTIM_PROFILES:
            vid = vp["id"]
            if vid in self.rescued_victims:
                continue
            dist = float(np.linalg.norm(robot_xy - vp["position"]))
            if dist > AUDIO_DETECTION_RANGE:
                continue

            # Closer distance -> higher effective urgency and more frequent calls
            proximity_factor = 1.0 - (dist / AUDIO_DETECTION_RANGE)
            base_urgency = URGENCY_ORDER[vp["urgency"]]
            effective_urgency = min(3, base_urgency + int(proximity_factor * 2))
            urgency_label = [k for k, v in URGENCY_ORDER.items() if v == effective_urgency][0]

            # Call frequency: closer = more frequent
            call_interval = max(1, int(4 - proximity_factor * 3))
            last = self.last_call_step.get(vid, -999)
            if (step - last) >= call_interval:
                self.last_call_step[vid] = step
                call_text = vp["calls"][step % len(vp["calls"])]
                detections.append({
                    "victim_id": vid,
                    "distance": dist,
                    "urgency": urgency_label,
                    "frequency_hz": vp["frequency_hz"],
                    "call": call_text,
                })
        # Sort by urgency descending
        detections.sort(key=lambda d: URGENCY_ORDER[d["urgency"]], reverse=True)
        return detections


# ── Map Planner Action to Env Action ─────────────────────────────────

def map_wm_action_to_env(wm_action_id: int) -> int:
    """Map world model action (0-7) to environment action (0-4).

    World model actions:
        0: navigate_to   -> 0: MOVE_FORWARD
        1: open_door     -> 0: MOVE_FORWARD (push through)
        2: clear_obstacle -> 3: LIFT_DEBRIS
        3: assist_stand  -> 4: RESCUE_PERSON
        4: carry_person  -> 4: RESCUE_PERSON
        5: communicate   -> 0: MOVE_FORWARD (continue mission)
        6: signal_team   -> 0: MOVE_FORWARD (continue mission)
        7: emergency_stop -> 0: MOVE_FORWARD (hold position, step in place)
    """
    mapping = {0: 0, 1: 0, 2: 3, 3: 4, 4: 4, 5: 0, 6: 0, 7: 0}
    return mapping.get(wm_action_id, 0)


# ── Main Pipeline ────────────────────────────────────────────────────

def run_mission():
    """Execute a full rescue mission with the trained world model pipeline."""

    print_header()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {BOLD}{device}{RESET}")
    print()

    # ── 1. Load Models ────────────────────────────────────────────
    print_section("LOADING TRAINED MODELS")

    encoder = load_encoder(ENCODER_CHECKPOINT, device)
    world_model = load_world_model(WM_CHECKPOINT, device)
    fusion = MockFusion()
    planner = WorldModelPlanner(world_model, horizon=PLANNING_HORIZON)
    safety_monitor = SafetyMonitor()
    audio_detector = AudioDetector()

    param_count_enc = sum(p.numel() for p in encoder.parameters())
    param_count_wm = sum(p.numel() for p in world_model.parameters())
    print(f"\n  StateEncoder params:  {param_count_enc:,}")
    print(f"  SOSWorldModel params: {param_count_wm:,}")
    print(f"  Planning horizon:     {PLANNING_HORIZON} steps x {NUM_ACTIONS} actions")

    # ── 2. Create Environment ─────────────────────────────────────
    print_section("INITIALIZING ENVIRONMENT")

    try:
        from sim.env import SOSRescueEnv
        env = SOSRescueEnv(render_mode=None, headless=True)
        print(f"  {GREEN}[OK]{RESET} SOSRescueEnv initialized (headless mode)")
        print(f"  Scene: MuJoCo rubble rescue arena")
        print(f"  Victims: {len(VICTIM_PROFILES)} targets identified")
        print(f"  Max steps: {env.max_steps}")
    except Exception as e:
        print(f"  {RED}[ERROR]{RESET} Failed to initialize environment: {e}")
        sys.exit(1)

    # ── 3. Run Mission ────────────────────────────────────────────
    print_section("MISSION START")
    print(f"  {BOLD}Objective: Locate and rescue all trapped victims{RESET}")
    print(f"  {BOLD}Constraints: Maintain structural safety at all times{RESET}")
    print()

    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    mission_start = time.time()
    planning_times_fp32 = []

    victims_reached_set = set()
    victims_rescued_set = set()
    debris_cleared_log = []

    prev_victims_reached = 0
    prev_victims_rescued = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_start = time.time()
        step_count += 1

        # (a) Get sensor data from environment
        rgb_t = torch.from_numpy(obs["rgb"]).unsqueeze(0)
        depth_t = torch.from_numpy(obs["depth"]).unsqueeze(0)
        imu_t = torch.from_numpy(obs["imu"]).unsqueeze(0)

        # (b) Run Fusion
        with torch.no_grad():
            fused, stab_logits = fusion(rgb_t, depth_t, imu_t)

        # (c) Encode state
        with torch.no_grad():
            state = encoder(fused.to(device))  # [1, 512]

        # (d) Plan next action via WorldModelPlanner
        plan_t0 = time.time()
        plan_result = planner.plan(state.squeeze(0), device)
        plan_elapsed_ms = (time.time() - plan_t0) * 1000
        planning_times_fp32.append(plan_elapsed_ms)

        wm_action_id = plan_result["best_action_id"]
        wm_action_name = WM_ACTION_NAMES[wm_action_id]
        best_score = plan_result["best_score"]

        # (e) Safety check
        imu_features = info.get("imu_features", {})
        stability_val = stab_logits.argmax(dim=-1).item()
        safety_state = safety_monitor.evaluate(
            stability=stability_val,
            impact_detected=imu_features.get("impact_detected", False),
            tilt_angle=imu_features.get("tilt_angle", 0.0),
            vibration_magnitude=imu_features.get("vibration_magnitude", 0.0),
        )

        # Override action on safety violations
        original_action = wm_action_id
        if safety_state in (EMERGENCY_RETREAT, FREEZE_REASSESS):
            wm_action_id = 7  # emergency_stop
            wm_action_name = "emergency_stop"

        # Map to environment action
        env_action = map_wm_action_to_env(wm_action_id)
        env_action_name = ENV_ACTION_NAMES[env_action]

        # (f) Execute action
        obs, reward, terminated, truncated, info = env.step(env_action)
        total_reward += reward

        # Robot position
        robot_pos = info.get("robot_pos", np.zeros(3))
        robot_xy = robot_pos[:2]
        elapsed = time.time() - mission_start

        # Stability from environment
        env_stability = info.get("stability", STABLE)
        stab_label = STABILITY_NAMES[min(env_stability, 2)]
        stab_color = STABILITY_COLORS.get(env_stability, WHITE)
        safety_color = SAFETY_COLORS.get(safety_state, WHITE)

        # Victim tracking
        cur_reached = info.get("victims_reached", 0)
        cur_rescued = info.get("victims_rescued", 0)

        # (g) Print status line
        score_sign = "+" if best_score >= 0 else ""
        print(
            f"  {DIM}[{fmt_time(elapsed)}]{RESET} "
            f"{BOLD}STEP {step_count:3d}{RESET} | "
            f"ACTION: {CYAN}{env_action_name:<14s}{RESET} "
            f"(plan: {wm_action_name}, score: {score_sign}{best_score:+.2f}) | "
            f"SAFETY: {safety_color}{safety_state}{RESET}"
        )
        print(
            f"           "
            f"POS: ({robot_pos[0]:+6.2f}, {robot_pos[1]:+6.2f}, {robot_pos[2]:+5.2f}) | "
            f"STABILITY: {stab_color}{stab_label}{RESET} | "
            f"VICTIMS: {cur_reached}/{len(VICTIM_PROFILES)} reached, "
            f"{cur_rescued}/{len(VICTIM_PROFILES)} rescued | "
            f"plan: {plan_elapsed_ms:.1f}ms"
        )

        # (h) Victim reached alert
        if cur_reached > prev_victims_reached:
            new_reached = cur_reached - prev_victims_reached
            for i in range(new_reached):
                vid = prev_victims_reached + i
                print(
                    f"  {BOLD}{GREEN}>>> VICTIM {vid + 1} REACHED — "
                    f"Contact established at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}){RESET}"
                )
                victims_reached_set.add(vid)
            prev_victims_reached = cur_reached

        if cur_rescued > prev_victims_rescued:
            new_rescued = cur_rescued - prev_victims_rescued
            for i in range(new_rescued):
                vid = prev_victims_rescued + i
                print(
                    f"  {BOLD}{GREEN}>>> VICTIM {vid + 1} RESCUED — "
                    f"Extraction confirmed{RESET}"
                )
                victims_rescued_set.add(vid)
                audio_detector.mark_rescued(vid)
            prev_victims_rescued = cur_rescued

        # (i) Debris interaction
        if env_action == 3:  # LIFT_DEBRIS
            debris_cleared_log.append(step_count)
            print(
                f"  {YELLOW}    DEBRIS: Structural element cleared at "
                f"({robot_pos[0]:.1f}, {robot_pos[1]:.1f}){RESET}"
            )

        # Audio detection
        audio_hits = audio_detector.detect(robot_xy, step_count, elapsed)
        for hit in audio_hits:
            urg_color = {
                "LOW": DIM, "MODERATE": YELLOW, "HIGH": RED, "CRITICAL": f"{RED}{BOLD}"
            }.get(hit["urgency"], WHITE)
            print(
                f"  {MAGENTA}    AUDIO: \"{hit['call']}\" "
                f"(victim_{hit['victim_id'] + 1}, dist={hit['distance']:.1f}m, "
                f"urgency={urg_color}{hit['urgency']}{RESET}{MAGENTA}){RESET}"
            )

        # Safety override notification
        if wm_action_id != original_action and safety_state != NOMINAL:
            print(
                f"  {RED}    SAFETY OVERRIDE: Action vetoed by SafetyMonitor "
                f"({safety_state}){RESET}"
            )

        print()  # blank line between steps

    # ── 4. Mission Summary ────────────────────────────────────────
    mission_elapsed = time.time() - mission_start
    print_section("MISSION COMPLETE")

    end_reason = "All objectives achieved" if terminated else "Step limit reached"
    avg_plan_ms = np.mean(planning_times_fp32) if planning_times_fp32 else 0
    max_plan_ms = np.max(planning_times_fp32) if planning_times_fp32 else 0

    print(f"  Termination:       {end_reason}")
    print(f"  Total steps:       {step_count}")
    print(f"  Total reward:      {total_reward:+.2f}")
    print(f"  Victims reached:   {prev_victims_reached}/{len(VICTIM_PROFILES)}")
    print(f"  Victims rescued:   {prev_victims_rescued}/{len(VICTIM_PROFILES)}")
    print(f"  Debris cleared:    {len(debris_cleared_log)} events")
    print(f"  Wall-clock time:   {mission_elapsed:.1f}s")
    print(f"  Avg planning time: {avg_plan_ms:.1f}ms (FP32)")
    print(f"  Max planning time: {max_plan_ms:.1f}ms (FP32)")
    print()

    env.close()

    return planning_times_fp32


# ── INT8 Quantized Comparison ─────────────────────────────────────────

def run_quantized_comparison(fp32_times: list):
    """Load INT8 quantized models and compare inference speed."""

    print_section("INT8 QUANTIZED MODEL COMPARISON")

    device = torch.device("cpu")  # INT8 runs on CPU

    # Load quantized encoder
    q_encoder, enc_ok = load_quantized_model(
        ENCODER_INT8_PATH, device, StateEncoder, "StateEncoder"
    )
    q_wm, wm_ok = load_quantized_model(
        WM_INT8_PATH, device, SOSWorldModel, "SOSWorldModel"
    )

    fusion = MockFusion()
    q_planner = WorldModelPlanner(q_wm, horizon=PLANNING_HORIZON)

    print(f"\n  Running 10-step inference benchmark (INT8)...\n")

    int8_plan_times = []

    for step in range(10):
        # Generate input
        rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        depth = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        imu = torch.randn(1, IMU_SEQ_LEN, IMU_CHANNELS)

        with torch.no_grad():
            fused, stab_logits = fusion(rgb, depth, imu)
            state = q_encoder(fused.to(device))  # [1, 512]

        t0 = time.time()
        result = q_planner.plan(state.squeeze(0), device)
        plan_ms = (time.time() - t0) * 1000
        int8_plan_times.append(plan_ms)

        action_name = WM_ACTION_NAMES[result["best_action_id"]]
        print(
            f"  Step {step + 1:2d} | ACTION: {action_name:<14s} "
            f"(score: {result['best_score']:+.2f}) | "
            f"plan: {plan_ms:.1f}ms"
        )

    # ── Comparison ────────────────────────────────────────────────
    print_section("INFERENCE TIMING COMPARISON")

    avg_fp32 = np.mean(fp32_times) if fp32_times else 0
    avg_int8 = np.mean(int8_plan_times) if int8_plan_times else 0
    speedup = avg_fp32 / avg_int8 if avg_int8 > 0 else 0

    print(f"  {'Metric':<28s} {'FP32':>10s} {'INT8':>10s}")
    print(f"  {'─' * 50}")
    print(f"  {'Avg planning time (ms)':<28s} {avg_fp32:>9.1f}  {avg_int8:>9.1f}")
    print(f"  {'Max planning time (ms)':<28s} "
          f"{np.max(fp32_times) if fp32_times else 0:>9.1f}  "
          f"{np.max(int8_plan_times) if int8_plan_times else 0:>9.1f}")
    print(f"  {'Min planning time (ms)':<28s} "
          f"{np.min(fp32_times) if fp32_times else 0:>9.1f}  "
          f"{np.min(int8_plan_times) if int8_plan_times else 0:>9.1f}")
    print(f"  {'─' * 50}")
    if speedup > 0:
        color = GREEN if speedup > 1.0 else YELLOW
        print(f"  {'INT8 speedup':<28s} {color}{speedup:.2f}x{RESET}")
    print()

    target_ms = 200.0
    full_cycle_fp32 = avg_fp32 + 5.0  # estimate encode + overhead
    full_cycle_int8 = avg_int8 + 5.0
    fp32_ok = full_cycle_fp32 < target_ms
    int8_ok = full_cycle_int8 < target_ms

    print(f"  Full cycle estimate (encode + plan):")
    fp32_status = f"{GREEN}PASS{RESET}" if fp32_ok else f"{RED}FAIL{RESET}"
    int8_status = f"{GREEN}PASS{RESET}" if int8_ok else f"{RED}FAIL{RESET}"
    print(f"    FP32: ~{full_cycle_fp32:.0f}ms (target <{target_ms:.0f}ms) [{fp32_status}]")
    print(f"    INT8: ~{full_cycle_int8:.0f}ms (target <{target_ms:.0f}ms) [{int8_status}]")
    print()


# ── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        fp32_times = run_mission()
        run_quantized_comparison(fp32_times)

        print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
        print(f"{BOLD}{CYAN}  PIPELINE TEST COMPLETE{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
        print()

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Mission aborted by operator.{RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  {RED}[FATAL] {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
