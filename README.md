# SOS-Answered: Autonomous Humanoid Rescue in Collapsed Structures

```
  ███████╗ ██████╗ ███████╗       █████╗ ███╗   ██╗███████╗██╗    ██╗███████╗██████╗ ███████╗██████╗
  ██╔════╝██╔═══██╗██╔════╝      ██╔══██╗████╗  ██║██╔════╝██║    ██║██╔════╝██╔══██╗██╔════╝██╔══██╗
  ███████╗██║   ██║███████╗█████╗███████║██╔██╗ ██║███████╗██║ █╗ ██║█████╗  ██████╔╝█████╗  ██║  ██║
  ╚════██║██║   ██║╚════██║╚════╝██╔══██║██║╚██╗██║╚════██║██║███╗██║██╔══╝  ██╔══██╗██╔══╝  ██║  ██║
  ███████║╚██████╔╝███████║      ██║  ██║██║ ╚████║███████║╚███╔███╔╝███████╗██║  ██║███████╗██████╔╝
  ╚══════╝ ╚═════╝ ╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═════╝
```

> **A multimodal AI system that enables a Unitree G1 humanoid robot to autonomously navigate rubble, detect trapped victims through audio and vision, clear debris with physical manipulation, and execute rescue operations — all powered by a learned world model that simulates futures before committing to action.**

---

## The Problem

```
                     ┌──────────────────────────────────────┐
                     │      EARTHQUAKE / BUILDING COLLAPSE  │
                     │                                      │
                     │   ████  Rubble    ████  Debris       │
                     │   ████            ████               │
                     │        "Help!"              ████     │
                     │   ████        Victim trapped ████    │
                     │   ████     under debris      ████    │
                     │        "Can anyone hear me?"         │
                     │   ████                       ████    │
                     │   Unstable     ████   Danger         │
                     │   structure    ████   zones          │
                     └──────────────────────────────────────┘

     After a structural collapse, the first 72 hours are critical.
     Human rescuers face life-threatening conditions: unstable walls,
     toxic dust, secondary collapses. Every minute matters.

     SOS-Answered sends a humanoid robot FIRST.
```

**71,000+ people** died in the 2023 Turkey-Syria earthquake. Rescue teams couldn't reach victims fast enough through collapsed structures. SOS-Answered changes this — the robot goes in first, assesses structural stability, locates victims by their calls for help, physically removes debris, and signals human teams exactly where to focus.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOS-ANSWERED FULL PIPELINE                       │
│                                                                     │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐                     │
│  │  RGB CAM  │   │ DEPTH CAM │   │    IMU    │   SENSOR INPUTS     │
│  │ 224×224×3 │   │ 224×224×3 │   │ 100×9     │   (500 Hz)          │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                     │
│        │               │               │                            │
│        └───────────────┼───────────────┘                            │
│                        ▼                                            │
│           ┌────────────────────────┐                                │
│           │   MULTIMODAL FUSION    │                                │
│           │                        │                                │
│           │  Audio Encoder (256d)  │   ◄── SmolLM2-135M + LoRA     │
│           │  Depth Encoder (256d)  │       (RL fine-tuned)          │
│           │  IMU Encoder   (256d)  │                                │
│           │        ↓               │                                │
│           │  Concat → 768d         │                                │
│           │  Project → [492,2048]  │                                │
│           └───────────┬────────────┘                                │
│                       ▼                                             │
│           ┌────────────────────────┐                                │
│           │    STATE ENCODER       │    657K params                  │
│           │  [492,2048] → [512]    │    INT8 quantized (651 KB)     │
│           │  Linear→GELU→Pool→LN  │                                │
│           └───────────┬────────────┘                                │
│                       ▼                                             │
│           ┌────────────────────────┐                                │
│           │   WORLD MODEL (GRU)   │    2.4M params                  │
│           │                        │    INT8 quantized (6.8 MB)     │
│           │  State[512] + Action   │                                │
│           │       ↓                │                                │
│           │  Embed(8→64) → Concat  │                                │
│           │  [576] → FC → GRU      │                                │
│           │       ↓                │                                │
│           │  ┌──────────────────┐  │                                │
│           │  │ 4 Prediction     │  │                                │
│           │  │ Heads:           │  │                                │
│           │  │  next_state[512] │  │                                │
│           │  │  reward[1]       │  │                                │
│           │  │  done[1]         │  │                                │
│           │  │  stability[3]   │  │   ◄── 3× weighted loss          │
│           │  └──────────────────┘  │                                │
│           └───────────┬────────────┘                                │
│                       ▼                                             │
│           ┌────────────────────────┐                                │
│           │  WORLD MODEL PLANNER   │                                │
│           │                        │                                │
│           │  For each action (8):  │                                │
│           │    Simulate 5 steps    │                                │
│           │    Sum rewards         │                                │
│           │    Check stability     │                                │
│           │                        │                                │
│           │  Select: max(score)    │                                │
│           └───────────┬────────────┘                                │
│                       ▼                                             │
│           ┌────────────────────────┐                                │
│           │    SAFETY MONITOR      │                                │
│           │                        │                                │
│           │  IMMINENT_COLLAPSE?    │──── EMERGENCY_RETREAT           │
│           │  Impact detected?      │──── FREEZE_REASSESS             │
│           │  Tilt > 0.15 rad?     │──── SLOW_CAUTIOUS               │
│           │  Vibration > 2.5?     │──── SLOW_CAUTIOUS               │
│           │  Otherwise            │──── NOMINAL ✓                    │
│           └───────────┬────────────┘                                │
│                       ▼                                             │
│              EXECUTE ON G1 ROBOT                                    │
│              Log experience → overnight fine-tune                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Multimodal Perception System

SOS-Answered processes **three sensor modalities simultaneously** to understand the disaster environment:

### Audio: Detecting Victims Through Sound

```
  Microphone Input (16 kHz)
         │
         ▼
  ┌──────────────────────────┐
  │   Mel Spectrogram        │   128 bins × 126 frames
  │   ┌──┬──┬──┬──┬──┬──┐   │
  │   │  │▓▓│  │▓▓│  │  │   │   "Help! I'm trapped!"
  │   │  │▓▓│▓▓│▓▓│▓▓│  │   │
  │   │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│   │     → HELP_PLEA (class 0)
  │   │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│   │     → urgency: CRITICAL
  │   └──┴──┴──┴──┴──┴──┘   │     → distance: 1.2m
  └──────────┬───────────────┘
             ▼
  Audio Encoder → 256-dim embedding
```

**6 Audio Classes Detected:**

| Class | Label | What It Means |
|:---:|:---|:---|
| 0 | `HELP_PLEA` | Human voice calling for help — victim located |
| 1 | `ELECTRICAL_HISS` | Live electrical hazard nearby |
| 2 | `WATER_PIPE` | Broken water main — flooding risk |
| 3 | `STRUCTURAL_STRESS` | Building still shifting — collapse risk |
| 4 | `SILENCE` | No signals — area likely clear |
| 5 | `AMBIENT` | Background noise — non-critical |

### Vision: RGB + Depth Perception

```
  ┌────────────────┐    ┌────────────────┐
  │   RGB Camera   │    │  Depth Camera  │
  │   224 × 224    │    │   224 × 224    │
  │                │    │                │
  │  ┌──────────┐  │    │  ░░▒▒▓▓██████  │
  │  │  Victim  │  │    │  ░░▒▒▓▓██████  │  near ░░  far ██
  │  │  under   │  │    │  ░░▒▒▓▓██████  │
  │  │  rubble  │  │    │  ░░░░▒▒▓▓████  │
  │  └──────────┘  │    │  ░░░░▒▒▒▒▓▓██  │
  └────────────────┘    └────────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         Depth Encoder → 256-dim embedding
```

### IMU: Structural Stability Assessment

```
  IMU Buffer: 100 timesteps × 9 channels @ 100 Hz

  ┌─────────────────────────────────────────────────┐
  │ Accel X  ═══════════╗                           │
  │ Accel Y  ═══════════╬══► vibration_magnitude    │
  │ Accel Z  ═══════════╝                           │
  │ Gyro  X  ═══════╗                               │
  │ Gyro  Y  ═══════╬═════► tilt_angle (rad)        │
  │ Gyro  Z  ═══════╝                               │
  │ Lin Acc X ═══╗                                  │
  │ Lin Acc Y ═══╬═════════► impact_detected (bool) │
  │ Lin Acc Z ═══╝                                  │
  └─────────────────────────────────────────────────┘
           │
           ▼
  IMU Encoder → 256-dim embedding

  ┌─────────────────────────────────────────┐
  │  STABILITY CLASSIFICATION               │
  │                                         │
  │  vibration < 2.0,  tilt < 0.15 rad     │
  │  ───────────────────────────────────    │
  │  ██████████████░░░░░░░░░░░░░░░░░░░░    │
  │  ▲ STABLE      ▲ UNSTABLE  ▲ COLLAPSE  │
  │  Safe to       Proceed     ABORT NOW   │
  │  proceed       with care               │
  └─────────────────────────────────────────┘
```

---

## Small Language Model (SLM) with RL Fine-Tuning

### Base Model

| Property | Value |
|:---|:---|
| **Model** | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| **Parameters** | 135 million |
| **Architecture** | Llama-style transformer |
| **Hidden size** | 576 |
| **Attention heads** | 9 (3 KV heads, grouped-query) |
| **Layers** | 30 |
| **Intermediate size** | 1536 |
| **Context window** | 8,192 tokens |
| **Precision** | bfloat16 |

### RL / LoRA Fine-Tuning

The SLM was fine-tuned using **PEFT (Parameter-Efficient Fine-Tuning)** with LoRA adapters to specialize it for rescue scenario understanding:

```
  ┌─────────────────────────────────────────────┐
  │         SmolLM2-135M-Instruct               │
  │                                             │
  │  ┌─────────┐    ┌────────┐                  │
  │  │ q_proj  │◄───│ LoRA A │  r=16            │
  │  │         │    │ LoRA B │  α=32            │
  │  ├─────────┤    ├────────┤  dropout=0.05    │
  │  │ k_proj  │◄───│ LoRA A │                  │
  │  │         │    │ LoRA B │  Only 0.2% of    │
  │  ├─────────┤    ├────────┤  params trained  │
  │  │ v_proj  │◄───│ LoRA A │                  │
  │  │         │    │ LoRA B │                  │
  │  ├─────────┤    ├────────┤                  │
  │  │ o_proj  │◄───│ LoRA A │                  │
  │  │         │    │ LoRA B │                  │
  │  └─────────┘    └────────┘                  │
  │                                             │
  │  Frozen base weights + trained LoRA adapters │
  └─────────────────────────────────────────────┘
```

**Training Configuration:**

| Parameter | Value |
|:---|:---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| Batch size | 16 |
| Epochs | 10 |
| Optimizer | AdamW |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |

The fine-tuning teaches the model to:
- Classify audio signals (help pleas vs structural sounds vs silence)
- Interpret multimodal rescue contexts
- Generate structured action recommendations from sensor descriptions

---

## World Model: Simulating Futures Before Acting

The world model answers: **"If I take action A in state S, what will state S' look like?"**

```
  Current State                    Simulated Futures (5 steps ahead)
  ┌──────────┐
  │ State    │     Action 0: navigate_to
  │  [512]   │──── ├─ step1: reward=+0.5, stable ✓
  │          │     ├─ step2: reward=+0.3, stable ✓
  │          │     ├─ step3: reward=+10.0, victim reached! ✓
  │          │     ├─ step4: reward=+0.1, stable ✓
  │          │     └─ step5: reward=+0.2, stable ✓   TOTAL: +11.1 ◄── BEST
  │          │
  │          │     Action 2: clear_obstacle
  │          │──── ├─ step1: reward=-0.1, stable ✓
  │          │     ├─ step2: reward=+0.0, stable ✓
  │          │     └─ step3: reward=+5.0, zone mapped    TOTAL: +4.9
  │          │
  │          │     Action 7: emergency_stop
  │          │──── ├─ step1: reward=-0.1, stable ✓
  │          │     └─ ...                                TOTAL: -0.5
  │          │
  │          │     Action 0 (toward unstable zone):
  │          │──── ├─ step1: reward=-5.0, UNSTABLE ⚠
  │          │     ├─ step2: reward=-30.0, COLLAPSE! ✕
  │          │     └─ ABORTED                            TOTAL: -35.0
  └──────────┘

  Decision: Execute Action 0 (navigate_to) — score +11.1
```

### World Model Architecture

```
  State [512]  +  Action [8 → 64]
       │                │
       └──────┬─────────┘
              ▼
       Concatenate [576]
              │
       Linear(576→512) + LayerNorm + GELU
              │
       Linear(512→512) + GELU
              │
       GRUCell(512, 512)  ◄── Hidden state carries temporal memory
              │
       ┌──────┼──────────┬────────────┐
       ▼      ▼          ▼            ▼
   next_state reward    done     stability
    [512]     [1]       [1]        [3]
                                 STABLE=0
                                 UNSTABLE=1
                                 COLLAPSE=2
                                  (3× loss weight)
```

**Key Design Decision:** Stability prediction loss is weighted **3x higher** than other losses during training. Getting structural stability wrong is catastrophically dangerous — a false "stable" prediction could send the robot into a collapsing structure.

### Training Pipeline

```
  Stage 1: Collect Experience
  ┌──────────────────────────────────────┐
  │  MuJoCo Sim → 50,000 timesteps      │
  │  Store (s, a, s', r, done, stab)     │
  │  → experience_data/buffer.h5         │
  └──────────────────────────────────────┘
                    │
                    ▼
  Stage 2: Train StateEncoder (Autoencoder)
  ┌──────────────────────────────────────┐
  │  [492,2048] → Encoder → [512]       │
  │  [512] → Decoder → [492,2048]       │
  │  Loss: MSE reconstruction           │
  │  100 epochs, lr=1e-3, batch=64      │
  │  → checkpoints/encoder_best.pt      │
  │  Target: MSE < 0.1                  │
  └──────────────────────────────────────┘
                    │
                    ▼
  Stage 3: Train World Model
  ┌──────────────────────────────────────┐
  │  Encoder FROZEN (no gradients)       │
  │  Encode all states → [512]           │
  │  Train GRU transition model          │
  │  50 epochs, lr=1e-4, batch=64       │
  │  Gradient clip: max_norm=1.0         │
  │                                      │
  │  Loss =  1.0 × state_MSE            │
  │       +  1.0 × reward_MSE           │
  │       +  1.0 × done_BCE             │
  │       +  3.0 × stability_CE  ◄──    │
  │                                      │
  │  → checkpoints/world_model_final.pt  │
  └──────────────────────────────────────┘
                    │
                    ▼
  Stage 4: Quantize for G1 Deployment
  ┌──────────────────────────────────────┐
  │  INT8 dynamic quantization           │
  │  Encoder: 2.5 MB → 651 KB           │
  │  World Model: 9.2 MB → 6.8 MB       │
  │  → exported/encoder_int8.pt          │
  │  → exported/world_model_int8.pt      │
  └──────────────────────────────────────┘
```

---

## Physics Simulation: Real Bipedal Walking

The G1 walks using **pure physics** — no teleporting, no sliding. Each step is a discrete mechanical sequence:

```
  WALKING GAIT CYCLE (6 phases per stride)

  Phase 1: SHIFT WEIGHT LEFT        Phase 2: LIFT RIGHT LEG

      ┌──┐                              ┌──┐
      │  │                              │  │
     /    \                            /    \
    /      \                          /      \  ┌─┐
   │        │                        │        │ │ │ ← knee 0.25 rad
   │        │                        │        │ └─┘
   │        │                        │
  ─┴────────┴─                      ─┴──────────
   ◄── COM shifts                    Swing leg lifts off ground

  Phase 3: SWING FORWARD            Phase 4: SHIFT WEIGHT RIGHT

      ┌──┐                              ┌──┐
      │  │                              │  │
     /    \                            /    \
    /      \    ┌─┐                   /      \
   │        │   │ │                  │        │
   │        │   │ │ ← placed ahead  │        │
   │            │ │                  │        │
  ─┴────────────┴─┘                 ─┴────────┴─
   Stance ankle pushes                COM shifts ──►

  Repeat phases 5-6 for left leg...
```

**Key Parameters:**

| Parameter | Value | Purpose |
|:---|:---|:---|
| Stride frequency | 0.5 Hz | Slow, deliberate steps |
| Knee lift | 0.25 rad | Visible foot clearance |
| Hip reach | 0.06 rad | Forward step length |
| Stance push | 0.08 rad | Hip extension for propulsion |
| Ankle push | 0.08 rad | Plantarflexion thrust |
| Weight shift | 0.03 rad | Lateral COM transfer |
| Arm swing | 0.20 rad | Counter-balance |

### Debris Manipulation Sequence

```
  CROUCH          REACH           GRAB            STAND           TOSS

    ┌──┐           ┌──┐           ┌──┐            ┌──┐           ┌──┐
    │  │           │  │           │  │            │  │         ──/│  │\──
   /    \         /    \         /    \          /    \       /  /    \  \
  │      │     ──/      \──   ──/      \──     │      │     │ │      │ │
  │      │    /  │      │  \ /  │      │  \    │      │     │ │      │ │
   \    /    │   │      │   │   │  ██  │   │    │      │       │      │
    \  /     │    \    /    │    \ ████ /   │    │  ██  │       │      │
     \/           \/             \/             │ ████ │       │      │
  ───────     ───────       ───────        ─────┴──────┴──  ───┴──────┴───
  Knees bend   Arms reach   Fingers curl    Legs straighten   Arms spread,
  0.5 rad      forward      to grip         carrying object   fingers release
```

---

## Mission Planning Agent

The `RescueMissionAgent` generates sequential task plans by weighing **proximity + audio urgency**:

```
  Score(victim) = urgency_weight × ───────── + proximity_weight × ─────────
                                    dist + 0.1                     dist + 0.1

  ┌──────────────────────────────────────────────────┐
  │  MISSION PLAN (generated at runtime)             │
  │                                                  │
  │  Robot @ (0, 0)                                  │
  │                                                  │
  │  V1 @ (3.0, 0.1)  dist=3.0m  score=0.65  ◄──   │
  │  V3 @ (-2.0,-2.5) dist=3.2m  score=0.61         │
  │  V4 @ (-3.0, 3.0) dist=4.2m  score=0.47         │
  │  V2 @ (5.0, 3.0)  dist=5.8m  score=0.34         │
  │                                                  │
  │  Execution Order:                                │
  │  ┌──────────────────────────────────────────┐    │
  │  │  1. TURN_TO  V1  (bearing: +2°)         │    │
  │  │  2. WALK     V1  (3.0m)                 │    │
  │  │  3. GRAB     debris_a                   │    │
  │  │  4. SETTLE   (recover balance)          │    │
  │  │  5. GRAB     debris_b                   │    │
  │  │  6. SETTLE   (recover balance)          │    │
  │  │  7. RESCUE   V1  ✓                      │    │
  │  │  8. TURN_TO  V3  (bearing: -135°)       │    │
  │  │  9. WALK     V3  (5.5m)                 │    │
  │  │  10. GRAB    debris_a ...               │    │
  │  │  ... (replans after each rescue)        │    │
  │  └──────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────┘
```

### Audio Detection System

```
  Distance to victim    Audio urgency         Console output
  ─────────────────────────────────────────────────────────
  < 1.5m               CRITICAL              "I'm right here!"
  < 3.0m               HIGH                  "Help! Can you hear me?"
  < 5.0m               MODERATE              "Is anyone there?"
  > 5.0m               (not detected)         —
```

---

## Reward Structure

```
  ┌─────────────────────────────────────────────────────────┐
  │                    REWARD SIGNALS                        │
  │                                                         │
  │  POSITIVE                     │  NEGATIVE               │
  │  ─────────                    │  ─────────              │
  │  +25.0  Victim rescued ★★★   │  -30.0  IMMINENT        │
  │  +10.0  Victim reached ★★    │         COLLAPSE        │
  │  +5.0   New zone mapped ★    │  -20.0  Structural      │
  │                               │         collapse        │
  │                               │  -15.0  Robot fell      │
  │                               │  -5.0   Unstable zone   │
  │                               │  -5.0   Stability=1     │
  │                               │  -0.1   Per timestep    │
  │                               │         (time pressure) │
  └─────────────────────────────────────────────────────────┘
```

---

## Running the System

### Prerequisites

```bash
pip install torch torchvision mujoco gymnasium h5py numpy tqdm transformers peft
```

### Live Demo (MuJoCo Viewer)

```bash
# Autonomous rescue mission with physics-based walking
mjpython run_viewer.py
```

The robot will:
1. Plan a route to the nearest/loudest victim
2. Walk using discrete bipedal steps (one leg at a time)
3. Navigate around collapsed walls
4. Crouch down, grab debris covering a victim
5. Stand up and toss the debris aside
6. Rescue the victim
7. Replan and navigate to the next victim

### Full Pipeline Test (World Model + Planning)

```bash
python3 test_pipeline.py
```

Runs the complete Fusion → StateEncoder → WorldModel → Planner → SafetyMonitor pipeline with real trained checkpoints and console output showing every decision in real-time.

---

## Model Checkpoints

```
checkpoints/
├── encoder_best.pt          657K params   2.5 MB   StateEncoder
├── decoder_best.pt          1.0B params   3.8 GB   StateDecoder (training only)
└── world_model_final.pt     2.4M params   9.2 MB   SOSWorldModel (GRU)

exported/
├── encoder_int8.pt                        651 KB   INT8 quantized for G1
└── world_model_int8.pt                    6.8 MB   INT8 quantized for G1

voice_checkpoints/
├── merged_model/                                   SmolLM2-135M merged
└── lora_adapter/                                   LoRA adapter (r=16, α=32)
```

**Full inference cycle: <200ms on CPU** (8 actions × 5 steps per planning cycle)

---

## Project Structure

```
sos-answered/
├── CLAUDE.md                  # Build instructions & architecture spec
├── README.md                  # This file
├── config.py                  # All hyperparameters
├── requirements.txt           # Dependencies
│
├── models/
│   ├── state_encoder.py       # StateEncoder + StateDecoder
│   └── world_model.py         # SOSWorldModel (GRU transition)
│
├── planning/
│   ├── planner.py             # WorldModelPlanner (5-step lookahead)
│   └── safety_monitor.py      # Hard safety rules + IMU thresholds
│
├── sim/
│   ├── scene.xml              # MuJoCo rubble rescue environment
│   ├── env.py                 # Gymnasium env + G1WalkController
│   ├── agent.py               # RescueMissionAgent (task planner)
│   ├── vla.py                 # RescueVLA (raycast perception)
│   ├── sensors.py             # IMU buffer + camera renderer
│   └── rewards.py             # Reward function
│
├── training/
│   ├── collect_experience.py  # Sim rollout data collection
│   ├── train_encoder.py       # Autoencoder training
│   └── train_world_model.py   # World model training
│
├── inference/
│   └── sos_runtime.py         # Full online inference loop
│
├── export/
│   └── export.py              # INT8 quantization for G1
│
├── cadenza/                   # Locomotion library (Go1/Go2/G1)
│   ├── locomotion/            # Gait engines, balance, kinematics
│   ├── vla_steer/             # FAISS-based VLA steering
│   ├── actions/               # Action primitive library
│   └── sim.py                 # MuJoCo simulation interface
│
├── test_pipeline.py           # End-to-end pipeline test
├── test_gym_render.py         # Environment render tests
└── run_viewer.py              # Autonomous rescue demo
```

---

## Impact Potential

**Search and rescue** is a $130B+ global industry growing at 8% annually. Current robotic solutions are wheeled/tracked — they can't navigate the rubble, stairs, and confined spaces that humanoids can.

```
  CURRENT APPROACH                    SOS-ANSWERED
  ────────────────                    ────────────
  Human rescuers enter first          Robot enters first
  Risk of secondary collapse          Robot assesses stability
  Limited by darkness, dust           RGB + Depth + Audio sensing
  Fatigue after hours                 Operates continuously
  Communication delays                Real-time data to command
  Reactive decisions                  World model predicts futures
```

**Long-term vision:**
- Deploy fleet of G1 robots immediately after disaster
- Robots map collapse zone, locate all victims, assess structural stability
- Human teams receive exact victim locations + safe entry routes
- 10x faster victim location in the critical first 72 hours

---

## Creativity and Originality

| Innovation | Why It's Novel |
|:---|:---|
| **World model for rescue** | First system to learn a transition model specifically for disaster environments — predicting structural collapse, victim proximity, and action outcomes |
| **Multimodal fusion for victim detection** | Combines vision, depth, audio (help calls), and IMU (structural vibration) in a single learned embedding |
| **SLM with RL for rescue context** | Fine-tuned SmolLM2-135M with LoRA to understand rescue-specific audio and generate action recommendations — not a generic chatbot but a specialized rescue agent |
| **Physics-based humanoid walking** | Discrete step-by-step gait with stance/swing gain switching — the robot lifts one leg at a time, transfers weight, and plants it ahead |
| **Safety-first planning** | Stability prediction weighted 3× in training loss — the robot refuses to enter zones it predicts will collapse, even if a victim is there |
| **Crouch-grab-throw manipulation** | Full kinematic chain: squat → reach → grip fingers → stand → toss debris — using all 43 actuators of the G1 |

---

## Technical Specifications Summary

```
┌────────────────────────────────────────────────────────┐
│  ROBOT         Unitree G1 Humanoid, 35 kg, 0.75m tall │
│  ACTUATORS     43 (12 legs + 3 waist + 28 arms/hands) │
│  SENSORS       RGB 224×224, Depth 224×224, IMU 9-axis  │
│  WORLD MODEL   GRU-based, 2.4M params, 5-step horizon │
│  SLM           SmolLM2-135M + LoRA (r=16, α=32)       │
│  FUSION        3-modal → [492, 2048] → [512] encoded  │
│  PLANNING      8 actions × 5 steps, <200ms on CPU     │
│  DEPLOYMENT    INT8 quantized (encoder 651KB + WM 6.8MB│
│  WALKING       Pure physics, 0.5 Hz stride, 0.013 m/s │
│  SAFETY        3-class stability, 4 safety states      │
└────────────────────────────────────────────────────────┘
```

---

*Built for the hackathon. Designed for the real world. Because every minute matters when someone is trapped.*
