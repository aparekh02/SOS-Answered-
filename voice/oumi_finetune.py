"""Oumi fine-tuning pipeline for the voice agent.

Fine-tunes SmolLM2-135M via LoRA to classify audio events in rescue scenarios.
The model learns to map audio feature descriptions to rescue-relevant classifications:
    - HELP_PLEA:          human voice, screaming, crying, tapping patterns
    - ELECTRICAL_HISS:    electrical arcing, buzzing, sparking
    - WATER_PIPE:         pressurized water leak, hissing, dripping
    - STRUCTURAL_STRESS:  creaking, cracking, grinding of load-bearing elements
    - SILENCE:            no significant audio
    - AMBIENT:            wind, distant traffic, non-critical background

Usage:
    # Generate training data
    python voice/oumi_finetune.py --generate-data --output voice_data/

    # Fine-tune on A100 (VESSL AI)
    python voice/oumi_finetune.py --train --data voice_data/ --output voice_checkpoints/

    # Export fine-tuned model
    python voice/oumi_finetune.py --export --checkpoint voice_checkpoints/
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUDIO_CLASS_NAMES, OUMI_BASE_MODEL, OUMI_LORA_R, OUMI_LORA_ALPHA,
    OUMI_LORA_LR, OUMI_FINETUNE_EPOCHS, OUMI_FINETUNE_BATCH_SIZE,
)


# ── Training data generation ────────────────────────────────────────

AUDIO_DESCRIPTIONS = {
    "HELP_PLEA": [
        "Faint human voice shouting 'help' repeatedly from beneath rubble",
        "Rhythmic tapping pattern on metal pipe — SOS morse code",
        "Weak crying and moaning sounds from behind collapsed wall",
        "Muffled screaming from approximately 3 meters below debris",
        "Child's voice calling for parents, intermittent, very weak",
        "Adult male voice, strained, repeating 'here' every few seconds",
        "Tapping on concrete — three short, three long, three short (SOS)",
        "Whispered plea for water, barely audible through dust",
        "Multiple voices overlapping, distressed, from underneath structure",
        "Banging on metal surface in regular intervals — deliberate signal",
    ],
    "ELECTRICAL_HISS": [
        "Continuous high-frequency buzzing from exposed wire bundle",
        "Intermittent sparking and crackling from damaged junction box",
        "Loud electrical arcing — sounds like welding, very dangerous",
        "Low hum with occasional snap from severed power line",
        "Buzzing intensifies when robot approaches east wall section",
        "Transformer hum — steady 60Hz tone, moderate intensity",
        "Electrical popping sounds from behind drywall, smell of ozone",
        "High voltage arc flash sound — rapid crackling bursts",
    ],
    "WATER_PIPE": [
        "Pressurized water spraying from burst pipe — loud hissing",
        "Steady dripping onto concrete floor — water accumulating",
        "Gurgling sound from broken pipe inside wall cavity",
        "High-pressure gas or water leak — sustained hiss from ceiling",
        "Water flowing rapidly through debris — potential flooding risk",
        "Intermittent spray from damaged sprinkler head",
        "Steam hissing from hot water pipe breach",
        "Rushing water sound increasing in volume — pipe failure progressing",
    ],
    "STRUCTURAL_STRESS": [
        "Deep creaking from load-bearing beam overhead — imminent failure",
        "Concrete cracking sounds — sharp snapping from support column",
        "Metal groaning under stress — steel beam deforming",
        "Settling sounds — debris shifting and compacting slowly",
        "Loud crack followed by dust fall — partial collapse nearby",
        "Rhythmic creaking in floor joists — structural fatigue",
        "Grinding sound from displaced foundation elements",
        "Progressive cracking — frequency increasing, collapse likely",
    ],
    "SILENCE": [
        "No significant audio detected — ambient noise below threshold",
        "Near-complete silence — dust settling, no active sounds",
        "Very quiet — only robot's own servo noise audible",
        "Dead silence in sealed room — no external sound penetration",
    ],
    "AMBIENT": [
        "Wind whistling through broken windows — moderate gusts",
        "Distant sirens and emergency vehicle activity",
        "Rain hitting exposed surfaces — steady precipitation",
        "Helicopter overhead — rescue team aerial survey",
        "General urban background noise — traffic, distant construction",
        "Bird sounds from outside damaged structure",
        "Crowd noise from rescue staging area outside",
    ],
}

SYSTEM_PROMPT = """You are a rescue robot audio classifier. Analyze the audio description and classify it into exactly one category. Respond with only the category name.

Categories:
- HELP_PLEA: Human voices calling for help, tapping signals, distress sounds
- ELECTRICAL_HISS: Electrical arcing, buzzing, sparking from damaged wiring
- WATER_PIPE: Pressurized water leaks, hissing, dripping from broken pipes
- STRUCTURAL_STRESS: Creaking, cracking, grinding from load-bearing elements
- SILENCE: No significant audio detected
- AMBIENT: Background noise, weather, distant activity"""


def generate_training_data(output_dir: str):
    """Generate JSONL training data for Oumi SFT fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)

    train_samples = []
    val_samples = []

    for class_name, descriptions in AUDIO_DESCRIPTIONS.items():
        for i, desc in enumerate(descriptions):
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this audio: {desc}"},
                    {"role": "assistant", "content": class_name},
                ]
            }
            # 80/20 train/val split
            if i % 5 == 0:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    # Write JSONL
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(train_samples)} train + {len(val_samples)} val samples")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    return train_path, val_path


def generate_oumi_config(data_dir: str, output_dir: str, config_path: str = None):
    """Generate Oumi YAML training config for LoRA fine-tuning."""
    if config_path is None:
        config_path = os.path.join(output_dir, "oumi_train_config.yaml")

    os.makedirs(output_dir, exist_ok=True)

    config = f"""# Oumi LoRA fine-tuning config for SOS voice classifier
# Run: oumi train -c {config_path}

model:
  model_name: "{OUMI_BASE_MODEL}"
  dtype: "bfloat16"
  trust_remote_code: true

data:
  train:
    datasets:
      - dataset_name: "json"
        dataset_path: "{os.path.abspath(data_dir)}/train.jsonl"
  val:
    datasets:
      - dataset_name: "json"
        dataset_path: "{os.path.abspath(data_dir)}/val.jsonl"

training:
  output_dir: "{os.path.abspath(output_dir)}"
  num_train_epochs: {OUMI_FINETUNE_EPOCHS}
  per_device_train_batch_size: {OUMI_FINETUNE_BATCH_SIZE}
  learning_rate: {OUMI_LORA_LR}
  warmup_ratio: 0.1
  weight_decay: 0.01
  logging_steps: 5
  save_strategy: "epoch"
  eval_strategy: "epoch"
  use_peft: true
  gradient_checkpointing: true

peft:
  peft_type: "lora"
  lora_r: {OUMI_LORA_R}
  lora_alpha: {OUMI_LORA_ALPHA}
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "o_proj"
    - "k_proj"
"""

    with open(config_path, "w") as f:
        f.write(config)

    print(f"Oumi config written to {config_path}")
    return config_path


def generate_vessl_run_script(data_dir: str, output_dir: str):
    """Generate a shell script to run fine-tuning on VESSL AI A100."""
    script_path = os.path.join(output_dir, "run_finetune_vessl.sh")
    os.makedirs(output_dir, exist_ok=True)

    script = f"""#!/bin/bash
set -e

# VESSL AI A100 fine-tuning script for SOS voice classifier
source /opt/conda/etc/profile.d/conda.sh
conda activate base

echo "=== Installing Oumi ==="
pip install oumi

echo "=== GPU Check ==="
python -c "import torch; print(f'CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0)}}')"

echo "=== Starting LoRA Fine-tuning ==="
cd /root/sos-answered

# Generate training data
python voice/oumi_finetune.py --generate-data --output voice_data/

# Run Oumi fine-tuning
oumi train -c voice_checkpoints/oumi_train_config.yaml

echo "=== Fine-tuning Complete ==="
ls -lh voice_checkpoints/
"""

    with open(script_path, "w") as f:
        f.write(script)

    os.chmod(script_path, 0o755)
    print(f"VESSL run script: {script_path}")
    return script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oumi voice agent fine-tuning")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate training data JSONL files")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate Oumi YAML training config")
    parser.add_argument("--generate-vessl-script", action="store_true",
                        help="Generate VESSL AI run script")
    parser.add_argument("--output", type=str, default="voice_data/",
                        help="Output directory for data")
    parser.add_argument("--checkpoint-dir", type=str, default="voice_checkpoints/",
                        help="Output directory for checkpoints and config")
    args = parser.parse_args()

    if args.generate_data:
        generate_training_data(args.output)

    if args.generate_config:
        generate_oumi_config(args.output, args.checkpoint_dir)

    if args.generate_vessl_script:
        generate_vessl_run_script(args.output, args.checkpoint_dir)

    if not (args.generate_data or args.generate_config or args.generate_vessl_script):
        # Default: generate everything
        train_path, val_path = generate_training_data(args.output)
        config_path = generate_oumi_config(args.output, args.checkpoint_dir)
        script_path = generate_vessl_run_script(args.output, args.checkpoint_dir)
        print(f"\nAll artifacts generated. To fine-tune on VESSL AI:")
        print(f"  1. Upload project to VESSL")
        print(f"  2. Run: bash {script_path}")
