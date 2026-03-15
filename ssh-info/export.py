"""Export quantized StateEncoder + SOSWorldModel for G1 deployment (INT8)."""

import os
import sys
import torch
import torch.quantization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FUSION_SEQ_LEN, FUSION_DIM, STATE_DIM, NUM_ACTIONS, GRU_HIDDEN_DIM
from models.state_encoder import StateEncoder
from models.world_model import SOSWorldModel

# Set quantization engine (qnnpack works on ARM/mobile, fbgemm on x86)
torch.backends.quantized.engine = "qnnpack"


def quantize_encoder(encoder: StateEncoder, save_path: str):
    """Quantize StateEncoder to INT8 using dynamic quantization."""
    encoder.eval()
    quantized = torch.quantization.quantize_dynamic(
        encoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized.state_dict(), save_path)
    print(f"Quantized encoder saved to {save_path}")

    # Verify
    x = torch.randn(1, FUSION_SEQ_LEN, FUSION_DIM)
    with torch.no_grad():
        z_orig = encoder(x)
        z_quant = quantized(x)
    diff = (z_orig - z_quant).abs().mean().item()
    print(f"  Quantization diff (mean abs): {diff:.6f}")
    print(f"  Output shape: {z_quant.shape}")

    return quantized


def quantize_world_model(wm: SOSWorldModel, save_path: str):
    """Quantize SOSWorldModel to INT8 using dynamic quantization."""
    wm.eval()
    quantized = torch.quantization.quantize_dynamic(
        wm, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized.state_dict(), save_path)
    print(f"Quantized world model saved to {save_path}")

    # Verify
    state = torch.randn(1, STATE_DIM)
    action = torch.tensor([0])
    h = torch.zeros(1, GRU_HIDDEN_DIM)
    with torch.no_grad():
        ns_orig, r_orig, d_orig, st_orig, h_orig = wm(state, action, h)
        ns_quant, r_quant, d_quant, st_quant, h_quant = quantized(state, action, h)
    diff = (ns_orig - ns_quant).abs().mean().item()
    print(f"  Quantization diff (mean abs next_state): {diff:.6f}")
    print(f"  next_state shape: {ns_quant.shape}")
    print(f"  reward shape: {r_quant.shape}")
    print(f"  done shape: {d_quant.shape}")
    print(f"  stability shape: {st_quant.shape}")

    return quantized


def export(encoder_path: str = "checkpoints/encoder_best.pt",
           wm_path: str = "checkpoints/world_model_final.pt",
           output_dir: str = "exported"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load models
    encoder = StateEncoder().to(device)
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        print(f"Loaded encoder from {encoder_path}")
    else:
        print(f"WARNING: {encoder_path} not found, using random weights")

    wm = SOSWorldModel().to(device)
    if os.path.exists(wm_path):
        wm.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
        print(f"Loaded world model from {wm_path}")
    else:
        print(f"WARNING: {wm_path} not found, using random weights")

    # Quantize
    print("\n--- Quantizing StateEncoder ---")
    quantize_encoder(encoder, os.path.join(output_dir, "encoder_int8.pt"))

    print("\n--- Quantizing SOSWorldModel ---")
    quantize_world_model(wm, os.path.join(output_dir, "world_model_int8.pt"))

    print(f"\nExport complete. Files in {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="checkpoints/encoder_best.pt")
    parser.add_argument("--world-model", type=str, default="checkpoints/world_model_final.pt")
    parser.add_argument("--output-dir", type=str, default="exported")
    args = parser.parse_args()

    export(args.encoder, args.world_model, args.output_dir)
