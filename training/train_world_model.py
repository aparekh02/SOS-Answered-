"""Train SOSWorldModel supervised on experience buffer (with frozen StateEncoder)."""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WM_LR, WM_BATCH_SIZE, WM_EPOCHS, WM_GRAD_CLIP,
    LOSS_WEIGHT_STATE, LOSS_WEIGHT_REWARD, LOSS_WEIGHT_DONE, LOSS_WEIGHT_STABILITY,
)
from models.state_encoder import StateEncoder
from models.world_model import SOSWorldModel
from data.experience_buffer import ExperienceBuffer


def train(buffer_path: str, encoder_path: str, epochs: int = WM_EPOCHS,
          batch_size: int = WM_BATCH_SIZE, lr: float = WM_LR,
          save_dir: str = "checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen encoder
    encoder = StateEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("StateEncoder loaded and frozen")

    # World model
    wm = SOSWorldModel().to(device)
    optimizer = torch.optim.Adam(wm.parameters(), lr=lr)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    buf = ExperienceBuffer(buffer_path)
    print(f"Buffer size: {len(buf)}")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        wm.train()

        batch = buf.sample(min(batch_size, len(buf)))

        s_raw = torch.tensor(batch["s"], dtype=torch.float32, device=device)
        s1_raw = torch.tensor(batch["s1"], dtype=torch.float32, device=device)
        actions = torch.tensor(batch["a"], dtype=torch.long, device=device)
        rewards = torch.tensor(batch["r"], dtype=torch.float32, device=device)
        dones = torch.tensor(batch["d"], dtype=torch.float32, device=device)
        stabs = torch.tensor(batch["stab"], dtype=torch.long, device=device)

        # Encode states
        with torch.no_grad():
            s_enc = encoder(s_raw)       # [batch, 512]
            s1_enc = encoder(s1_raw)     # [batch, 512]

        h = wm.init_hidden(s_enc.shape[0], device)

        # Forward
        pred_next, pred_reward, pred_done, pred_stab, _ = wm(s_enc, actions, h)

        # Losses
        loss_state = mse_loss(pred_next, s1_enc) * LOSS_WEIGHT_STATE
        loss_reward = mse_loss(pred_reward, rewards) * LOSS_WEIGHT_REWARD
        loss_done = bce_loss(pred_done, dones) * LOSS_WEIGHT_DONE
        loss_stability = ce_loss(pred_stab, stabs) * LOSS_WEIGHT_STABILITY

        total_loss = loss_state + loss_reward + loss_done + loss_stability

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), WM_GRAD_CLIP)
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} — total: {total_loss.item():.4f} "
                  f"(state: {loss_state.item():.4f}, reward: {loss_reward.item():.4f}, "
                  f"done: {loss_done.item():.4f}, stab: {loss_stability.item():.4f})")

        if epoch % 10 == 0:
            torch.save(wm.state_dict(), os.path.join(save_dir, f"world_model_epoch{epoch}.pt"))

    torch.save(wm.state_dict(), os.path.join(save_dir, "world_model_final.pt"))
    print(f"Training complete. Saved to {save_dir}/")

    return wm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default="experience_data/buffer.h5")
    parser.add_argument("--encoder", type=str, default="checkpoints/encoder_best.pt")
    parser.add_argument("--epochs", type=int, default=WM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=WM_BATCH_SIZE)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args.buffer, args.encoder, args.epochs, args.batch_size, save_dir=args.save_dir)
