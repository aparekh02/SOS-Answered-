"""Train StateEncoder + StateDecoder as autoencoder on experience buffer data."""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AE_LR, AE_BATCH_SIZE, AE_EPOCHS, FUSION_SEQ_LEN, FUSION_DIM
from models.state_encoder import StateEncoder, StateDecoder
from data.experience_buffer import ExperienceBuffer


def train(buffer_path: str, epochs: int = AE_EPOCHS, batch_size: int = AE_BATCH_SIZE,
          lr: float = AE_LR, save_dir: str = "checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = StateEncoder().to(device)
    decoder = StateDecoder().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    buf = ExperienceBuffer(buffer_path)
    print(f"Buffer size: {len(buf)}")

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()

        batch = buf.sample(min(batch_size, len(buf)))
        s = torch.tensor(batch["s"], dtype=torch.float32, device=device)

        # Forward
        z = encoder(s)           # [batch, 512]
        recon = decoder(z)       # [batch, 492, 2048]

        loss = criterion(recon, s)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} — MSE: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder_best.pt"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder_best.pt"))

    print(f"Training complete. Best MSE: {best_loss:.6f}")
    print(f"Checkpoints saved to {save_dir}/")

    return encoder, decoder, best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default="experience_data/buffer.h5")
    parser.add_argument("--epochs", type=int, default=AE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=AE_BATCH_SIZE)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args.buffer, args.epochs, args.batch_size, save_dir=args.save_dir)
