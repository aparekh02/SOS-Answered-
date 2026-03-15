"""AudioEncoder — lightweight CNN that maps mel spectrograms to [256] embeddings."""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_EMBED_DIM, MEL_N_MELS


class AudioEncoder(nn.Module):
    """Small CNN for audio feature extraction from log-mel spectrograms.

    Architecture designed for INT8 quantization and edge deployment on G1.
    Input:  [batch, 1, 128, T]  (log-mel spectrogram)
    Output: [batch, 256]        (audio embedding)
    """

    def __init__(self, embed_dim: int = AUDIO_EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: [1, 128, T] → [32, 64, T//2]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2: [32, 64, T//2] → [64, 32, T//4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3: [64, 32, T//4] → [128, 16, T//8]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 4: [128, 16, T//8] → [256, 8, T//16]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [256, 1, 1]
        )
        self.fc = nn.Linear(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [batch, 1, n_mels, time_frames] → [batch, embed_dim]"""
        x = self.conv(mel)           # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)   # [batch, 256]
        return self.norm(self.fc(x)) # [batch, embed_dim]


if __name__ == "__main__":
    from config import AUDIO_WINDOW_SAMPLES, MEL_HOP_LENGTH

    device = torch.device("cpu")
    encoder = AudioEncoder().to(device)

    # Simulate mel spectrogram input
    time_frames = AUDIO_WINDOW_SAMPLES // MEL_HOP_LENGTH + 1  # ~126
    mel = torch.randn(4, 1, MEL_N_MELS, time_frames, device=device)
    print(f"Input mel:  {list(mel.shape)}")

    out = encoder(mel)
    print(f"Output:     {list(out.shape)} — expected [4, {AUDIO_EMBED_DIM}]")
    assert out.shape == (4, AUDIO_EMBED_DIM)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    print("AudioEncoder OK")
