"""MultimodalFusion — replaces MockFusion with real sensor encoders.

Fuses three streams:
    Audio embedding [256] + Depth embedding [256] + IMU embedding [256]
    → concat [768] → project to [492, 2048] + stability logits [3]

This is the real Fusion model that the world model pipeline expects.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FUSION_SEQ_LEN, FUSION_DIM, MULTIMODAL_EMBED_DIM,
    FUSION_PROJECT_HIDDEN, AUDIO_EMBED_DIM, DEPTH_EMBED_DIM, IMU_EMBED_DIM,
    AUDIO_WINDOW_SAMPLES, MEL_N_MELS, IMG_SIZE, IMU_SEQ_LEN, IMU_CHANNELS,
)
from voice.audio_processor import mel_spectrogram
from voice.audio_encoder import AudioEncoder
from voice.voice_classifier import VoiceClassifier
from sensors.depth_encoder import DepthEncoder
from sensors.imu_encoder import IMUEncoder


class MultimodalFusion(nn.Module):
    """Fuses audio + depth + IMU into the [492, 2048] embedding expected by StateEncoder.

    Online pipeline:
        Audio waveform → mel spectrogram → AudioEncoder → [256]
        Depth frame    → DepthEncoder → [256]
        IMU buffer     → IMUEncoder   → [256]
        Concat [768]   → project to [492, 2048]
        Also predicts stability logits [3] from fused representation.
    """

    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.depth_encoder = DepthEncoder()
        self.imu_encoder = IMUEncoder()
        self.voice_classifier = VoiceClassifier()

        # Project fused [768] → [492 * 2048]
        self.projector = nn.Sequential(
            nn.Linear(MULTIMODAL_EMBED_DIM, FUSION_PROJECT_HIDDEN),
            nn.GELU(),
            nn.Linear(FUSION_PROJECT_HIDDEN, FUSION_SEQ_LEN * FUSION_DIM),
        )

        # Stability head from fused embedding
        self.stability_head = nn.Sequential(
            nn.Linear(MULTIMODAL_EMBED_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, audio_mel: torch.Tensor, depth: torch.Tensor,
                imu: torch.Tensor) -> dict:
        """
        audio_mel: [batch, 1, 128, T]   log-mel spectrogram
        depth:     [batch, 1, 224, 224]  single-channel depth
        imu:       [batch, 100, 9]       IMU buffer

        Returns dict with:
            fused:            [batch, 492, 2048]  fusion embedding for StateEncoder
            stability_logits: [batch, 3]          stability class prediction
            audio_embedding:  [batch, 256]        raw audio features
            depth_embedding:  [batch, 256]        raw depth features
            imu_embedding:    [batch, 256]        raw IMU features
            voice_result:     dict                voice classification output
        """
        # Encode each modality
        audio_emb = self.audio_encoder(audio_mel)    # [batch, 256]
        depth_emb = self.depth_encoder(depth)        # [batch, 256]
        imu_emb = self.imu_encoder(imu)              # [batch, 256]

        # Voice classification (runs on audio embedding)
        voice_result = self.voice_classifier(audio_emb)

        # Fuse
        fused_emb = torch.cat([audio_emb, depth_emb, imu_emb], dim=-1)  # [batch, 768]

        # Project to world model input shape
        projected = self.projector(fused_emb)        # [batch, 492*2048]
        fused = projected.view(-1, FUSION_SEQ_LEN, FUSION_DIM)  # [batch, 492, 2048]

        # Stability prediction
        stability_logits = self.stability_head(fused_emb)  # [batch, 3]

        return {
            "fused": fused,
            "stability_logits": stability_logits,
            "audio_embedding": audio_emb,
            "depth_embedding": depth_emb,
            "imu_embedding": imu_emb,
            "voice_result": voice_result,
        }

    def forward_legacy(self, rgb, depth_3ch, imu):
        """Legacy interface matching MockFusion signature for backward compatibility.

        rgb:   [batch, 3, 224, 224]  — ignored (replaced by audio)
        depth_3ch: [batch, 3, 224, 224] — uses first channel only
        imu:   [batch, 100, 9]

        Returns: fused [batch, 492, 2048], stability_logits [batch, 3]
        """
        batch = depth_3ch.shape[0]
        device = depth_3ch.device

        # Use first channel of depth
        depth_1ch = depth_3ch[:, 0:1, :, :]  # [batch, 1, 224, 224]

        # Generate silence mel as placeholder when no audio available
        silence = torch.zeros(batch, AUDIO_WINDOW_SAMPLES, device=device)
        audio_mel = mel_spectrogram(silence)

        result = self.forward(audio_mel, depth_1ch, imu)
        return result["fused"], result["stability_logits"]


if __name__ == "__main__":
    device = torch.device("cpu")
    fusion = MultimodalFusion().to(device)

    batch = 2
    audio_mel = torch.randn(batch, 1, MEL_N_MELS, 126, device=device)
    depth = torch.randn(batch, 1, IMG_SIZE, IMG_SIZE, device=device)
    imu = torch.randn(batch, IMU_SEQ_LEN, IMU_CHANNELS, device=device)

    print(f"Inputs:")
    print(f"  audio_mel: {list(audio_mel.shape)}")
    print(f"  depth:     {list(depth.shape)}")
    print(f"  imu:       {list(imu.shape)}")

    result = fusion(audio_mel, depth, imu)

    print(f"\nOutputs:")
    print(f"  fused:            {list(result['fused'].shape)} — expected [{batch}, {FUSION_SEQ_LEN}, {FUSION_DIM}]")
    print(f"  stability_logits: {list(result['stability_logits'].shape)} — expected [{batch}, 3]")
    print(f"  audio_embedding:  {list(result['audio_embedding'].shape)} — expected [{batch}, {AUDIO_EMBED_DIM}]")
    print(f"  depth_embedding:  {list(result['depth_embedding'].shape)} — expected [{batch}, {DEPTH_EMBED_DIM}]")
    print(f"  imu_embedding:    {list(result['imu_embedding'].shape)} — expected [{batch}, {IMU_EMBED_DIM}]")
    print(f"  voice class:      {result['voice_result']['class_name']}")
    print(f"  voice severity:   {result['voice_result']['severity'].tolist()}")

    assert result["fused"].shape == (batch, FUSION_SEQ_LEN, FUSION_DIM)
    assert result["stability_logits"].shape == (batch, 3)

    # Test legacy interface
    print(f"\nLegacy interface test:")
    rgb = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    depth_3ch = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    fused, stab = fusion.forward_legacy(rgb, depth_3ch, imu)
    print(f"  fused: {list(fused.shape)}, stability: {list(stab.shape)}")
    assert fused.shape == (batch, FUSION_SEQ_LEN, FUSION_DIM)

    params = sum(p.numel() for p in fusion.parameters())
    print(f"\nTotal parameters: {params:,} ({params/1e6:.1f}M)")
    print("MultimodalFusion OK")
