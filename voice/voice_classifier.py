"""VoiceClassifier — classifies audio into rescue-relevant categories with severity scoring."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUDIO_EMBED_DIM, NUM_AUDIO_CLASSES, AUDIO_CLASS_NAMES,
    AUDIO_HELP_PLEA, AUDIO_ELECTRICAL_HISS, AUDIO_WATER_PIPE,
    AUDIO_STRUCTURAL_STRESS,
)


class VoiceClassifier(nn.Module):
    """Classifies audio embeddings into 6 rescue-relevant categories.

    Classes:
        0: HELP_PLEA           — human voice calling for help
        1: ELECTRICAL_HISS     — live wire / electrical arcing
        2: WATER_PIPE          — pressurized water leak
        3: STRUCTURAL_STRESS   — creaking, cracking of load-bearing elements
        4: SILENCE             — no significant audio
        5: AMBIENT             — background noise, non-critical

    Also outputs a severity score [0-1] indicating danger level.
    """

    def __init__(self, embed_dim: int = AUDIO_EMBED_DIM,
                 num_classes: int = NUM_AUDIO_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio_embedding: torch.Tensor) -> dict:
        """
        audio_embedding: [batch, 256]
        Returns dict with:
            logits:     [batch, 6]   — class logits
            probs:      [batch, 6]   — class probabilities
            class_id:   [batch]      — predicted class index
            class_name: list[str]    — predicted class names
            severity:   [batch]      — danger severity score [0-1]
        """
        logits = self.classifier(audio_embedding)       # [batch, 6]
        probs = F.softmax(logits, dim=-1)               # [batch, 6]
        class_id = logits.argmax(dim=-1)                # [batch]
        severity = self.severity_head(audio_embedding).squeeze(-1)  # [batch]

        class_names = [AUDIO_CLASS_NAMES[i] for i in class_id.tolist()]

        return {
            "logits": logits,
            "probs": probs,
            "class_id": class_id,
            "class_name": class_names,
            "severity": severity,
        }

    def is_critical(self, result: dict) -> bool:
        """Check if detected audio indicates immediate danger."""
        critical_classes = {AUDIO_HELP_PLEA, AUDIO_ELECTRICAL_HISS,
                           AUDIO_WATER_PIPE, AUDIO_STRUCTURAL_STRESS}
        return any(c.item() in critical_classes for c in result["class_id"])


if __name__ == "__main__":
    device = torch.device("cpu")
    classifier = VoiceClassifier().to(device)

    # Test single sample
    emb = torch.randn(1, AUDIO_EMBED_DIM, device=device)
    result = classifier(emb)
    print(f"logits:     {list(result['logits'].shape)}")
    print(f"probs:      {list(result['probs'].shape)}")
    print(f"class_id:   {result['class_id'].item()}")
    print(f"class_name: {result['class_name']}")
    print(f"severity:   {result['severity'].item():.4f}")

    # Test batch
    batch_emb = torch.randn(8, AUDIO_EMBED_DIM, device=device)
    batch_result = classifier(batch_emb)
    print(f"\nBatch logits:  {list(batch_result['logits'].shape)}")
    print(f"Batch classes: {batch_result['class_name']}")
    print(f"Batch severity: {batch_result['severity'].tolist()}")
    assert batch_result["logits"].shape == (8, NUM_AUDIO_CLASSES)

    # Test is_critical
    print(f"\nis_critical: {classifier.is_critical(result)}")

    params = sum(p.numel() for p in classifier.parameters())
    print(f"Parameters: {params:,}")
    print("VoiceClassifier OK")
