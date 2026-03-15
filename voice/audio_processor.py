"""Audio preprocessing — ring buffer + mel spectrogram extraction."""

import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUDIO_SAMPLE_RATE, AUDIO_WINDOW_SAMPLES, MEL_N_MELS,
    MEL_N_FFT, MEL_HOP_LENGTH,
)


class AudioBuffer:
    """Ring buffer that accumulates raw 16kHz audio into 2-second windows."""

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE,
                 window_samples: int = AUDIO_WINDOW_SAMPLES):
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.buffer = np.zeros(window_samples, dtype=np.float32)
        self._write_idx = 0

    def push(self, samples: np.ndarray):
        """Push new audio samples into the ring buffer."""
        n = len(samples)
        if n >= self.window_samples:
            self.buffer[:] = samples[-self.window_samples:]
            self._write_idx = 0
        else:
            end = self._write_idx + n
            if end <= self.window_samples:
                self.buffer[self._write_idx:end] = samples
            else:
                first = self.window_samples - self._write_idx
                self.buffer[self._write_idx:] = samples[:first]
                self.buffer[:n - first] = samples[first:]
            self._write_idx = end % self.window_samples

    def get_window(self) -> np.ndarray:
        """Return the current 2-second window in temporal order."""
        return np.roll(self.buffer, -self._write_idx).copy()


def mel_spectrogram(waveform: torch.Tensor, sample_rate: int = AUDIO_SAMPLE_RATE,
                    n_mels: int = MEL_N_MELS, n_fft: int = MEL_N_FFT,
                    hop_length: int = MEL_HOP_LENGTH) -> torch.Tensor:
    """Compute log-mel spectrogram from raw waveform.

    Args:
        waveform: [samples] or [batch, samples] float32 tensor
    Returns:
        [batch, 1, n_mels, time_frames] log-mel spectrogram
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    batch = waveform.shape[0]

    # STFT
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window, return_complex=True)
    power = stft.abs() ** 2  # [batch, n_fft//2+1, time_frames]

    # Mel filterbank
    fmin, fmax = 0.0, sample_rate / 2.0
    mel_low = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_points = torch.linspace(mel_low, mel_high, n_mels + 2, device=waveform.device)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = (hz_points * n_fft / sample_rate).long()

    n_freqs = n_fft // 2 + 1
    filterbank = torch.zeros(n_mels, n_freqs, device=waveform.device)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if center > left:
            filterbank[i, left:center] = torch.linspace(0, 1, int(center - left), device=waveform.device)
        if right > center:
            filterbank[i, center:right] = torch.linspace(1, 0, int(right - center), device=waveform.device)

    # Apply filterbank
    mel_spec = torch.matmul(filterbank, power)  # [batch, n_mels, time_frames]
    log_mel = torch.log(mel_spec.clamp(min=1e-9))

    return log_mel.unsqueeze(1)  # [batch, 1, n_mels, time_frames]


if __name__ == "__main__":
    # Test AudioBuffer
    buf = AudioBuffer()
    chunk = np.random.randn(8000).astype(np.float32)  # 0.5s chunk
    buf.push(chunk)
    buf.push(chunk)
    buf.push(chunk)
    buf.push(chunk)
    window = buf.get_window()
    print(f"AudioBuffer window shape: {window.shape} — expected ({AUDIO_WINDOW_SAMPLES},)")
    assert window.shape == (AUDIO_WINDOW_SAMPLES,)

    # Test mel spectrogram
    waveform = torch.randn(AUDIO_WINDOW_SAMPLES)
    mel = mel_spectrogram(waveform)
    print(f"Mel spectrogram shape: {list(mel.shape)} — expected [1, 1, {MEL_N_MELS}, ~{AUDIO_WINDOW_SAMPLES // MEL_HOP_LENGTH + 1}]")
    assert mel.shape[0] == 1
    assert mel.shape[1] == 1
    assert mel.shape[2] == MEL_N_MELS

    # Batch test
    batch_wav = torch.randn(4, AUDIO_WINDOW_SAMPLES)
    batch_mel = mel_spectrogram(batch_wav)
    print(f"Batch mel shape: {list(batch_mel.shape)}")
    assert batch_mel.shape[0] == 4

    print("AudioProcessor OK")
