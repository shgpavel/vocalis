# util.py
from __future__ import annotations
import wave
import numpy as np


def write_wav(path: str, pcm16: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())


def rms_level(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(((x.astype(np.float32) / 32768.0) ** 2).mean() ** 0.5)


def downsample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    r = x.size % 3
    if r:
        x = np.pad(x, (0, 3 - r), mode="edge")
    y = x.reshape(-1, 3).mean(axis=1)
    return y.astype(np.int16)
