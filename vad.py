# vad.py
from __future__ import annotations
import numpy as np
import torch
from silero_vad import load_silero_vad
from dataclasses import dataclass


@dataclass
class VadState:
    prob: float = 0.0
    is_speech: bool = False


class SileroVAD:
    def __init__(self, rate: int, chunk: int, thresh_on: float, thresh_off: float):
        self.rate = rate
        self.chunk = chunk
        self.th_on = thresh_on
        self.th_off = thresh_off
        self.model = load_silero_vad()
        self.state = VadState()

    def step(self, pcm16_16k: np.ndarray) -> VadState:
        x = torch.from_numpy(pcm16_16k.astype(np.float32)) / 32768.0
        m = 0.0
        s = x.numel()
        for i in range(0, s - self.chunk + 1, self.chunk):
            p = float(self.model(x[i : i + self.chunk], self.rate).item())
            if p > m:
                m = p
                if m > self.th_on:
                    break
        if self.state.is_speech:
            is_sp = m > self.th_off
        else:
            is_sp = m > self.th_on
        self.state = VadState(prob=m, is_speech=is_sp)
        return self.state
