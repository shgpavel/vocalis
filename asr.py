# asr.py
from __future__ import annotations
import tempfile
import os
import threading
import queue
import time
import numpy as np
import torch
import whisper
from dataclasses import dataclass
from typing import Callable, Optional
from .util import write_wav


@dataclass
class Transcript:
    start: str
    end: str
    text: str


class Transcriber(threading.Thread):
    def __init__(
        self,
        model_name: str,
        language: Optional[str],
        device_hint: Optional[str],
        on_text: Callable[[Transcript], None],
    ):
        super().__init__(daemon=True)
        self.q: queue.Queue[tuple[np.ndarray, int, float, float]] = queue.Queue(
            maxsize=8
        )
        self.stop_ev = threading.Event()
        self.on_text = on_text
        dev = device_hint or self._detect_device()
        self.model = whisper.load_model(model_name).to(dev)
        self.language = language

    def _detect_device(self) -> str:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def enqueue(self, pcm16: np.ndarray, rate: int, ts0: float, ts1: float) -> bool:
        try:
            self.q.put_nowait((pcm16, rate, ts0, ts1))
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                pcm16, rate, t0, t1 = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            try:
                write_wav(tmp.name, pcm16, rate)
                r = self.model.transcribe(
                    tmp.name,
                    language=self.language or None,
                    task="transcribe",
                    fp16=False,
                    verbose=False,
                )
                text = (r.get("text", "") or "").strip()
                if text:
                    self.on_text(
                        Transcript(
                            start=time.strftime(
                                "%Y-%m-%dT%H:%M:%S", time.localtime(t0)
                            ),
                            end=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t1)),
                            text=text,
                        )
                    )
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass

    def stop(self) -> None:
        self.stop_ev.set()
