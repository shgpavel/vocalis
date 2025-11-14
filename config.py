# config.py
from dataclasses import dataclass
import os


@dataclass
class Config:
    mic_rate: int = int(os.getenv("MIC_RATE", "48000"))
    vad_rate: int = 16000
    vad_chunk: int = 512
    frame_ms: int = int(os.getenv("FRAME_MS", "120"))
    silence_ms: int = int(os.getenv("SILENCE_MS", "700"))
    preroll_ms: int = int(os.getenv("PRE_ROLL_MS", "240"))
    thresh_on: float = float(os.getenv("VAD_THRESH_ON", os.getenv("VAD_THRESH", "0.5")))
    thresh_off: float = float(os.getenv("VAD_THRESH_OFF", "0.35"))
    whisper_model: str = os.getenv("WHISPER_MODEL", "large")
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "ru")
    ui_fps: int = int(os.getenv("UI_FPS", "30"))
    device_env: str | None = os.getenv("DEVICE_INDEX")
