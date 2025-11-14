# app.py
from __future__ import annotations
import time
import collections
import threading
import numpy as np
import sounddevice as sd
import curses
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import Config
from .devices import list_input_devices, pick_first_working_device
from .util import rms_level, downsample_48k_to_16k
from .vad import SileroVAD
from .asr import Transcriber, Transcript
from .ui import UI

DeviceInfo = Tuple[int, str, int, float]


@dataclass
class Shared:
    last_text: str = ""
    level: float = 0.0
    prob: float = 0.0
    state: str = "idle"
    transcripts: list[Transcript] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shared = Shared()
        self.devices: List[DeviceInfo] = list_input_devices()
        self.current = self._choose_device()
        self.vad = SileroVAD(cfg.vad_rate, cfg.vad_chunk, cfg.thresh_on, cfg.thresh_off)
        self.transcriber = Transcriber(
            cfg.whisper_model, cfg.whisper_language, None, self._on_text
        )
        self.ui = UI(cfg.ui_fps)

    def _choose_device(self) -> int:
        if not self.devices:
            return -1
        if self.cfg.device_env is not None:
            try:
                idx = int(self.cfg.device_env)
                sd.check_input_settings(
                    device=idx, channels=1, samplerate=self.cfg.mic_rate, dtype="int16"
                )
                return idx
            except Exception:
                pass
        pick = pick_first_working_device(self.cfg.mic_rate)
        return pick if pick is not None else self.devices[0][0]

    def _on_text(self, tr: Transcript) -> None:
        with self.shared.lock:
            self.shared.last_text = tr.text
            self.shared.transcripts.append(tr)

    def open_stream(self, dev: int, frame_samples: int):
        return sd.InputStream(
            device=dev,
            dtype="int16",
            channels=1,
            samplerate=self.cfg.mic_rate,
            blocksize=frame_samples,
        )

    def safe_reopen(self, stream, new_dev: int, frame_samples: int):
        try:
            if stream is not None:
                try:
                    stream.abort()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        finally:
            stream = self.open_stream(new_dev, frame_samples)
            stream.start()
            return stream

    def run(self, stdscr) -> None:
        if not self.devices:
            stdscr.addstr(0, 0, "Нет входных устройств")
            stdscr.refresh()
            time.sleep(2)
            return
        curses.curs_set(0)
        stdscr.nodelay(True)
        self.transcriber.start()
        sd.default.dtype = "int16"
        sd.default.channels = 1
        sd.default.samplerate = self.cfg.mic_rate
        frame_samples = int(self.cfg.mic_rate * self.cfg.frame_ms / 1000)
        silence_limit_frames = self.cfg.silence_ms // self.cfg.frame_ms
        preroll_len = self.cfg.preroll_ms // self.cfg.frame_ms
        preroll = collections.deque(maxlen=preroll_len)
        in_segment = False
        frames = []
        silence_frames = 0
        segment_start_ts = None

        stream = self.open_stream(self.current, frame_samples)
        stream.start()
        last_draw = 0.0
        try:
            while True:
                try:
                    audio, _ = stream.read(frame_samples)
                except Exception:
                    stream = self.safe_reopen(stream, self.current, frame_samples)
                    audio, _ = stream.read(frame_samples)
                buf = audio.reshape(-1).astype(np.int16)
                lvl = rms_level(buf)
                vad_pcm = downsample_48k_to_16k(buf)
                st = self.vad.step(vad_pcm)
                with self.shared.lock:
                    self.shared.level = lvl
                    self.shared.prob = st.prob
                if not in_segment:
                    with self.shared.lock:
                        self.shared.state = "idle"
                    preroll.append(buf)
                    if st.is_speech:
                        in_segment = True
                        with self.shared.lock:
                            self.shared.state = "recording"
                        frames = list(preroll) + [buf]
                        silence_frames = 0
                        segment_start_ts = time.time()
                else:
                    with self.shared.lock:
                        self.shared.state = "recording"
                    frames.append(buf)
                    if st.is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= silence_limit_frames:
                            pcm16 = np.concatenate(frames)
                            t0 = segment_start_ts or time.time()
                            t1 = time.time()
                            self.transcriber.enqueue(pcm16, self.cfg.mic_rate, t0, t1)
                            in_segment = False
                            frames = []
                            silence_frames = 0
                            preroll.clear()
                now = time.time()
                if now - last_draw >= 1.0 / self.cfg.ui_fps:
                    with self.shared.lock:
                        lt = self.shared.last_text
                        stt = self.shared.state
                        pr = self.shared.prob
                        lv = self.shared.level
                    self.ui.draw(stdscr, self.devices, self.current, lv, pr, stt, lt)
                    last_draw = now
                try:
                    ch = stdscr.getch()
                except Exception:
                    ch = -1
                if ch == ord("q"):
                    break
                if ch == ord("n") or ch == ord("p"):
                    pos_map = {d[0]: i for i, d in enumerate(self.devices)}
                    pos = pos_map[self.current]
                    if ch == ord("n"):
                        pos = (pos + 1) % len(self.devices)
                    else:
                        pos = (pos - 1) % len(self.devices)
                    self.current = self.devices[pos][0]
                    stream = self.safe_reopen(stream, self.current, frame_samples)
        finally:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self.transcriber.stop()
