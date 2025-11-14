# devices.py
from __future__ import annotations
import sounddevice as sd
from typing import List, Tuple

DeviceInfo = Tuple[int, str, int, float]


def list_input_devices(limit: int | None = None) -> List[DeviceInfo]:
    devs = sd.query_devices()
    res = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            res.append(
                (i, d["name"], d["max_input_channels"], float(d["default_samplerate"]))
            )
    return res if limit is None else res[:limit]


def pick_first_working_device(mic_rate: int) -> int | None:
    devices = list_input_devices()
    for idx, _, _, _ in devices:
        try:
            sd.check_input_settings(
                device=idx, channels=1, samplerate=mic_rate, dtype="int16"
            )
            return idx
        except Exception:
            continue
    return devices[0][0] if devices else None
