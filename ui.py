# ui.py
from __future__ import annotations
import curses
from typing import List, Tuple

DeviceInfo = Tuple[int, str, int, float]


class UI:
    def __init__(self, fps: int):
        self.fps = fps

    def draw(
        self,
        stdscr,
        devices: List[DeviceInfo],
        current_idx: int,
        level: float,
        prob: float,
        state: str,
        last_text: str,
    ) -> None:
        stdscr.erase()
        stdscr.addstr(0, 0, "Mics:")
        row = 1
        for idx, name, ch, sr in devices[:30]:
            tag = "*> " if idx == current_idx else "   "
            stdscr.addstr(row, 0, f"{tag}[{idx}] {name}  ch={ch}  def_sr={sr}")
            row += 1
        row += 1
        stdscr.addstr(row, 0, f"Current device: {current_idx}")
        row += 1
        bar_len = 50
        lv = min(1.0, level * 8.0)
        filled = int(lv * bar_len)
        stdscr.addstr(
            row,
            0,
            "Level: [" + "#" * filled + "-" * (bar_len - filled) + f"] {level:.3f}",
        )
        row += 1
        stdscr.addstr(row, 0, f"VAD prob: {prob:.3f}  State: {state}")
        row += 1
        preview = (last_text or "")[-120:]
        stdscr.addstr(row, 0, f"Last text: {preview}")
        row += 2
        stdscr.addstr(row, 0, "Actions: n=next p=prev q=exit")
        stdscr.refresh()
