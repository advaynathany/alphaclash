"""Thin adb wrapper: grab frames from and send taps to an Android emulator."""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Device:
    serial: str | None = None  # None = the only connected device

    def _adb(self, *args: str, binary: bool = False) -> bytes | str:
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)
        out = subprocess.run(cmd, check=True, capture_output=True).stdout
        return out if binary else out.decode()

    def screen_size(self) -> tuple[int, int]:
        # "Physical size: 1080x2340"
        w, h = self._adb("shell", "wm", "size").strip().split()[-1].split("x")
        return int(w), int(h)

    def screenshot(self) -> np.ndarray:
        """RGB frame as an HxWx3 uint8 array. ~5-10 fps; swap for scrcpy if too slow."""
        png = self._adb("exec-out", "screencap", "-p", binary=True)
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def tap(self, x: int, y: int) -> None:
        self._adb("shell", "input", "tap", str(x), str(y))

    def drag(self, x0: int, y0: int, x1: int, y1: int, ms: int = 150) -> None:
        """Card placement: drag from a hand slot onto the arena."""
        self._adb("shell", "input", "swipe", str(x0), str(y0), str(x1), str(y1), str(ms))
