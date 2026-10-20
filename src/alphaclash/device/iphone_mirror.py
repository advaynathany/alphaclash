"""macOS iPhone Mirroring backend: same interface as device.adb.Device.

Coordinates passed to tap/drag are pixels in the frame returned by screenshot(),
so perception code never needs to know which backend it is talking to.
Needs Screen Recording + Accessibility permission for the terminal.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import Quartz
from PIL import Image

OWNER = "iPhone Mirroring"


def _find_window() -> dict:
    wins = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    for w in wins:
        if w.get("kCGWindowOwnerName") == OWNER and w.get("kCGWindowLayer") == 0:
            return w
    raise RuntimeError(f"No on-screen '{OWNER}' window. Open the app and keep it visible.")


@dataclass
class Device:
    _win: dict = field(default_factory=_find_window, repr=False)
    _scale: float = 1.0  # frame pixels per screen point (2.0 on Retina)

    @property
    def window_id(self) -> int:
        return int(self._win["kCGWindowNumber"])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Window x, y, w, h in screen points."""
        b = self._win["kCGWindowBounds"]
        return b["X"], b["Y"], b["Width"], b["Height"]

    def refresh(self) -> None:
        """Re-read window position (call if the window was moved)."""
        self._win = _find_window()

    def screen_size(self) -> tuple[int, int]:
        h, w = self.screenshot().shape[:2]
        return w, h

    def screenshot(self) -> np.ndarray:
        """RGB frame of the mirrored phone window (no shadow)."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            subprocess.run(
                ["screencapture", "-x", "-o", f"-l{self.window_id}", str(p)], check=True
            )
            frame = np.asarray(Image.open(p).convert("RGB"))
        self._scale = frame.shape[1] / self.bounds[2]
        return frame

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        wx, wy, _, _ = self.bounds
        return wx + x / self._scale, wy + y / self._scale

    @staticmethod
    def _post(kind: int, pt: tuple[float, float]) -> None:
        ev = Quartz.CGEventCreateMouseEvent(None, kind, pt, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def focus(self) -> None:
        subprocess.run(["open", "-a", OWNER], check=True)
        time.sleep(0.3)

    def tap(self, x: int, y: int) -> None:
        pt = self._to_screen(x, y)
        self._post(Quartz.kCGEventMouseMoved, pt)
        self._post(Quartz.kCGEventLeftMouseDown, pt)
        time.sleep(0.05)
        self._post(Quartz.kCGEventLeftMouseUp, pt)

    def drag(self, x0: int, y0: int, x1: int, y1: int, ms: int = 150, steps: int = 12) -> None:
        """Card placement: press on a hand slot, move onto the arena, release."""
        a, b = self._to_screen(x0, y0), self._to_screen(x1, y1)
        self._post(Quartz.kCGEventMouseMoved, a)
        self._post(Quartz.kCGEventLeftMouseDown, a)
        for i in range(1, steps + 1):
            t = i / steps
            self._post(Quartz.kCGEventLeftMouseDragged, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            time.sleep(ms / 1000 / steps)
        self._post(Quartz.kCGEventLeftMouseUp, b)
