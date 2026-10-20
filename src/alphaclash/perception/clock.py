"""Read the match clock ("Time left: M:SS", top right in real matches).

Training Camp shows no clock, so read() returns None there. Digits are white with
a dark outline; the string is centered, so it moves when a "1" narrows it. We find
the colon (the only short blob), take the minute digit left of it and split the
seconds right of it, then match each glyph against templates in data/digits/
(built by scripts/build_digit_templates.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CLOCK_BOX = (540, 197, 630, 229)  # x0, y0, x1, y1 around the digits, 652x1436 frame
GLYPH = (12, 20)  # template size (w, h)
DIGIT_DIR = Path(__file__).resolve().parents[3] / "data" / "digits"
MATCH_MIN = 0.75


def _mask(frame: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = CLOCK_BOX
    return frame[y0:y1, x0:x1].astype(int).min(2) > 170


def _blobs(m: np.ndarray) -> list[tuple[int, int]]:
    cols = m.any(0)
    out, start = [], None
    for x, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = x
        elif not c and start is not None:
            out.append((start, x))
            start = None
    return out


def _glyph(m: np.ndarray, x0: int, x1: int) -> np.ndarray:
    rows = np.where(m[:, x0:x1].any(1))[0]
    g = m[rows[0] : rows[-1] + 1, x0:x1].astype(np.uint8) * 255
    v = np.asarray(Image.fromarray(g).resize(GLYPH), float).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def glyphs(frame: np.ndarray) -> list[np.ndarray] | None:
    """[minute, tens, ones] glyph vectors, or None if no M:SS layout is visible."""
    m = _mask(frame)
    blobs = [b for b in _blobs(m) if m[:, b[0] : b[1]].sum() >= 6]
    heights = [np.ptp(np.where(m[:, a:b].any(1))[0]) + 1 for a, b in blobs]
    colon = [i for i, (h, (a, b)) in enumerate(zip(heights, blobs)) if h < 22 and b - a <= 9]
    if len(colon) != 1:
        return None
    c = colon[0]
    left, right = blobs[:c], blobs[c + 1 :]
    if len(left) != 1 or not right:
        return None
    if len(right) == 1:  # seconds digits touching: split in the middle
        a, b = right[0]
        right = [(a, (a + b) // 2), ((a + b) // 2, b)]
    if len(right) != 2:
        return None
    return [_glyph(m, *left[0]), _glyph(m, *right[0]), _glyph(m, *right[1])]


class Clock:
    def __init__(self) -> None:
        paths = sorted(DIGIT_DIR.glob("*.npy"))
        self.digits = [int(p.stem) for p in paths]
        self.mat = np.stack([np.load(p) for p in paths]) if paths else np.zeros((0, GLYPH[0] * GLYPH[1]))

    def _digit(self, g: np.ndarray) -> int | None:
        if not len(self.mat):
            return None
        s = self.mat @ g
        i = int(s.argmax())
        return self.digits[i] if s[i] >= MATCH_MIN else None

    def read(self, frame: np.ndarray) -> int | None:
        """Seconds left, or None (no clock, overtime, unreadable)."""
        g = glyphs(frame)
        if g is None:
            return None
        d = [self._digit(x) for x in g]
        if None in d or d[1] > 5:
            return None
        return d[0] * 60 + d[1] * 10 + d[2]
