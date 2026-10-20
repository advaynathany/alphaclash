"""Read the end-of-match screen: crowns per side, and so who won.

Layout (settled, after the banners slide in): enemy's red banner at the top, my
blue banner below, "VS" between. Each side's crowns sit on the three cushions
above its banner, gold when earned, empty otherwise.

Measured only on wins (3 Training Camp / friendly-free matches in the first
recording). The enemy crown row (ENEMY_CROWN_Y) is inferred from the mirror
layout, not yet seen with crowns in it: check it on the first recorded loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphaclash.perception.screen import REF_H, REF_W

RED_BANNER_Y = (482, 595)
BLUE_BANNER_Y = (843, 918)
CROWN_SLOTS = [(120, 250), (255, 395), (400, 535)]
MY_CROWN_Y = (700, 840)
ENEMY_CROWN_Y = (340, 480)
GOLD_MIN = 0.07  # gold fraction of a cushion slot holding a crown (measured ~0.16, empty 0.0)


@dataclass
class Result:
    my_crowns: int
    enemy_crowns: int

    @property
    def outcome(self) -> str:
        """'win' | 'loss' | 'draw' by crowns (a tiebreak on tower HP would show as a draw here)."""
        if self.my_crowns == self.enemy_crowns:
            return "draw"
        return "win" if self.my_crowns > self.enemy_crowns else "loss"


def _banner(frame: np.ndarray, y: tuple[int, int], color: str) -> bool:
    a = frame[y[0] : y[1], 60:592].astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "red":
        m = (r > 110) & (g < 65) & (b > 45) & (b < 100) & (r > b + 40)
    else:
        m = (b > 140) & (r < 75) & (g > 70) & (g < 120)
    return (m.mean(1) > 0.6).mean() > 0.3  # names and badges break up many rows


def _crowns(frame: np.ndarray, y: tuple[int, int]) -> int:
    n = 0
    for x0, x1 in CROWN_SLOTS:
        a = frame[y[0] : y[1], x0:x1].astype(int)
        n += int(((a[..., 0] > 200) & (a[..., 1] > 140) & (a[..., 2] < 120)).mean() > GOLD_MIN)
    return n


def read_result(frame: np.ndarray) -> Result | None:
    """The match result if this frame is the settled result screen, else None."""
    assert frame.shape[:2] == (REF_H, REF_W), "normalize the frame first (Screen.normalize)"
    if not (_banner(frame, RED_BANNER_Y, "red") and _banner(frame, BLUE_BANNER_Y, "blue")):
        return None
    return Result(_crowns(frame, MY_CROWN_Y), _crowns(frame, ENEMY_CROWN_Y))
