"""Smooth per-frame tower readings over time.

Single frames are noisy: troops and spells cover bars (None), the match-start
zoom shifts everything, and a covered badge looks like a destroyed tower. The
game's own rules fix most of it: HP never goes up within a match, and a
destroyed tower stays destroyed.
"""

from __future__ import annotations

from collections import deque

from alphaclash.perception.screen import TOWER_BARS, Hud

WINDOW = 5  # frames in the median filter
DESTROY_FRAMES = 3  # consecutive 0.0 readings before a tower counts as destroyed
NEW_MATCH_GAP = 6  # out-of-battle frames that end a match
START_GRACE = 8  # in-battle frames after a (re)start where 0.0 is ignored (arena zoom-in)


class TowerTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hp = {name: 1.0 for name in TOWER_BARS}
        self._hist: dict[str, deque[float]] = {n: deque(maxlen=WINDOW) for n in TOWER_BARS}
        self._zeros = {n: 0 for n in TOWER_BARS}
        self._out_of_battle = 0
        self._in_battle = 0

    def update(self, hud: Hud) -> dict[str, float]:
        if not hud.in_battle:
            self._out_of_battle += 1
            if self._out_of_battle == NEW_MATCH_GAP:
                self.reset()
            return self.hp
        self._out_of_battle = 0
        self._in_battle += 1

        for name, v in hud.towers.items():
            if v is None or self.hp[name] == 0.0:
                continue
            if v == 0.0:
                if self._in_battle <= START_GRACE:
                    continue
                self._zeros[name] += 1
                if self._zeros[name] >= DESTROY_FRAMES:
                    self.hp[name] = 0.0
                continue
            self._zeros[name] = 0
            h = self._hist[name]
            h.append(v)
            med = sorted(h)[len(h) // 2]
            self.hp[name] = min(self.hp[name], med)  # HP only goes down
        return self.hp
