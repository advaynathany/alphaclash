"""Print what perception reads off the live game, a few times a second.

Usage: uv run scripts/live_hud.py [yolo_weights]
With weights, also lists the units on the arena (name, side, tile).
"""

import sys
import time

from alphaclash.device.iphone_mirror import Device
from alphaclash.perception.result import read_result
from alphaclash.perception.screen import Screen
from alphaclash.perception.tracker import TowerTracker

dev, screen, towers = Device(), Screen(), TowerTracker()
detector = None
if len(sys.argv) > 1:
    from alphaclash.perception.detector import Detector

    detector = Detector(sys.argv[1])
while True:
    t = time.time()
    frame = dev.screenshot()
    hud = screen.read(frame)
    hp = towers.update(hud)
    if hud.in_battle:
        hand = " | ".join(c or "-" for c in hud.hand)
        tw = " ".join(f"{hp[k]:.2f}" for k in ("enemy_left", "enemy_king", "enemy_right", "my_left", "my_king", "my_right"))
        print(
            f"{'--:--' if hud.clock is None else f'{hud.clock // 60}:{hud.clock % 60:02d}'}  "
            f"elixir {hud.elixir:4.1f}  hand [{hand}]  next {hud.next_card or '-'}  "
            f"towers eL/eK/eR mL/mK/mR {tw}  ({time.time() - t:.2f}s)"
        )
        if detector:
            side = {"mine": "+", "enemy": "-", None: "?"}
            units = detector.detect(frame)
            print("   units " + (", ".join(f"{side[u.side]}{u.name}@{u.tile[0]:.0f},{u.tile[1]:.0f}" for u in units) or "none"))
    elif result := read_result(screen.normalize(frame)):
        print(f"result: {result.outcome}  crowns me {result.my_crowns} - {result.enemy_crowns} enemy")
    else:
        print("not in battle")
