"""Print what perception reads off the live game, a few times a second.

Usage: uv run scripts/live_hud.py
"""

import time

from alphaclash.device.iphone_mirror import Device
from alphaclash.perception.screen import Screen
from alphaclash.perception.tracker import TowerTracker

dev, screen, towers = Device(), Screen(), TowerTracker()
while True:
    t = time.time()
    hud = screen.read(dev.screenshot())
    hp = towers.update(hud)
    if hud.in_battle:
        hand = " | ".join(c or "-" for c in hud.hand)
        tw = " ".join(f"{hp[k]:.2f}" for k in ("enemy_left", "enemy_king", "enemy_right", "my_left", "my_king", "my_right"))
        print(
            f"elixir {hud.elixir:4.1f}  hand [{hand}]  next {hud.next_card or '-'}  "
            f"towers eL/eK/eR mL/mK/mR {tw}  ({time.time() - t:.2f}s)"
        )
    else:
        print("not in battle")
