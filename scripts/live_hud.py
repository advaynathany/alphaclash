"""Print what perception reads off the live game, a few times a second.

Usage: uv run scripts/live_hud.py
"""

import time

from alphaclash.device.iphone_mirror import Device
from alphaclash.perception.screen import Screen

dev, screen = Device(), Screen()
while True:
    t = time.time()
    hud = screen.read(dev.screenshot())
    if hud.in_battle:
        hand = " | ".join(c or "-" for c in hud.hand)
        print(f"elixir {hud.elixir:4.1f}  hand [{hand}]  next {hud.next_card or '-'}  ({time.time() - t:.2f}s)")
    else:
        print("not in battle")
