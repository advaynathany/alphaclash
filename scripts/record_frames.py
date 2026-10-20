"""Save emulator frames to data/frames/<session>/ for labeling the perception model.

Usage: uv run scripts/record_frames.py --fps 2 --seconds 300
Play (or watch) a Training Camp match while this runs.
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from alphaclash.device.adb import Device

p = argparse.ArgumentParser()
p.add_argument("--fps", type=float, default=2.0)
p.add_argument("--seconds", type=float, default=300)
p.add_argument("--serial", default=None)
args = p.parse_args()

dev = Device(args.serial)
out = Path("data/frames") / time.strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True)
print(f"screen {dev.screen_size()} -> {out}")

end, i = time.time() + args.seconds, 0
while time.time() < end:
    t = time.time()
    Image.fromarray(dev.screenshot()).save(out / f"{i:06d}.png")
    i += 1
    time.sleep(max(0.0, 1 / args.fps - (time.time() - t)))
print(f"saved {i} frames")
