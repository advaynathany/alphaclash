"""Save emulator frames to data/frames/<session>/ for labeling the perception model.

Usage: uv run scripts/record_frames.py --fps 2 --seconds 300
Play (or watch) a Training Camp match while this runs.
Default backend is macOS iPhone Mirroring; --backend adb for an Android emulator.
"""

import argparse
import time
from pathlib import Path

from PIL import Image


p = argparse.ArgumentParser()
p.add_argument("--fps", type=float, default=2.0)
p.add_argument("--seconds", type=float, default=300)
p.add_argument("--serial", default=None)
p.add_argument("--backend", choices=["adb", "iphone"], default="iphone")
args = p.parse_args()

if args.backend == "adb":
    from alphaclash.device.adb import Device

    dev = Device(args.serial)
else:
    from alphaclash.device.iphone_mirror import Device

    dev = Device()
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
