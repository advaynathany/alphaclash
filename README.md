# AlphaClash

A learning agent for Clash Royale that plays the real game on a real device.

## Ground rules
- **Never on ladder.** The bot plays only Training Camp and friendly battles between my own alt accounts. Automation breaks Supercell's terms of service; keep the risk to alts I'm fine losing, and never put a bot against people who don't know.
- The real game runs at 1x speed (~400 games/day per device), so most learning comes from recorded human matches, not online self-play.

## Pipeline
| Stage | What | Status |
|---|---|---|
| 0. Device | iPhone Mirroring (`device/iphone_mirror.py`) now; USB Android phone + adb (`device/adb.py`) for training | working |
| 1. Perception | Frames to game state: troops (YOLO), tower HP, elixir, hand cards, clock, result | working, being scored |
| 2. Imitation | Run perception on top-player replays (YouTube / TV Royale), learn a policy by behavior cloning or offline RL | next |
| 3. Online RL | Fine-tune on device vs Training Camp and alt friendlies | — |
| 4. Self-play sim (optional) | Fast sim calibrated against device recordings for scale | — |

## Perception
Everything downstream depends on reading the screen correctly, so this is where the work so far has gone.

- **HUD** (`perception/screen.py`): in-battle detection, elixir from the bar, the four hand cards and the next card.
- **Towers** (`perception/tracker.py`): HP from health-bar fill, smoothed by a temporal tracker so a single occluded frame doesn't read as a destroyed tower.
- **Clock and result** (`perception/clock.py`, `result.py`): match clock from digit templates labelled automatically from frame timestamps; crowns per side on the result screen.
- **Units** (`perception/detector.py`, `side_net.py`): the arena is cropped to the view used by the KataCR dataset, a YOLO model finds units, and a small CNN decides whose side each unit is on. Detections are mapped to tile coordinates. Towers take their side from their half of the arena, not from the classifier.
- **Training and scoring**: `scripts/build_yolo_dataset.py` turns KataCR's hand-labelled frames into a YOLO dataset, a Colab notebook trains it, and `tools/label_tool.html` with `scripts/label_server.py` labels our own device frames (pre-labelled by the model) so the detector is scored on the screens it will actually see, not only on the dataset it was trained on.
- `scripts/live_hud.py` overlays all of it on the live game.

## Action and state spaces (v0)
- Decision every 0.5 s: no-op or (hand slot 1-4, placement cell on a coarse grid of my half).
- State: per-cell unit channels (type, side, HP), my elixir/hand/next card, tower HPs, time, opponent's cards seen so far.

## Devices
- **Google's Android emulator does not work:** Clash Royale's anti-tamper detects it and exits on launch (`Exiting: 02`). Don't try to spoof around it; that is anti-cheat evasion.
- **Now:** macOS iPhone Mirroring. Grant Terminal Screen Recording + Accessibility. Keep the window on the current desktop. It drives the real cursor.
- **For training:** a cheap Android phone over USB (`brew install --cask android-platform-tools`, enable USB debugging), adb + scrcpy.

## Setup
1. `uv sync`
2. Open iPhone Mirroring with Clash Royale on an alt account, in Training Camp.
3. `uv run scripts/record_frames.py --fps 2 --seconds 300`
4. `uv run scripts/live_hud.py` to watch perception on the live game.

## Layout
```
src/alphaclash/device/      screen capture and input: iPhone Mirroring, adb
src/alphaclash/perception/  HUD, towers, clock, result, unit detector
scripts/                    recording, dataset building, training, evaluation, live HUD
tools/label_tool.html       labelling UI for device frames
notebooks/                  Colab training for the unit detector
```

Unit labels for the detector come from the [KataCR](https://github.com/wty-yy/KataCR) dataset.
