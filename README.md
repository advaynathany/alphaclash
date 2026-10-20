# AlphaClash

A learning agent for Clash Royale that plays the real game on a real device.

## Ground rules
- **Never on ladder.** The bot plays only Training Camp and friendly battles between my own alt accounts. Automation breaks Supercell's terms of service; keep the risk to alts I'm fine losing, and never put a bot against people who don't know.
- The real game runs at 1x speed (~400 games/day per device), so most learning comes from recorded human matches, not online self-play.

## Pipeline
| Stage | What | Status |
|---|---|---|
| 0. Device | iPhone Mirroring (`device/iphone_mirror.py`) now; USB Android phone + adb (`device/adb.py`) for training | scaffolded |
| 1. Perception | Frames to game state: troops (YOLO), towers HP, elixir, hand cards, timer | — |
| 2. Imitation | Run perception on top-player replays (YouTube / TV Royale), learn a policy by behavior cloning or offline RL | — |
| 3. Online RL | Fine-tune on device vs Training Camp and alt friendlies | — |
| 4. Self-play sim (optional) | Fast sim calibrated against device recordings for scale | — |

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
