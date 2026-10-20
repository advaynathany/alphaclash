# AlphaClash

A learning agent for Clash Royale that plays the real game on an Android emulator.

## Ground rules
- **Never on ladder.** The bot plays only Training Camp and friendly battles between my own alt accounts. Automation breaks Supercell's terms of service; keep the risk to alts I'm fine losing, and never put a bot against people who don't know.
- The real game runs at 1x speed (~400 games/day per emulator), so most learning comes from recorded human matches, not online self-play.

## Pipeline
| Stage | What | Status |
|---|---|---|
| 0. Device | Emulator + adb: frame capture and taps/drags (`device/adb.py`) | scaffolded |
| 1. Perception | Frames to game state: troops (YOLO), towers HP, elixir, hand cards, timer | — |
| 2. Imitation | Run perception on top-player replays (YouTube / TV Royale), learn a policy by behavior cloning or offline RL | — |
| 3. Online RL | Fine-tune on the emulator vs Training Camp and alt friendlies | — |
| 4. Self-play sim (optional) | Fast sim calibrated against emulator recordings for scale | — |

## Action and state spaces (v0)
- Decision every 0.5 s: no-op or (hand slot 1-4, placement cell on a coarse grid of my half).
- State: per-cell unit channels (type, side, HP), my elixir/hand/next card, tower HPs, time, opponent's cards seen so far.

## Setup
1. Install Android Studio (or `brew install --cask android-platform-tools` for just adb), create an arm64 Pixel emulator with Google Play, install Clash Royale, log into an alt.
2. `uv sync`
3. `adb devices`, then `uv run scripts/record_frames.py --fps 2 --seconds 300` during a Training Camp match.
