"""Local labeling tool for our own arena frames (the detector's honest test set).

Serves tools/label_tool.html on http://127.0.0.1:8765. Boxes are stored per
image as data/eval/labels/<stem>.json: [{"name", "side", "xyxy"}], side "mine" |
"enemy" | null, xyxy in arena-crop pixels. An image with no saved labels opens
with the model's guesses from data/eval/prelabels/ (scripts/prelabel_eval.py),
so labeling is mostly correcting.

Usage: uv run scripts/label_server.py
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from alphaclash.perception.detector import NON_UNITS

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/eval"
IMAGES, LABELS, PRELABELS = EVAL / "images", EVAL / "labels", EVAL / "prelabels"
HTML = ROOT / "tools/label_tool.html"
NAMES = [n for _, n in re.findall(r"^\s+(\d+): (\S+)", (ROOT / "data/yolo/dataset.yaml").read_text(), re.M)]
CLASSES = sorted(n for n in NAMES if n not in NON_UNITS)
STEM = re.compile(r"^[\w.-]+$")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode())

    def _stem(self, prefix: str) -> str | None:
        stem = self.path[len(prefix) :]
        return stem if STEM.match(stem) and (IMAGES / f"{stem}.jpg").exists() else None

    def do_GET(self) -> None:
        if self.path == "/":
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/list":
            stems = sorted(p.stem for p in IMAGES.glob("*.jpg"))
            return self._json({"classes": CLASSES, "images": [{"stem": s, "done": (LABELS / f"{s}.json").exists()} for s in stems]})
        if self.path.startswith("/api/image/") and (stem := self._stem("/api/image/")):
            return self._send(200, (IMAGES / f"{stem}.jpg").read_bytes(), "image/jpeg")
        if self.path.startswith("/api/labels/") and (stem := self._stem("/api/labels/")):
            for src, kind in ((LABELS, "saved"), (PRELABELS, "model")):
                if (p := src / f"{stem}.json").exists():
                    return self._json({"source": kind, "boxes": json.loads(p.read_text())})
            return self._json({"source": "empty", "boxes": []})
        self._send(404, b"{}")

    def do_PUT(self) -> None:
        if not (self.path.startswith("/api/labels/") and (stem := self._stem("/api/labels/"))):
            return self._send(404, b"{}")
        boxes = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        clean = [
            {"name": b["name"], "side": b["side"] if b["side"] in ("mine", "enemy") else None,
             "xyxy": [round(float(v), 1) for v in b["xyxy"]]}
            for b in boxes if b["name"] in CLASSES and len(b["xyxy"]) == 4
        ]
        LABELS.mkdir(parents=True, exist_ok=True)
        (LABELS / f"{stem}.json").write_text(json.dumps(clean, indent=0))
        self._json({"saved": len(clean)})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print("labeling tool on http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
