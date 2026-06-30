from __future__ import annotations

import argparse
import base64
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from prefab_housing.wallface import (
    empty_wall_face_design,
    load_wall_face_design,
    render_wall_face_preview,
    save_wall_face_design,
    serialise_wall_face_design,
    wall_face_design_from_dict,
    wall_face_design_to_dict,
)
from voxel_renderer.block_registry import BLOCK_COLOURS, BLOCK_TEXTURE_MAP

REPO_ROOT = Path(__file__).resolve().parent.parent
EDITOR_HTML = REPO_ROOT / "prefab-housing" / "editor" / "wallface_editor.html"
DEFAULT_DESIGN = REPO_ROOT / "prefab-housing" / "designs" / "modular_default.wallface"


def _block_index() -> list[dict[str, object]]:
    known = sorted(set(BLOCK_COLOURS) | set(BLOCK_TEXTURE_MAP))
    return [
        {
            "id": block_id,
            "has_texture": block_id in BLOCK_TEXTURE_MAP,
        }
        for block_id in known
    ]


def _render_preview(design_payload: dict[str, object]) -> dict[str, str]:
    design = wall_face_design_from_dict(design_payload)
    return render_wall_face_preview(design)


class _EditorHandler(BaseHTTPRequestHandler):
    design_path: Path = DEFAULT_DESIGN

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(EDITOR_HTML.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/blocks":
            self._send_json({"blocks": _block_index()})
            return
        if parsed.path == "/api/design":
            params = parse_qs(parsed.query)
            width = int(params.get("width", [8])[0])
            height = int(params.get("height", [6])[0])
            path = self.design_path if self.design_path.exists() else None
            design = load_wall_face_design(path) if path is not None else empty_wall_face_design(width, height)
            self._send_json(
                {
                    "design": wall_face_design_to_dict(design),
                    "path": str(path) if path is not None else None,
                    "text": serialise_wall_face_design(design),
                    "preview": _render_preview(wall_face_design_to_dict(design)),
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8")) if body else {}
        if parsed.path == "/api/save":
            design = wall_face_design_from_dict(payload["design"])
            target = Path(str(payload.get("path") or self.design_path))
            save_wall_face_design(target, design)
            self.design_path = target
            self._send_json(
                {
                    "ok": True,
                    "path": str(target),
                    "text": serialise_wall_face_design(design),
                    "preview": _render_preview(wall_face_design_to_dict(design)),
                }
            )
            return
        if parsed.path == "/api/load":
            source = Path(str(payload["path"]))
            design = load_wall_face_design(source)
            self.design_path = source
            self._send_json(
                {
                    "ok": True,
                    "path": str(source),
                    "design": wall_face_design_to_dict(design),
                    "text": serialise_wall_face_design(design),
                    "preview": _render_preview(wall_face_design_to_dict(design)),
                }
            )
            return
        if parsed.path == "/api/preview":
            design_payload = payload["design"]
            self._send_json({"ok": True, "preview": _render_preview(design_payload)})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the wall-face texture editor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--design",
        default=str(DEFAULT_DESIGN),
        help="Initial .wallface design file to edit.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _EditorHandler.design_path = Path(args.design)
    server = ThreadingHTTPServer((args.host, args.port), _EditorHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"wallface_editor={url}")
    print(f"design={_EditorHandler.design_path}")
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
