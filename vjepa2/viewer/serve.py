#!/usr/bin/env python3
"""Serve completed run manifests and assets for the local research viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from viewer.inventory import CSV_FIELDS


STATIC_DIR = Path(__file__).resolve().parent / "static"


def run_key(path: Path, existing: set[str]) -> str:
    base = path.name or "run"
    key = base
    index = 2
    while key in existing:
        key = f"{base}-{index}"
        index += 1
    return key


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def load_inventory(run_dir: Path) -> dict[str, Any]:
    inventory_path = run_dir / "inventory.json"
    if not inventory_path.exists():
        return {"schema_version": 1, "summary": {"n_items": 0}, "items": []}
    return json.loads(inventory_path.read_text())


def write_inventory_files(run_dir: Path, inventory: dict[str, Any]) -> None:
    import csv

    summary = inventory.setdefault("summary", {})
    items = inventory.get("items") or []
    summary["visual_status"] = {
        key: sum(1 for item in items if item.get("visual_status") == key)
        for key in sorted({item.get("visual_status", "unknown") for item in items})
    }

    (run_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    with (run_dir / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item.get(key) for key in CSV_FIELDS})


def content_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def make_handler(runs: dict[str, Path]):
    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "VJEPAViewer/0.1"

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/runs":
                self.send_json({"runs": [{"id": key, "path": str(path)} for key, path in runs.items()]})
                return
            if parsed.path == "/api/manifest":
                self.handle_manifest(parsed)
                return
            if parsed.path == "/api/inventory":
                self.handle_inventory(parsed)
                return
            if parsed.path.startswith("/assets/"):
                self.handle_asset(parsed.path)
                return
            self.handle_static(parsed.path)

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/inventory/item":
                self.handle_inventory_item_update(parsed)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def handle_manifest(self, parsed: urllib.parse.ParseResult) -> None:
            query = urllib.parse.parse_qs(parsed.query)
            key = query.get("run", [next(iter(runs))])[0]
            run_dir = runs.get(key)
            if run_dir is None:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown run: {key}")
                return
            try:
                manifest = load_manifest(run_dir)
            except (OSError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self.send_json({"run_id": key, "manifest": manifest})

        def handle_inventory(self, parsed: urllib.parse.ParseResult) -> None:
            query = urllib.parse.parse_qs(parsed.query)
            key = query.get("run", [next(iter(runs))])[0]
            run_dir = runs.get(key)
            if run_dir is None:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown run: {key}")
                return
            try:
                inventory = load_inventory(run_dir)
            except (OSError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self.send_json({"run_id": key, "inventory": inventory})

        def handle_inventory_item_update(self, parsed: urllib.parse.ParseResult) -> None:
            query = urllib.parse.parse_qs(parsed.query)
            key = query.get("run", [next(iter(runs))])[0]
            run_dir = runs.get(key)
            if run_dir is None:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown run: {key}")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                spec = payload["spec"]
                updates = payload.get("updates") or {}
                inventory = load_inventory(run_dir)
                item = next((row for row in inventory.get("items", []) if row.get("spec") == spec), None)
                if item is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"unknown inventory item: {spec}")
                    return
                allowed = {"visual_status", "visual_notes", "violation_type", "object_type"}
                for field, value in updates.items():
                    if field in allowed:
                        item[field] = str(value or "")
                curation = item.setdefault("curation", {})
                curation["visual_status"] = item.get("visual_status", "")
                curation["visual_failure"] = item.get("visual_status", "")
                write_inventory_files(run_dir, inventory)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_json({"run_id": key, "item": item})

        def handle_static(self, request_path: str) -> None:
            if request_path in {"", "/"}:
                request_path = "/index.html"
            path = self.safe_join(STATIC_DIR, request_path.lstrip("/"))
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(path)

        def handle_asset(self, request_path: str) -> None:
            parts = request_path.split("/", 3)
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            key = urllib.parse.unquote(parts[2])
            rel = urllib.parse.unquote(parts[3])
            run_dir = runs.get(key)
            if run_dir is None:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown run: {key}")
                return
            path = self.safe_join(run_dir, rel)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(path)

        def safe_join(self, root: Path, rel: str) -> Path | None:
            normalized = posixpath.normpath("/" + rel).lstrip("/")
            candidate = (root / normalized).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                return None
            return candidate

        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_file(self, path: Path) -> None:
            size = path.stat().st_size
            range_header = self.headers.get("Range")
            start = 0
            end = size - 1
            status = HTTPStatus.OK

            if range_header:
                parsed = self.parse_range(range_header, size)
                if parsed is None:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                start, end = parsed
                status = HTTPStatus.PARTIAL_CONTENT

            length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header("Content-Type", content_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return

            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def parse_range(self, header: str, size: int) -> tuple[int, int] | None:
            if not header.startswith("bytes=") or size <= 0:
                return None
            spec = header.removeprefix("bytes=").split(",", 1)[0].strip()
            if "-" not in spec:
                return None
            start_text, end_text = spec.split("-", 1)
            try:
                if start_text == "":
                    suffix = int(end_text)
                    if suffix <= 0:
                        return None
                    return max(0, size - suffix), size - 1
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            except ValueError:
                return None
            if start < 0 or start >= size or end < start:
                return None
            return start, min(end, size - 1)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return ViewerHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Run directory containing manifest.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: dict[str, Path] = {}
    for raw_path in args.run:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Run directory does not exist: {path}")
        if not (path / "manifest.json").exists():
            raise SystemExit(f"Run directory has no manifest.json: {path}")
        key = run_key(path, set(runs))
        runs[key] = path

    handler = make_handler(runs)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {len(runs)} run(s) at {url}")
    print("Runs: " + ", ".join(f"{key}={path}" for key, path in runs.items()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
