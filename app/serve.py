#!/usr/bin/env python3
"""
Development server for the semantic search app.

Run it with no arguments. The port is fixed on purpose:

    python3 app/serve.py

Why the port is not automatic
-----------------------------
IndexedDB is scoped to an *origin*, which is scheme + host + port. Serving on
8768 one day and 8769 the next produces two unrelated origins, each with its own
empty index — so previously indexed documents simply vanish, and stale copies
accumulate in DevTools. A server that quietly falls back to "the next free port"
would cause exactly that, so this one fails loudly instead.

Why the headers matter
----------------------
Multi-threaded WebAssembly needs SharedArrayBuffer, which the browser only
grants to a cross-origin-isolated page. Without COOP/COEP the runtime silently
drops to a single thread — several times slower, with no error anywhere.
"""

import http.server
import socketserver
import socket
import subprocess
import sys
from pathlib import Path

PORT = 8768
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, plus the two headers that enable multi-threaded WASM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("X-Content-Type-Options", "nosniff")
        # The app is edited live; never let a stale module linger.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        """Capture selftest.html results so a headless run can be scripted.

        Confined to /result and to loopback, because this server exists only for
        local development and should never accept anything else.
        """
        if self.path != "/result":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace")
        (ROOT / "result.json").write_text(body)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        # Log everything. An earlier filter hid non-GET lines, which meant a
        # 501 on every POST went unnoticed and looked like a silent test hang.
        sys.stderr.write(f"  {fmt % args}\n")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def describe_occupant(port: int) -> str:
    """Best-effort: name the process already holding the port."""
    try:
        output = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return output[1] if len(output) > 1 else ""
    except Exception:
        return ""


def main() -> int:
    if not port_is_free(PORT):
        occupant = describe_occupant(PORT)
        print(f"Port {PORT} is already in use.", file=sys.stderr)
        if occupant:
            print(f"  held by: {occupant}", file=sys.stderr)
        print(
            f"\nThis server will not start on a different port. The browser ties your\n"
            f"index to the origin http://127.0.0.1:{PORT}, so moving ports would hide\n"
            f"every document you have already indexed.\n\n"
            f"Free it, then try again:\n"
            f"    kill $(lsof -t -iTCP:{PORT} -sTCP:LISTEN)",
            file=sys.stderr,
        )
        return 1

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Local Semantic Search  →  http://127.0.0.1:{PORT}/")
        print("cross-origin isolated (multi-threaded WASM enabled)")
        print("Ctrl-C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
