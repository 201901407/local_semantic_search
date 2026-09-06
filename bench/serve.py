#!/usr/bin/env python3
"""Static server for the in-browser benchmark.

Sets COOP/COEP so SharedArrayBuffer is available -- without these,
onnxruntime-web silently falls back to SINGLE-THREADED wasm and the
measured throughput would be several times too low with no error.
Also accepts POST /result to capture the page's findings.
"""
import http.server, socketserver, json, sys, datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        with open("result.json", "w") as f:
            f.write(body)
        print("\n=== RESULT RECEIVED ===")
        print(body)
        print("=== END RESULT ===", flush=True)
        self.send_response(204); self.end_headers()

    def log_message(self, fmt, *args):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {fmt % args}", flush=True)

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
    print(f"serving {PORT} with COOP/COEP", flush=True)
    httpd.serve_forever()
