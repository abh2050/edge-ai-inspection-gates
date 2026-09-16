"""Serve the evidence dashboard on the local host with caching disabled.

A rebuilt data file must reach the browser immediately. Python's default handler sends no
cache directives, so a browser may reuse a stale data.json and render an older run.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "dashboard"


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Serve dashboard files and forbid every cache from holding a copy."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    """Bind the loopback interface only; the dashboard never serves beyond this host."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    handler = partial(NoCacheHandler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Evidence dashboard at http://127.0.0.1:{args.port}/ (ctrl-c to stop).")
        print("Caching is disabled, so a rebuilt data file reaches the browser on reload.")
        server.serve_forever()


if __name__ == "__main__":
    main()
