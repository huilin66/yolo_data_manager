from __future__ import annotations

import argparse

from yolo_data_manager.web.launcher import run_web


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m yolo_data_manager.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=5174)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--api-only", action="store_true")
    args = parser.parse_args()
    return run_web(
        host=args.host,
        port=args.port,
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        open_browser=args.open_browser,
        api_only=args.api_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
