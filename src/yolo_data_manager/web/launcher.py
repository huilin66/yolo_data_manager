"""Standalone launcher for the YDM FastAPI backend and React frontend."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from urllib.error import URLError


class WebStartupError(RuntimeError):
    """Raised when the standalone web workspace cannot be started."""


def _find_npm() -> str | None:
    candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    return next((shutil.which(candidate) for candidate in candidates if shutil.which(candidate)), None)


def _ensure_port_free(host: str, port: int, service: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise WebStartupError(f"{service} port {host}:{port} is already in use") from exc


def _start_process(command: list[str], cwd: Path, name: str) -> subprocess.Popen:
    print(f"-> Starting {name}...")
    kwargs: dict[str, object] = {"cwd": str(cwd)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    return subprocess.Popen(command, **kwargs)


def _wait_for_http(url: str, process: subprocess.Popen, name: str, timeout: float = 30) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise WebStartupError(f"{name} exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    print(f"-> {name} is ready: {url}")
                    return
        except (OSError, URLError, TimeoutError):
            time.sleep(0.5)
    raise WebStartupError(f"{name} did not become ready within {timeout:g} seconds: {url}")


def _terminate(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"-> Stopping {name}...")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:  # noqa: BLE001 - shutdown should continue for either process
        process.kill()


def run_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8091,
    frontend_host: str = "127.0.0.1",
    frontend_port: int = 5174,
    open_browser: bool = False,
    api_only: bool = False,
) -> int:
    """Run YDM as a local standalone web application.

    The frontend and backend remain independent processes.  This makes the
    application useful by itself today and leaves a clean launch boundary for
    a future MultiAnno menu entry.
    """

    backend_process: subprocess.Popen | None = None
    frontend_process: subprocess.Popen | None = None
    frontend_dir = Path(__file__).parent / "frontend"
    backend_url = f"http://{host}:{port}"
    frontend_url = f"http://{frontend_host}:{frontend_port}"

    try:
        _ensure_port_free(host, port, "YDM API")
        backend_process = _start_process(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "yolo_data_manager.web.server:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            Path.cwd(),
            "YDM API (FastAPI)",
        )
        _wait_for_http(f"{backend_url}/api/health", backend_process, "YDM API")

        if api_only:
            print(f"YDM API is running at {backend_url}. Press Ctrl+C to stop.")
            while backend_process.poll() is None:
                time.sleep(0.5)
            return int(backend_process.returncode or 0)

        npm = _find_npm()
        if not npm:
            raise WebStartupError("npm was not found; install Node.js or use --api-only")
        if not (frontend_dir / "package.json").is_file():
            raise WebStartupError(f"YDM frontend package not found: {frontend_dir}")
        if not (frontend_dir / "node_modules").is_dir():
            raise WebStartupError(
                f"YDM frontend dependencies are not installed. Run 'npm install' in {frontend_dir}"
            )
        _ensure_port_free(frontend_host, frontend_port, "YDM frontend")
        frontend_process = _start_process(
            [
                npm,
                "run",
                "dev",
                "--",
                "--host",
                frontend_host,
                "--port",
                str(frontend_port),
                "--strictPort",
            ],
            frontend_dir,
            "YDM frontend (Vite)",
        )
        _wait_for_http(frontend_url, frontend_process, "YDM frontend")
        print(f"YDM is ready: {frontend_url}")
        if open_browser:
            webbrowser.open(frontend_url)
        while backend_process.poll() is None and frontend_process.poll() is None:
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate(frontend_process, "YDM frontend")
        _terminate(backend_process, "YDM API")
