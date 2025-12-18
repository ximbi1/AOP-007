from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_LLAMA_DIR = Path("/home/ximbi/MODELOSIA/llama.cpp/build")
DEFAULT_LLAMA_BINARY = DEFAULT_LLAMA_DIR / "bin" / "llama-server"
DEFAULT_MODEL_PATH = Path("~/MODELOSIA/models/granite/granite-4.0-h-1b-Q4_0.gguf").expanduser()
DEFAULT_ARGS = [
    "-m",
    str(DEFAULT_MODEL_PATH),
    "-t",
    "10",
    "-c",
    "25096",
    "-b",
    "512",
    "-ngl",
    "0",
    "--temp",
    "0.2",
    "--top_k",
    "80",
    "--top_p",
    "0.9",
    "--repeat_penalty",
    "1.05",
    "--mirostat",
    "0",
    "--port",
    "8080",
    "--host",
    "0.0.0.0",
]
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_LD_PATH = "/opt/intel/oneapi/mkl/2025.3/lib/intel64"


@dataclass
class BackendSettings:
    base_url: str = DEFAULT_BASE_URL
    llama_binary: Path = DEFAULT_LLAMA_BINARY
    llama_args: list[str] = None  # type: ignore[assignment]
    auto_start: bool = True
    ld_library_path: str = DEFAULT_LD_PATH

    def __post_init__(self) -> None:
        if self.llama_args is None:
            self.llama_args = list(DEFAULT_ARGS)


class BackendManager:
    """Starts and supervises llama-server for the agent."""

    def __init__(self, settings: Optional[BackendSettings] = None):
        self.settings = settings or BackendSettings()
        self.process: Optional[subprocess.Popen[str]] = None
        self._started_here = False

    def __enter__(self) -> "BackendManager":
        if not self._probe():
            if not self.settings.auto_start:
                raise RuntimeError(
                    "llama-server no está disponible y auto_start está deshabilitado"
                )
            self._start()
            self._started_here = True
            self._wait_until_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self._started_here and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    # ------------------------------------------------------------------
    def _start(self) -> None:
        binary = Path(self.settings.llama_binary)
        if not binary.exists():
            raise FileNotFoundError(f"No se encontró llama-server en {binary}")
        workdir = binary.parent.parent if binary.parent.name == "bin" else binary.parent
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH")
        if existing:
            env["LD_LIBRARY_PATH"] = f"{self.settings.ld_library_path}:{existing}"
        else:
            env["LD_LIBRARY_PATH"] = self.settings.ld_library_path
        cmd = [str(binary), *self.settings.llama_args]
        log_dir = Path(os.path.expanduser("~/.cache/opencode"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llama_server.log"
        self.process = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=log_path.open("w", encoding="utf-8", buffering=1),
            stderr=subprocess.STDOUT,
            env=env,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 45
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            ready, last_error = self._probe(return_error=True)
            if ready:
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"llama-server no respondió en {self.settings.base_url} ({last_error})."
        )

    def _probe(self, return_error: bool = False) -> bool | tuple[bool, Optional[str]]:
        url = f"{self.settings.base_url.rstrip('/')}/health"
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status < 500:
                    return (True, None) if return_error else True
        except Exception as exc:  # noqa: BLE001
            if return_error:
                return False, str(exc)
            return False
        return (False, "status >=500") if return_error else False


def call_llama(base_url: str, payload: Dict[str, object], timeout: int = 120) -> Dict[str, object]:
    import urllib.request

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
        return json.loads(body)
