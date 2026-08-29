"""
telemetry.py -- CPU / GPU / memory sampling for the dashboard's right panel.

PROJECT.md section 7.1 wants "live system panel -- per-network active state,
CPU and GPU load, whether each module is running correctly".

Section 11 left GPU telemetry open: "No GPU backend is wired up; needs
torch.cuda or nvidia-smi on the target machine." Both are wired here, tried
in that order, and when neither answers the panel says so rather than showing
a fake zero.

Sampling runs on its own daemon thread. Reading CPU load blocks for a moment,
and the UI thread must never block -- same rule as the activation queue.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import config


@dataclass
class SystemSample:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = "none"
    gpu_percent: float | None = None       # None = backend cannot report it
    gpu_mem_percent: float | None = None
    gpu_source: str = "unavailable"
    # True when a CUDA GPU exists but torch cannot use it (CPU-only build).
    gpu_idle_capacity: bool = False
    threads: int = 0
    # False until the sampler has produced a real reading. The panel shows
    # "sampling..." rather than a zero that looks like a measured idle system.
    valid: bool = False
    timestamp: float = field(default_factory=time.time)

    def gpu_line(self, compact: bool = False) -> str:
        """One line of GPU state. `compact` fits the 26-char dashboard panel."""
        if not self.gpu_available:
            return "GPU     none" if compact else "GPU     none detected (CPU inference)"
        util = "  n/a" if self.gpu_percent is None else f"{self.gpu_percent:5.1f}%"
        mem = "  n/a" if self.gpu_mem_percent is None else f"{self.gpu_mem_percent:5.1f}%"
        if compact:
            return f"GPU   {util} mem {mem}"
        return f"GPU     {util}  mem {mem}  [{self.gpu_source}]"

    def gpu_note(self) -> str:
        if self.gpu_idle_capacity:
            return "GPU present but torch is a CPU-only build -- not in use"
        return ""


def _probe_nvidia_smi() -> tuple[str, float, float] | None:
    """(name, utilisation %, memory %) from nvidia-smi, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    if len(parts) < 4:
        return None
    try:
        name, util, used, total = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    return name, util, (used / total * 100.0 if total else 0.0)


class TelemetryMonitor:
    """Samples on a background thread; the UI reads the latest snapshot."""

    # nvidia-smi costs well over a second per call on some machines, which is
    # far too slow for the CPU/RAM cadence. It gets its own slow clock and the
    # result is cached in between.
    GPU_INTERVAL = 5.0

    def __init__(self, interval: float = config.UI_TELEMETRY_MS / 1000.0):
        self.interval = interval
        self._sample = SystemSample()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._torch_cuda = self._detect_torch_cuda()
        self._gpu_cache: tuple[str, float, float] | None = None
        self._gpu_checked_at = 0.0

    @staticmethod
    def _detect_torch_cuda() -> tuple[bool, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False, "none"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> SystemSample:
        with self._lock:
            return self._sample

    def _loop(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None  # type: ignore[assignment]

        process = psutil.Process() if psutil else None

        while not self._stop.is_set():
            sample = SystemSample(valid=True)

            if psutil is not None:
                # Blocking read, which is exactly why this is not on the UI thread.
                sample.cpu_percent = psutil.cpu_percent(interval=self.interval)
                memory = psutil.virtual_memory()
                sample.ram_percent = memory.percent
                sample.ram_used_gb = (memory.total - memory.available) / 1e9
                sample.ram_total_gb = memory.total / 1e9
                if process is not None:
                    sample.threads = process.num_threads()
            else:
                time.sleep(self.interval)

            has_cuda, cuda_name = self._torch_cuda

            now = time.time()
            if now - self._gpu_checked_at >= self.GPU_INTERVAL:
                self._gpu_cache = _probe_nvidia_smi()
                self._gpu_checked_at = now
            probe = self._gpu_cache

            if probe is not None:
                name, util, mem = probe
                sample.gpu_available = True
                sample.gpu_name = name
                sample.gpu_percent = util
                sample.gpu_mem_percent = mem
                sample.gpu_source = "nvidia-smi"
                # A CUDA card is here, but torch may still be a CPU-only build.
                # Say so rather than implying the GPU is doing the work.
                sample.gpu_idle_capacity = not has_cuda
            elif has_cuda:
                sample.gpu_available = True
                sample.gpu_name = cuda_name
                sample.gpu_source = "torch.cuda"
                try:
                    import torch

                    free, total = torch.cuda.mem_get_info()
                    sample.gpu_mem_percent = (1 - free / total) * 100.0
                except Exception:
                    pass
                # torch reports memory, never utilisation. Left as None so the
                # panel prints n/a instead of an invented number.

            with self._lock:
                self._sample = sample


__all__ = ["TelemetryMonitor", "SystemSample"]
