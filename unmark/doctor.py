"""`unmark doctor [--gpu]`: check, one line each, that what clean needs works on this computer."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from unmark import fill, media

TORCH = """import torch
if not torch.cuda.is_available():
    raise SystemExit(f"torch {torch.__version__} sees no NVIDIA card (its CUDA version: {torch.version.cuda})")
card = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} sees {card.name}, {card.total_memory / 2**30:.1f} GB")"""


def _tail(e: Exception) -> str:
    """The last few lines of an error, on one line."""
    return " | ".join(str(e).strip().splitlines()[-3:])


def _sample(n: int = 60, w: int = 256, h: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n RGB frames of soft colour waves drifting 2 pixels a frame, the same with a solid block, and the block."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = np.stack([np.stack([128 + 60 * np.sin((xx + 2 * i) / (30 + 10 * c) + c) * np.cos(yy / (25 + 8 * c))
                                for c in range(3)], axis=-1) for i in range(n)]).round().astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[h // 2 - 16:h // 2 + 16, w // 2 - 24:w // 2 + 24] = True
    seen = clean.copy()
    seen[:, mask] = 230
    return clean, seen, mask


def _gpu(checkout: Path, python: str, report: Callable[[bool | None, str, str], None]) -> None:
    """Ask the checkout's torch about the card, then time ProPainter on the sample, watching the card's memory."""
    p = subprocess.run([python, "-c", TORCH], check=False, cwd=checkout, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    said = (p.stdout + p.stderr).strip()
    if "No module named 'torch'" in said:
        report(False, "graphics card", f"torch is not installed in {python}, so ProPainter cannot run")
    else:
        report(p.returncode == 0, "graphics card", said.splitlines()[-1] if said else f"no answer ({p.returncode})")

    smi, peak, stop = shutil.which("nvidia-smi"), [], threading.Event()

    def watch() -> None:
        while True:  # one reading straight away, so even a short run has one
            try:
                q = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   check=False, capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                return
            peak.extend(int(v) for v in q.stdout.split() if v.isdigit())
            if stop.wait(0.2):
                return

    watcher = threading.Thread(target=watch, daemon=True)
    if smi:
        watcher.start()
    clean, seen, mask = _sample()
    start = time.perf_counter()
    try:
        out = np.stack(list(fill.fill(iter(seen), mask)))
    except RuntimeError as e:
        report(False, "ProPainter sample", _tail(e))
        return
    finally:
        stop.set()
        if smi:
            watcher.join()
    took = time.perf_counter() - start
    mse = np.mean((out[:, mask].astype(np.float64) - clean[:, mask]) ** 2)
    memory = f", card memory in use peaked at {max(peak)} MiB (all programs)" if peak else ""
    report(True, "ProPainter sample", f"{len(out)} frames of {mask.shape[1]}x{mask.shape[0]} filled in {took:.1f} s, "
                                      f"{10 * np.log10(255 ** 2 / max(mse, 1e-10)):.1f} dB inside the block{memory}")


def doctor(gpu: bool) -> int:
    """Run every check, printing one line each. Returns the exit code."""
    failures = 0

    def report(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal failures
        failures += ok is False
        print(f"[{'ok' if ok else 'FAIL' if ok is False else 'warn'}] {label}{': ' + detail if detail else ''}")

    try:
        for name in ("ffmpeg", "ffprobe"):
            path, version = media.version(name)
            report(True, name, f"{path} ({version})")
        filters = media.listed("filters")
        missing = [f for f in media.FILTERS if f not in filters]
        report(not missing, "filters", "missing " + ", ".join(missing) if missing else ", ".join(media.FILTERS))
        encoders = media.listed("encoders")
        name, why = media.encoder()  # libx264 is required either way: explicit crf and the fallback both use it
        report("libx264" in encoders, "encoder",
               (f"{name} ({why})" if why else name) if "libx264" in encoders else "libx264 is missing")
        with tempfile.TemporaryDirectory(prefix="unmark-doctor-") as tmp:
            sample, out = Path(tmp) / "sample.mp4", Path(tmp) / "out.mp4"
            media.sample_clip(sample)
            start = time.perf_counter()
            try:
                # every decode and encode clean uses: sampled, cropped, scene cuts, and the full patch
                list(media.planes(sample, 10))
                list(media.crops(sample, (16, 16, 64, 32)))
                media.cuts(sample)
                n = media.patch(sample, out, lambda frame: None)
                report(True, "end-to-end run", f"a {n}-frame test clip went through every ffmpeg command unmark uses "
                                               f"in {time.perf_counter() - start:.1f} s")
            except RuntimeError as e:
                report(False, "end-to-end run", _tail(e))
    except RuntimeError as e:
        report(False, "ffmpeg", _tail(e))

    checkout = os.environ.get("UNMARK_PROPAINTER")
    if not checkout:
        report(None, "ProPainter", "UNMARK_PROPAINTER is not set, so solid marks will use the softer simple fill")
        if gpu:
            report(False, "graphics card", "--gpu needs ProPainter; set UNMARK_PROPAINTER to its folder (on Windows: "
                                           "setx UNMARK_PROPAINTER C:\\ProPainter, then open a new window)")
    else:
        try:
            python = fill.checkout_python(Path(checkout))
            report(True, "ProPainter", f"{checkout}, run by {python}")
        except RuntimeError as e:
            report(False, "ProPainter", str(e))
        else:
            if gpu:
                _gpu(Path(checkout), python, report)
    print("doctor: " + ("all required checks passed" if not failures else f"{failures} check(s) failed"))
    return 1 if failures else 0
