"""Synthetic clips with a known clean truth, for end-to-end tests. Uses ffmpeg directly, never unmark itself."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def clean_frames(n: int = 150, w: int = 320, h: int = 180, seed: int = 0) -> np.ndarray:
    """n independent smooth RGB pictures (n, h, w, 3) uint8 whose brightness and colours vary a lot between frames."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, h, w, 3), np.uint8)
    for i in range(n):
        coarse = rng.normal(0, 1, (h // 20, w // 20, 3)).astype(np.float32)
        detail = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC) * 30
        level = rng.uniform(40, 190) + rng.normal(0, 20, 3)
        out[i] = np.clip(detail + level, 0, 255).round().astype(np.uint8)
    return out


def text_opacity(w: int, h: int, box: tuple[int, int, int, int], text: str = "UM", peak: float = 0.45) -> np.ndarray:
    """A (h, w) float opacity map: thick soft-edged letters filling box (x, y, bw, bh), zero elsewhere."""
    x, y, bw, bh = box
    ink = np.zeros((bh, bw), np.uint8)
    cv2.putText(ink, text, (4, bh - 8), cv2.FONT_HERSHEY_SIMPLEX, bh / 32, 255, max(2, bh // 8), cv2.LINE_AA)
    ink = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.0)
    full = np.zeros((h, w), np.float32)
    full[y:y + bh, x:x + bw] = ink / ink.max() * peak
    return full


def stamp(frames: np.ndarray, opacity: np.ndarray, colour: float = 200.0) -> np.ndarray:
    """Blend a see-through overlay the way an editor does: seen = opacity * colour + (1 - opacity) * real."""
    a = opacity[None, :, :, None]
    return np.clip(a * colour + (1 - a) * frames, 0, 255).round().astype(np.uint8)


def encode(frames: np.ndarray, path: Path, fps: int = 30, audio: bool = True) -> Path:
    """Encode RGB frames losslessly to yuv420p H.264, optionally with a sine audio track."""
    n, h, w, _ = frames.shape
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={n / fps}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, input=frames.tobytes(), check=True)
    return Path(path)


def decode_luma(path: Path, w: int, h: int) -> np.ndarray:
    """Every frame's luma plane as (n, h, w) uint8."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    size = w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)
    return np.frombuffer(raw, np.uint8).reshape(-1, size)[:, :w * h].reshape(-1, h, w)


def has_audio(path: Path) -> bool:
    """True when the file has at least one audio stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for 8-bit data."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
