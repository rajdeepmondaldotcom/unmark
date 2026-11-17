"""Synthetic clips with a known clean truth, for end-to-end tests. Uses ffmpeg directly, never unmark itself."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# whether this computer has the hardware encoder unmark prefers: an Apple-silicon Mac whose ffmpeg lists it
HEVC = sys.platform == "darwin" and platform.machine() == "arm64" and "hevc_videotoolbox" in subprocess.run(
    ["ffmpeg", "-hide_banner", "-encoders"], check=False, capture_output=True, text=True).stdout


def failing_ffmpeg(folder: Path) -> Path:
    """A stand-in ffmpeg (a shell script) in folder that fails whenever hevc_videotoolbox is asked for, like a busy or
    broken hardware encoder, and runs the real ffmpeg otherwise."""
    fake = folder / "ffmpeg"
    fake.write_text(f'#!/bin/sh\ncase "$*" in *hevc_videotoolbox*) echo "no encoder here" >&2; exit 1;; esac\n'
                    f'exec "{shutil.which("ffmpeg")}" "$@"\n')
    fake.chmod(0o755)
    return fake


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


def footage(n: int = 300, w: int = 320, h: int = 180, scenes: int = 5, seed: int = 0) -> np.ndarray:
    """Temporally coherent RGB footage (n, h, w, 3) uint8: scenes with hard cuts, each panning or zooming over a
    detailed texture while its brightness drifts, and a different brightness in each scene."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, h, w, 3), np.uint8)
    for s, idx in enumerate(np.array_split(np.arange(n), scenes)):
        tw, th = 2 * w, 2 * h
        texture = np.zeros((th, tw, 3), np.float32)
        for cell, amp, planes in ((64, 25, 3), (16, 20, 1), (3, 10, 1)):  # colour patches, then grey detail and grain
            coarse = rng.normal(0, 1, (th // cell + 1, tw // cell + 1, planes)).astype(np.float32)
            texture += cv2.resize(coarse, (tw, th), interpolation=cv2.INTER_CUBIC).reshape(th, tw, -1) * amp
        for _ in range(60):  # hard-edged objects
            x0, y0 = int(rng.integers(0, tw)), int(rng.integers(0, th))
            size, shade = int(rng.integers(4, 40)), tuple(float(c) for c in rng.normal(0, 40, 3))
            if rng.random() < 0.5:
                cv2.rectangle(texture, (x0, y0), (x0 + size, y0 + int(rng.integers(2, 30))), shade, -1)
            else:
                cv2.circle(texture, (x0, y0), size // 2, shade, -1, cv2.LINE_AA)
        start, end = rng.uniform(50, 180, 2)  # brightness at the start and end of the scene
        zoom = s % 2 == 1
        vx, vy = rng.uniform(-3, 3, 2)
        for k, i in enumerate(idx):
            t = k / max(1, len(idx) - 1)
            if zoom:
                m = cv2.getRotationMatrix2D((tw / 2, th / 2), 0, 1 + 0.5 * t)
                m[:, 2] -= (tw - w) / 2, (th - h) / 2
            else:
                m = np.float32([[1, 0, -w / 2 - vx * k], [0, 1, -h / 2 - vy * k]])
            view = cv2.warpAffine(texture, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out[i] = np.clip(view + start + (end - start) * t, 0, 255).round().astype(np.uint8)
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


def encode(frames: np.ndarray, path: Path, fps: int = 30, audio: bool = True, crf: int | None = None) -> Path:
    """Encode RGB frames to yuv420p H.264, losslessly unless crf is given, optionally with a sine audio track."""
    n, h, w, _ = frames.shape
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={n / fps}", "-c:a", "aac"]
    quality = ["-qp", "0"] if crf is None else ["-crf", str(crf), "-preset", "medium"]
    args += ["-c:v", "libx264", *quality, "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, input=frames.tobytes(), check=True)
    return Path(path)


def decode(path: Path, w: int, h: int) -> np.ndarray:
    """Every frame as raw yuv420p bytes, (n, w * h * 3 // 2) uint8 for even w and h."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, w * h * 3 // 2)


def decode_luma(path: Path, w: int, h: int) -> np.ndarray:
    """Every frame's luma plane as (n, h, w) uint8."""
    return decode(path, w, h)[:, :w * h].reshape(-1, h, w)


def has_audio(path: Path) -> bool:
    """True when the file has at least one audio stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for 8-bit data."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def smooth(n: int, w: int = 320, h: int = 180, seed: int = 0) -> np.ndarray:
    """n RGB frames (n, h, w, 3) uint8 of soft colour waves drifting about a pixel per frame, their look set by seed."""
    rng = np.random.default_rng(seed)
    level, periods, phases = rng.uniform(70, 180), rng.uniform(30, 90, (3, 2)), rng.uniform(0, 6, 3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, h, w, 3), np.uint8)
    for i in range(n):
        for c, ((fx, fy), p) in enumerate(zip(periods, phases)):
            wave = np.sin((xx + i) / fx + p) + np.sin((yy + i / 2) / fy)  # no still rows or columns
            out[i, ..., c] = np.clip(level + 35 * wave, 0, 255).round()
    return out


def still(n: int, w: int = 320, h: int = 180, scenes: int = 4, seed: int = 0) -> np.ndarray:
    """n RGB frames (n, h, w, 3) uint8 in scenes with hard cuts, each one smooth colour gradient, nearly still: it
    drifts by a few levels and a few pixels over the whole scene, like a sky, a wall or a title card."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, h, w, 3), np.uint8)
    for idx in np.array_split(np.arange(n), scenes):
        level, slope, angle = rng.uniform(40, 200, 3), rng.uniform(-40, 40, 3), rng.uniform(0, np.pi)
        for k, i in enumerate(idx):
            t = k / max(1, len(idx) - 1)
            ramp = ((xx + 4 * t) * np.cos(angle) + yy * np.sin(angle)) / max(w, h)
            out[i] = np.clip(level + 4 * t + slope * ramp[..., None], 0, 255).round()
    return out
