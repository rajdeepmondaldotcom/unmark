"""The maths: measure a see-through overlay (seen = offset + keep * real) from many frames, then invert it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from unmark.media import Frame

PAD = 16  # luma pixels of plain background kept around the box to interpolate from
OVERLAY = 0.97  # keep below this counts as overlay; plain background measures within a percent or two of 1
FLOOR = 0.15  # shortcut: near-solid pixels are inverted with keep >= 0.15, upgrade when solid overlays get their own fill


def region(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """The even-aligned padded area (x, y, w, h) around box whose frames find() needs."""
    x, y, w, h = box
    x0, y0 = max(0, x - PAD), max(0, y - PAD)
    x1, y1 = min(width & ~1, x + w + PAD), min(height & ~1, y + h + PAD)
    return x0, y0, x1 - x0, y1 - y0


@dataclass
class Mark:
    """A measured overlay: per plane, where it sits in the full frame, which pixels it covers, and keep / offset."""
    parts: list[tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]]

    def erase(self, frame: Frame) -> None:
        """Clean one full-size frame in place; pixels outside the overlay are left exactly as they were."""
        for plane, (where, sel, keep, offset) in zip((frame.y, frame.u, frame.v), self.parts):
            p = plane[where]
            p[sel] = np.clip(np.rint((p[sel] - offset) / keep), 0, 255)


def _fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate img inward over mask from the surrounding pixels."""
    return cv2.inpaint(img.astype(np.float32), mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def find(frames: Iterable[Frame], box: tuple[int, int, int, int], area: tuple[int, int, int, int]) -> list[Mark]:
    """Measure the overlay inside box from frames cropped to area (from region()); both in full-frame luma pixels."""
    sums = sq = None
    n = 0
    for f in frames:
        planes = [p.astype(np.float64) for p in (f.y, f.u, f.v)]
        sums = planes if sums is None else [s + p for s, p in zip(sums, planes)]
        sq = [p * p for p in planes] if sq is None else [s + p * p for s, p in zip(sq, planes)]
        n += 1
    if n < 2:
        raise RuntimeError("the video has too few frames to measure the overlay")
    mean = [s / n for s in sums]
    var = [np.maximum(q / n - m * m, 1e-6) for q, m in zip(sq, mean)]

    x, y, w, h = box
    ax, ay, aw, ah = area
    inside = np.zeros((ah, aw), bool)
    inside[y - ay:y - ay + h, x - ax:x - ax + w] = True
    overlay = inside
    for _ in range(3):  # each later pass interpolates from background that sits closer to the overlay
        unknown = inside & cv2.dilate(overlay.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        keep = np.where(unknown, np.sqrt(var[0] / np.maximum(_fill(var[0], unknown), 1e-6)), 1.0).clip(0, 1)
        overlay = inside & (keep < OVERLAY)
    keep = np.maximum(keep, FLOOR)
    offset = mean[0] - keep * _fill(mean[0], unknown)

    # chroma variance is small and noisy, so chroma reuses the luma keep and solves only its offset
    blocks = (ah // 2, 2, aw // 2, 2)
    keep_c = keep.reshape(blocks).mean(axis=(1, 3))
    overlay_c = overlay.reshape(blocks).any(axis=(1, 3))
    unknown_c = unknown.reshape(blocks).any(axis=(1, 3))
    luma = (np.s_[ay:ay + ah, ax:ax + aw], overlay, keep[overlay], offset[overlay])
    chroma = np.s_[ay // 2:(ay + ah) // 2, ax // 2:(ax + aw) // 2]
    parts = [luma] + [(chroma, overlay_c, keep_c[overlay_c], (m - keep_c * _fill(m, unknown_c))[overlay_c])
                      for m in mean[1:]]
    return [Mark(parts)]
