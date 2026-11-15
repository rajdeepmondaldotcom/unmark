"""The maths: find fixed overlays from many frames, measure a see-through one (seen = offset + keep * real), invert it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

from unmark.media import Frame

PAD = 16  # luma pixels of plain background kept around the box to interpolate from
OVERLAY = 0.97  # keep below this counts as overlay; plain background measures within a percent or two of 1
FLOOR = 0.15  # shortcut: near-solid pixels are inverted with keep >= 0.15, upgrade when solid overlays get their own fill

MEAN_EDGE = 3.0  # luma levels per pixel: the averaged picture is smoother than this, a fixed overlay's edge is not
VAR_EDGE = 0.25  # natural-log units per pixel, the same test on the log of the temporal variance
BOX_PAD = 2  # luma pixels added around a found region
RING = 8  # luma pixels of surrounding picture compared with a region
STILL = 4.0  # temporal variance (levels squared) below which the ring is a still scene, so nothing can be measured
SOLID = 0.05  # overlay variance below this share of the ring's means the overlay hides the picture completely
LOW = 10  # percentile of a region's variance taken as its overlay pixels; overlays cover at least this share of it
LARGEST = 0.05  # regions bigger than this share of the frame are scenery, not an overlay
SPAN = 0.9  # regions this wide or tall relative to the frame are letterbox or pillarbox bars
SMALLEST = 8  # luma pixels; narrower or shorter regions are specks
CLOSE = 0.02  # gaps up to this share of the frame's shorter side are bridged, so letters join into one region


@dataclass
class Mark:
    """A fixed overlay: its box (x, y, w, h, even-aligned luma pixels), kind, peak opacity, and per plane where it
    sits in the full frame, which pixels it covers, and keep / offset (empty for a solid mark)."""
    box: tuple[int, int, int, int]
    kind: str
    opacity: float
    parts: list[tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)

    def erase(self, frame: Frame) -> None:
        """Clean one full-size frame in place; pixels outside the overlay are left exactly as they were."""
        if self.kind == "solid":
            raise RuntimeError("this mark is solid (it hides the picture completely), and removing solid marks is not "
                               "supported yet")
        for plane, (where, sel, keep, offset) in zip((frame.y, frame.u, frame.v), self.parts):
            p = plane[where]
            p[sel] = np.clip(np.rint((p[sel] - offset) / keep), 0, 255)


def _fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate img inward over mask from the surrounding pixels."""
    return cv2.inpaint(img.astype(np.float32), mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def _stats(frames: Iterable[Frame]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per plane, the temporal mean and variance of frames, from exact integer sums."""
    sums = sq = None
    n = 0
    for f in frames:
        planes = (f.y, f.u, f.v)
        if sums is None:
            sums = [np.zeros(p.shape, np.uint32) for p in planes]
            sq = [np.zeros(p.shape, np.uint32) for p in planes]  # exact up to 66,000 frames of 255 squared
        for s, q, p in zip(sums, sq, planes):
            s += p
            q += np.square(p, dtype=np.uint32)
        n += 1
    if n < 2:
        raise RuntimeError("the video has too few frames to measure the overlay")
    mean = [s / n for s in sums]
    return mean, [np.maximum(q / n - m * m, 0) for q, m in zip(sq, mean)]


def _measure(mean: list[np.ndarray], var: list[np.ndarray], box: tuple[int, int, int, int]) -> Mark:
    """Measure the see-through overlay inside box from full-frame temporal mean and variance."""
    height, width = mean[0].shape
    x, y, w, h = box
    ax, ay = max(0, x - PAD), max(0, y - PAD)
    aw, ah = min(width & ~1, x + w + PAD) - ax, min(height & ~1, y + h + PAD) - ay
    luma = np.s_[ay:ay + ah, ax:ax + aw]
    chroma = np.s_[ay // 2:(ay + ah) // 2, ax // 2:(ax + aw) // 2]
    m0, v0 = mean[0][luma], np.maximum(var[0][luma], 1e-6)

    inside = np.zeros((ah, aw), bool)
    inside[y - ay:y - ay + h, x - ax:x - ax + w] = True
    overlay = inside
    for _ in range(3):  # each later pass interpolates from background that sits closer to the overlay
        unknown = inside & cv2.dilate(overlay.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        keep = np.where(unknown, np.sqrt(v0 / np.maximum(_fill(v0, unknown), 1e-6)), 1.0).clip(0, 1)
        overlay = inside & (keep < OVERLAY)
    opacity = float(1 - keep[overlay].min()) if overlay.any() else 0.0
    keep = np.maximum(keep, FLOOR)
    offset = m0 - keep * _fill(m0, unknown)

    # chroma variance is small and noisy, so chroma reuses the luma keep and solves only its offset
    blocks = (ah // 2, 2, aw // 2, 2)
    keep_c = keep.reshape(blocks).mean(axis=(1, 3))
    overlay_c = overlay.reshape(blocks).any(axis=(1, 3))
    unknown_c = unknown.reshape(blocks).any(axis=(1, 3))
    parts = [(luma, overlay, keep[overlay], offset[overlay])]
    parts += [(chroma, overlay_c, keep_c[overlay_c], (m[chroma] - keep_c * _fill(m[chroma], unknown_c))[overlay_c])
              for m in mean[1:]]
    return Mark(box, "see-through", opacity, parts)


def _edges(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude per pixel."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1) / 8
    return np.hypot(gx, gy)


def _regions(mean: np.ndarray, var: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes around the sharp structure a fixed overlay leaves in the averaged picture or its variance."""
    height, width = mean.shape
    sharp = (_edges(mean.astype(np.float32)) > MEAN_EDGE) | (_edges(np.log(var + 1).astype(np.float32)) > VAR_EDGE)
    k = max(3, round(CLOSE * min(width, height))) | 1
    sharp = cv2.morphologyEx(sharp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(sharp)
    boxes = []
    for x, y, w, h, _ in stats[1:n]:
        if w * h > LARGEST * width * height or w >= SPAN * width or h >= SPAN * height or min(w, h) < SMALLEST:
            continue
        x0, y0 = max(0, x - BOX_PAD) & ~1, max(0, y - BOX_PAD) & ~1
        x1, y1 = min(width & ~1, x + w + BOX_PAD + 1 & ~1), min(height & ~1, y + h + BOX_PAD + 1 & ~1)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return boxes


def find(frames: Iterable[Frame], box: tuple[int, int, int, int] | None = None) -> list[Mark]:
    """Every fixed overlay in full-size frames, each measured; with box (even-aligned luma pixels), just that one."""
    mean, var = _stats(frames)
    if box:
        return [_measure(mean, var, box)]
    found = []
    for b in sorted(_regions(mean[0], var[0]), key=lambda b: (b[1], b[0])):  # top to bottom, then left to right
        x, y, w, h = b
        x0, y0 = max(0, x - RING), max(0, y - RING)
        around = var[0][y0:y + h + RING, x0:x + w + RING].copy()
        around[y - y0:y - y0 + h, x - x0:x - x0 + w] = np.nan
        ring = np.nanmedian(around)
        if ring < STILL:
            continue
        if np.percentile(var[0][y:y + h, x:x + w], LOW) < SOLID * ring:
            found.append(Mark(b, "solid", 1.0))
        else:
            found.append(_measure(mean, var, b))
    return found
