"""The maths: find fixed overlays from many frames, measure a see-through one (seen = offset + keep * real), invert it."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from unmark.media import Frame

PAD = 16  # luma pixels of plain background kept around the box to interpolate from
OVERLAY = 0.97  # keep below this counts as overlay; plain background measures within a percent or two of 1
FLAT = 24.0  # luma levels: an overlay whose colour's interquartile spread is below this is one flat colour
APART = 32.0  # luma levels: the flat colour's opacity is solved from the means only this far from the picture behind
# shortcut: near-solid pixels of a see-through overlay are inverted with keep >= 0.15, upgrade by filling them instead
FLOOR = 0.15
PEAK = 10  # percentile of an overlay's keep taken as its peak; the very lowest few are noise or passing scenery

MEAN_EDGE = 3.0  # luma levels per half-size pixel: a stretch's averaged picture is smoother, a fixed overlay's edge not
VAR_EDGE = 0.25  # natural-log units per pixel, the same test on the log of the temporal variance
RING = 8  # luma pixels of surrounding picture compared with a region
STILL = 4.0  # temporal variance (levels squared) below which the ring is a still scene, so nothing can be measured
SOLID = 0.05  # overlay variance below this share of the ring's means the overlay hides the picture completely
LOW = 10  # percentile of a region's variance taken as its overlay pixels; overlays cover at least this share of it
LARGEST = 0.08  # regions bigger than this share of the frame are scenery, not an overlay
SPAN = 0.9  # regions this wide or tall relative to the frame are letterbox or pillarbox bars
SMALLEST = 0.02  # share of the frame's shorter side; narrower or shorter regions are specks
NOTHING = "nothing see-through or solid was measured there; check the box, or leave --box out to find overlays"
CLOSE = 0.02  # gaps up to this share of the frame's shorter side are bridged, so letters join into one region
# shortcut: an overlay on screen for less than about 70% of the video is not found, upgrade if real ones come and go
GROUPS = 8  # stretches of the video, at least, whose structure is compared; a scene's own is in few of them
AGREE = 0.7  # share of those stretches an overlay's outline must be sharp in, with the same sign


@dataclass
class Mark:
    """A fixed overlay: its box (x, y, w, h, even-aligned luma pixels), kind, peak opacity, and per plane where it
    sits in the full frame, which pixels it covers, and keep / offset (a solid mark: luma only, its unchanging
    pixels, empty keep / offset)."""
    box: tuple[int, int, int, int]
    kind: str
    opacity: float
    parts: list[tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)

    def erase(self, frame: Frame) -> None:
        """Clean one full-size frame in place, unless that would leave the overlay's outline sharper than it was (so
        the overlay is not in this frame); pixels outside the overlay are left exactly as they were."""
        if self.kind == "solid":
            raise RuntimeError("a solid mark hides the picture completely, so it is filled in, not erased")
        real = self._luma(frame)
        if real is None:
            return
        frame.y[self.parts[0][0]] = real
        for plane, (where, sel, keep, offset) in zip((frame.u, frame.v), self.parts[1:]):
            p = plane[where]
            p[sel] = np.clip(np.rint((p[sel] - offset) / keep), 0, 255)

    def _luma(self, frame: Frame) -> np.ndarray | None:
        """The cleaned luma around the overlay, or None when inverting would leave its outline sharper than it is."""
        where, sel, keep, offset = self.parts[0]
        seen = frame.y[where].astype(np.float32)
        real = seen.copy()
        real[sel] = np.clip(np.rint((seen[sel] - offset) / keep), 0, 255)
        outline = _outline(sel)[0]
        return real if _edges(real)[outline].mean() < _edges(seen)[outline].mean() else None

    def refined(self, frames: Iterable[Frame]) -> Mark:
        """This see-through mark measured again on only the frames it is in, so frames without it (an intro, say) do
        not water the measurement down; a solid mark is returned as it is."""
        if self.kind == "solid":
            return self
        if not self.parts[0][1].any():
            raise RuntimeError(NOTHING)
        return _measure(*_stats((f for f in frames if self._luma(f) is not None), NOTHING), self.box)

    def visibility(self, frames: Iterable[Frame]) -> float:
        """Average luma gradient on the overlay's outline over that on nearby picture; 1.0 means the outline is gone."""
        where, sel = self.parts[0][:2]
        outline, near = _outline(sel)
        on = off = 0.0
        for f in frames:
            g = _edges(f.y[where].astype(np.float32))
            on += g[outline].mean()
            off += g[near].mean()
        return on / max(off, 1e-6)


def _outline(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pixels of mask sel within 2 of its edge, and the pixels more than 2 away from it outside."""
    m, k = sel.astype(np.uint8), np.ones((5, 5), np.uint8)
    return sel & ~cv2.erode(m, k).astype(bool), ~cv2.dilate(m, k).astype(bool)


def _fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate img inward over mask from the surrounding pixels."""
    return cv2.inpaint(img.astype(np.float32), mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def _stats(frames: Iterable[Frame], few: str = "the video has too few frames to measure the overlay"
           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per plane, the temporal mean and variance of frames, from exact integer sums; fewer than two raise few."""
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
        raise RuntimeError(few)
    mean = [s / n for s in sums]
    return mean, [np.maximum(q / n - m * m, 0) for q, m in zip(sq, mean)]


def _area(box: tuple[int, int, int, int], height: int, width: int) -> tuple[tuple, tuple, np.ndarray]:
    """The luma and chroma slices of box padded by PAD, and which pixels of that area are inside box."""
    x, y, w, h = box
    ax, ay = max(0, x - PAD), max(0, y - PAD)
    aw, ah = min(width & ~1, x + w + PAD) - ax, min(height & ~1, y + h + PAD) - ay
    inside = np.zeros((ah, aw), bool)
    inside[y - ay:y - ay + h, x - ax:x - ax + w] = True
    return np.s_[ay:ay + ah, ax:ax + aw], np.s_[ay // 2:(ay + ah) // 2, ax // 2:(ax + aw) // 2], inside


def _solid(box: tuple[int, int, int, int], var: np.ndarray, ring: float) -> Mark:
    """A solid mark; the pixels of its box that barely change are its outline for visibility."""
    luma, _, inside = _area(box, *var.shape)
    return Mark(box, "solid", 1.0, [(luma, inside & (var[luma] < SOLID * ring), np.empty(0), np.empty(0))])


def _measure(mean: list[np.ndarray], var: list[np.ndarray], box: tuple[int, int, int, int]) -> Mark:
    """Measure the see-through overlay inside box from full-frame temporal mean and variance."""
    luma, chroma, inside = _area(box, *mean[0].shape)
    ah, aw = inside.shape
    m0, v0 = mean[0][luma], np.maximum(var[0][luma], 1e-6)
    overlay = inside
    for _ in range(3):  # each later pass interpolates from background that sits closer to the overlay
        unknown = inside & cv2.dilate(overlay.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        keep = np.where(unknown, np.sqrt(v0 / np.maximum(_fill(v0, unknown), 1e-6)), 1.0).clip(0, 1)
        overlay = inside & (keep < OVERLAY)
    opacity = float(1 - np.percentile(keep[overlay], PEAK)) if overlay.any() else 0.0
    behind = _fill(m0, unknown)
    core = overlay & (1 - keep > opacity / 2)
    if core.any():
        colour = (m0[core] - keep[core] * behind[core]) / (1 - keep[core])
        q1, c, q3 = np.percentile(colour, [25, 50, 75])
        if q3 - q1 < FLAT:  # one flat colour: re-solve opacity from the means, far less noisy than the variance ratio
            apart = np.abs(c - behind) >= APART  # elsewhere that division is mostly noise, so the ratio's keep stays
            keep = np.where(unknown & apart, 1 - ((m0 - behind) / np.where(apart, c - behind, 1)).clip(0, 1), keep)
            overlay = inside & (keep < OVERLAY)
            opacity = float(1 - np.percentile(keep[overlay], PEAK)) if overlay.any() else 0.0
    keep = np.maximum(keep, FLOOR)
    offset = m0 - keep * behind

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


def _grouped(frames: Iterable[Frame], chunks: list[list]) -> Iterator[Frame]:
    """Pass frames on, adding each one's half-size luma to chunks: [count, sum, sum of squares] for each of GROUPS to
    2 * GROUPS stretches of the video in order, all as long as each other but the last."""
    size = 1
    for f in frames:
        if not chunks or chunks[-1][0] == size:
            if len(chunks) == 2 * GROUPS:  # the number of frames is not known ahead, so neighbours merge as they come
                chunks[:] = [[a + b for a, b in zip(*pair)] for pair in zip(chunks[::2], chunks[1::2])]
                size *= 2
            h, w = f.y.shape[0] // 2, f.y.shape[1] // 2
            chunks.append([0, np.zeros((h, w), np.uint32), np.zeros((h, w), np.uint32)])
        y = cv2.resize(f.y, chunks[-1][1].shape[::-1], interpolation=cv2.INTER_AREA)
        chunks[-1][0] += 1
        chunks[-1][1] += y
        chunks[-1][2] += np.square(y, dtype=np.uint32)
        yield f


def _lasting_edges(maps: list[np.ndarray]) -> np.ndarray:
    """Per pixel, the gradient magnitude that at least AGREE of maps share, sign included, in x and in y."""
    parts = []
    for dx in (1, 0):
        g = np.empty((len(maps), *maps[0].shape), np.float32)  # filled and sorted in place to keep memory down
        for i, m in enumerate(maps):
            g[i] = cv2.Sobel(m, cv2.CV_32F, dx, 1 - dx) / 8
        g.sort(axis=0)
        low, high = g[int((1 - AGREE) * (len(g) - 1))], g[-1 - int((1 - AGREE) * (len(g) - 1))]
        parts.append(np.maximum(np.maximum(low, -high), 0))
    return np.hypot(*parts)


def _regions(chunks: list[list], height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Boxes (luma pixels of a height x width frame) around the sharp structure a fixed overlay leaves in the averaged
    picture or its variance in most stretches of the video; a scene's own structure is in only some of them."""
    sharp = _lasting_edges([(s / n).astype(np.float32) for n, s, _ in chunks]) > MEAN_EDGE
    sharp |= _lasting_edges([np.log(np.maximum(q / n - (s / n) ** 2, 0) + 1).astype(np.float32)
                             for n, s, q in chunks]) > VAR_EDGE
    h, w = sharp.shape
    k = max(3, round(CLOSE * min(w, h))) | 1
    sharp = cv2.morphologyEx(sharp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(sharp)
    boxes = []
    for x, y, bw, bh, _ in stats[1:n]:
        if bw * bh > LARGEST * w * h or bw >= SPAN * w or bh >= SPAN * h or min(bw, bh) < SMALLEST * min(w, h):
            continue
        # the gradient already reaches a half-size pixel (two luma pixels) past the outline, so no padding is added
        boxes.append((int(2 * x), int(2 * y), int(min(width & ~1, 2 * (x + bw)) - 2 * x),
                      int(min(height & ~1, 2 * (y + bh)) - 2 * y)))
    return boxes


def find(frames: Callable[[], Iterable[Frame]], box: tuple[int, int, int, int] | None = None) -> list[Mark]:
    """Every fixed overlay in full-size frames, each measured once (enough to list them); with box (even-aligned luma
    pixels), just that one, refined. frames() gives a fresh run of the same frames each call."""
    if box:
        return [_classify(*_stats(frames()), box, True).refined(frames())]
    chunks = []
    mean, var = _stats(_grouped(frames(), chunks))
    boxes = sorted(_regions(chunks, *mean[0].shape), key=lambda b: (b[1], b[0]))  # top to bottom, then left to right
    return [m for m in (_classify(mean, var, b, False) for b in boxes) if m]


def _classify(mean: list[np.ndarray], var: list[np.ndarray], box: tuple[int, int, int, int], given: bool
              ) -> Mark | None:
    """The overlay in box as measured on all frames, solid or see-through; None for a found (not given) box in a
    still part of the picture, where nothing can be measured."""
    x, y, w, h = box
    x0, y0 = max(0, x - RING), max(0, y - RING)
    around = var[0][y0:y + h + RING, x0:x + w + RING].copy()
    around[y - y0:y - y0 + h, x - x0:x - x0 + w] = np.nan
    ring = np.nanmedian(around)
    if ring < STILL and not given:
        return None
    if np.percentile(var[0][y:y + h, x:x + w], LOW) < SOLID * ring:
        return _solid(box, var[0], ring)
    return _measure(mean, var, box)
