"""The maths: find fixed overlays in many frames, measure a see-through one (seen = offset + keep * real), invert it."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import NamedTuple

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
HOLD = 4  # every this-many-th sampled frame is kept out of measuring, to choose the measurement that cleans it best
# luma levels (the smoothing's range): compression leaves ringing at the overlay's edges and inverting makes it larger;
# the least smoothing that leaves the outline no sharper than the picture around it is kept, on frames held out
SMOOTH = (0.0, 2.0, 4.0, 8.0, 16.0)
REACH = 7  # pixels across the window that smoothing averages over
FAINT = 1.0  # luma levels per pixel: an outline gradient this weak cannot be seen, so visibility counts it as none

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
STILL_THERE = ("the picture around the box barely changes during the video, so the overlay cannot be measured; it "
               "needs a video whose picture moves or changes behind it")
CLOSE = 0.02  # gaps up to this share of the frame's shorter side are bridged, so letters join into one region
# shortcut: an overlay on screen for less than about 70% of the video is not found, upgrade if real ones come and go
GROUPS = 8  # stretches of the video, at least, whose structure is compared; a scene's own is in few of them
AGREE = 0.7  # share of those stretches an overlay's outline must be sharp in, with the same sign


class Part(NamedTuple):
    """A mark in one plane: where it sits in the full frame, which of those pixels it covers, and their keep and
    offset (empty for a solid mark)."""
    where: tuple[slice, slice]
    sel: np.ndarray
    keep: np.ndarray
    offset: np.ndarray


@dataclass
class Mark:
    """A fixed overlay: its box (x, y, w, h, even-aligned luma pixels), kind ("see-through" or "solid") and peak
    opacity. Make one with find; a see-through one is removed with erase, a solid one must be filled in."""
    box: tuple[int, int, int, int]
    kind: str
    opacity: float
    parts: list[Part] = field(default_factory=list, repr=False)  # luma, then u and v (a solid mark: luma only)
    smooth: float = 0.0

    def erase(self, frame: Frame) -> None:
        """Clean one full-size frame in place, unless that would leave the overlay's outline sharper than it was (so
        the overlay is not in this frame); pixels outside the overlay are left exactly as they were."""
        if self.kind == "solid":
            raise RuntimeError("a solid mark hides the picture completely, so it is filled in, not erased")
        luma = self.parts[0].where
        real = self._clean(frame.y[luma].astype(np.float32))
        if real is None:
            return
        frame.y[luma] = real
        for plane, (where, sel, keep, offset) in zip((frame.u, frame.v), self.parts[1:]):
            p = plane[where]
            p[sel] = np.clip(np.rint((p[sel] - offset) / keep), 0, 255)
            if self.smooth:
                p[sel] = cv2.bilateralFilter(p, REACH, self.smooth, REACH / 2.5)[sel]

    def _clean(self, seen: np.ndarray) -> np.ndarray | None:
        """seen, the float32 luma around the overlay, cleaned; None when that would leave its outline sharper."""
        _, sel, keep, offset = self.parts[0]
        real = seen.copy()
        real[sel] = np.clip(np.rint((seen[sel] - offset) / keep), 0, 255)
        outline = _outline(self._core())[0]
        if _edges(real)[outline].mean() >= _edges(seen)[outline].mean():
            return None
        if self.smooth:
            real[sel] = cv2.bilateralFilter(real.astype(np.uint8), REACH, self.smooth, REACH / 2.5)[sel]
        return real

    def _core(self) -> np.ndarray:
        """The pixels the overlay dims noticeably; a mark may also correct the faint skirt around them."""
        _, sel, keep, _ = self.parts[0]
        core = sel.copy()
        if self.kind != "solid":
            core[sel] = keep < OVERLAY
        return core

    def refined(self, frames: Iterable[Frame]) -> Mark:
        """This see-through mark measured again on only those of frames it is in, the way that cleans best; a
        solid mark is returned as it is. Raises RuntimeError when nothing can be measured."""
        if self.kind == "solid":
            return self
        if not self.parts[0].sel.any():
            raise RuntimeError(NOTHING)
        luma, chroma = self.parts[0].where, self.parts[1].where
        # shortcut: the padded crops of every sampled frame are kept in memory (about 50 MB for a 420x70 box), upgrade
        # by keeping only a sample of them if very large boxes are given
        fit, held = [], []

        def split(frames: Iterable[Frame]) -> Iterator[Frame]:
            for i, f in enumerate(f for f in frames if self._clean(f.y[luma].astype(np.float32)) is not None):
                if i % HOLD:
                    fit.append(Frame(f.y[luma].copy(), f.u[chroma].copy(), f.v[chroma].copy()))
                else:
                    held.append(f.y[luma].astype(np.float32))
                yield f

        measured = _measure(*_stats(split(frames), NOTHING), self.box)
        if not held or not measured.parts[0].sel.any():
            return measured

        def score(m: Mark) -> float:  # on the same outline for both
            return measured._visibility(r if (r := m._clean(c)) is not None else c for c in held)

        best = min(measured, _fitted(measured, fit), key=score)  # unsmoothed, as smoothing evens them out
        smoothed = [replace(best, smooth=s) for s in SMOOTH]
        return next((m for m in smoothed if score(m) <= 1), smoothed[-1])

    def visibility(self, frames: Iterable[Frame]) -> float:
        """How much sharper the overlay's outline is than nearby picture, averaged over frames; 1.0 means it is gone."""
        return self._visibility(f.y[self.parts[0].where].astype(np.float32) for f in frames)

    def _visibility(self, lumas: Iterable[np.ndarray]) -> float:
        """visibility on the luma around the overlay. Gradients count from FAINT, so the faint noise of a smooth
        picture does not read as a visible outline."""
        outline, near = _outline(self._core())
        ratios = [(g[outline].mean() + FAINT) / (g[near].mean() + FAINT) for g in map(_edges, lumas)]
        return float(np.mean(ratios)) if ratios else 1.0


def _outline(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pixels of mask sel within 2 of its edge, and the pixels more than 2 away from it outside."""
    m, k = sel.astype(np.uint8), np.ones((5, 5), np.uint8)
    return sel & ~cv2.erode(m, k).astype(bool), ~cv2.dilate(m, k).astype(bool)


def _inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate img inward over mask from the surrounding pixels."""
    return cv2.inpaint(img.astype(np.float32), mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def _line(seen: np.ndarray, behind: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per column, keep and offset of the line seen = offset + keep * behind across rows, fitted again twice without
    the points far off it (frames where the picture behind was not interpolated well); with keep, only offset."""
    use = np.ones(seen.shape, bool)
    for _ in range(3):
        n = use.sum(0)
        mb, ms = (use * behind).sum(0) / n, (use * seen).sum(0) / n
        k = keep if keep is not None else (
            (use * (behind - mb) * (seen - ms)).sum(0) / np.maximum((use * (behind - mb) ** 2).sum(0), 1e-6))
        k = np.clip(k, FLOOR, 1)
        off = ms - k * mb
        miss = np.abs(seen - off - k * behind)
        use = miss <= 2 * np.median(miss, axis=0) + 1
    return k, off


def _fitted(mark: Mark, crops: list[Frame]) -> Mark:
    """mark measured again from the padded crops of frames: per pixel, a line fit of what is seen against the picture
    behind as interpolated from that frame's own surroundings. Unlike _measure it needs only a few different
    pictures behind the overlay, but they must be smooth enough to interpolate across it. It also covers the faint
    skirt just outside mark's pixels."""
    luma, overlay = mark.parts[0].where, mark.parts[0].sel
    k = np.ones((5, 5), np.uint8)
    sel = cv2.dilate(overlay.astype(np.uint8), k)
    unknown = cv2.dilate(sel, k).astype(bool)
    (ys, xs), (x, y, w, h) = luma, mark.box
    inside = np.zeros(sel.shape, bool)
    inside[y - ys.start:y - ys.start + h, x - xs.start:x - xs.start + w] = True
    sel = sel.astype(bool) & inside  # the skirt too, but nothing outside the box the user was promised changes
    keep, offset = _line(*(np.stack(v) for v in zip(*((s[sel], _inpaint(s, unknown)[sel])
                                                       for s in (c.y.astype(np.float32) for c in crops)))))
    full = np.ones(sel.shape, np.float32)
    full[sel] = keep
    ah, aw = sel.shape
    keep_c = full.reshape(ah // 2, 2, aw // 2, 2).mean(axis=(1, 3))
    sel_c, unknown_c = (m.reshape(ah // 2, 2, aw // 2, 2).any(axis=(1, 3)) for m in (sel, unknown))
    parts = [Part(luma, sel, keep, offset)]
    for chroma, planes in zip((p.where for p in mark.parts[1:]), ([c.u for c in crops], [c.v for c in crops])):
        seen, behind = zip(*((p[sel_c], _inpaint(p, unknown_c)[sel_c]) for p in (q.astype(np.float32) for q in planes)))
        parts.append(Part(chroma, sel_c, keep_c[sel_c], _line(np.stack(seen), np.stack(behind), keep_c[sel_c])[1]))
    return Mark(mark.box, mark.kind, mark.opacity, parts)


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


def _area(box: tuple[int, int, int, int], width: int, height: int) -> tuple[tuple, tuple, np.ndarray]:
    """The luma and chroma slices of box padded by PAD, and which pixels of that area are inside box."""
    x, y, w, h = box
    ax, ay = max(0, x - PAD), max(0, y - PAD)
    aw, ah = min(width & ~1, x + w + PAD) - ax, min(height & ~1, y + h + PAD) - ay
    inside = np.zeros((ah, aw), bool)
    inside[y - ay:y - ay + h, x - ax:x - ax + w] = True
    return np.s_[ay:ay + ah, ax:ax + aw], np.s_[ay // 2:(ay + ah) // 2, ax // 2:(ax + aw) // 2], inside


def _solid(box: tuple[int, int, int, int], var: np.ndarray, ring: float) -> Mark:
    """A solid mark; the pixels of its box that barely change are its outline for visibility."""
    luma, _, inside = _area(box, *var.shape[::-1])
    return Mark(box, "solid", 1.0, [Part(luma, inside & (var[luma] < SOLID * ring), np.empty(0), np.empty(0))])


def _measure(mean: list[np.ndarray], var: list[np.ndarray], box: tuple[int, int, int, int]) -> Mark:
    """Measure the see-through overlay inside box from full-frame temporal mean and variance."""
    luma, chroma, inside = _area(box, *mean[0].shape[::-1])
    ah, aw = inside.shape
    m0, v0 = mean[0][luma], np.maximum(var[0][luma], 1e-6)
    overlay = inside
    for _ in range(3):  # each later pass interpolates from background that sits closer to the overlay
        unknown = inside & cv2.dilate(overlay.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        keep = np.where(unknown, np.sqrt(v0 / np.maximum(_inpaint(v0, unknown), 1e-6)), 1.0).clip(0, 1)
        overlay = inside & (keep < OVERLAY)
    opacity = float(1 - np.percentile(keep[overlay], PEAK)) if overlay.any() else 0.0
    behind = _inpaint(m0, unknown)
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
    parts = [Part(luma, overlay, keep[overlay], offset[overlay])]
    parts += [Part(chroma, overlay_c, keep_c[overlay_c],
                   (m[chroma] - keep_c * _inpaint(m[chroma], unknown_c))[overlay_c]) for m in mean[1:]]
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


def _regions(chunks: list[list], width: int, height: int) -> list[tuple[int, int, int, int]]:
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
    # top to bottom, then left to right
    boxes = sorted(_regions(chunks, *mean[0].shape[::-1]), key=lambda b: (b[1], b[0]))
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
    if ring < STILL:
        if given:
            raise RuntimeError(STILL_THERE)
        return None
    if np.percentile(var[0][y:y + h, x:x + w], LOW) < SOLID * ring:
        return _solid(box, var[0], ring)
    return _measure(mean, var, box)
