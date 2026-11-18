"""The maths: find fixed overlays in many frames, measure a see-through one (seen = offset + keep * real), invert it."""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
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
# robust spreads (1.4826 x the median absolute deviation) by which a frame's mark on the stray pixels, against the
# pixels beside them, must stand out to be measured without: on the real video's 1,000 sampled frames, the 7 with the
# game's map screen across the overlay sat 45 to 56 spreads out and every other frame -22 to +13
PRINTED = 30.0
FAINT = 1.0  # luma levels per pixel: an outline gradient this weak cannot be seen, so visibility counts it as none
# Compression leaves errors of a few levels at and inside the overlay's letters, different in every frame, which undoing
# it keeps. The part shaped like the overlay's edges softened is fitted and taken out; the rest is judged by comparing
# the undone letters with the same letter shape moved onto nearby picture without the overlay (the yardstick): coarse
# and fine detail are each kept where the letters are no rougher than the yardstick and filled from around where rougher
RIM = 2  # luma pixels around the overlay that compression errors spill into, cleaned too and never predicted from
SPREAD = 6.0  # luma pixels: the spread of the average of the surrounding picture carried across the letters
WIDE = 15.0  # luma pixels: a wider average, for inside closed letters (an "e", an "A") where SPREAD finds nothing
WIDE_SHARE = 0.03  # its weight against SPREAD's, so it only counts where SPREAD has nothing to average
SOFT = 1.5  # luma pixels: the blur compression softens the overlay's edges by; how much it does is fitted locally
SOFTEST = 2.0  # most of that softening fitted, so real edges crossing the letters cannot drive it far
SURE = 12.0  # luma levels: compression errors stay within this; larger differences are real picture and are kept
NEAR = 12.0  # luma pixels: the spread over which the softening is fitted and roughness compared
FINE = 2.5  # luma pixels: detail finer than this is judged apart from the coarser, and holds the overlay's edges
# levels squared: the letters' roughness above the yardstick's that can be compression error, coarse and fine; more is
# real picture (an interface bar or menu text crossing the letters), which is kept
NOISE = (16.0, 9.0)
CALM = (0.25, 0.01)  # levels squared added to both roughnesses, coarse and fine, so near-flat pictures compare as alike
# share of the letters' roughness that is real (not compression error), coarse and fine: at or below the first they
# are filled from around, at or above the second kept, blended between
FILL = ((0.3, 0.8), (0.1, 0.5))
PRESENT = 0.5  # share of its full strength the overlay must read at in a frame to be there and be undone
WEAK = 2.0  # luma levels: an overlay adding less than this on average cannot be read, so it is assumed to be there
# The real video's overlay dims colour more than brightness, by an amount that changes from scene to scene (probably
# it was mixed in before the picture's colours were mapped), so each frame's colour strength is read from the frame
# itself: the overlay's pattern in what is seen against what the measured overlay should add, both between these
# spreads
STROKE = (1.0, 5.0)  # luma pixels: finer is noise, coarser is picture the fill from around gets wrong
# colour levels squared per colour pixel of the overlay: how strong the pattern must read before a frame's own
# strength outweighs the measured one; weaker frames (little colour behind) read off by up to half, so this is the
# least that left the measured strength's cleaning no worse on known-right clips of the real video
SHRINK = 1.15
STRONGEST = 2.0  # most times the measured colour strength a frame can read, and its inverse the least

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
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # per frame size, see _plan

    def erase(self, frame: Frame) -> None:
        """Clean one full-size frame in place, unless the overlay is not in it; pixels outside the box are left
        exactly as they were."""
        if self.kind == "solid":
            raise RuntimeError("a solid mark hides the picture completely, so it is filled in, not erased")
        plans = self._planned(frame)
        seen, base = _behind(frame.y, plans[0])
        if not _present(seen, base, plans[0]):
            return
        _clean(frame.y, seen, base, plans[0])
        for plane, plan in zip((frame.u, frame.v), plans[1:]):
            seen, base = _behind(plane, plan)
            _clean(plane, seen, base, _strength(seen, base, plan))

    def _planned(self, frame: Frame) -> list[Plan]:
        """The parts laid out for frame's size, worked out once per size."""
        if frame.y.shape not in self._plans:
            self._plans[frame.y.shape] = [_plan(p, self.box, plane.shape, i > 0)
                                          for i, (p, plane) in enumerate(zip(self.parts, (frame.y, frame.u, frame.v)))]
        return self._plans[frame.y.shape]

    def save(self, path: Path, notes: dict) -> None:
        """Write this mark and notes (plain values) to path as .npz, all at once."""
        arrays = {"box": np.array(self.box), "kind": np.array(self.kind), "opacity": np.array(self.opacity),
                  "notes": np.array(json.dumps(notes))}
        for i, p in enumerate(self.parts):
            arrays |= {f"where{i}": np.array([(s.start, s.stop) for s in p.where]), f"sel{i}": p.sel,
                       f"keep{i}": p.keep, f"offset{i}": p.offset}
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            tmp.replace(path)  # a stop while writing never leaves half a file
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def load(path: Path) -> tuple[Mark, dict] | None:
        """The mark and notes save wrote to path; None when there is none or it cannot be read."""
        try:
            with np.load(path, allow_pickle=False) as f:  # never pickle: a changed file could run code
                parts = [Part(tuple(slice(int(a), int(b)) for a, b in f[f"where{i}"]), f[f"sel{i}"], f[f"keep{i}"],
                              f[f"offset{i}"]) for i in range(sum(k.startswith("sel") for k in f.files))]
                return (Mark(tuple(int(v) for v in f["box"]), str(f["kind"]), float(f["opacity"]), parts),
                        json.loads(str(f["notes"])))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return None

    def fingerprint(self) -> bytes:
        """Bytes that differ whenever anything erase reads from this mark does."""
        return (repr((self.box, self.kind, [p.where for p in self.parts])).encode()
                + b"".join(a.tobytes() for p in self.parts for a in p[1:]))

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
        (ys, xs), chroma = self.parts[0].where, self.parts[1].where
        # shortcut: the padded crops of every sampled frame are kept in memory (about 70 MB for a 420x70 box), to
        # measure again without the frames that print something else in; upgrade by keeping only a sample of them if
        # very large boxes are given
        fit, held, rows = [], [], []

        def split(frames: Iterable[Frame]) -> Iterator[Frame]:
            i = 0
            for f in frames:
                plan = self._planned(f)[0]
                if not _present(*_behind(f.y, plan), plan):
                    continue
                if i % HOLD:
                    fit.append(Frame(f.y[ys, xs].copy(), f.u[chroma].copy(), f.v[chroma].copy()))
                else:  # with room above and below for the yardstick
                    rows[:] = max(0, 2 * ys.start - ys.stop), min(f.y.shape[0] & ~1, 2 * ys.stop - ys.start)
                    held.append(Frame(f.y[rows[0]:rows[1], xs].copy(), f.u[chroma].copy(), f.v[chroma].copy()))
                i += 1
                yield f

        mean, var = _stats(split(frames), NOTHING)
        measured = _measure(mean, var, self.box)
        if not measured.parts[0].sel.any():
            raise RuntimeError(NOTHING)
        if not held:
            return measured

        def best(measured: Mark, fit: list[Frame], held: list[Frame]) -> tuple[Mark, np.ndarray]:
            """Whichever of measured and its line fit cleans held best, and measured's stray pixels: overlay the
            line fit reads as not dimmed."""
            fitted = _fitted(measured, fit)
            keep = np.ones(measured.parts[0].sel.shape, np.float32)
            keep[fitted.parts[0].sel] = fitted.parts[0].keep

            def score(m: Mark) -> float:  # on the same outline for both
                (my, mx), *rest = m.parts[0]
                part = Part(np.s_[my.start - rows[0]:my.stop - rows[0], mx.start - xs.start:mx.stop - xs.start], *rest)
                x, y, w, h = m.box
                plan = _plan(part, (x - xs.start, y - rows[0], w, h), (rows[1] - rows[0], xs.stop - xs.start), False)

                def cleaned(crop: Frame) -> np.ndarray:
                    crop = crop.y.copy()
                    _clean(crop, *_behind(crop, plan), plan)
                    return crop[part.where].astype(np.float32)

                return measured._visibility(map(cleaned, held))

            return min(measured, fitted, key=score), measured.parts[0].sel & (keep >= OVERLAY)

        chosen, stray = best(measured, fit, held)
        area = slice(ys.start - rows[0], ys.stop - rows[0])  # the held crops' rows that are the padded area
        crops = fit + [Frame(h.y[area], h.u, h.v) for h in held]
        out = _printed(measured.parts[0].sel, stray, mean[0][ys, xs], [c.y for c in crops])
        if not out.any():
            return chosen
        # measure again without them: their stats replace those of the padded area, the only part _measure reads
        kept_mean, kept_var = _stats(c for c, o in zip(crops, out) if not o)
        for full, part, where in zip(mean + var, kept_mean + kept_var, ((ys, xs), chroma, chroma) * 2):
            full[where] = part
        again = _measure(mean, var, self.box)
        if not again.parts[0].sel.any():
            return chosen
        second, fewer = best(again, [c for c, o in zip(fit, out) if not o],
                             [h for h, o in zip(held, out[len(fit):]) if not o])
        return second if fewer.sum() < stray.sum() else chosen

    def visibility(self, frames: Iterable[Frame]) -> float:
        """How much sharper the overlay's outline is than nearby picture, averaged over frames; 1.0 means it is gone."""
        return self._visibility(f.y[self.parts[0].where].astype(np.float32) for f in frames)

    def _visibility(self, lumas: Iterable[np.ndarray]) -> float:
        """visibility on the luma around the overlay. Gradients count from FAINT, so the faint noise of a smooth
        picture does not read as a visible outline."""
        outline, near = _outline(self._core())
        ratios = [(g[outline].mean() + FAINT) / (g[near].mean() + FAINT) for g in map(_edges, lumas)]
        return float(np.mean(ratios)) if ratios else 1.0


class Plan(NamedTuple):
    """A see-through mark's part laid out for one plane size, as maps over a window of the plane around it."""
    win: tuple[slice, slice]  # the part's padded area, made taller to hold the yardstick
    sel: np.ndarray  # the overlay's pixels
    offset: np.ndarray  # its offset, 0 elsewhere
    dim: np.ndarray  # 1 - keep, 0 elsewhere
    gain: np.ndarray  # 1 / keep, 1 elsewhere
    write: np.ndarray  # the pixels changed: the overlay and its rim, inside the box
    known: np.ndarray  # 1 on the pixels the picture behind is predicted from: neither letters nor yardstick
    norm: np.ndarray  # what that prediction is normalised by
    letters: np.ndarray  # 1 on the overlay and its rim
    yard: np.ndarray | None  # letters moved onto nearby picture without the overlay; None when there is no room
    near: np.ndarray  # letters spread over NEAR, shrunk as _shrink does, to normalise by
    shift: int  # rows from the yardstick to the letters
    clear: np.ndarray  # 0 on the pixel either side of the overlay's edge, 1 elsewhere
    clear_norm: np.ndarray  # clear spread over SOFT
    scale: int  # luma pixels per pixel of this plane, which the spreads (in luma pixels) are divided by
    area: slice  # the window's rows that are the part's padded area


def _shrink(img: np.ndarray, sigma: float) -> tuple[np.ndarray, int]:
    """Gaussian blur of each channel, and the factor it is shrunk by: a wide one is worked out at a quarter or half
    the size, as it has no fine detail to lose."""
    f = 4 if sigma >= 12 else 2 if sigma >= 6 else 1
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma), 1
    h, w = img.shape[:2]
    if h % f or w % f:  # whole multiples resize far faster
        img = cv2.copyMakeBorder(img, 0, -h % f, 0, -w % f, cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(cv2.resize(img, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0),
                            sigma / f), f


def _grow(small: np.ndarray, f: int, shape: tuple[int, ...]) -> np.ndarray:
    """small, shrunk by f, back at shape."""
    if f == 1:
        return small
    return cv2.resize(small, (small.shape[1] * f, small.shape[0] * f), interpolation=cv2.INTER_LINEAR)[:shape[0],
                                                                                                      :shape[1]]


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of img by sigma pixels (see _shrink)."""
    return _grow(*_shrink(img, sigma), img.shape)


def _plan(part: Part, box: tuple[int, int, int, int], shape: tuple[int, int], half: bool) -> Plan:
    """part laid out for a plane of shape (full size, or half for colour)."""
    s = 2 if half else 1
    rim = RIM // s
    (ys, xs), sel = part.where, part.sel
    letters = cv2.dilate(sel.astype(np.uint8), np.ones((2 * rim + 1,) * 2, np.uint8)).astype(bool)
    rows = np.flatnonzero(letters.any(1))
    t0, t1 = ys.start + rows[0], ys.start + rows[-1] + 1
    shift, margin = t1 - t0 + rim, int(np.ceil(2 * SPREAD / s))
    up, down = t0 - shift, shape[0] - t1 - shift  # rows left beyond the yardstick above and below
    if up < down:
        shift = -shift
    top, bottom = (max(0, up - margin), ys.stop) if shift > 0 else (ys.start, min(shape[0], t1 - shift + margin))
    h, w = bottom - top, xs.stop - xs.start
    area = np.s_[ys.start - top:ys.stop - top, :]

    def lay(values, fill: float) -> np.ndarray:
        out = np.full((h, w), fill, np.float32)
        out[area][sel] = values
        return out

    mask = np.zeros((h, w), bool)
    mask[area] = letters
    inside = np.zeros((h, w), bool)
    x, y, bw, bh = (v // s for v in box)
    inside[y - top:y - top + bh, x - xs.start:x - xs.start + bw] = True
    # shortcut: an overlay without its own height of room above or below it is undone without the yardstick (kept
    # as undone), upgrade by moving the yardstick sideways if overlays that tall come
    yard = np.roll(mask, -shift, 0) if max(up, down) >= 0 else None
    known = (~(mask | yard if yard is not None else mask)).astype(np.float32)
    letters_f = mask.astype(np.float32)
    core = lay(part.keep < OVERLAY, False).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    clear = 1 - (cv2.dilate(core, k) - cv2.erode(core, k)).astype(np.float32)
    return Plan(np.s_[top:bottom, xs], lay(True, False).astype(bool), lay(part.offset, 0), lay(1 - part.keep, 0),
                lay(1 / part.keep, 1), mask & inside, known,
                np.maximum(_blur(known, SPREAD / s) + WIDE_SHARE * _blur(known, WIDE / s), 1e-6), letters_f, yard,
                np.maximum(_shrink(letters_f, NEAR / s)[0], 1e-6),
                shift, clear, np.maximum(_blur(clear, SOFT / s), 1e-6), s, area[0])


def _behind(plane: np.ndarray, p: Plan) -> tuple[np.ndarray, np.ndarray]:
    """In p's window: the plane as seen, and the picture around the letters carried across them."""
    seen = plane[p.win].astype(np.float32)
    known = (seen - p.offset) * p.gain * p.known
    return seen, (_blur(known, SPREAD / p.scale) + WIDE_SHARE * _blur(known, WIDE / p.scale)) / p.norm


def _present(seen: np.ndarray, base: np.ndarray, p: Plan) -> bool:
    """Whether the overlay is in this frame: its strength read from what it adds to the picture carried across, as a
    whole and at its edges only. The whole is misread where the picture is about as bright as the overlay and does
    not quite follow the carried-across average; the edges, where strong real edges cross the letters."""
    # shortcut: in a frame whose picture is about as bright as the overlay, busy, and crossed by strong edges, both
    # readings can miss it and the frame is left alone; upgrade by reading against a sharper prediction of the picture
    rows, sel = p.area, p.sel[p.area]
    adds, extra = p.offset[rows] - p.dim[rows] * base[rows], seen[rows] - base[rows]
    if np.mean(adds[sel] ** 2) < WEAK ** 2:
        return True
    smooth = _blur(np.dstack([extra, adds]), FINE)
    pairs = ((extra, adds), (extra - smooth[..., 0], adds - smooth[..., 1]))
    return any(float(e[sel] @ a[sel]) >= PRESENT * float(a[sel] @ a[sel]) for e, a in pairs)


def _strength(seen: np.ndarray, base: np.ndarray, p: Plan) -> Plan:
    """p with the overlay as strong as this frame shows it: one factor for the whole overlay, shrunk towards the
    measured strength by SHRINK."""
    # shortcut: one factor per frame; on a strongly coloured, textured scene it can read too strong, upgrade to a
    # factor that varies across the overlay if that leaves a visible trace
    band = lambda x: _blur(x, STROKE[0] / p.scale) - _blur(x, STROKE[1] / p.scale)
    shows, adds = band(seen - base)[p.sel], band(p.offset - p.dim * base)[p.sel]
    prior = SHRINK * adds.size
    s = (float(shows @ adds) + prior) / (float(adds @ adds) + prior)
    s = min(max(s, 1 / STRONGEST), STRONGEST, (1 - FLOOR) / max(float(p.dim.max()), 1e-6))
    return p._replace(offset=p.offset * s, dim=p.dim * s, gain=1 / (1 - p.dim * s))


def _kept(coarse: np.ndarray, fine: np.ndarray, p: Plan) -> tuple[np.ndarray | float, np.ndarray | float]:
    """How much of the coarse and the fine detail to keep: by the share of the letters' roughness that is real
    picture, the most the yardstick's roughness and NOISE allow."""
    if p.yard is None:
        return 1.0, 1.0
    energies = np.zeros((*coarse.shape, 4), np.float32)
    for i, d in enumerate((coarse, fine)):
        e = energies[..., 2 * i] = d * d
        if p.shift > 0:  # the yardstick's moved over the letters
            energies[p.shift:, :, 2 * i + 1] = e[:-p.shift]
        else:
            energies[:p.shift, :, 2 * i + 1] = e[-p.shift:]
    rough, f = _shrink(energies * p.letters[..., None], NEAR / p.scale)
    rough /= p.near[..., None]
    kept = []
    for which in (0, 1):
        letters, yard = rough[..., 2 * which], rough[..., 2 * which + 1]
        real = letters - np.clip(letters - yard, 0, NOISE[which])
        lo, hi = FILL[which]
        kept.append(np.clip(((real + CALM[which]) / (letters + CALM[which]) - lo) / (hi - lo), 0, 1))
    kept = _grow(np.dstack(kept), f, coarse.shape)
    return kept[..., 0], kept[..., 1]


def _clean(plane: np.ndarray, seen: np.ndarray, base: np.ndarray, p: Plan) -> None:
    """Undo the overlay in plane's window, taking out the compression error the yardstick shows is there."""
    adds = p.offset - p.dim * base
    soft = _blur(adds, SOFT / p.scale) - adds  # what softening the overlay's edges adds
    miss = np.clip(seen - base - adds, -SURE, SURE) * p.letters
    fit, f = _shrink(np.dstack([miss * soft, soft * soft * p.letters]), NEAR / p.scale)
    amount = _grow(np.clip(fit[..., 0] / (fit[..., 1] + 1e-3), 0, SOFTEST), f, miss.shape)
    z = (seen - amount * soft - p.offset) * p.gain
    # what is left at the edge itself is a thin line, so the pixel either side is filled from its neighbours, unless
    # that changes it more than compression can have (a real edge crossing)
    edge = (_blur(z * p.clear, SOFT / p.scale) / p.clear_norm - z) * (1 - p.clear)
    z += edge * np.clip(2 - np.abs(edge) / SURE, 0, 1)
    coarse = _blur(z, FINE / p.scale) - base
    fine = z - base - coarse
    keep_coarse, keep_fine = _kept(coarse, fine, p)
    out = base + keep_coarse * coarse + keep_fine * fine
    plane[p.win][p.write] = np.clip(np.rint(out[p.write]), 0, 255)


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


def _printed(sel: np.ndarray, stray: np.ndarray, mean: np.ndarray, lumas: list[np.ndarray]) -> np.ndarray:
    """Which of lumas (crops of the padded area, averaging mean) printed stray (overlay pixels sel the line fit reads
    as not dimmed) into the variance: a few alike, very high-contrast frames, such as a menu lying across the box, make
    the pixels beside stray vary far more than stray itself in those frames only."""
    beside = cv2.dilate(stray.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~sel
    if not stray.any() or not beside.any():
        return np.zeros(len(lumas), bool)
    gap = np.array([d[beside].mean() - d[stray].mean() for d in ((y - mean) ** 2 for y in lumas)])
    middle = np.median(gap)
    return gap - middle > PRINTED * max(1.4826 * np.median(np.abs(gap - middle)), 1e-6)


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
