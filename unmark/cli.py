"""Command line: `unmark clean VIDEO [--box x,y,w,h | --mark N] [--out PATH]`."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np

from unmark import doctor, fill, marks, media

SAMPLES = 1000  # frames measured; more average out the picture, fewer read faster
CHECKS = 100  # frames sampled from the input and the written output to check how visible the outline is
VISIBLE = 1.25  # outline visibility above this counts as still visible; clean pictures measure about 1.0
ROOM = 32  # fewest pixels of real picture kept around a solid mark for the filler to work from
WIDEST = 640  # pixels; the picture sent to the filler is kept this wide or less where the box allows, for ProPainter
COLOURS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]  # BGR


def _box(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Parse x,y,w,h and widen it to even luma pixels so it lines up with the half-size colour planes."""
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise RuntimeError(f"--box must be four whole numbers x,y,w,h (for example 20,20,120,40), not {text!r}")
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise RuntimeError(f"--box {text} does not fit inside the {width}x{height} video")
    x0, y0 = x & ~1, y & ~1
    x1, y1 = min(x + w + 1 & ~1, width & ~1), min(y + h + 1 & ~1, height & ~1)
    return x0, y0, x1 - x0, y1 - y0


def _crop(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """box padded with real picture for the filler, inside the frame, at an even place, sides multiples of 8."""
    x, y, w, h = box
    pad = max(ROOM, min(max(w, h) // 2, (WIDEST - w) // 2))

    def side(a: int, n: int, size: int) -> tuple[int, int]:
        n = min(n + 2 * pad + 7 & ~7, size & ~7)
        return min(max(0, a - pad), size - n) & ~1, n

    (x0, w0), (y0, h0) = side(x, w, width), side(y, h, height)
    return x0, y0, w0, h0


def _rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # shortcut: OpenCV converts with BT.601 limited range whatever the tags say; only the filled pixels see it,
    # upgrade by converting with the video's own matrix and range if filled colours look off on BT.709 or full range
    return cv2.cvtColor(np.concatenate((y.ravel(), u.ravel(), v.ravel())).reshape(-1, y.shape[1]),
                        cv2.COLOR_YUV2RGB_I420)


def _filler(src: Path, mark: marks.Mark, width: int, height: int) -> Callable[[media.Frame], None]:
    """An edit that pastes in the filled picture under a solid mark, one frame at a time from a second decode."""
    x, y, w, h = crop = _crop(mark.box, width, height)
    bx, by, bw, bh = mark.box
    if bx + bw > x + w or by + bh > y + h:
        raise RuntimeError("the box spans almost the whole width or height of the video, which is too big to fill in")
    mask = np.zeros((h, w), bool)
    # shortcut: the whole box is filled, upgrade to the mark's own pixels if the fill blurs too much of a large box
    mask[by - y:by - y + bh, bx - x:bx - x + bw] = True
    if os.environ.get("UNMARK_PROPAINTER"):
        cuts = media.cuts(src)
    else:
        cuts = []
        print("UNMARK_PROPAINTER is not set, so the hidden picture is filled with the simple OpenCV fill, which looks "
              "softer than ProPainter")
    filled = fill.fill((_rgb(f.y, f.u, f.v) for f in media.crops(src, crop)), mask, cuts)
    luma, chroma = np.s_[y:y + h, x:x + w], np.s_[y // 2:(y + h) // 2, x // 2:(x + w) // 2]

    def edit(frame: media.Frame) -> None:
        planes = frame.y[luma], frame.u[chroma], frame.v[chroma]
        new = next(filled)
        changed = (new != _rgb(*planes)).any(axis=2)
        yuv = cv2.cvtColor(new, cv2.COLOR_RGB2YUV_I420).ravel()
        planes[0][changed] = yuv[:w * h].reshape(h, w)[changed]
        changed = changed.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))  # 4:2:0 shares colour per 2x2 block
        for plane, part in zip(planes[1:], np.split(yuv[w * h:], 2)):
            plane[changed] = part.reshape(h // 2, w // 2)[changed]

    return edit


def _keep_one(frames: Iterable[media.Frame], kept: list[np.ndarray]) -> Iterator[media.Frame]:
    """Pass frames through, leaving in kept a copy of one from the second half of the run."""
    for i, f in enumerate(frames, 1):
        if i & (i - 1) == 0:  # powers of two: only a few copies, the last one at least halfway through
            kept[:] = [f.y.copy()]
        yield f


def _picture(luma: np.ndarray, found: list[marks.Mark], path: Path) -> None:
    """Save the frame in grey with each mark's box drawn and numbered in its own colour."""
    img = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
    size = max(1, luma.shape[0] // 360)
    for i, m in enumerate(found, 1):
        x, y, w, h = m.box
        colour = COLOURS[(i - 1) % len(COLOURS)]
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), colour, size)
        cv2.putText(img, str(i), (x + 2, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * size, colour, size, cv2.LINE_AA)
    try:  # not cv2.imwrite: it returns False instead of raising, and on Windows fails on non-ASCII paths
        cv2.imencode(".png", img)[1].tofile(path)
    except OSError as e:
        raise RuntimeError(f"could not save {path}, the picture of the overlays found ({e.strerror}); run on a copy of "
                           "the video in a folder you can write to") from None


def _show(path: Path) -> None:
    """Open the picture in the system viewer; if that fails the user still has its path."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _progress() -> Callable[[int, int], None]:
    """A progress callback for media.patch that prints one line per part, with the time left at this run's speed."""
    first: list = []

    def show(done: int, total: int) -> None:
        if not first:
            first[:] = done, time.monotonic()
            if done:
                print(f"continuing an unfinished earlier run, which had already written {done} frames")
            return
        left = max(total - done, 0) * (time.monotonic() - first[1]) / (done - first[0])
        print(f"{min(done / max(total, 1), 1):.0%} written, about {left / 60:.0f} min left")

    return show


def clean(video: str, box_text: str | None, mark: int | None, out: str | None) -> int:
    """Find or measure the overlay and write a copy of the video without it."""
    src = Path(video)
    info = media.probe(src)
    suffix = src.suffix if src.suffix.lower() in media.MUXERS else ".mkv"
    dst = Path(out) if out else src.with_name(f"{src.stem}.clean{suffix}")
    if dst.suffix.lower() not in media.MUXERS:
        raise RuntimeError(f"--out must end in one of {', '.join(media.MUXERS)}, the kinds of video unmark can write")
    if dst.exists() and dst.samefile(src):  # not resolve(): disks that ignore case see CLIP.mp4 as clip.mp4
        raise RuntimeError("--out must differ from the input video")
    if box_text is not None:
        chosen = marks.find(lambda: media.planes(src, SAMPLES), _box(box_text, info["width"], info["height"]))[0]
    else:
        kept: list[np.ndarray] = []
        found = marks.find(lambda: _keep_one(media.planes(src, SAMPLES), kept))
        if not found:
            raise RuntimeError("no fixed overlay was found; if you can see one, give its position with --box x,y,w,h")
        picture = src.with_name(f"{src.stem}.marks.png")
        try:
            _picture(kept[0], found, picture)
        except RuntimeError as e:
            if mark is None:  # the user needs the picture to choose
                raise
            print(f"warning: {e}")
        for i, m in enumerate(found, 1):
            x, y, w, h = m.box
            print(f"{i}: {m.kind} at {x},{y}, size {w}x{h}, opacity {m.opacity:.0%}")
        if mark is not None:
            answer = str(mark)
        elif sys.stdin.isatty():
            _show(picture)
            answer = input(f"Which overlay should go (1-{len(found)}, see {picture})? ")
        else:
            raise RuntimeError(f"run again with --mark N, where N is the number of the overlay to remove "
                               f"(the boxes are drawn in {picture})")
        if not answer.strip().isdigit() or not 1 <= int(answer) <= len(found):
            raise RuntimeError(f"there is no overlay {answer.strip()}; choose a number from 1 to {len(found)} "
                               f"(the boxes are drawn in {picture})")
        chosen = found[int(answer) - 1].refined(media.planes(src, SAMPLES))

    before = chosen.visibility(media.planes(src, CHECKS))
    if before <= VISIBLE:  # an outline that cannot be seen in the input means the measurement found only noise
        raise RuntimeError(marks.NOTHING)
    if chosen.kind == "solid":
        # shortcut: a solid fill is written in one go and cannot continue after a stop, as its filler only runs from
        # the first frame on; upgrade by starting the filler at each part if long videos with solid overlays come
        n = media.patch(src, dst, _filler(src, chosen, info["width"], info["height"]), progress=_progress())
    else:
        cv2.setNumThreads(1)  # the area cleaned per frame is small: OpenCV's threads cost more than they save there
        key = hashlib.sha256(repr((chosen.box, chosen.kind, [p.where for p in chosen.parts])).encode()
                             + b"".join(a.tobytes() for p in chosen.parts for a in p[1:])).hexdigest()
        n = media.patch(src, dst, chosen.erase, resume=key, progress=_progress())
    print(f"wrote {dst} ({n} frames)")
    after = chosen.visibility(media.planes(dst, CHECKS))
    print(f"overlay outline visibility (1.0 = gone): input {before:.2f}, output {after:.2f}")
    if after > VISIBLE:
        print("warning: the overlay's outline is still visible in the output; look at it before using it")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for `unmark`. Mistakes end in one plain sentence and exit code 1; --debug shows the traceback."""
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so a subcommand's default does not overwrite `unmark --debug <cmd>`
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                        help="show the full Python traceback when something fails")
    p = argparse.ArgumentParser(prog="unmark", description="Remove a fixed-position overlay from a video.",
                                parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean", help="find a fixed overlay, or take the box you give, and remove it", parents=[common])
    c.add_argument("video")
    where = c.add_mutually_exclusive_group()
    where.add_argument("--box", help="overlay box in pixels: x,y,w,h; without it, overlays are found automatically")
    where.add_argument("--mark", type=int, help="number of the found overlay to remove, as listed")
    c.add_argument("--out", help="output path; default <name>.clean<suffix> next to the input")
    d = sub.add_parser("doctor", help="check that ffmpeg (and ProPainter, if set up) work on this computer",
                       parents=[common])
    d.add_argument("--gpu", action="store_true", help="also fill a short sample with ProPainter on the graphics card")
    a = p.parse_args(argv)
    try:
        return doctor.doctor(a.gpu) if a.cmd == "doctor" else clean(a.video, a.box, a.mark, a.out)
    except (KeyboardInterrupt, EOFError):  # Ctrl-C, or Ctrl-D at the question
        print("\nstopped; any finished parts of the cleaned video are kept, so running the same command again "
              "continues from them")
        return 130
    except Exception as e:  # the last line of defence for non-technical users
        if getattr(a, "debug", False):
            raise
        print(f"error: {e} (add --debug to see the technical details)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
