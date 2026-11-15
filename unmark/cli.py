"""Command line: `unmark clean VIDEO [--box x,y,w,h | --mark N] [--out PATH]`."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np

from unmark import fill, marks, media

SAMPLES = 1000  # frames measured; more average out the picture, fewer read faster
CHECKS = 100  # frames sampled from the input and the written output to check how visible the outline is
VISIBLE = 1.25  # outline visibility above this counts as still visible; clean pictures measure about 1.0
ROOM = 32  # fewest pixels of real picture kept around a solid mark for the filler to work from
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
    pad = max(ROOM, max(w, h) // 2)

    def side(a: int, n: int, size: int) -> tuple[int, int]:
        n = min(n + 2 * pad + 7 & ~7, size & ~7)
        return min(max(0, a - pad), size - n) & ~1, n

    (x0, w0), (y0, h0) = side(x, w, width), side(y, h, height)
    return x0, y0, w0, h0


def _rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # shortcut: OpenCV converts with BT.601 limited range whatever the tags say; only the filled pixels see it
    return cv2.cvtColor(np.concatenate((y.ravel(), u.ravel(), v.ravel())).reshape(-1, y.shape[1]),
                        cv2.COLOR_YUV2RGB_I420)


def _filler(src: Path, mark: marks.Mark, width: int, height: int) -> Callable[[media.Frame], None]:
    """An edit that pastes in the filled picture under a solid mark, one frame at a time from a second decode."""
    x, y, w, h = crop = _crop(mark.box, width, height)
    bx, by, bw, bh = mark.box
    if bx + bw > x + w or by + bh > y + h:
        raise RuntimeError("the box spans almost the whole width or height of the video, which is too big to fill in")
    mask = np.zeros((h, w), bool)
    mask[by - y:by - y + bh, bx - x:bx - x + bw] = True  # shortcut: the whole box, a per-pixel mask would keep more
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
    cv2.imwrite(str(path), img)


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


def clean(video: str, box_text: str | None, mark: int | None, out: str | None) -> int:
    """Find or measure the overlay and write a copy of the video without it."""
    src = Path(video)
    info = media.probe(src)
    dst = Path(out) if out else src.with_name(f"{src.stem}.clean{src.suffix if src.suffix.lower() in media.MUXERS else '.mkv'}")
    if dst.resolve() == src.resolve():
        raise RuntimeError("--out must differ from the input video")
    if box_text is not None:
        chosen = marks.find(lambda: media.planes(src, SAMPLES), _box(box_text, info["width"], info["height"]))[0]
    else:
        kept: list[np.ndarray] = []
        found = marks.find(lambda: _keep_one(media.planes(src, SAMPLES), kept))
        if not found:
            raise RuntimeError("no fixed overlay was found; if you can see one, give its position with --box x,y,w,h")
        picture = src.with_name(f"{src.stem}.marks.png")
        _picture(kept[0], found, picture)
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

    # shortcut: each check decodes the whole file again, upgrade with keyframe seeks if long videos feel slow
    before = chosen.visibility(media.planes(src, CHECKS))
    if before <= VISIBLE:  # an outline that cannot be seen in the input means the measurement found only noise
        raise RuntimeError(marks.NOTHING)
    edit = _filler(src, chosen, info["width"], info["height"]) if chosen.kind == "solid" else chosen.erase
    n = media.patch(src, dst, edit)
    print(f"wrote {dst} ({n} frames)")
    after = chosen.visibility(media.planes(dst, CHECKS))
    print(f"overlay outline visibility (1.0 = gone): input {before:.2f}, output {after:.2f}")
    if after > VISIBLE:
        print("warning: the overlay's outline is still visible in the output; look at it before using it")
    return 0


def _tail(e: Exception) -> str:
    """The last few lines of an error, on one line."""
    return " | ".join(str(e).strip().splitlines()[-3:])


TORCH = """import torch
if not torch.cuda.is_available():
    raise SystemExit(f"torch {torch.__version__} sees no NVIDIA card (its CUDA version: {torch.version.cuda})")
card = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} sees {card.name}, {card.total_memory / 2**30:.1f} GB")"""


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
    """Check, one line each, that what clean needs works on this computer. Returns the exit code."""
    failures = 0

    def report(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal failures
        failures += ok is False
        print(f"[{'ok' if ok else 'FAIL' if ok is False else 'warn'}] {label}{': ' + detail if detail else ''}")

    try:
        for name in ("ffmpeg", "ffprobe"):
            path = media._bin(name)
            version = media._run([path, "-hide_banner", "-version"]).stdout.splitlines()[0].split(" Copyright")[0]
            report(True, name, f"{path} ({version})")
        ff = media._bin("ffmpeg")
        filters = {line.split()[1] for line in media._run([ff, "-hide_banner", "-filters"]).stdout.splitlines()
                   if line.startswith(" ") and len(line.split()) > 2}
        missing = [f for f in media.FILTERS if f not in filters]
        report(not missing, "filters", "missing " + ", ".join(missing) if missing else ", ".join(media.FILTERS))
        encoders = {line.split()[1] for line in media._run([ff, "-hide_banner", "-encoders"]).stdout.splitlines()
                    if line.startswith(" ") and len(line.split()) > 2}
        report("libx264" in encoders, "encoder", "libx264" if "libx264" in encoders else "libx264 is missing")
        with tempfile.TemporaryDirectory(prefix="unmark-doctor-") as tmp:
            sample, out = Path(tmp) / "sample.mp4", Path(tmp) / "out.mp4"
            media._run([ff, "-hide_banner", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=2",
                        "-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
                        "yuv420p", "-c:a", "aac", str(sample)])
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
        return doctor(a.gpu) if a.cmd == "doctor" else clean(a.video, a.box, a.mark, a.out)
    except (KeyboardInterrupt, EOFError):  # Ctrl-C, or Ctrl-D at the question
        print("\nstopped; nothing was written")
        return 130
    except Exception as e:  # the last line of defence for non-technical users
        if getattr(a, "debug", False):
            raise
        print(f"error: {e} (add --debug to see the technical details)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
