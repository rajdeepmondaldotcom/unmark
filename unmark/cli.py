"""Command line: `unmark clean VIDEO --box x,y,w,h [--out PATH]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unmark import marks, media

SAMPLES = 400  # frames measured; more average out the picture, fewer read faster


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


def clean(video: str, box_text: str, out: str | None) -> int:
    """Measure the see-through overlay inside the box and write a copy of the video without it."""
    src = Path(video)
    info = media.probe(src)
    box = _box(box_text, info["width"], info["height"])
    dst = Path(out) if out else src.with_name(f"{src.stem}.clean{src.suffix if src.suffix.lower() in media.MUXERS else '.mkv'}")
    if dst.resolve() == src.resolve():
        raise RuntimeError("--out must differ from the input video")
    area = marks.region(box, info["width"], info["height"])
    found = marks.find(media.planes(src, SAMPLES, area), box, area)

    def edit(frame: media.Frame) -> None:
        for m in found:
            m.erase(frame)

    n = media.patch(src, dst, edit)
    print(f"wrote {dst} ({n} frames)")
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
    c = sub.add_parser("clean", help="remove a see-through overlay inside a given box", parents=[common])
    c.add_argument("video")
    c.add_argument("--box", required=True, help="overlay box in pixels: x,y,w,h")
    c.add_argument("--out", help="output path; default <name>.clean<suffix> next to the input")
    a = p.parse_args(argv)
    try:
        return clean(a.video, a.box, a.out)
    except KeyboardInterrupt:
        print("\nstopped; nothing was written")
        return 130
    except Exception as e:  # the last line of defence for non-technical users
        if getattr(a, "debug", False):
            raise
        print(f"error: {e} (add --debug to see the technical details)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
