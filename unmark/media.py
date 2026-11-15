"""Every ffmpeg / ffprobe call lives here. Arguments are lists, never shell strings."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

# shortcut: only 8-bit 4:2:0 is read and written, upgrade when a real source arrives in 10-bit or 4:2:2/4:4:4
FORMATS = {"yuv420p", "yuvj420p"}
# probe key -> ffmpeg option that stamps it on the raw frames going into the encoder
COLOURS = {"color_range": "-color_range", "color_space": "-colorspace", "color_primaries": "-color_primaries",
           "color_transfer": "-color_trc"}
MUXERS = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska"}


@dataclass
class Frame:
    """One picture's planes as uint8 arrays; u and v are half size (4:2:0)."""
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _bin(name: str) -> str:
    return os.environ.get(f"UNMARK_{name.upper()}") or shutil.which(name) or name


def _not_found(name: str) -> RuntimeError:
    return RuntimeError(f"{name} was not found; install ffmpeg or set UNMARK_FFMPEG and UNMARK_FFPROBE to the full "
                        "paths of ffmpeg and ffprobe")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a command, raising RuntimeError with the stderr tail on failure."""
    # utf-8 explicitly: the Windows default code page would garble non-ASCII paths in ffprobe's output
    try:
        p = subprocess.run(args, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:  # raised here, not by the caller: on Windows the error does not name the program
        raise _not_found(args[0]) from None
    if p.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} failed ({p.returncode}): {p.stderr[-2000:]}")
    return p


def probe(src) -> dict:
    """Probe a video: width, height, fps (rational string), pix_fmt, duration_s, and the colour tags it has."""
    src = Path(src)
    if not src.is_file():
        raise RuntimeError(f"{src} does not exist")
    data = json.loads(_run([_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                            "-show_streams", str(src)]).stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{src.name} has no video stream")
    fps = video.get("avg_frame_rate", "0/0")
    if fps in ("0/0", "0/1"):
        fps = video.get("r_frame_rate", "0/0")
    # shortcut: 10-bit (so all HDR) is refused rather than squeezed to 8-bit, upgrade when a real one needs cleaning
    if any(b in video.get("pix_fmt", "") for b in ("10le", "10be")):  # not a bare "10": yuv410p is 8-bit
        raise RuntimeError(f"{src.name} is a 10-bit or HDR video, which unmark cannot clean yet; export an ordinary "
                           "8-bit copy first")
    # shortcut: ffmpeg turns decoded frames upright, so the size and box would not match; upgrade with -noautorotate
    if any(d.get("rotation") for d in video.get("side_data_list", [])):
        raise RuntimeError(f"{src.name} is a rotated (phone) video, which unmark cannot clean yet; export an upright "
                           "copy first")
    if video.get("pix_fmt") not in FORMATS:
        raise RuntimeError(f"{src.name} stores its picture as {video.get('pix_fmt')}; only 8-bit 4:2:0 video "
                           "(yuv420p) is supported for now")
    return {
        "width": video["width"],
        "height": video["height"],
        "fps": fps,
        "pix_fmt": video["pix_fmt"],
        "duration_s": float(data.get("format", {}).get("duration") or video.get("duration") or 0),
        # seconds from the file's first packet to the video's; ffmpeg counts each input's clock from its own start
        "video_start_s": float(video.get("start_time") or 0) - float(data.get("format", {}).get("start_time") or 0),
    } | {k: video[k] for k in COLOURS if video.get(k, "unknown") != "unknown"}


def _views(buf: bytearray, w: int, h: int) -> Frame:
    """Writable numpy views of the three planes inside buf."""
    cw, ch = (w + 1) // 2, (h + 1) // 2
    a = np.frombuffer(buf, np.uint8)
    return Frame(a[:w * h].reshape(h, w), a[w * h:w * h + cw * ch].reshape(ch, cw),
                 a[w * h + cw * ch:].reshape(ch, cw))


def _popen(args: list[str], log, **kw) -> subprocess.Popen:
    try:
        return subprocess.Popen(args, stderr=log, **kw)
    except FileNotFoundError:
        raise _not_found(args[0]) from None


def _fail(p: subprocess.Popen, log) -> RuntimeError:
    log.seek(0)
    return RuntimeError(f"ffmpeg failed ({p.returncode}): {log.read().decode('utf-8', 'replace')[-2000:]}")


def _decode(src: Path, pix_fmt: str, vf: str, log) -> subprocess.Popen:
    return _popen([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src), "-map", "0:v:0",
                   "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"], log, stdout=subprocess.PIPE)


def _frames(p: subprocess.Popen, w: int, h: int) -> Iterator[tuple[bytearray, Frame]]:
    """Read raw frames from p into one reused buffer."""
    buf = bytearray(w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2))
    frame, view = _views(buf, w, h), memoryview(buf)
    while True:
        got = 0
        while got < len(buf) and (n := p.stdout.readinto(view[got:])):
            got += n
        if got < len(buf):
            return  # end of stream; a partial trailing frame cannot happen with rawvideo, so it is ignored
        yield buf, frame


def planes(src, count: int, box: tuple[int, int, int, int] | None = None) -> Iterator[Frame]:
    """About count frames spread evenly over the video (all if fewer), optionally cropped to box (x, y, w, h).
    Each yielded Frame is valid only until the next one."""
    src = Path(src)
    info = probe(src)
    # shortcut: assumes the video runs to the end of the file, upgrade if sources with long audio tails appear
    span = info["duration_s"] - info["video_start_s"]
    rate = Fraction(count) / Fraction(span if span > 0 else 1).limit_denominator(1000)
    filters = ["setpts=PTS-STARTPTS", f"fps={rate}"] if rate < Fraction(info["fps"]) else []
    w, h = info["width"], info["height"]
    if box:
        x, y, w, h = box
        filters.append(f"crop={w}:{h}:{x}:{y}")
    with tempfile.TemporaryFile() as log:
        p = _decode(src, info["pix_fmt"], ",".join(filters) or "null", log)
        try:
            for _, frame in _frames(p, w, h):
                yield frame
        finally:
            p.stdout.close()
            p.wait()
        if p.returncode != 0:
            raise _fail(p, log)


def patch(src, dst, edit: Callable[[Frame], None], crf: int = 16) -> int:
    """Stream every frame of src through edit (in place) into dst, copying the audio untouched. Returns frames written.
    crf=0 is lossless."""
    src, dst = Path(src), Path(dst)
    info = probe(src)
    w, h, pf = info["width"], info["height"], info["pix_fmt"]
    part = dst.with_name(dst.name + ".part")
    enc_args = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", pf, "-s", f"{w}x{h}", "-framerate", info["fps"],
                *(a for k, o in COLOURS.items() if k in info for a in (o, info[k])), "-i", "-",
                "-i", str(src), "-map", "0:v", "-map", "1:a?",
                # the pipe carries no timestamps; a fine time base keeps a start between two frame ticks exact
                "-vf", f"settb=1/90000,setpts=PTS+{info['video_start_s']}/TB", "-enc_time_base:v", "filter",
                "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pf,
                "-c:a", "copy", "-f", MUXERS.get(dst.suffix.lower(), "matroska"), str(part)]
    n = 0
    try:
        with tempfile.TemporaryFile() as dlog, tempfile.TemporaryFile() as elog:
            # fps makes variable-rate input constant; starting at 0 stops it padding a late start with copies of frame 0
            dec = _decode(src, pf, f"setpts=PTS-STARTPTS,fps={info['fps']}", dlog)
            enc = _popen(enc_args, elog, stdin=subprocess.PIPE)
            try:
                for buf, frame in _frames(dec, w, h):
                    edit(frame)
                    enc.stdin.write(buf)
                    n += 1
            except OSError:  # BrokenPipeError, or EINVAL on Windows
                pass  # the encoder died; its log says why
            finally:
                dec.stdout.close()
                dec.wait()
                try:
                    enc.stdin.close()
                except OSError:
                    pass
                enc.wait()
            if enc.returncode != 0:
                raise _fail(enc, elog)
            if dec.returncode != 0:
                raise _fail(dec, dlog)
        part.replace(dst)
    finally:
        part.unlink(missing_ok=True)
    return n
