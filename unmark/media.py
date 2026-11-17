"""Every ffmpeg / ffprobe call lives here. Arguments are lists, never shell strings."""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
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
FILTERS = ["null", "setpts", "fps", "crop", "scale", "scdet", "metadata", "settb"]  # every filter used below
CUT_WIDTH = 160  # pixels; scene cuts are looked for on a picture this wide, which is plenty and cheap
# shortcut: scdet score (0-100) above which a frame starts a new scene, set on synthetic clips (a soft cut scores 8.9,
# smooth motion 0.1; ffmpeg's default 10 misses that cut), upgrade by calibrating on real game footage
CUT_SCORE = 5
CHUNK_S = 300  # seconds of video in each part of a resumable patch; a stopped run loses at most the part it was on
STEADY = 500  # packets checked to tell a steady frame rate from a variable one
# frames per sample above which each sample is read with its own seek instead of decoding everything: a seek decodes
# about half a keyframe gap (2 to 6 s on real 60 fps footage, so about 120 frames) plus starting ffmpeg
SEEK = 200
SEEKERS = 4  # seeks run at once; each ffmpeg is multi-threaded already, and each holds one decoded frame
CRF = 16  # libx264's crf when none is given
# Apple silicon's HEVC encoder at QUALITY measured closer to the source than libx264 at CRF, smaller and faster
# shortcut: hevc_nvenc on Windows with an NVIDIA card probably helps the same way but has never been run, upgrade by
# measuring it there the way videotoolbox was measured
HARDWARE = "hevc_videotoolbox"
QUALITY = 80  # HARDWARE's -q:v, 1 to 100
KEY_S = 2  # seconds between keyframes
SPARE = 1.05  # the joined video holds the sound too, so this much more than the parts' size must be free to join them
# finished parts' names; the digit keeps out the "._" companion files macOS adds beside each on exFAT and FAT disks
PARTS = "[0-9]*-*.nut"
# video stream tags that describe the source's encoding, not the picture, so the cleaned video must not copy them
STALE = {"DURATION", "BPS", "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES", "ENCODER", "_STATISTICS_WRITING_APP",
         "_STATISTICS_WRITING_DATE_UTC", "_STATISTICS_TAGS"}


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


def version(name: str) -> tuple[str, str]:
    """Where ffmpeg or ffprobe (name) is run from, and the first line of its version, without the copyright."""
    path = _bin(name)
    return path, _run([path, "-hide_banner", "-version"]).stdout.splitlines()[0].split(" Copyright")[0]


def listed(kind: str) -> set[str]:
    """The names ffmpeg lists for kind: "filters" or "encoders"."""
    out = _run([_bin("ffmpeg"), "-hide_banner", f"-{kind}"]).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith(" ") and len(line.split()) > 2}


def encoder() -> tuple[str, str]:
    """The encoder patch uses when no crf is given, and why: this Mac's hardware HEVC encoder if a tiny test encode
    works, else libx264 (with no reason when there is no hardware encoder to try)."""
    if sys.platform != "darwin" or platform.machine() != "arm64" or HARDWARE not in listed("encoders"):
        return "libx264", ""
    try:
        _run([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
              "testsrc2=s=64x48:d=0.1", "-c:v", HARDWARE, "-f", "null", "-"])
    except RuntimeError as e:
        return "libx264", f"{HARDWARE} is listed but did not start: {str(e).strip().splitlines()[-1]}"
    return HARDWARE, "this Mac's hardware encoder"


def sample_clip(path) -> None:
    """Write a 2-second 320x180 clip with sound to path, made by ffmpeg from its own test pattern."""
    _run([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=2",
          "-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
          "-c:a", "aac", str(path)])


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
        "tags": video.get("tags", {}),
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


def _decode(src: Path, pix_fmt: str, vf: str, log, ss: float | None = None, frames: int = 0) -> subprocess.Popen:
    """Decode src from ss seconds after the file's start (all of it without ss), at most frames frames if given."""
    return _popen([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
                   *(["-ss", f"{ss:.6f}"] if ss is not None else []), "-i", str(src), "-map", "0:v:0", "-vf", vf,
                   *(["-frames:v", str(frames)] if frames else []), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                  log, stdout=subprocess.PIPE)


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


def planes(src, count: int) -> Iterator[Frame]:
    """About count frames spread evenly over the video (all if fewer). Each yielded Frame is valid only until the
    next one."""
    src = Path(src)
    info = probe(src)
    # shortcut: assumes the video runs to the end of the file, upgrade if sources with long audio tails appear
    span = info["duration_s"] - info["video_start_s"]
    if span * Fraction(info["fps"]) > SEEK * count:
        yield from _seeks(src, info, [info["video_start_s"] + (k + 0.5) * span / count for k in range(count)])
        return
    rate = Fraction(count) / Fraction(span if span > 0 else 1).limit_denominator(1000)
    filters = ["setpts=PTS-STARTPTS", f"fps={rate}"] if rate < Fraction(info["fps"]) else []
    yield from _stream(src, info["pix_fmt"], ",".join(filters) or "null", info["width"], info["height"])


def _seeks(src: Path, info: dict, times: list[float]) -> Iterator[Frame]:
    """The first frame at or after each time (seconds from the file's start), in order; times past the end give none."""
    def one(t: float) -> list[Frame]:
        frames = _stream(src, info["pix_fmt"], "null", info["width"], info["height"], ss=t, frames=1)
        return [Frame(f.y.copy(), f.u.copy(), f.v.copy()) for f in frames]

    with ThreadPoolExecutor(SEEKERS) as pool:
        window: deque = deque()  # not pool.map: it would read every sample ahead and keep them all in memory
        for t in times:
            window.append(pool.submit(one, t))
            if len(window) > SEEKERS:
                yield from window.popleft().result()
        while window:
            yield from window.popleft().result()


def _patch_filters(info: dict) -> str:
    """The filters patch decodes with: fps makes variable-rate input constant; starting at 0 stops it padding a late
    start with copies of frame 0."""
    return f"setpts=PTS-STARTPTS,fps={info['fps']}"


def crops(src, box: tuple[int, int, int, int]) -> Iterator[Frame]:
    """Every frame exactly as patch passes it to edit, cropped to box (x, y, w, h, even); each valid until the next."""
    src = Path(src)
    info = probe(src)
    x, y, w, h = box
    yield from _stream(src, info["pix_fmt"], f"{_patch_filters(info)},crop={w}:{h}:{x}:{y}", w, h)


def cuts(src) -> list[int]:
    """Numbers of the frames where a new scene starts, counted the way patch counts them."""
    src = Path(src)
    vf = f"{_patch_filters(probe(src))},scale={CUT_WIDTH}:-2,scdet=threshold={CUT_SCORE}"
    out = _run([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src), "-map", "0:v:0",
                "-vf", f"{vf},metadata=mode=print:key=lavfi.scd.time:file=-", "-f", "null", "-"]).stdout
    return [int(line.split()[0][6:]) for line in out.splitlines() if line.startswith("frame:")]


def _stream(src: Path, pix_fmt: str, vf: str, w: int, h: int, ss: float | None = None, frames: int = 0
            ) -> Iterator[Frame]:
    with tempfile.TemporaryFile() as log:
        p = _decode(src, pix_fmt, vf, log, ss, frames)
        try:
            for _, frame in _frames(p, w, h):
                yield frame
        finally:
            p.stdout.close()
            p.wait()
        if p.returncode != 0:
            raise _fail(p, log)


def _steady(src: Path, fps: Fraction) -> bool:
    """True when the video's first packets come one frame apart; Matroska reports a steady rate even when it is not."""
    # shortcut: only the first STEADY packets are looked at, upgrade if sources that change rate later on appear
    out = _run([_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{STEADY}",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(src)]).stdout
    t = sorted(float(v) for v in out.split() if v != "N/A")
    return len(t) > 1 and all(abs(b - a - 1 / fps) < 0.25 / fps for a, b in itertools.pairwise(t))


def _vanished(func, path, exc: BaseException) -> None:
    """For rmtree: on exFAT and FAT disks macOS removes a file's "._" companion itself, before rmtree gets to it."""
    if not isinstance(exc, FileNotFoundError):
        raise exc


def patch(src, dst, edit: Callable[[Frame], None], crf: int | None = None, resume: str | None = None,
          progress: Callable[[int, int], None] | None = None) -> int:
    """Stream every frame of src through edit (in place) into dst, copying the audio untouched. Returns frames in dst.
    Without crf it uses the best encoder this computer has (see encoder); a crf number means libx264 at that crf, and
    crf=0 is lossless. With resume, a key naming the edit, a constant-rate video is written in parts of CHUNK_S seconds
    that a stopped run keeps, so a later run with the same key continues from them; edit then sees each part from its
    own seek. progress(frames done, about how many in all) is called before the first part and after each one."""
    src, dst = Path(src), Path(dst)
    info = probe(src)
    fps = Fraction(info["fps"])
    # shortcut: a variable-rate video is written in one part and cannot resume, as a seek there cannot be sure to land
    # on a part's first frame; upgrade by seeking to each part's own time if long variable-rate videos come
    each = round(CHUNK_S * fps) if resume is not None and _steady(src, fps) else 0  # frames per part; 0: just one
    work, joined = dst.with_name(dst.name + ".parts"), dst.with_name(dst.name + ".part")
    run = {"key": resume, "size": src.stat().st_size, "mtime": src.stat().st_mtime_ns, "crf": crf, "frames": each}
    try:
        old = json.loads((work / "run.json").read_text("utf-8"))
        same = {k: v for k, v in old.items() if k != "encoder"} == run
    except (OSError, ValueError, AttributeError):
        old, same = {}, False
    # an earlier run that fell back to libx264 keeps it, so its parts are not thrown away if the hardware is back
    run["encoder"] = "libx264" if crf is not None or same and old.get("encoder") == "libx264" else encoder()[0]
    finished = sorted(work.glob(PARTS)) if resume is not None else []
    if not (each and same and old.get("encoder") == run["encoder"]):  # never mix parts from different runs
        if finished and same and old.get("encoder") == HARDWARE:
            raise RuntimeError(f"{HARDWARE} did not start this time, and the finished parts in {work} were made with "
                               "it; they are kept, so try again (or delete that folder to start over without it)")
        if finished:
            raise RuntimeError(f"{work} holds finished parts of an earlier run that differs from this one (another "
                               "video, overlay, setting or version of unmark); they are kept: to continue them, run "
                               "the earlier command again, or delete that folder to start over")
        if work.exists():
            shutil.rmtree(work, onexc=_vanished)
        work.mkdir()
        (work / "run.json").write_text(json.dumps(run), "utf-8")
    done = [int(p.stem.split("-")[1]) for p in finished]  # frames in each finished part
    total = round(info["duration_s"] * fps)
    hardware = run["encoder"] == HARDWARE
    quality = ["-q:v", str(QUALITY)] if hardware else ["-crf", str(CRF if crf is None else crf)]
    codec = ["-c:v", run["encoder"], *quality]
    try:
        if progress:
            progress(sum(done), total)
        while not done or (each and done[-1] == each):
            n = _encode(src, work / f"{len(done):06d}", edit, info, codec, each, len(done) * each)
            if not n:
                break
            done.append(n)
            if progress:
                progress(sum(done), total)
        need = SPARE * sum(p.stat().st_size for p in work.glob(PARTS))
        free = shutil.disk_usage(dst.parent).free
        if need > free:  # the join needs the parts and the whole joined video on the disk at once
            raise RuntimeError(f"joining the cleaned video needs about {math.ceil((need - free) / 1e9)} GB more free "
                               f"space on the disk of {dst}; " + (
                                   "the finished parts are kept, so free that and the same command finishes the job"
                                   if each else "free that and run the same command again"))
        (work / "parts.txt").write_text("".join(f"file {k:06d}-{n}.nut\nduration {float(n / fps):.6f}\n"
                                                for k, n in enumerate(done)), "utf-8")
        # shortcut: subtitles and attachments are copied only into the source's own kind of file, as one the output
        # cannot hold would fail the whole join; upgrade by converting subtitles if people ask for them across kinds
        same = dst.suffix.lower() == src.suffix.lower()
        # the video comes from the parts, so its tags are copied from the source, less those that describe its old
        # encoding (Matroska's duration and mkvmerge's statistics, read as "BPS-eng" and the like)
        tags = {k: v for k, v in info["tags"].items() if k.split("-")[0].upper() not in STALE}
        _run([_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
              "-itsoffset", f"{info['video_start_s']:.6f}", "-f", "concat", "-i", str(work / "parts.txt"),
              "-i", str(src), "-map", "0:v", "-map", "1:a?", *(["-map", "1:s?", "-map", "1:t?"] if same else []),
              "-map_metadata", "1", "-map_chapters", "1",
              *(a for k, v in tags.items() for a in ("-metadata:s:v:0", f"{k}={v}")), "-c", "copy",
              # hvc1, not ffmpeg's default hev1: Apple's players open only hvc1 HEVC in .mp4 and .mov
              *(["-tag:v", "hvc1"] if hardware and dst.suffix.lower() != ".mkv" else []),
              "-f", MUXERS[dst.suffix.lower()], str(joined)])
        joined.replace(dst)
    except BaseException:
        if not each:  # nothing a later run could pick up
            shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        joined.unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    return sum(done)


def _encode(src: Path, name: Path, edit: Callable[[Frame], None], info: dict, codec: list[str], frames: int,
            first: int) -> int:
    """Encode, video only, with the codec arguments, up to frames frames (all without frames) from frame number first
    into name-<count>.nut; returns the count, and writes no file for none."""
    w, h, pf = info["width"], info["height"], info["pix_fmt"]
    tmp = name.with_name(f"{name.name}.{os.getpid()}.tmp")  # its own name, so two runs never write into one file
    # half a frame early: containers round times (Matroska to 1 ms), and a frame rounded down must not be skipped
    ss = info["video_start_s"] + (first - 0.5) / Fraction(info["fps"]) if first else None
    enc_args = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", pf, "-s", f"{w}x{h}", "-framerate", info["fps"],
                *(a for k, o in COLOURS.items() if k in info for a in (o, info[k])), "-i", "-",
                # a fine time base, so the late-start shift when the parts are joined is not rounded to a whole frame
                "-vf", "settb=1/90000", "-enc_time_base:v", "filter", *codec,
                "-g", str(round(KEY_S * Fraction(info["fps"]))), "-pix_fmt", pf, "-f", "nut", str(tmp)]
    n = 0
    try:
        with tempfile.TemporaryFile() as dlog, tempfile.TemporaryFile() as elog:
            dec = _decode(src, pf, _patch_filters(info), dlog, ss, frames)
            enc = _popen(enc_args, elog, stdin=subprocess.PIPE)
            try:
                for buf, frame in _frames(dec, w, h):
                    edit(frame)
                    try:
                        enc.stdin.write(buf)
                    except OSError:  # BrokenPipeError, or EINVAL on Windows
                        break  # the encoder died; its log says why
                    n += 1
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
        if n:
            with open(tmp, "rb+") as f:  # on the disk before it looks finished, so a lost disk cannot cut it short
                os.fsync(f.fileno())
            tmp.replace(name.with_name(f"{name.name}-{n}.nut"))
    finally:
        tmp.unlink(missing_ok=True)
    return n
