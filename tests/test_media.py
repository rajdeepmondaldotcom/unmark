"""media.probe / planes / patch against real ffmpeg on tiny generated clips. Expectations come from ffmpeg directly."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from helpers import HEVC, failing_ffmpeg

from unmark import media

W, H = 64, 48
SIZE = W * H * 3 // 2  # one yuv420p frame


def ff(*args: str) -> bytes:
    """Run ffmpeg directly and return its stdout."""
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True,
                          check=True).stdout


def raw(path: Path, pix_fmt: str = "yuv420p") -> bytes:
    """Every decoded frame of path's first video stream as raw planar bytes, with no rate or colour change."""
    return ff("-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-")


def make(path: Path, frames: np.ndarray, fps: int = 25, pix_fmt: str = "yuv420p", extra: tuple = (),
         codec: str = "libx264", quality: tuple = ("-qp", "0")) -> Path:
    """Encode raw yuv frames (n, SIZE) losslessly (unless quality says otherwise), followed by any extra ffmpeg
    arguments (inputs, maps, tags)."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
                    "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *extra, "-c:v", codec, *quality,
                    "-pix_fmt", pix_fmt, str(path)], input=frames.tobytes(), check=True, capture_output=True)
    return path


def noise(n: int, seed: int = 0) -> np.ndarray:
    """n frames of random 8-bit planes: any rounding or conversion anywhere shows up as a changed byte."""
    return np.random.default_rng(seed).integers(16, 236, (n, SIZE), np.uint8)


def test_do_nothing_patch_is_exact(tmp_path):
    src = make(tmp_path / "src.mp4", noise(20))
    out = tmp_path / "out.mp4"
    media.patch(src, out, lambda f: None, crf=0)
    assert raw(out) == raw(src)


def test_an_edit_that_fails_ends_in_its_own_error_and_leaves_no_file(tmp_path):
    src = make(tmp_path / "src.mp4", noise(20))
    out = tmp_path / "out.mp4"

    def edit(f: media.Frame) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        media.patch(src, out, edit)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src.mp4"]


def frame_count(path: Path) -> int:
    """Decoded video frames, counted by ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return int(out.strip())


def lumas(path: Path) -> np.ndarray:
    """Every decoded frame's luma plane, (n, H, W)."""
    return np.frombuffer(raw(path), np.uint8).reshape(-1, SIZE)[:, :W * H].reshape(-1, H, W)


def test_each_edit_lands_on_its_own_frame(tmp_path):
    src = make(tmp_path / "src.mp4", noise(40, seed=1))
    calls = 0

    def edit(f: media.Frame) -> None:
        nonlocal calls
        f.y[:16, :16] = calls
        calls += 1

    out = tmp_path / "out.mp4"
    n = media.patch(src, out, edit, crf=0)
    assert n == calls == frame_count(src) == frame_count(out) == 40
    assert (lumas(out)[:, :16, :16] == np.arange(40)[:, None, None]).all()


def audio_md5(path: Path) -> bytes:
    """md5 of every audio packet, copied without decoding."""
    return ff("-i", str(path), "-map", "0:a", "-c", "copy", "-f", "md5", "-")


def test_audio_is_identical_bit_for_bit(tmp_path):
    two = ("-f", "lavfi", "-i", "sine=frequency=440:duration=1.6", "-f", "lavfi", "-i", "anoisesrc=d=1.6:a=0.1",
           "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:a:0", "aac", "-c:a:1", "libopus")
    one = ("-f", "lavfi", "-i", "sine=frequency=440:duration=1.6", "-c:a", "aac")
    for name, extra in (("two.mkv", two), ("one.mp4", one)):
        src = make(tmp_path / name, noise(40), extra=extra)
        out = tmp_path / f"out-{name}"
        media.patch(src, out, lambda f: None)
        assert audio_md5(out) == audio_md5(src), name


def video_tags(path: Path) -> dict:
    """The first video stream's colour fields as ffprobe reports them."""
    keys = ("pix_fmt", "color_range", "color_space", "color_primaries", "color_transfer")
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=" + ",".join(keys), "-of", "default=nw=1", str(path)], capture_output=True,
                         text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.split())


def test_full_range_round_trips_exact_with_its_tag(tmp_path):
    full = np.random.default_rng(2).integers(0, 256, (20, SIZE), np.uint8)  # values a range squeeze would clip
    # H.264 reports full range as yuvj420p; FFV1 keeps yuv420p and only the range tag says full
    for name, codec, pf in (("h264.mp4", "libx264", "yuvj420p"), ("ffv1.mkv", "ffv1", "yuv420p")):
        src = make(tmp_path / name, full, pix_fmt=pf, extra=("-color_range", "pc"), codec=codec)
        assert video_tags(src)["color_range"] == "pc" and video_tags(src)["pix_fmt"] == pf
        out = tmp_path / f"out-{name}"
        media.patch(src, out, lambda f: None, crf=0)
        assert video_tags(out)["color_range"] == "pc", name
        assert raw(out, pf) == raw(src, pf), name


def test_colour_tags_are_kept(tmp_path):
    tags = {"color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709"}
    src = make(tmp_path / "src.mp4", noise(20, seed=3), extra=(
        "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"))
    assert tags.items() <= video_tags(src).items()
    out = tmp_path / "out.mp4"
    media.patch(src, out, lambda f: None, crf=0)
    assert tags.items() <= video_tags(out).items()
    assert raw(out) == raw(src)


def start(path: Path, kind: str) -> float:
    """start_time of the first stream of kind (v or a), from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"{kind}:0", "-show_entries",
                          "stream=start_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return float(out.split()[0])  # MPEG-TS repeats the stream under its program


def test_late_video_stays_in_sync_with_audio(tmp_path):
    video = make(tmp_path / "v.mkv", noise(25))
    for name in ("src.mkv", "src.ts"):  # MPEG-TS files start their clock well after zero
        src = tmp_path / name
        ff("-itsoffset", "0.5", "-i", str(video), "-f", "lavfi", "-i", "sine=duration=2", "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", str(src))
        gap = start(src, "v") - start(src, "a")
        assert abs(gap - 0.5) < 0.05  # sanity: the source really starts late
        for out in (tmp_path / f"{name}.mkv", tmp_path / f"{name}.mp4"):
            media.patch(src, out, lambda f: None, crf=0)
            assert abs(start(out, "v") - start(out, "a") - gap) <= 0.5 / 25, out.name
            assert raw(out) == raw(src), out.name


def video_pts(path: Path) -> list[float]:
    """Presentation time of every video frame, in display order, from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return sorted(float(t) for t in out.split())


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def test_variable_frame_rate_keeps_duration_audio_and_sync(tmp_path):
    ticks = 200  # a 100 fps grid over 2 s; tick i is painted luma 20 + i so every picture names its own time
    frames = np.zeros((ticks, SIZE), np.uint8)
    frames[:, W * H:] = 128
    frames[:, :W * H] = (20 + np.arange(ticks))[:, None]
    kept = [i for i in range(ticks) if i % 24 in (0, 4, 12)]  # gaps of 40, 80 and 120 ms
    keep = "+".join(f"eq(mod(n\\,24)\\,{k})" for k in (0, 4, 12))
    src = make(tmp_path / "src.mkv", frames, fps=100, extra=(
        "-f", "lavfi", "-i", "sine=duration=2", "-map", "0:v", "-map", "1:a", "-c:a", "aac",
        "-vf", f"select='{keep}'", "-fps_mode", "vfr"))
    t_src = video_pts(src)
    assert np.allclose(np.diff(t_src), np.diff(kept) / 100, atol=0.002)  # sanity: really variable
    out = tmp_path / "out.mkv"
    media.patch(src, out, lambda f: None, crf=0)

    t_out = video_pts(out)
    step = float(np.median(np.diff(t_out)))
    assert abs(duration(out) - duration(src)) <= step
    assert audio_md5(out) == audio_md5(src)
    shown = (lumas(out)[:, 0, 0].astype(int) - 20) / 100  # the source time each output picture was painted for
    assert len(shown) == len(t_out)

    def active(t: float) -> float:
        """Source time of the picture on screen at t seconds into the source video."""
        return max([k / 100 for k in kept if k / 100 <= t + 1e-6] or [0])

    for t, s in zip(t_out, shown):
        t -= t_out[0]
        assert active(t - step) <= s <= active(t + step), (t, s)


def test_ten_bit_video_is_refused_in_plain_words(tmp_path):
    src = tmp_path / "ten.mkv"
    ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:d=0.2", "-pix_fmt", "yuv420p10le", "-c:v", "libx264", str(src))
    with pytest.raises(RuntimeError, match="10-bit or HDR"):
        media.probe(src)


def test_rotated_phone_video_is_refused_in_plain_words(tmp_path):
    flat = make(tmp_path / "flat.mp4", noise(5))
    src = tmp_path / "rotated.mp4"
    ff("-display_rotation", "90", "-i", str(flat), "-c", "copy", str(src))
    assert "rotation" in subprocess.run(["ffprobe", "-v", "error", "-show_streams", str(src)], capture_output=True,
                                        text=True, check=True).stdout  # sanity: the tag is really there
    with pytest.raises(RuntimeError, match="rotat"):
        media.probe(src)


def test_planes_samples_the_whole_video(tmp_path):
    frames = noise(50, seed=4)  # 2 s at 25 fps; random pictures, so each identifies its frame
    src = make(tmp_path / "src.mp4", frames)
    ys = frames[:, :W * H].reshape(-1, H, W)
    us = frames[:, W * H:W * H * 5 // 4].reshape(-1, H // 2, W // 2)
    got = []
    for f in media.planes(src, 5):
        i = next(i for i in range(50) if (ys[i] == f.y).all())
        assert (us[i] == f.u).all()
        got.append(i)
    assert 4 <= len(got) <= 6, got
    assert got[0] <= 5 and got[-1] >= 40 and got == sorted(got), got


def test_planes_on_late_video_does_not_repeat_its_first_frame(tmp_path):
    frames = noise(25, seed=5)
    video, src = make(tmp_path / "v.mkv", frames), tmp_path / "src.mkv"
    ff("-itsoffset", "1", "-i", str(video), "-f", "lavfi", "-i", "sine=duration=2", "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", str(src))
    ys = frames[:, :W * H].reshape(-1, H, W)
    got = [next(i for i in range(25) if (ys[i] == f.y).all()) for f in media.planes(src, 5)]
    assert len(set(got)) == len(got) >= 4 and got[-1] >= 18, got


def test_crops_line_up_with_the_frames_patch_edits(tmp_path):
    video = make(tmp_path / "v.mkv", noise(25, seed=6))
    late = tmp_path / "late.mkv"
    ff("-itsoffset", "0.5", "-i", str(video), "-f", "lavfi", "-i", "sine=duration=2", "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", str(late))
    keep = "+".join(f"eq(mod(n\\,24)\\,{k})" for k in (0, 4, 12))
    vfr = make(tmp_path / "vfr.mkv", noise(100, seed=7), fps=100, extra=("-vf", f"select='{keep}'", "-fps_mode", "vfr"))
    x, y, w, h = 8, 6, 24, 16
    luma, chroma = np.s_[y:y + h, x:x + w], np.s_[y // 2:(y + h) // 2, x // 2:(x + w) // 2]
    for src in (late, vfr):
        seen = []
        media.patch(src, tmp_path / "out.mkv", lambda f, seen=seen: seen.append((f.y[luma].copy(), f.v[chroma].copy())))
        got = [(f.y.copy(), f.v.copy()) for f in media.crops(src, (x, y, w, h))]
        assert len(got) == len(seen) > 0, src.name
        assert all((a == c).all() and (b == d).all() for (a, b), (c, d) in zip(got, seen)), src.name


def drift(n: int, fx: float, level: int, speed: float = 0.1) -> np.ndarray:
    """n yuv frames of a soft luma pattern sliding speed pixels per frame, grey chroma."""
    yy, xx = np.mgrid[:H, :W]
    frames = np.full((n, SIZE), 128, np.uint8)
    for i in range(n):
        frames[i, :W * H] = np.clip(level + 50 * np.sin((xx + i * speed) / fx) * np.cos(yy / 11), 0, 255).ravel()
    return frames


def test_cuts_finds_a_hard_cut_in_patch_numbering(tmp_path):
    keep = "+".join(f"eq(mod(n\\,24)\\,{k})" for k in (0, 4, 12))  # variable rate, so numbering is easy to get wrong
    for name, frames in (("cut.mkv", np.concatenate([drift(100, 7, 80), drift(100, 13, 170)])),
                         ("smooth.mkv", drift(200, 7, 120))):
        src = make(tmp_path / name, frames, fps=100, extra=("-vf", f"select='{keep}'", "-fps_mode", "vfr"))
        levels = []
        media.patch(src, tmp_path / "out.mkv", lambda f, levels=levels: levels.append(f.y.mean()))
        want = [i for i in range(1, len(levels)) if (levels[i - 1] < 125) != (levels[i] < 125)]
        got = media.cuts(src)
        assert len(got) == len(want) == (name == "cut.mkv") and all(abs(g - c) <= 1 for g, c in zip(got, want)), \
            (name, got, want)


def counted(n: int) -> np.ndarray:
    """n yuv frames whose luma shows each frame's number in binary, one 8x8 block per bit, so lossy coding keeps it."""
    frames = np.full((n, SIZE), 128, np.uint8)
    bits = (np.arange(n)[:, None] >> np.arange(12)) & 1  # 12 bits: up to 4096 frames
    blocks = np.where(bits, 200, 40).astype(np.uint8).reshape(n, 2, 6)
    luma = np.kron(blocks, np.ones((1, 8, 8), np.uint8))  # (n, 16, 48)
    frames[:, :W * H] = np.pad(luma, ((0, 0), (0, H - 16), (0, W - 48)), constant_values=128).reshape(n, -1)
    return frames


def number(y: np.ndarray) -> int:
    """The frame number a counted() luma shows."""
    return int(sum(1 << b for b in range(12) if y[b // 6 * 8 + 4, b % 6 * 8 + 4] > 120))


def test_planes_far_apart_seeks_each_sample_and_matches_a_full_decode(tmp_path):
    n = 2400  # 40 s at 60 fps: 4 samples are 600 frames apart, past the seek rule (the short clips stay under it)
    src = make(tmp_path / "src.mp4", counted(n), fps=60, extra=("-g", "120", "-preset", "ultrafast"))
    full = np.frombuffer(raw(src), np.uint8).reshape(-1, SIZE)
    assert len(full) == n and [number(f[:W * H].reshape(H, W)) for f in full[::599]] == list(range(0, n, 599))
    got = []
    for f in media.planes(src, 4):
        got.append(number(f.y))
        assert (np.concatenate((f.y.ravel(), f.u.ravel(), f.v.ravel())) == full[got[-1]]).all(), got
    assert got == [300, 900, 1500, 2100]  # the first frame at or after the middle of each quarter


def test_planes_seeks_on_late_video_and_skips_samples_past_its_end(tmp_path):
    video, src = make(tmp_path / "v.mkv", counted(2400), fps=60, extra=("-g", "120")), tmp_path / "late [4K] x.mkv"
    ff("-itsoffset", "1", "-i", str(video), "-f", "lavfi", "-i", "sine=duration=61", "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", str(src))  # video from 1 s to 41 s, sound to 61 s
    full = np.frombuffer(raw(src), np.uint8).reshape(-1, SIZE)
    got = []
    for f in media.planes(src, 4):  # asks for 7.5, 22.5, 37.5 and 52.5 s into the video
        got.append(number(f.y))
        assert (np.concatenate((f.y.ravel(), f.u.ravel(), f.v.ravel())) == full[got[-1]]).all(), got
    # the file's duration includes the sound's last packet, a few ms past 61 s, so a sample may land one frame later
    assert len(got) == 3 and all(0 <= g - want <= 1 for g, want in zip(got, (450, 1350, 2250))), got


SOUND = ("-f", "lavfi", "-i", "sine=duration=2", "-map", "0:v", "-map", "1:a", "-c:a", "aac")


def late(path: Path, frames: np.ndarray) -> Path:
    """A clip whose video starts 0.5 s after its sound."""
    video = make(path.with_name("v-" + path.name), frames)
    ff("-itsoffset", "0.5", "-i", str(video), *SOUND[:4], *SOUND[4:8], "-c:v", "copy", "-c:a", "aac", str(path))
    return path


def b_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=has_b_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out)


CLIPS = {  # name: how to make it from frames
    "30.mp4": lambda p, f: make(p, f, fps=30, extra=SOUND),
    "ntsc.mp4": lambda p, f: make(p, f, fps="30000/1001", extra=SOUND),
    "60.mkv": lambda p, f: make(p, f, fps=60, extra=SOUND),  # Matroska rounds every time to 1 ms
    "bframes.mp4": lambda p, f: make(p, f, fps=30, extra=SOUND, quality=("-crf", "10")),  # lossless has none
    "late.mkv": late,
}


def numbering() -> tuple[Callable, list[int]]:
    """An edit that paints each frame's number into its corner, and the list of numbers it has painted."""
    seen: list[int] = []

    def edit(f: media.Frame) -> None:
        f.y[:8, :8] = len(seen)
        seen.append(len(seen))

    return edit, seen


@pytest.mark.parametrize("name", CLIPS)
def test_parts_give_the_same_video_as_one_part(tmp_path, monkeypatch, name):
    monkeypatch.setattr(media, "CHUNK_S", 0.25)  # 6 to 15 frames per part
    src = CLIPS[name](tmp_path / f"src-{name}", noise(50, seed=8))
    if name == "bframes.mp4":
        assert b_frames(src) > 0  # sanity
    one = tmp_path / f"one-{name}"
    edit, seen = numbering()
    n = media.patch(src, one, edit, crf=0)
    folder = tmp_path / "out [4K 60FPS PS5 PRO] it's"
    folder.mkdir()
    out = folder / f"clean [4K 60FPS] {name}"
    edit, seen = numbering()
    progress = []
    assert media.patch(src, out, edit, crf=0, resume="key", progress=lambda *a: progress.append(a)) == n
    assert n == len(seen) == frame_count(src) == frame_count(out) == 50
    assert progress[0][0] == 0 and len(progress) >= 5 and progress[-1][0] == n  # really in parts
    assert raw(out) == raw(one)
    assert audio_md5(out) == audio_md5(src)
    assert video_pts(out) == video_pts(one)
    assert start(out, "v") - start(out, "a") == start(one, "v") - start(one, "a")
    assert [p.name for p in folder.iterdir()] == [out.name]


def test_parts_that_have_b_frames_join_with_the_same_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.25)
    src = make(tmp_path / "src.mp4", noise(50, seed=12), fps=30, extra=SOUND, quality=("-crf", "10"))
    one, out = tmp_path / "one.mp4", tmp_path / "out.mp4"
    # libx264 at a lossy crf, which makes B-frames; the Mac's hardware HEVC encoder makes none at its defaults
    assert media.patch(src, one, invert, crf=16) == media.patch(src, out, invert, crf=16, resume="k") == 50
    assert b_frames(out) > 0 and frame_count(out) == 50
    assert video_pts(out) == video_pts(one) and audio_md5(out) == audio_md5(src)


def invert(f: media.Frame) -> None:
    f.y[:8, :8] = 255 - f.y[:8, :8]


def stopping(at: int) -> tuple[Callable, list[int]]:
    """An edit that inverts a corner and fails as an unplugged disk would on its at-th frame (never with 0), and the
    list it adds one entry to per frame it is given."""
    calls: list[int] = []

    def edit(f: media.Frame) -> None:
        calls.append(1)
        if len(calls) == at:
            raise OSError("disk unplugged")
        invert(f)

    return edit, calls


def test_an_unfinished_run_continues_from_its_finished_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src = make(tmp_path / "src.mp4", noise(40, seed=9), fps=30, extra=SOUND)
    want = tmp_path / "want.mp4"
    media.patch(src, want, invert, crf=0)
    out, progress = tmp_path / "out.mp4", []
    edit, _ = stopping(15)
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, crf=0, resume="a")
    assert not out.exists()
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="a", progress=lambda *a: progress.append(a)) == 40
    assert len(calls) == 40 - 12 and progress[0][0] == 12  # the two finished parts were kept and used
    assert raw(out) == raw(want) and audio_md5(out) == audio_md5(src)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4", "src.mp4", "want.mp4"]


def test_a_continued_run_on_an_exfat_disk_does_not_count_the_companion_files_as_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src = make(tmp_path / "src.mp4", noise(40, seed=9), fps=30, extra=SOUND)
    want = tmp_path / "want.mp4"
    media.patch(src, want, invert, crf=0)
    out = tmp_path / "out.mp4"
    edit, _ = stopping(15)
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, crf=0, resume="a")
    parts = sorted(p for p in out.with_name("out.mp4.parts").iterdir() if p.suffix == ".nut")
    assert len(parts) == 2  # sanity
    for p in parts:  # what macOS puts beside each file on exFAT and FAT disks
        p.with_name("._" + p.name).write_bytes(b"\0\5\26\7" + bytes(4092))
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="a") == 40
    assert len(calls) == 40 - 12 and raw(out) == raw(want)


def test_a_stale_folder_is_replaced_when_a_file_in_it_vanishes_before_it_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src, out = make(tmp_path / "src.mp4", noise(40, seed=9), fps=30, extra=SOUND), tmp_path / "out.mp4"
    edit, _ = stopping(3)  # stopped before its first part, so its folder is replaced without asking
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, crf=0, resume="a")
    work = out.with_name("out.mp4.parts")
    pair = {"run.json": "._run.json", "._run.json": "run.json"}
    (work / "._run.json").write_bytes(bytes(4096))
    unlink, vanished = os.unlink, []

    def disk_unlink(path, *, dir_fd=None):
        """os.unlink on a disk that also removes the other of a file and its companion, as macOS does on exFAT; both
        ways round, so the second one rmtree listed is gone whichever order this disk lists them in."""
        unlink(path, dir_fd=dir_fd)
        head, name = os.path.split(path)
        if name in pair:
            try:
                unlink(other := os.path.join(head, pair[name]), dir_fd=dir_fd)
                vanished.append(other)
            except FileNotFoundError:  # already gone
                pass

    monkeypatch.setattr(os, "unlink", disk_unlink)
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="b") == 40 and len(calls) == 40
    assert vanished  # sanity: a listed file really was gone when rmtree got to it


def test_finished_parts_of_a_different_run_are_never_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src, out = make(tmp_path / "src.mp4", noise(40, seed=9), fps=30, extra=SOUND), tmp_path / "out.mp4"
    edit, _ = stopping(15)
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, crf=0, resume="a")
    with pytest.raises(RuntimeError, match=r"out\.mp4\.parts.*kept.*run the earlier command again.*delete"):
        media.patch(src, out, invert, crf=0, resume="b")
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="a") == 40 and len(calls) == 40 - 12  # still there


def test_a_run_stopped_before_its_first_part_is_replaced_without_asking(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src, out = make(tmp_path / "src.mp4", noise(40, seed=9), fps=30, extra=SOUND), tmp_path / "out.mp4"
    edit, _ = stopping(3)
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, crf=0, resume="a")
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="b") == 40 and len(calls) == 40


def test_a_video_of_exactly_whole_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part, 6 parts
    src = make(tmp_path / "src.mp4", noise(36, seed=10), fps=30, extra=SOUND)
    one, out = tmp_path / "one.mp4", tmp_path / "out.mp4"
    assert media.patch(src, one, invert, crf=0) == media.patch(src, out, invert, crf=0, resume="k") == 36
    assert raw(out) == raw(one) and frame_count(out) == 36


def stream(path: Path) -> tuple[str, str]:
    """The first video stream's codec name and tag, from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,codec_tag_string", "-of", "csv=p=0", str(path)], capture_output=True,
                         text=True, check=True).stdout
    return tuple(out.strip().split(","))


needs_hevc = pytest.mark.skipif(not HEVC, reason="needs an Apple-silicon Mac whose ffmpeg lists hevc_videotoolbox")


@needs_hevc
def test_default_on_this_mac_is_hevc_with_its_tags_frames_and_audio(tmp_path):
    tags = {"color_range": "tv", "color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709"}
    src = make(tmp_path / "src.mp4", drift(50, 7, 120), extra=(
        *SOUND, "-vf", "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"))
    assert tags.items() <= video_tags(src).items()
    for name in ("out.mp4", "out.mov", "out.mkv"):
        out = tmp_path / name
        assert media.patch(src, out, invert) == 50
        assert stream(out)[0] == "hevc" and (stream(out)[1] == "hvc1") == (name != "out.mkv"), name
        assert frame_count(out) == 50 and audio_md5(out) == audio_md5(src), name
        assert tags.items() <= video_tags(out).items(), name
        y = lumas(out).astype(float)
        assert np.abs(y[:, 8:, 8:] - lumas(src)[:, 8:, 8:]).mean() < 1, name  # close to the source outside the edit


@needs_hevc
def test_a_resumed_default_run_matches_an_uninterrupted_one(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 5 frames per part at 25 fps
    src = late(tmp_path / "src.mkv", drift(50, 7, 120))
    one = tmp_path / "one.mp4"
    assert media.patch(src, one, invert) == 50
    edit, calls = stopping(13)
    out = tmp_path / "out.mp4"
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, resume="k")
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, resume="k") == 50
    assert len(calls) == 40  # the two finished parts were kept and used
    assert stream(out) == stream(one) == ("hevc", "hvc1")
    assert frame_count(out) == 50 and video_pts(out) == video_pts(one) and audio_md5(out) == audio_md5(src)
    assert start(out, "v") - start(out, "a") == start(one, "v") - start(one, "a")
    assert abs(start(out, "v") - start(out, "a") - (start(src, "v") - start(src, "a"))) <= 0.5 / 25


def test_an_explicit_crf_gives_h264_and_crf_0_stays_exact(tmp_path):
    src = make(tmp_path / "src.mp4", noise(20, seed=13), extra=SOUND)
    for crf in (0, 16):
        out = tmp_path / f"out-{crf}.mp4"
        assert media.patch(src, out, lambda f: None, crf=crf) == 20
        assert stream(out)[0] == "h264", crf
    assert raw(tmp_path / "out-0.mp4") == raw(src)


@needs_hevc
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in ffmpeg is a shell script")
def test_a_hardware_encoder_that_will_not_start_falls_back_to_libx264(tmp_path, monkeypatch):
    monkeypatch.setenv("UNMARK_FFMPEG", str(failing_ffmpeg(tmp_path)))
    monkeypatch.setattr(media, "CHUNK_S", 0.2)
    why = "hevc_videotoolbox is listed but did not start: ffmpeg failed (1): no encoder here"
    assert media.encoder() == ("libx264", why)
    src = make(tmp_path / "src.mp4", noise(20, seed=14), extra=SOUND)
    out = tmp_path / "out.mp4"
    assert media.patch(src, out, invert, resume="k") == 20
    assert stream(out)[0] == "h264" and frame_count(out) == 20


@needs_hevc
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in ffmpeg is a shell script")
def test_finished_hardware_parts_are_kept_when_the_hardware_encoder_does_not_start_once(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 5 frames per part at 25 fps
    src, out = make(tmp_path / "src.mp4", drift(50, 7, 120), extra=SOUND), tmp_path / "out.mp4"
    edit, calls = stopping(13)
    with pytest.raises(OSError, match="unplugged"):
        media.patch(src, out, edit, resume="k")
    monkeypatch.setenv("UNMARK_FFMPEG", str(failing_ffmpeg(tmp_path)))  # busy this time
    with pytest.raises(RuntimeError, match="hevc_videotoolbox did not start this time.*kept.*try again"):
        media.patch(src, out, edit, resume="k")
    monkeypatch.delenv("UNMARK_FFMPEG")
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, resume="k") == 50
    assert len(calls) == 40 and stream(out)[0] == "hevc"  # the two finished parts were still there


def test_a_join_that_will_not_fit_on_the_disk_says_how_much_to_free_and_keeps_the_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)  # 6 frames per part
    src, out = make(tmp_path / "src.mp4", noise(40, seed=15), fps=30, extra=SOUND), tmp_path / "out.mp4"
    usage = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(path)._replace(free=0))
    with pytest.raises(RuntimeError, match=r"about [\d.]+ GB.*kept.*same command"):
        media.patch(src, out, invert, crf=0, resume="k")
    assert not out.exists()
    monkeypatch.setattr(shutil, "disk_usage", usage)
    edit, calls = stopping(0)
    assert media.patch(src, out, edit, crf=0, resume="k") == 40 and calls == []  # every part was kept


def test_a_variable_rate_video_is_written_in_one_part(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.2)
    keep = "+".join(f"eq(mod(n\\,24)\\,{k})" for k in (0, 4, 12))
    src = make(tmp_path / "src.mkv", noise(100, seed=11), fps=100, extra=(*SOUND, "-vf", f"select='{keep}'",
                                                                          "-fps_mode", "vfr"))
    one, out, progress = tmp_path / "one.mkv", tmp_path / "out.mkv", []
    n = media.patch(src, one, invert, crf=0)
    assert media.patch(src, out, invert, crf=0, resume="k", progress=lambda *a: progress.append(a)) == n
    assert raw(out) == raw(one) and video_pts(out) == video_pts(one) and len(progress) == 2


def tagged(folder: Path) -> Path:
    """A 1.6 s .mkv with a title and artist, two chapters, a subtitle, a font attachment and a language on each
    stream; the video also has a tag of its own."""
    meta, srt, font = folder / "meta.txt", folder / "sub.srt", folder / "font.ttf"
    meta.write_text(";FFMETADATA1\ntitle=Night Drive\nartist=Test Band\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=800\ntitle=Opening\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=800\nEND=1600\ntitle=Ending\n")
    srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nHello there\n")
    font.write_bytes(b"not really a font")
    return make(folder / "src.mkv", noise(40, seed=16), extra=(
        "-f", "lavfi", "-i", "sine=duration=1.6", "-i", str(srt), "-i", str(meta), "-attach", str(font),
        "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3", "-map_chapters", "3",
        "-metadata:s:v:0", "language=hin", "-metadata:s:v:0", "VIEW=left", "-metadata:s:a:0", "language=jpn",
        "-metadata:s:s:0", "language=ita", "-metadata:s:t:0", "mimetype=application/x-truetype-font",
        "-c:a", "aac", "-c:s", "srt"))


def described(path: Path) -> dict:
    """ffprobe's format, streams and chapters, with tag names in lower case."""
    out = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters",
                                     "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    for s in [*out["streams"], out["format"]]:
        s["tags"] = {k.lower(): v for k, v in s.get("tags", {}).items()}
    return out


def test_the_same_container_keeps_tags_chapters_subtitles_attachments_and_stream_languages(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 0.25)  # in parts, so the resumable join is the one checked
    src, out = tagged(tmp_path), tmp_path / "out.mkv"
    assert media.patch(src, out, invert, crf=0, resume="k") == 40
    got = described(out)
    assert {"title": "Night Drive", "artist": "Test Band"}.items() <= got["format"]["tags"].items()
    assert [c["tags"]["title"] for c in got["chapters"]] == ["Opening", "Ending"]
    kinds = {s["codec_type"]: s for s in got["streams"]}
    assert sorted(kinds) == ["attachment", "audio", "subtitle", "video"] and len(got["streams"]) == 4
    assert kinds["video"]["tags"]["language"] == "hin" and kinds["video"]["tags"]["view"] == "left"
    assert kinds["audio"]["tags"]["language"] == "jpn" and kinds["subtitle"]["tags"]["language"] == "ita"
    assert kinds["attachment"]["tags"]["filename"] == "font.ttf"
    assert b"Hello there" in ff("-i", str(out), "-map", "0:s", "-f", "srt", "-")
    assert kinds["video"]["tags"]["duration"].startswith("00:00:01.6")  # the output's own 40 frames at 25 fps
    assert frame_count(out) == 40 and audio_md5(out) == audio_md5(src)


def test_another_container_still_succeeds_and_keeps_picture_sound_tags_and_chapters(tmp_path):
    src, out = tagged(tmp_path), tmp_path / "out.mp4"  # .mp4 cannot hold this subtitle, so it must be left out
    assert media.patch(src, out, lambda f: None, crf=0) == 40  # in one piece, the way a variable-rate video is
    got = described(out)
    assert {"title": "Night Drive", "artist": "Test Band"}.items() <= got["format"]["tags"].items()
    assert [c["tags"]["title"] for c in got["chapters"]] == ["Opening", "Ending"]
    picture_and_sound = [(s["codec_type"], s["tags"]["language"]) for s in got["streams"]
                         if s["codec_type"] != "data"]  # .mp4 stores chapters in a data track of their own
    assert picture_and_sound == [("video", "hin"), ("audio", "jpn")]
    assert raw(out) == raw(src) and audio_md5(out) == audio_md5(src)
