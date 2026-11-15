"""media.probe / planes / patch against real ffmpeg on tiny generated clips. Expectations come from ffmpeg directly."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

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
         codec: str = "libx264") -> Path:
    """Encode raw yuv frames (n, SIZE) losslessly, followed by any extra ffmpeg arguments (inputs, maps, tags)."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
                    "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *extra, "-c:v", codec, "-qp", "0",
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
