"""fill.fill on generated crops with a known clean truth, through OpenCV and through a stand-in ProPainter script."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from unmark import fill

W = H = 64
FAKE = Path(__file__).parent / "fake_propainter"


def smooth(n: int, w: int = W, h: int = H) -> np.ndarray:
    """n RGB frames (n, h, w, 3) of a soft colour pattern drifting one pixel per frame: the clean truth."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, h, w, 3), np.uint8)
    for i in range(n):
        x = xx + i
        for c, (fx, fy) in enumerate(((40, 55), (65, 35), (50, 80))):
            out[i, ..., c] = np.clip(128 + 60 * np.sin(x / fx + c) * np.cos(yy / fy), 0, 255).round()
    return out


def block(w: int = W, h: int = H, size: int = 16) -> np.ndarray:
    """A centred square mask."""
    mask = np.zeros((h, w), bool)
    mask[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2] = True
    return mask


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def check_fills_block(out: list[np.ndarray], clean: np.ndarray, seen: np.ndarray, mask: np.ndarray) -> None:
    """Inside the mask much closer to the truth than the stamped input; outside the grown mask untouched."""
    assert len(out) == len(clean)
    out = np.stack(out)
    assert psnr(out[:, mask], clean[:, mask]) >= psnr(seen[:, mask], clean[:, mask]) + 10
    reach = fill.GROW + fill.FEATHER
    near = cv2.dilate(mask.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)).astype(bool)
    assert (out[:, ~near] == seen[:, ~near]).all()


def test_opencv_fills_a_solid_block(monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)
    clean, mask = smooth(12), block()
    seen = clean.copy()
    seen[:, mask] = 255
    check_fills_block(list(fill.fill(iter(seen), mask)), clean, seen, mask)


@pytest.fixture
def fake(monkeypatch, tmp_path) -> Path:
    """Point fill at the stand-in ProPainter checkout; returns the log file it appends one JSON line per run to."""
    monkeypatch.setenv("UNMARK_PROPAINTER", str(FAKE))
    monkeypatch.setenv("UNMARK_PROPAINTER_PYTHON", sys.executable)
    log = tmp_path / "runs.jsonl"
    monkeypatch.setenv("FAKE_PROPAINTER_LOG", str(log))
    return log


def runs(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_propainter_fills_a_solid_block(fake):
    clean, mask = smooth(12), block()
    seen = clean.copy()
    seen[:, mask] = 255
    check_fills_block(list(fill.fill(iter(seen), mask)), clean, seen, mask)
    (run,) = runs(fake)
    assert run["frames"] == 12 and Path(run["cwd"]).resolve() == FAKE.resolve()
    assert {"--save_frames", "--fp16"} <= set(run["argv"])


def numbered(n: int, w: int = 32, h: int = 32) -> np.ndarray:
    """n flat grey frames, each carrying its own number in the top-left pixel (red = n % 256, green = n // 256)."""
    out = np.full((n, h, w, 3), 90, np.uint8)
    out[:, 0, 0, 0] = np.arange(n) % 256
    out[:, 0, 0, 1] = np.arange(n) // 256
    return out


def number(pixel) -> int:
    return int(pixel[0]) + 256 * int(pixel[1])


def test_long_stream_is_cut_into_overlapping_pieces_that_respect_scene_cuts(fake):
    frames = numbered(900)
    out = list(fill.fill(iter(frames), block(32, 32, 8), cuts=[450]))
    assert [number(f[0, 0]) for f in out] == list(range(900))
    pieces = [[number(c) for c in r["corners"]] for r in runs(fake)]
    assert all(p == list(range(p[0], p[0] + len(p))) and len(p) <= fill.PIECE for p in pieces)
    assert all(p[-1] < 450 or p[0] >= 450 for p in pieces)
    assert sorted({i for p in pieces for i in p}) == list(range(900))
    for a, b in itertools.pairwise(pieces):
        assert b[0] == (450 if a[-1] == 449 else a[-1] - 9)


def test_one_frame_scene_is_filled_without_propainter(fake):
    clean, mask = smooth(6), block()
    seen = clean.copy()
    seen[:, mask] = 255
    out = list(fill.fill(iter(seen), mask, cuts=[2, 3]))  # frame 2 is a scene on its own
    check_fills_block(out, clean, seen, mask)
    assert [r["frames"] for r in runs(fake)] == [2, 3]


@pytest.mark.parametrize("how, says", [("exit", "ProPainter failed.*CUDA out of memory"),
                                       ("short", "ProPainter returned 11 frames for 12")])
def test_a_failing_script_ends_in_a_plain_error(fake, monkeypatch, how, says):
    monkeypatch.setenv("FAKE_PROPAINTER_BREAK", how)
    with pytest.raises(RuntimeError, match=says):
        list(fill.fill(iter(smooth(12)), block()))


def test_a_folder_that_is_not_a_checkout_is_refused(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("UNMARK_PROPAINTER", str(tmp_path))
    with pytest.raises(RuntimeError, match="not a ProPainter checkout"):
        list(fill.fill(iter(smooth(4)), block()))


def test_a_checkout_without_its_python_is_refused(fake, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER_PYTHON")  # the stand-in has no .venv
    with pytest.raises(RuntimeError, match="UNMARK_PROPAINTER_PYTHON"):
        list(fill.fill(iter(smooth(4)), block()))


def test_overlap_frames_fade_from_one_piece_to_the_next(fake, monkeypatch):
    monkeypatch.setenv("FAKE_PROPAINTER_FILL", "mean")  # each run fills with its frames' average colour
    frames = numbered(410)
    corner = frames[:, 0, 0].copy()
    frames[:] = (np.arange(410) // 2)[:, None, None, None]  # brightness i // 2
    frames[:, 0, 0] = corner
    out = list(fill.fill(iter(frames), block(32, 32, 8)))
    assert [number(f[0, 0]) for f in out] == list(range(410))
    centre = np.array([int(f[16, 16, 0]) for f in out])
    first, second = centre[0], centre[-1]  # pieces 0-399 and 390-409 average brightness 99.75 and 199.5
    assert abs(first - 99.75) <= 1 and abs(second - 199.5) <= 1
    assert (centre[:390] == first).all() and (centre[400:] == second).all()
    ramp = centre[389:401]
    assert (np.diff(ramp) > 0).all() and np.diff(ramp).max() <= (second - first) / 5
