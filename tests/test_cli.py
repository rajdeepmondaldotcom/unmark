import json
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    clean_frames,
    decode,
    decode_luma,
    encode,
    footage,
    has_audio,
    psnr,
    smooth,
    stamp,
    still,
    text_opacity,
)

from unmark import cli, fill, marks, media

W, H = 320, 180
BOX = (208, 16, 96, 48)  # near the top-right corner, like a typical logo
MARGIN = 32  # at least the tool's own padding
PANEL = (32, 100, 64, 40)  # an opaque panel, lower left


def check_cleaned(clean, marked, out, box=BOX, bar=35):
    """The output matches the clean clip inside box and the marked clip well away from it."""
    c, m, o = (decode_luma(p, W, H) for p in (clean, marked, out))
    assert len(o) == len(m)
    x, y, w, h = box
    inside = np.s_[:, y:y + h, x:x + w]
    before, after = psnr(m[inside], c[inside]), psnr(o[inside], c[inside])
    far = np.ones((H, W), bool)
    far[max(0, y - MARGIN):y + h + MARGIN, max(0, x - MARGIN):x + w + MARGIN] = False
    untouched = psnr(o[:, far], m[:, far])
    print(f"inside box: marked {before:.2f} dB, output {after:.2f} dB; outside region: {untouched:.2f} dB")
    assert after >= bar and after >= before + 10
    assert untouched > 40
    assert has_audio(out)


def clips(tmp_path):
    real = clean_frames(150, W, H, seed=1)
    return encode(real, tmp_path / "clean.mp4"), encode(stamp(real, text_opacity(W, H, BOX)), tmp_path / "marked.mp4")


def test_clean_removes_see_through_overlay(tmp_path):
    clean, marked = clips(tmp_path)
    out = tmp_path / "out.mp4"
    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0
    check_cleaned(clean, marked, out)


def test_clean_removes_an_overlay_the_colour_of_the_average_picture(tmp_path):
    real = clean_frames(150, W, H, seed=1)
    grey = float(real.mean())  # the overlay's colour minus the picture behind it averages to about zero
    clean, marked = encode(real, tmp_path / "clean.mp4"), encode(stamp(real, text_opacity(W, H, BOX), grey),
                                                                   tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0

    check_cleaned(clean, marked, out)


def test_clean_beats_delogo_on_compressed_footage(tmp_path, capsys):
    real = footage(300, W, H, seed=5)
    truth = decode_luma(encode(real, tmp_path / "clean.mp4", audio=False), W, H)
    marked = encode(stamp(real, text_opacity(W, H, BOX)), tmp_path / "marked.mp4", crf=28)
    out, delogo = tmp_path / "out.mp4", tmp_path / "delogo.mp4"
    x, y, w, h = BOX
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(marked), "-vf",
                    f"delogo=x={x}:y={y}:w={w}:h={h}", "-c:v", "libx264", "-qp", "0", str(delogo)], check=True)

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0

    inside = np.s_[:, y:y + h, x:x + w]
    ceiling = encode(real, tmp_path / "clean28.mp4", audio=False, crf=28)
    before, after, theirs, best = (psnr(decode_luma(p, W, H)[inside], truth[inside])
                                   for p in (marked, out, delogo, ceiling))
    print(f"inside box vs lossless clean: marked {before:.2f} dB, unmark {after:.2f} dB, delogo {theirs:.2f} dB, "
          f"clean at crf 28 {best:.2f} dB")
    assert after >= theirs + 3 and after > before
    said = capsys.readouterr().out
    shown = re.search(r"outline visibility \(1\.0 = gone\): input ([\d.]+), output ([\d.]+)", said)
    assert shown and float(shown[1]) > 1.5 and float(shown[2]) < 1.15 and "still visible" not in said


def test_clean_leaves_no_outline_on_smooth_still_scenes(tmp_path, capsys):
    real = still(160, W, H, scenes=4)
    truth = decode_luma(encode(real, tmp_path / "clean.mp4", audio=False), W, H)
    marked = encode(stamp(real, text_opacity(W, H, BOX), 215), tmp_path / "marked.mp4", audio=False, crf=23)
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0

    x, y, w, h = BOX
    o, t = (v[:, y:y + h, x:x + w] for v in (decode_luma(out, W, H), truth))
    worst = min(psnr(a, b) for a, b in zip(o, t))
    print(f"inside box vs lossless clean: average {psnr(o, t):.2f} dB, worst frame {worst:.2f} dB")
    # outline-free results measure 49.2 to 49.8 dB, worst frame 43.9 to 45.7, over seeds 0 to 2; the faint outline
    # this test was written against measured 45.3 and 41.8
    assert psnr(o, t) >= 48 and worst >= 43
    said = capsys.readouterr().out
    shown = re.search(r"outline visibility \(1\.0 = gone\): input ([\d.]+), output ([\d.]+)", said)
    assert shown and float(shown[2]) <= cli.VISIBLE and "still visible" not in said


def test_clean_warns_when_the_outline_is_still_visible(tmp_path, capsys, monkeypatch):
    _, marked = clips(tmp_path)
    monkeypatch.setattr(marks.Mark, "erase", lambda self, frame: None)  # an erase that removes nothing

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(tmp_path / "out.mp4")]) == 0

    assert "still visible" in capsys.readouterr().out


def test_clean_finds_the_overlay_and_removes_the_chosen_mark(tmp_path):
    clean, marked = clips(tmp_path)
    out = tmp_path / "out.mp4"
    assert cli.main(["clean", str(marked), "--mark", "1", "--out", str(out)]) == 0
    check_cleaned(clean, marked, out)
    assert (tmp_path / "marked.marks.png").is_file()


def test_clean_asks_which_mark_at_a_terminal(tmp_path, monkeypatch):
    clean, marked = clips(tmp_path)
    out = tmp_path / "out.mp4"
    shown = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    monkeypatch.setattr(cli, "_show", shown.append)

    assert cli.main(["clean", str(marked), "--out", str(out)]) == 0

    assert shown == [tmp_path / "marked.marks.png"]
    check_cleaned(clean, marked, out)


def test_clean_says_when_no_overlay_is_found(tmp_path, capsys):
    clean, _ = clips(tmp_path)
    assert cli.main(["clean", str(clean), "--mark", "1"]) == 1
    said = capsys.readouterr().out
    assert "no fixed overlay was found" in said and "--box" in said


def test_clean_says_when_the_mark_number_is_not_listed(tmp_path, capsys):
    _, marked = clips(tmp_path)
    assert cli.main(["clean", str(marked), "--mark", "2"]) == 1
    assert "there is no overlay 2" in capsys.readouterr().out


def solid_clips(tmp_path, frames, panel=PANEL):
    """The clean clip and the same with an opaque white panel over panel."""
    marked = frames.copy()
    x, y, w, h = panel
    marked[:, y:y + h, x:x + w] = 230
    return encode(frames, tmp_path / "clean.mp4"), encode(marked, tmp_path / "solid.mp4")


def test_clean_fills_a_solid_mark(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)
    clean, solid = solid_clips(tmp_path, smooth(150, W, H, seed=1))
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(solid), "--mark", "1", "--out", str(out)]) == 0

    said = capsys.readouterr().out
    check_cleaned(clean, solid, out, PANEL, bar=28)  # 30.5 to 33.9 dB measured over seeds 1 to 5
    assert "1: solid" in said and "softer" in said
    shown = re.search(r"outline visibility \(1\.0 = gone\): input ([\d.]+), output ([\d.]+)", said)
    assert shown and float(shown[1]) > 10 and float(shown[2]) < cli.VISIBLE and "still visible" not in said


def test_clean_fills_a_solid_mark_with_propainter_one_scene_at_a_time(tmp_path, capsys, monkeypatch):
    log = tmp_path / "runs.jsonl"
    monkeypatch.setenv("UNMARK_PROPAINTER", str(Path(__file__).parent / "fake_propainter"))
    monkeypatch.setenv("UNMARK_PROPAINTER_PYTHON", sys.executable)
    monkeypatch.setenv("FAKE_PROPAINTER_LOG", str(log))
    cut = 70
    clean, solid = solid_clips(tmp_path, np.concatenate([smooth(cut, W, H, seed=1), smooth(80, W, H, seed=2)]))
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(solid), "--mark", "1", "--out", str(out)]) == 0

    assert "softer" not in capsys.readouterr().out
    check_cleaned(clean, solid, out, PANEL, bar=28)
    found = media.cuts(solid)
    assert len(found) == 1 and abs(found[0] - cut) <= 1, found
    sizes = [json.loads(line)["frames"] for line in log.read_text().splitlines()]
    starts = np.cumsum([0, *sizes])  # the clip is shorter than one piece, so pieces only split at cuts
    assert starts[-1] == 150 and found[0] in starts, sizes


def test_a_given_box_on_a_solid_mark_is_filled_and_nothing_else_changes(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)
    monkeypatch.setattr(media, "patch", partial(media.patch, crf=0))  # lossless, so untouched means byte-identical
    # saturated, so some colours do not survive a trip through RGB and back: only what the fill changed may be pasted
    vivid = np.clip((smooth(150, W, H, seed=3).astype(np.float32) - 128) * 3 + 128, 0, 255).astype(np.uint8)
    clean, solid = solid_clips(tmp_path, vivid)
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(solid), "--box", ",".join(map(str, PANEL)), "--out", str(out)]) == 0

    assert "softer" in capsys.readouterr().out
    check_cleaned(clean, solid, out, PANEL, bar=28)
    outside_unchanged(out, solid, PANEL, REACH)


REACH = fill.GROW + fill.FEATHER + 2  # pixels a solid fill's feathered edge reaches, plus the colour pixel it shares


def outside_unchanged(out, marked, box, reach=0):
    """Every byte of all three planes more than reach pixels outside box (even-aligned) is the same in out as in
    marked."""
    x, y, w, h = box
    keep = np.ones((H, W), bool)
    keep[max(0, y - reach):y + h + reach, max(0, x - reach):x + w + reach] = False
    o, m = (decode(p, W, H) for p in (out, marked))
    for plane, size, k in ((0, W * H, keep), (1, W * H // 4, keep[::2, ::2]), (2, W * H // 4, keep[::2, ::2])):
        start = 0 if plane == 0 else W * H + (plane - 1) * size
        a, b = (f[:, start:start + size].reshape(len(f), *k.shape) for f in (o, m))
        assert (a[:, k] == b[:, k]).all(), (plane, int((a[:, k] != b[:, k]).sum()))


def plate(box):
    """Opacity of letters on a see-through plate that fills box exactly, so the overlay reaches every side of it."""
    x, y, w, h = box
    opacity = text_opacity(W, H, box)
    opacity[y:y + h, x:x + w] = np.maximum(opacity[y:y + h, x:x + w], 0.25)
    return opacity


def test_cleaning_a_see_through_mark_changes_nothing_outside_its_box(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "patch", partial(media.patch, crf=0))  # lossless, so untouched means byte-identical
    real = still(160, W, H, scenes=4)  # smooth, so the fit from the surrounding picture is the one kept
    clean, marked = encode(real, tmp_path / "clean.mp4"), encode(stamp(real, plate(BOX)), tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0  # tight

    outside_unchanged(out, marked, BOX)
    check_cleaned(clean, marked, out)


def test_a_solid_panel_in_the_corner_of_the_frame_is_filled(tmp_path, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)
    monkeypatch.setattr(media, "patch", partial(media.patch, crf=0))
    corner = (256, 140, 64, 40)  # touching the right and bottom edges
    clean, solid = solid_clips(tmp_path, smooth(150, W, H, seed=1), corner)
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(solid), "--box", ",".join(map(str, corner)), "--out", str(out)]) == 0

    check_cleaned(clean, solid, out, corner, bar=28)
    outside_unchanged(out, solid, corner, REACH)


def test_a_see_through_overlay_in_the_corner_of_the_frame_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "patch", partial(media.patch, crf=0))
    corner = (224, 132, 96, 48)  # its plate reaches the right and bottom edges
    real = clean_frames(150, W, H, seed=1)
    clean, marked = encode(real, tmp_path / "clean.mp4"), encode(stamp(real, plate(corner)), tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, corner)), "--out", str(out)]) == 0

    check_cleaned(clean, marked, out, corner)
    outside_unchanged(out, marked, corner)


def test_a_given_box_with_nothing_in_it_is_refused_in_plain_words(tmp_path, capsys):
    _, marked = clips(tmp_path)  # the overlay is top right; PANEL, lower left, has only moving picture in it
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, PANEL)), "--out", str(out)]) == 1

    said = capsys.readouterr().out
    print(said)
    assert "nothing see-through or solid" in said and said.count("\n") == 1
    assert not out.exists()


@pytest.mark.parametrize("how", ["capitals", "hard link"])
def test_an_out_that_is_the_input_file_is_refused(tmp_path, capsys, how):
    src = clips(tmp_path)[1].rename(tmp_path / "clip.mp4")  # would be cleaned, so overwritten without the check
    before = src.read_bytes()
    out = tmp_path / "CLIP.mp4"
    if how == "hard link":
        out = tmp_path / "other.mp4"
        out.hardlink_to(src)
    elif not out.exists():
        pytest.skip("this disk tells capitals apart, so CLIP.mp4 is another file")

    assert cli.main(["clean", str(src), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 1

    assert "--out must differ from the input video" in capsys.readouterr().out
    assert src.read_bytes() == before


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores a folder's read-only flag")
def test_a_picture_that_cannot_be_saved_ends_in_a_plain_error(tmp_path, capsys, monkeypatch):
    _, marked = clips(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    tmp_path.chmod(0o555)
    try:
        assert cli.main(["clean", str(marked)]) == 1
    finally:
        tmp_path.chmod(0o755)

    said = capsys.readouterr().out
    assert not (tmp_path / "marked.marks.png").exists()
    assert "could not save" in said and "--mark N" not in said, said


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores a folder's read-only flag")
def test_with_mark_given_a_picture_that_cannot_be_saved_is_only_a_warning(tmp_path, capsys):
    (tmp_path / "shared").mkdir()
    _, marked = clips(tmp_path / "shared")
    out = tmp_path / "out.mp4"
    (tmp_path / "shared").chmod(0o555)
    try:
        assert cli.main(["clean", str(marked), "--mark", "1", "--out", str(out)]) == 0
    finally:
        (tmp_path / "shared").chmod(0o755)

    assert "warning: could not save" in capsys.readouterr().out and out.is_file()


def test_an_out_ending_that_cannot_be_written_is_refused(tmp_path, capsys):
    _, marked = clips(tmp_path)
    out = tmp_path / "clip.webm"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 1

    said = capsys.readouterr().out
    assert ".mp4, .mov, .mkv" in said and said.count("\n") == 1 and not out.exists()


def test_a_given_box_on_a_picture_that_barely_changes_says_so(tmp_path, capsys):
    real = still(150, W, H, scenes=1)  # one scene drifting a few levels: nothing moves behind the overlay
    marked = encode(stamp(real, text_opacity(W, H, BOX)), tmp_path / "still.mp4")

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(tmp_path / "o.mp4")]) == 1

    said = capsys.readouterr().out
    assert "barely changes" in said and said.count("\n") == 1, said


def test_the_picture_sent_to_propainter_is_at_most_640_wide(tmp_path, monkeypatch):
    log = tmp_path / "runs.jsonl"
    monkeypatch.setenv("UNMARK_PROPAINTER", str(Path(__file__).parent / "fake_propainter"))
    monkeypatch.setenv("UNMARK_PROPAINTER_PYTHON", sys.executable)
    monkeypatch.setenv("FAKE_PROPAINTER_LOG", str(log))
    w, h, box = 1280, 720, (400, 500, 400, 100)  # half its width of padding each side would be 800 wide
    frames = smooth(40, w, h, seed=1)
    x, y, bw, bh = box
    frames[:, y:y + bh, x:x + bw] = 230
    solid = encode(frames, tmp_path / "solid.mp4", audio=False)

    assert cli.main(["clean", str(solid), "--box", ",".join(map(str, box)), "--out", str(tmp_path / "o.mp4")]) == 0

    widths = {json.loads(line)["size"][0] for line in log.read_text().splitlines()}
    assert widths and all(bw + 2 * cli.ROOM <= v <= 640 for v in widths), widths


def test_clean_without_a_terminal_asks_for_mark(tmp_path, capsys, monkeypatch):
    _, marked = clips(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert cli.main(["clean", str(marked)]) == 1

    said = capsys.readouterr().out
    assert "--mark N" in said and str(tmp_path / "marked.marks.png") in said


@pytest.mark.parametrize("where", [["--box", ",".join(map(str, BOX))], ["--mark", "1"]])
def test_clean_leaves_frames_without_the_overlay_alone(tmp_path, where):
    real = clean_frames(300, W, H, seed=6)
    intro = len(real) // 5
    marked_frames = real.copy()
    marked_frames[intro:] = stamp(real[intro:], text_opacity(W, H, BOX))
    clean, marked = encode(real, tmp_path / "clean.mp4"), encode(marked_frames, tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), *where, "--out", str(out)]) == 0

    c, m, o = (decode_luma(p, W, H) for p in (clean, marked, out))
    x, y, w, h = BOX
    inside = np.s_[y:y + h, x:x + w]
    untouched = [psnr(o[i][inside], m[i][inside]) for i in range(intro)]
    cleaned = [psnr(o[i][inside], c[i][inside]) for i in range(intro, len(o))]
    print(f"frames without the overlay, output vs input: worst {min(untouched):.2f} dB; "
          f"frames with it, output vs clean: worst {min(cleaned):.2f} dB, median {np.median(cleaned):.2f} dB")
    assert min(untouched) > 40 and min(cleaned) >= 35


def test_a_stopped_clean_continues_where_it_stopped(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 1)  # 30 frames per part
    clean, marked = clips(tmp_path)
    out = tmp_path / "out.mp4"
    erase, calls, stop_at = marks.Mark.erase, [], 45  # Ctrl-C halfway through the second part

    def stopping(self, frame):
        calls.append(1)
        if len(calls) == stop_at:
            raise KeyboardInterrupt
        erase(self, frame)

    monkeypatch.setattr(marks.Mark, "erase", stopping)
    args = ["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]
    assert cli.main(args) == 130
    assert not out.exists() and "again" in capsys.readouterr().out
    calls.clear()
    stop_at = 0
    assert cli.main(args) == 0
    assert len(calls) == 150 - 30  # measured to the same key, so the first part was kept
    assert "continuing" in capsys.readouterr().out
    check_cleaned(clean, marked, out)


def reading(monkeypatch) -> list[str]:
    """The names of the videos media.planes is asked to read from, as it is asked."""
    names, planes = [], media.planes

    def counted(src, count):
        names.append(Path(src).name)
        return planes(src, count)

    monkeypatch.setattr(media, "planes", counted)
    return names


@pytest.mark.parametrize("where", [["--box", ",".join(map(str, BOX))], ["--mark", "1"]])
def test_a_continued_clean_does_not_measure_the_overlay_again(tmp_path, capsys, monkeypatch, where):
    monkeypatch.setattr(media, "CHUNK_S", 1)  # 30 frames per part
    monkeypatch.setattr(media, "patch", partial(media.patch, crf=0))  # lossless, so the same run gives the same bytes
    _, marked = clips(tmp_path)
    one, out = tmp_path / "one.mp4", tmp_path / "out.mp4"
    assert cli.main(["clean", str(marked), *where, "--out", str(one)]) == 0
    visible = re.search(r"input [\d.]+, output [\d.]+", capsys.readouterr().out)[0]
    erase, calls = marks.Mark.erase, []

    def stopping(self, frame):
        calls.append(1)
        if len(calls) == 45:  # Ctrl-C halfway through the second part
            raise KeyboardInterrupt
        erase(self, frame)

    monkeypatch.setattr(marks.Mark, "erase", stopping)
    assert cli.main(["clean", str(marked), *where, "--out", str(out)]) == 130
    monkeypatch.setattr(marks.Mark, "erase", erase)
    capsys.readouterr()
    read = reading(monkeypatch)

    assert cli.main(["clean", str(marked), *where, "--out", str(out)]) == 0

    said = capsys.readouterr().out
    assert read == ["out.mp4"], read  # only the check of the written video; nothing of the input was read to measure
    assert "earlier" in said and "continuing" in said and "see-through at" not in said, said
    assert decode(out, W, H).tobytes() == decode(one, W, H).tobytes() and visible in said
    left = {"clean.mp4", "marked.mp4", "one.mp4", "out.mp4"} | ({"marked.marks.png"} if "--mark" in where else set())
    assert {p.name for p in tmp_path.iterdir()} == left  # nothing of the stopped run is left behind


def test_a_stopped_clean_does_not_continue_with_changed_cleaning_code(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(media, "CHUNK_S", 1)  # 30 frames per part
    marked = clips(tmp_path)[1]
    out = tmp_path / "out.mp4"
    erase, calls = marks.Mark.erase, []

    def stopping(self, frame):
        calls.append(1)
        if len(calls) == 45:  # Ctrl-C halfway through the second part
            raise KeyboardInterrupt
        erase(self, frame)

    monkeypatch.setattr(marks.Mark, "erase", stopping)
    args = ["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]
    assert cli.main(args) == 130
    tuned = tmp_path / "tuned"  # the package as it would be after a constant in marks.py was tuned
    tuned.mkdir()
    for f in Path(cli.__file__).parent.glob("*.py"):
        (tuned / f.name).write_bytes(f.read_bytes() + (b"\n# tuned\n" if f.name == "marks.py" else b""))
    monkeypatch.setattr(cli, "__file__", str(tuned / "cli.py"))
    capsys.readouterr()
    read = reading(monkeypatch)
    assert cli.main(args) == 1
    said = capsys.readouterr().out
    assert "kept" in said and "delete" in said and not out.exists()
    assert "marked.mp4" in read  # the earlier run's measurement was not used by the changed code
