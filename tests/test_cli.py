import re
import subprocess

import numpy as np
from helpers import (
    clean_frames,
    decode_luma,
    encode,
    footage,
    has_audio,
    psnr,
    stamp,
    text_opacity,
)

from unmark import cli, marks

W, H = 320, 180
BOX = (208, 16, 96, 48)  # near the top-right corner, like a typical logo
MARGIN = 32  # at least the tool's own padding


def check_cleaned(clean, marked, out):
    """The output matches the clean clip inside BOX and the marked clip well away from it."""
    c, m, o = (decode_luma(p, W, H) for p in (clean, marked, out))
    assert len(o) == len(m)
    x, y, w, h = BOX
    inside = np.s_[:, y:y + h, x:x + w]
    before, after = psnr(m[inside], c[inside]), psnr(o[inside], c[inside])
    far = np.ones((H, W), bool)
    far[max(0, y - MARGIN):y + h + MARGIN, max(0, x - MARGIN):x + w + MARGIN] = False
    untouched = psnr(o[:, far], m[:, far])
    print(f"inside box: marked {before:.2f} dB, output {after:.2f} dB; outside region: {untouched:.2f} dB")
    assert after >= 35 and after >= before + 10
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


def test_clean_refuses_a_solid_mark(tmp_path, capsys):
    frames = clean_frames(150, W, H, seed=4)
    frames[:, 100:140, 30:90] = 230  # an opaque white panel
    solid = encode(frames, tmp_path / "solid.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(solid), "--mark", "1", "--out", str(out)]) == 1

    said = capsys.readouterr().out
    assert "1: solid" in said and "solid marks is not supported yet" in said
    assert not out.exists()


def test_clean_without_a_terminal_asks_for_mark(tmp_path, capsys, monkeypatch):
    _, marked = clips(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert cli.main(["clean", str(marked)]) == 1

    said = capsys.readouterr().out
    assert "--mark N" in said and str(tmp_path / "marked.marks.png") in said


def test_clean_leaves_frames_without_the_overlay_alone(tmp_path):
    real = clean_frames(300, W, H, seed=6)
    intro = len(real) // 5
    marked_frames = real.copy()
    marked_frames[intro:] = stamp(real[intro:], text_opacity(W, H, BOX))
    clean, marked = encode(real, tmp_path / "clean.mp4"), encode(marked_frames, tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0

    c, m, o = (decode_luma(p, W, H) for p in (clean, marked, out))
    x, y, w, h = BOX
    inside = np.s_[y:y + h, x:x + w]
    untouched = [psnr(o[i][inside], m[i][inside]) for i in range(intro)]
    cleaned = [psnr(o[i][inside], c[i][inside]) for i in range(intro, len(o))]
    print(f"frames without the overlay, output vs input: worst {min(untouched):.2f} dB; "
          f"frames with it, output vs clean: worst {min(cleaned):.2f} dB, median {np.median(cleaned):.2f} dB")
    assert min(untouched) > 40 and min(cleaned) >= 35
