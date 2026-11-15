import numpy as np
from helpers import (
    clean_frames,
    decode_luma,
    encode,
    has_audio,
    psnr,
    stamp,
    text_opacity,
)

from unmark import cli

W, H = 320, 180
BOX = (208, 16, 96, 48)  # near the top-right corner, like a typical logo
MARGIN = 32  # at least the tool's own padding


def test_clean_removes_see_through_overlay(tmp_path):
    real = clean_frames(150, W, H, seed=1)
    clean = encode(real, tmp_path / "clean.mp4")
    marked = encode(stamp(real, text_opacity(W, H, BOX)), tmp_path / "marked.mp4")
    out = tmp_path / "out.mp4"

    assert cli.main(["clean", str(marked), "--box", ",".join(map(str, BOX)), "--out", str(out)]) == 0

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
