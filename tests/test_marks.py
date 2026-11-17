import subprocess

import cv2
import numpy as np
import pytest
from helpers import clean_frames, encode, footage, psnr, stamp, still, text_opacity

from unmark import marks, media

W, H = 320, 180
TEXT = (200, 40, 96, 48)  # see-through text, upper right
PANEL = (24, 110, 64, 40)  # solid decoy with a fixed pattern on it, lower left, like a game HUD panel
BAR = 20  # letterbox bar height, top and bottom


def hud(frames: np.ndarray) -> np.ndarray:
    """Paint an opaque panel with a fixed checker pattern over every frame."""
    x, y, w, h = PANEL
    yy, xx = np.mgrid[:h, :w]
    frames[:, y:y + h, x:x + w] = (np.where((yy // 8 + xx // 8) % 2, 60, 170)[..., None] + [0, 20, 40])
    return frames


def letterbox(frames: np.ndarray) -> np.ndarray:
    """Black bars across the top and bottom of every frame."""
    frames[:, :BAR] = frames[:, -BAR:] = 0
    return frames


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    return iw * ih / (aw * ah + bw * bh - iw * ih)


def extent(opacity: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of every pixel the stamped overlay touches."""
    ys, xs = np.nonzero(opacity)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def test_find_tells_see_through_text_from_solid_panel_and_ignores_letterbox(tmp_path):
    opacity = text_opacity(W, H, TEXT, peak=0.45)
    clip = encode(letterbox(hud(stamp(clean_frames(150, W, H, seed=2), opacity))), tmp_path / "a.mp4", audio=False)

    found = marks.find(lambda: media.planes(clip, 1000))

    assert [m.kind for m in found] == ["see-through", "solid"]  # the text sits higher than the panel
    see = [m for m in found if m.kind == "see-through"]
    print(f"box {see[0].box} vs true {extent(opacity)}: IoU {iou(see[0].box, extent(opacity)):.3f}; "
          f"opacity {see[0].opacity:.3f} vs 0.45")
    assert iou(see[0].box, extent(opacity)) > 0.8
    assert abs(see[0].opacity - 0.45) <= 0.05
    solid = [m for m in found if m.kind == "solid"]
    assert len(solid) == 1 and iou(solid[0].box, PANEL) > 0.5
    assert not [m for m in found if m.box[1] < BAR or m.box[1] + m.box[3] > H - BAR]


def scenery(frames: np.ndarray, scenes: int, seed: int = 0) -> np.ndarray:
    """Give each scene of footage its own still, sharp, high-contrast structure, the way test patterns, a game's
    level or a static shot have: a full-height line, a full-width line (each scene's in a different place) and a few
    hard blocks, held for that scene."""
    rng = np.random.default_rng(seed)
    n, h, w, _ = frames.shape
    for s, idx in enumerate(np.array_split(np.arange(n), scenes)):
        part = frames[idx]
        x, y = w * (2 * s + 1) // (2 * scenes), h * (2 * s + 1) // (2 * scenes)
        part[:, :, x:x + 3] = rng.choice([10, 245])
        part[:, y:y + 3] = rng.choice([10, 245])
        for _ in range(6):
            bx, by, size = int(rng.integers(0, w - 30)), int(rng.integers(0, h - 30)), int(rng.integers(6, 30))
            part[:, by:by + size, bx:bx + size] = rng.choice([10, 245])
        frames[idx] = part
    return frames


def test_find_ignores_structure_held_for_only_one_scene(tmp_path):
    opacity = text_opacity(W, H, TEXT, peak=0.45)
    clip = encode(hud(stamp(scenery(clean_frames(300, W, H, seed=7), 5, seed=7), opacity)), tmp_path / "s.mp4",
                  audio=False)

    found = marks.find(lambda: media.planes(clip, 1000))

    print([(m.kind, m.box, round(m.opacity, 3)) for m in found])
    assert [m.kind for m in found] == ["see-through", "solid"]
    assert iou(found[0].box, extent(opacity)) > 0.8 and abs(found[0].opacity - 0.45) <= 0.05
    assert iou(found[1].box, PANEL) > 0.5


def test_find_on_ffmpeg_test_sources_lists_just_the_two_overlays(tmp_path):
    w, h, seconds = 1280, 720, 6  # per scene
    text, hud_box = (900, 610), (30, 520, 160, 140)
    ink = np.zeros((60, 330), np.uint8)
    cv2.putText(ink, "MyChannel", (6, 44), cv2.FONT_HERSHEY_DUPLEX, 1.4, 255, 3, cv2.LINE_AA)
    png = np.dstack([np.full((60, 330, 3), 215, np.uint8), (ink * 0.45).astype(np.uint8)])  # light grey, 45%
    cv2.imwrite(str(tmp_path / "mark.png"), png)
    opacity = np.zeros((h, w))
    opacity[text[1]:text[1] + 60, text[0]:text[0] + 330] = png[..., 3] / 255
    # each scene has still structure of its own (testsrc2's layout, settled cells of life), none lasts the whole video
    cut = f"trim=duration={seconds},setpts=PTS-STARTPTS,format=yuv420p"
    scenes = [f"testsrc2=s={w}x{h}:r=10,{cut}", f"mandelbrot=s={w // 2}x{h // 2}:r=10,scale={w}:{h},{cut}",
              (f"life=s={w // 4}x{h // 4}:r=10:seed=1:mold=20:life_color=#ffcc66:death_color=#223344,"
               f"scale={w}:{h},{cut}"),
              f"perlin=s={w}x{h}:r=10:random_mode=seed:seed=1,eq=contrast=1.4,{cut}"]
    x, y, bw, bh = hud_box
    graph = ";".join(f"{s}[s{i}]" for i, s in enumerate(scenes)) + ";" + "".join(f"[s{i}]" for i in range(4)) + (
        f"concat=n=4:v=1:a=0,drawbox=x={x}:y={y}:w={bw}:h={bh}:color=0x303030:t=fill,"
        f"drawbox=x={x}:y={y}:w={bw}:h={bh}:color=white:t=4[hud];[hud][0:v]overlay={text[0]}:{text[1]}")
    clip = tmp_path / "lavfi.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_path / "mark.png"),
                    "-filter_complex", graph, "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)

    found = marks.find(lambda: media.planes(clip, 1000))

    print([(m.kind, m.box, round(m.opacity, 3)) for m in found])
    assert [m.kind for m in found] == ["solid", "see-through"]  # the HUD sits higher than the text
    assert iou(found[0].box, hud_box) > 0.5
    # measuring, not finding, reads 0.47 to 0.51 here over life and perlin seeds 1 to 6: compression and still scenes
    assert iou(found[1].box, extent(opacity)) > 0.8 and abs(found[1].opacity - 0.45) <= 0.08


def test_find_skips_structure_inside_a_still_part_of_the_scene(tmp_path):
    frames = clean_frames(150, W, H, seed=3)
    frames[:, 40:140, 100:240] = 120  # a wall that never changes, too big to be an overlay
    frames[:, 80:100, 150:190] = 40  # a dark sign painted on it
    clip = encode(frames, tmp_path / "still.mp4", audio=False)

    assert marks.find(lambda: media.planes(clip, 1000)) == []


def test_visibility_tells_a_stamped_outline_from_the_picture_and_a_cleaned_one(tmp_path):
    box = (208, 16, 96, 48)
    real = footage(300, W, H, seed=5)
    clean = encode(real, tmp_path / "clean.mp4", audio=False, crf=28)
    marked = encode(stamp(real, text_opacity(W, H, box)), tmp_path / "marked.mp4", audio=False, crf=28)
    mark = marks.find(lambda: media.planes(marked, 1000), box)[0]

    def cleaned():
        for f in media.planes(marked, 1000):
            mark.erase(f)
            yield f

    seen, truth, out = (mark.visibility(frames) for frames in (media.planes(marked, 1000), media.planes(clean, 1000),
                                                                cleaned()))
    print(f"visibility: marked {seen:.2f}, clean {truth:.2f}, cleaned {out:.2f}")
    assert seen > 1.5
    assert abs(truth - 1) < 0.15 and abs(out - 1) < 0.15


def test_visibility_means_the_same_on_a_smooth_picture(tmp_path):
    box = (208, 16, 96, 48)
    real, opacity = still(160, W, H), text_opacity(W, H, box)
    mark = marks.find(lambda: media.planes(encode(stamp(real, opacity, 215), tmp_path / "m.mp4", audio=False, crf=23),
                                           1000), box)[0]

    def outline(share: float) -> float:
        """Visibility of the overlay stamped at share of its opacity: its outline steps up to 52 * share levels."""
        clip = encode(stamp(real, opacity * share, 215), tmp_path / f"{share}.mp4", audio=False, crf=23)
        return mark.visibility(media.planes(clip, 1000))

    unseen, seen = outline(0.02), outline(0.1)  # steps of at most 1 level, which the eye cannot find, and of 5
    print(f"visibility: 1-level outline {unseen:.2f}, 5-level outline {seen:.2f}")
    assert abs(unseen - 1) < 0.15 and seen > 1.5


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """The text overlay measured on lossless footage, and how to stamp it on YUV frames the way that footage had it."""
    tmp = tmp_path_factory.mktemp("measured")
    opacity = text_opacity(W, H, TEXT)
    mark = marks.find(lambda: media.planes(encode(stamp(footage(300, W, H, seed=5), opacity), tmp / "m.mp4",
                                                  audio=False), 1000), TEXT)[0]
    grey = next(iter(yuv(np.full((1, H, W, 3), 200, np.uint8), tmp)))  # the overlay's colour, converted as ffmpeg does
    return mark, opacity, grey


def yuv(frames: np.ndarray, tmp) -> list[media.Frame]:
    """RGB frames as the YUV frames ffmpeg makes of them."""
    return [media.Frame(f.y.copy(), f.u.copy(), f.v.copy())
            for f in media.planes(encode(frames, tmp / "yuv.mp4", audio=False), 1000)]


def stamped(real: media.Frame, opacity: np.ndarray, colour: media.Frame) -> media.Frame:
    """real with the overlay blended in each plane, colour at half size."""
    half = opacity.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    return media.Frame(*(np.clip(np.rint(a * c + (1 - a) * p.astype(np.float32)), 0, 255).astype(np.uint8)
                         for a, c, p in zip((opacity, half, half), (colour.y, colour.u, colour.v),
                                            (real.y, real.u, real.v))))


def compressed(real: media.Frame, opacity: np.ndarray, colour: media.Frame, rng: np.random.Generator,
               noise: float, planes: int = 1) -> tuple[media.Frame, list[np.ndarray]]:
    """real with the overlay, the first planes of y, u and v with errors like compression leaves, different in each
    frame: softened by a varying amount, and noise of about noise levels where the overlay's opacity changes. With the
    truth to compare with: those planes of real softened the same way."""
    seen, truth = stamped(real, opacity, colour), []
    half = opacity.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    for s, a, p, t in list(zip((1, 2, 2), (opacity, half, half), (seen.y, seen.u, seen.v), (real.y, real.u, real.v)))[
            :planes]:
        sigma = rng.uniform(0.6, 1.4) / s
        edge = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)) > 0.05
        errors = cv2.GaussianBlur(rng.normal(0, noise, a.shape).astype(np.float32), (0, 0), 0.7 / s) * 2 * edge
        soft = [cv2.GaussianBlur(q.astype(np.float32), (0, 0), sigma) for q in (p, t)]
        p[:] = np.clip(np.rint(soft[0] + errors), 0, 255)
        truth.append(np.clip(np.rint(soft[1]), 0, 255).astype(np.uint8))
    return seen, truth


def fine(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    return np.abs(y - cv2.GaussianBlur(y, (0, 0), 2))


def letters(opacity: np.ndarray) -> np.ndarray:
    """The overlay's pixels away from its edges."""
    return cv2.erode((opacity > 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def test_a_frame_crossed_by_strong_edges_under_the_overlay_is_cleaned(tmp_path, measured):
    mark, opacity, grey = measured
    real = footage(6, W, H, seed=8)
    x, y, w, h = TEXT
    real[:, y + 8:y + h - 8, x - 10:x + w + 10] = 235  # a bright menu bar with dark text on it across the letters
    for k in range(x - 8, x + w + 8, 9):
        real[:, y + 14:y + h - 14, k:k + 3] = 20
    truth = yuv(real, tmp_path)
    for t in truth:
        f = stamped(t, opacity, grey)
        before = psnr(f.y[y:y + h, x:x + w], t.y[y:y + h, x:x + w])
        mark.erase(f)
        after = psnr(f.y[y:y + h, x:x + w], t.y[y:y + h, x:x + w])
        print(f"inside box vs truth: marked {before:.2f} dB, cleaned {after:.2f} dB")
        assert after >= before + 10


def test_compression_errors_on_a_smooth_picture_leave_no_outline(tmp_path, measured):
    mark, opacity, grey = measured
    rng = np.random.default_rng(1)
    real = yuv(still(8, W, H, scenes=8, seed=3), tmp_path)
    seen, truth = zip(*(compressed(t, opacity, grey, rng, 3.0) for t in real))
    truth = [t[0] for t in truth]
    for f in seen:
        mark.erase(f)
    x, y, w, h = TEXT
    box = np.s_[y:y + h, x:x + w]
    ours = psnr(np.stack([f.y[box] for f in seen]), np.stack([t[box] for t in truth]))
    shown, clean = (mark.visibility(media.Frame(a, f.u, f.v) for a, f in zip(out, seen))
                    for out in ([f.y for f in seen], truth))
    print(f"inside box vs truth: cleaned {ours:.2f} dB; visibility cleaned {shown:.2f}, truth {clean:.2f}")
    assert ours > 50 and abs(shown - clean) < 0.1


def test_compression_errors_on_a_smooth_colour_picture_leave_no_colour_outline(tmp_path, measured):
    mark, opacity, grey = measured
    rng = np.random.default_rng(3)
    seen, truth = zip(*(compressed(t, opacity, grey, rng, 1.0, planes=3) for t in yuv(
        still(8, W, H, scenes=8, seed=5), tmp_path)))
    for f in seen:
        mark.erase(f)
    x, y, w, h = (v // 2 for v in TEXT)
    for i, name in ((1, "u"), (2, "v")):
        ours = psnr(np.stack([getattr(f, name)[y:y + h, x:x + w] for f in seen]),
                    np.stack([t[i][y:y + h, x:x + w] for t in truth]))
        print(f"{name} inside box vs truth: cleaned {ours:.2f} dB")
        assert ours > 50  # plain undoing, (seen - offset) / keep, measured 41 dB here


def test_compression_errors_on_a_textured_picture_keep_the_texture(tmp_path, measured):
    mark, opacity, grey = measured
    rng = np.random.default_rng(2)
    grain = rng.normal(0, 15, (8, H, W, 1))  # a grainy, busy picture, as in fast or dark game scenes
    truth = yuv(np.clip(footage(8, W, H, scenes=8, seed=4) + grain, 0, 255).astype(np.uint8), tmp_path)
    inside, kept = letters(opacity), []
    for real in truth:
        f, (t, ) = compressed(real, opacity, grey, rng, 1.0)
        mark.erase(f)
        kept.append(fine(f.y)[inside].mean() / fine(t)[inside].mean())
        print(f"fine detail inside the letters over truth's: cleaned {kept[-1]:.2f}")
    # the requirement: texture within 10% of the truth's, never matte (a little of the errors may stay, as texture
    # hides it)
    assert min(kept) >= 0.9 and 0.9 <= np.mean(kept) <= 1.1


def test_a_frame_without_the_overlay_is_left_exactly_as_it_was(tmp_path, measured):
    mark = measured[0]
    for f in yuv(np.concatenate([footage(4, W, H, seed=9), still(4, W, H, seed=9)]), tmp_path):
        before = [p.copy() for p in (f.y, f.u, f.v)]
        mark.erase(f)
        assert all((a == b).all() for a, b in zip(before, (f.y, f.u, f.v)))


def test_refining_on_frames_where_nothing_can_be_measured_says_so(tmp_path, measured):
    mark = measured[0]
    # no overlay, but a picture as bright as it on average, so every frame reads as having it, and busier inside the
    # box than around it, so nothing there is dimmed
    frames = np.full((12, H, W, 3), 200.0)
    x, y, w, h = TEXT
    frames[:, y:y + h, x:x + w] += np.random.default_rng(4).normal(0, 20, (12, h, w, 1))
    with pytest.raises(RuntimeError, match="nothing see-through or solid was measured"):
        mark.refined(yuv(np.clip(frames, 0, 255).astype(np.uint8), tmp_path))


def test_finding_reads_the_frames_once_and_a_given_box_twice(tmp_path):
    two = text_opacity(W, H, TEXT) + text_opacity(W, H, (24, 110, 96, 48), "OK")
    clip = encode(stamp(clean_frames(150, W, H, seed=2), two), tmp_path / "two.mp4", audio=False)
    reads = []

    def frames():
        reads.append(1)
        return media.planes(clip, 1000)

    found = marks.find(frames)
    assert [m.kind for m in found] == ["see-through", "see-through"] and len(reads) == 1
    reads.clear()
    marks.find(frames, found[1].box)
    assert len(reads) == 2  # once to measure, once more on only the frames the overlay is in
