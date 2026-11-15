import numpy as np
from helpers import clean_frames, encode, footage, stamp, text_opacity

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
