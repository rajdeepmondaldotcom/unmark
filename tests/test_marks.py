import subprocess

import cv2
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
              f"life=s={w // 4}x{h // 4}:r=10:seed=1:mold=20:life_color=#ffcc66:death_color=#223344,scale={w}:{h},{cut}",
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
