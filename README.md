# unmark

unmark removes a small overlay that sits in the same place for the whole video, like a logo or a name in a corner.
You give it a video of 2 to 30 minutes, and it writes a cleaned copy next to it. The original file is never changed.

Use it only on footage you own or have permission to edit.

## Install

You need two things on your computer:

- **uv**, which runs the Python part. See https://docs.astral.sh/uv/ for how to install it.
- **ffmpeg** and **ffprobe**, the video tools, on your PATH (typing `ffmpeg -version` in a terminal must work).
  If they are somewhere else, set `UNMARK_FFMPEG` and `UNMARK_FFPROBE` to their full paths.

Then, in the `unmark` folder:

```
uv sync
uv run unmark doctor
```

`doctor` checks that everything works on this computer and prints one line per check. The last line should be
`doctor: all required checks passed`. A `[warn]` line about ProPainter is fine; it only matters for solid overlays
(see "How it works").

On Windows, follow [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) instead.

## Use

There are three ways to run it.

**1. Let it find the overlay and ask you.**

```
uv run unmark clean holiday.mp4
```

It looks through the video for things that never move, then saves a picture called `holiday.marks.png` next to the
video. The picture is one frame in grey with a coloured, numbered box around each thing it found. It opens the picture
and asks which number should go. Type the number and press Enter.

**2. Tell it the number straight away.** If you already know the number (for example from an earlier run's
picture), give it with `--mark`:

```
uv run unmark clean holiday.mp4 --mark 1
```

**3. Give the box yourself.** If it did not find the overlay, or found the wrong area, give the box in pixels:
`x,y,w,h` is the distance from the left edge, the distance from the top edge, the width and the height. Make the box a
little bigger than the overlay.

```
uv run unmark clean holiday.mp4 --box 478,22,132,60
```

The cleaned video is called `holiday.clean.mp4`, next to the original. To choose another name, add
`--out other-name.mp4`. The sound is copied over untouched.

This is what it printed on a generated 20-second test clip with a see-through "UM" logo in the top right corner:

```
$ uv run unmark clean holiday.mp4 --mark 1
1: see-through at 482,32, size 80x46, opacity 49%
wrote holiday.clean.mp4 (600 frames)
overlay outline visibility (1.0 = gone): input 2.14, output 1.04
```

The first line is the list of what it found: the kind of overlay (see-through or solid), where it is, its size, and
how strongly it covers the picture.

If something goes wrong, it prints one line starting with `error:` that says what to do. Add `--debug` to see the
technical details, which help if you ask someone for help.

## Reading the result

The last line, "overlay outline visibility", measures how sharp the overlay's outline is compared with the picture
around it.

- **About 1.0** means the outline can no longer be told apart from the picture: the overlay is gone.
- **Above 1.25** means the outline can still be seen. Then it also prints
  `warning: the overlay's outline is still visible in the output; look at it before using it`.

The number for the input tells you how visible the overlay was to begin with. Either way, watch the cleaned video
before you use it. The number is a check, not a guarantee.

## How it works

1. It reads up to 1,000 frames spread over the whole video. Anything that changes from frame to frame is the video;
   anything that stays sharp in the same place is a fixed overlay.
2. For a **see-through** overlay, the picture behind it still shows. Because the picture behind keeps changing, the
   tool can measure, pixel by pixel, how much the overlay dims the picture and what colour it adds.
3. It then undoes that exactly in every frame: it subtracts the overlay's colour and brightens the picture back.
   Pixels away from the overlay are left exactly as they were.
4. A **solid** overlay hides the picture completely, so there is nothing to undo. The hidden area is filled in from the
   surrounding video instead: with ProPainter (an AI video filler) when it is set up on a computer with an NVIDIA
   graphics card, otherwise with a simpler fill from OpenCV that looks softer.

## Limits

These are honest limits of this version.

- **Only ordinary 8-bit video** (the common "yuv420p" format). 10-bit and HDR videos, and 4:2:2 or 4:4:4 videos, are
  refused. Export an ordinary copy first.
- **No rotated phone videos.** A phone video that is stored sideways and turned upright on playback is refused. Export
  an upright copy first.
- **Fixed overlays only.** An overlay that moves, fades in and out at different places, or changes shape cannot be
  removed. With a see-through overlay, frames that do not have it (an intro, say) are left alone; a solid overlay's box
  is filled in every frame.
- **The video must change enough.** Everything is measured from the picture changing behind the overlay. A very short
  video, or a still scene (a slideshow, a static screen), gives too little change, and it will say it found nothing to
  remove.
- **Very faint overlays** may not be found automatically. Give the box with `--box` instead.
- **Heavily compressed video** can keep a faint ghost of the overlay, because compression blurs the very detail that is
  measured.
- **Near-solid parts of a see-through overlay** (the overlay hides 85% or more of the picture there) are brightened back
  as far as they can be, not filled. Those spots can look noisy.
- **Solid overlays with the simple fill look soft.** The fill takes the whole box, not just the overlay's own pixels.
- **Speed.** It is not fast. The whole input video is decoded 4 times (5 when ProPainter fills a solid overlay, to find
  the scene cuts), and the cleaned video is decoded once more at the end to check it. It is re-encoded once with
  libx264. No timings on long videos have been measured yet.
- **Scene cuts** for ProPainter are found with a threshold set on test clips, not on real footage.
- **Untested:** it has never been run on Windows, and never with the real ProPainter. Both are written to work, but
  only the stand-in in `tests/fake_propainter` has been run.

## For developers

```
uv run pytest
uvx ruff check .
```

The tests build small videos with ffmpeg, with a known clean truth, and check the output against it.

- `unmark/cli.py`: the `clean` and `doctor` commands, the numbered picture, and pasting a fill into the frames.
- `unmark/marks.py`: the maths: finding fixed overlays, measuring a see-through one, inverting it, the visibility score.
- `unmark/media.py`: every ffmpeg and ffprobe call: probing, reading frames, finding scene cuts, writing the output.
- `unmark/fill.py`: filling a solid overlay, with ProPainter on pieces of each scene, or with OpenCV.
- `tests/fake_propainter/`: a stand-in for ProPainter's script, same arguments and output, used by the tests.

Known shortcuts are marked in the code with `# shortcut:`; list them with `grep -rn "shortcut:" unmark`.
