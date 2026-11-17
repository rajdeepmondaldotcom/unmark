# unmark

unmark removes a small overlay that sits in the same place for the whole video, like a logo or a name in a corner.
You give it a video, from a couple of minutes to several hours, and it writes a cleaned copy next to it. The original file is never changed.

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

The cleaned video is called `holiday.clean.mp4`, next to the original (`.clean.mkv` for videos that do not end in
`.mp4`, `.mov` or `.mkv`). To choose another name, add `--out other-name.mp4`; the name must end in `.mp4`, `.mov` or
`.mkv`, and must not be the input itself. The sound is copied over untouched, every sound track of it. The file's
tags (title, artist and so on), its chapters and the video's language are kept too, and so are subtitles and
attachments (such as fonts in an `.mkv`) when the cleaned video ends in the same way as the original (see "Limits").

This is what it printed on a generated 90-second 1080p test video (six scenes, a grey box in the lower left and a
see-through "MyChannelName" in the lower right):

```
$ uv run unmark clean holiday.mp4 --mark 2
1: solid at 38,798, size 244x204, opacity 100%
2: see-through at 1466,954, size 386x48, opacity 46%
wrote holiday.clean.mp4 (2700 frames)
overlay outline visibility (1.0 = gone): input 9.21, output 1.17
```

The first lines are the list of what it found: the kind of overlay (see-through or solid), where it is, its size, and
how strongly it covers the picture.

If something goes wrong, it prints one line starting with `error:` that says what to do. Add `--debug` to see the
technical details, which help if you ask someone for help.

## Reading the result

The last line, "overlay outline visibility", measures how sharp the overlay's outline is compared with the picture
around it, frame by frame, averaged over 100 frames. Differences of about a brightness level, which the eye cannot find,
count as nothing, so the number means the same on a plain sky as on busy footage.

- **About 1.0** means the outline can no longer be told apart from the picture: the overlay is gone.
- **Below 1.0** means the outline is a little smoother than the picture around it.
- **Above 1.25** means the outline can still be seen. Then it also prints
  `warning: the overlay's outline is still visible in the output; look at it before using it`.

As a guide: an outline 1 brightness level from the picture reads about 1.07 on a plain picture, one 5 levels away about
1.75. The number for the input tells you how visible the overlay was to begin with; if that is 1.25 or less, there is
nothing there to remove and it stops. Either way, watch the cleaned video before you use it. The number is a check, not
a guarantee.

## Picture quality

The cleaned video is re-encoded: every frame is compressed again, so outside the box it is very close to the original
but not bit-for-bit the same. Only the tests, which use a lossless setting, get exact copies.

On an Apple-silicon Mac it uses the Mac's hardware HEVC (H.265) encoder at quality 74. On other computers, or if that
encoder does not start, it uses libx264 (H.264) at crf 16. `unmark doctor` says which one this computer uses.

This is what each costs, measured on real pieces of one video (5 hours 33 minutes, 3840x2160, 60 fps, 35 GB as
downloaded), against the decoded original. Higher dB means closer to the original; VMAF 100 means it looks the same.

| setting | luma PSNR | VMAF | size of the whole video |
|---|---|---|---|
| Mac hardware HEVC, quality 74 (used on a Mac) | 46.5 / 46.8 dB | 96.0 / 99.8 | about 65 GB |
| Mac hardware HEVC, quality 80 | 48.2 / 48.4 dB | 96.7 / 99.8 | about 94 GB |
| Mac hardware HEVC, quality 68 | 45.3 / 45.6 dB | 94.2 / 99.8 | about 47 GB |
| libx264 crf 16 (used elsewhere) | 48.2 / 45.8 dB | not measured | 172 to 195 GB |
| lossless libx264 | exact | 100 | 1,500 to 2,100 GB |

The hardware rows are a fast, strongly coloured piece and a detailed one, 5 seconds each; the libx264 row is two other
12-second pieces. The 94 GB comes from a real run at quality 80 that wrote 53% of the video in 50 GB, and the other
hardware sizes are scaled from it. Quality 74 is the lowest setting whose worst frames stayed within one VMAF point of
quality 80's (95.2 on both pieces, against 96.2 and 95.8); at 68 the fast piece's worst frame fell to 92.8. libx265
and SVT-AV1 at their fast presets were tried too: at the same closeness they were no smaller on the detailed piece and
took 5 to 60 times the processor time per frame. The hardware encoder wrote 35 to 57 frames a second at this size on
an idle Apple M4 at quality 80, and far fewer while the graphics chip was busy with other work.

So a lossless copy of that video would need 1.5 to 2 TB, which is why it is not the default.

## How it works

1. It reads up to 1,000 frames spread over the whole video. Anything that changes from frame to frame is the video;
   anything that stays sharp in the same place is a fixed overlay.
2. For a **see-through** overlay, the picture behind it still shows. Because the picture behind keeps changing, the
   tool can measure, pixel by pixel, how much the overlay dims the picture and what colour it adds.
   It measures this two ways: from how much the picture changes behind the overlay compared with around it, and,
   frame by frame, from the picture just around the overlay, which predicts the picture behind it well when the
   picture is smooth. It keeps whichever cleans a quarter of the frames, set aside for the purpose, best.
3. It then undoes that in every frame that has the overlay: it subtracts the overlay's colour and brightens the
   picture back. How strongly the overlay dims colour is read again from each frame, because on a real video it
   dimmed colour more than brightness, by an amount that changed from scene to scene; frames with little colour keep
   the measured strength. Compression leaves small errors in and around the letters, which brightening makes stronger. So,
   pixel by pixel and frame by frame, where the picture around the letters is smooth the letters are filled from it,
   and where it is detailed the undone picture is kept. The cleaning changes nothing outside the overlay's box, but
   the whole video is still re-encoded (see "Picture quality").
4. A **solid** overlay hides the picture completely, so there is nothing to undo. The hidden area is filled in from the
   surrounding video instead: with ProPainter (an AI video filler) when it is set up on a computer with an NVIDIA
   graphics card, otherwise with a simpler fill from OpenCV that looks softer. The fill fades into the picture
   around the box, so pixels just outside the box change too: brightness up to 5 pixels from the box, and colour,
   which is stored once per 2x2 block of pixels, up to 6. Nothing further away changes.

## Limits

These are honest limits of this version.

- **Only ordinary 8-bit video** (the common "yuv420p" format). 10-bit and HDR videos, and 4:2:2 or 4:4:4 videos, are
  refused. Export an ordinary copy first.
- **No rotated phone videos.** A phone video that is stored sideways and turned upright on playback is refused. Export
  an upright copy first.
- **Fixed overlays only.** An overlay that moves, fades in and out at different places, or changes shape cannot be
  removed. With a see-through overlay, frames that do not have it (an intro, say) are left alone; a solid overlay's box
  is filled in every frame. A frame whose picture is about as bright as the overlay, busy, and crossed by strong
  edges can be misread as not having it, and is then left as it is.
- **An overlay with no room of its own height above or below it** (one nearly as tall as the frame) is only undone:
  its letters are not filled from the picture around them, so faint compression errors there can stay.
- **The video must change enough.** Everything is measured from the picture changing behind the overlay. A very short
  video, or one still scene (a slideshow, a static screen), gives too little change: it then says that the picture
  around the box barely changes, or that it found nothing to remove. A few different scenes are enough if the
  picture around the overlay is smooth.
- **Overlays on screen for less than about 70% of the video** are not found automatically. Give the box with `--box`
  instead.
- **Very faint overlays** may not be found automatically. Give the box with `--box` instead.
- **Compressed video** can keep a faint ghost of the overlay, because compression blurs the very detail that is
  measured. On a 1080p download-quality test video the ghost is about one brightness level, which you may just find on
  a large plain area of one colour if you look for it. On strongly coloured pictures (a bright red wall, say) a faint
  darker ghost of the letters can stay, and on strongly coloured, fast or textured scenes a faint colour ghost too: the
  colour strength read from such a frame can be a little off.
- **Near-solid parts of a see-through overlay** (the overlay hides 85% or more of the picture there) are brightened back
  as far as they can be, not filled. Those spots can look noisy.
- **Solid overlays with the simple fill look soft.** The fill takes the whole box, not just the overlay's own pixels,
  and is worked out on a half-size copy of the area around it, which makes it about four times faster.
- **Speed.** It is not fast. It reads the frames it measures twice, with `--box` or without: on a long video it jumps
  straight to each one, on a short one it decodes the whole video each time, which is quicker there. It reads 100
  frames once more to check how visible the overlay is. A solid overlay's fill decodes the whole video once more
  (twice with ProPainter, which also looks for scene cuts). The video is re-encoded once (see "Picture quality"), and
  100 frames of the cleaned video are read at the end to check it. It prints how far it has got after each part it
  writes, with a rough time left. On one machine, an Apple M4 Mac, a 90-second 1080p video took about 80 to 110
  seconds for a see-through overlay and about 100 seconds for a solid one with the simple fill, when it still
  re-encoded with libx264 there; the Mac's hardware encoder is faster. This is one machine; a slower computer will
  take longer. A run that continues after a stop (see "Stopping") skips the measuring and the first check, and starts
  writing straight away.
  <!-- real-video numbers go here -->
- **Stopping.** With a see-through overlay, the cleaned video is written in parts of 5 minutes of video, which are
  kept if it stops (Ctrl-C, sleep, a lost disk). Run the same command again, with the same `--box` or `--mark`, and it
  continues from the last finished part. It does not measure the overlay again: it uses the measurement it saved
  before writing, in a file named after the output with `.mark.npz` added, and says so. A solid overlay, and a video
  whose frame rate changes, are written in one go and cannot continue: they start over. The parts are kept in a folder
  named after the output with `.parts` added. The folder and the saved measurement are removed when the video is
  finished.
- **Unfinished parts are never deleted for you.** If a later run differs from the one that wrote them (another video,
  overlay, setting or version of unmark), it stops with an error saying the parts are kept: run the earlier command
  again to finish, or delete the `.parts` folder to start over. If the Mac's hardware encoder wrote them and does not
  start this time, it says so: try again.
- **Disk space.** Joining the parts into the finished video needs both on the disk at once, so about twice the size
  of the cleaned video must be free (see "Picture quality" for sizes). If it is not, it says roughly how many GB to
  free. Parts that can continue (see "Stopping") are kept, so after freeing that the same command finishes the job.
- **Subtitles and attachments** are kept only when the cleaned video has the same ending as the original (`.mkv` to
  `.mkv`, say), because another kind of file may not be able to hold them. With `--out` set to another ending, or for
  an original that is not `.mp4`, `.mov` or `.mkv`, they are left out. Other extra tracks (a timecode or data track,
  cover art) are always left out.
- **Scene cuts** for ProPainter are found with a threshold set on test clips, not on real footage.
- **Untested:** it has never been run on Windows, and never with the real ProPainter. Both are written to work, but
  only the stand-in in `tests/fake_propainter` has been run.

## For developers

```
uv run pytest
uvx ruff check .
```

The tests build small videos with ffmpeg, with a known clean truth, and check the output against it.

- `unmark/cli.py`: the command line and the `clean` command: the numbered picture, and pasting a fill into the frames.
- `unmark/doctor.py`: the `doctor` command's checks, including the `--gpu` ProPainter sample.
- `unmark/marks.py`: the maths: finding fixed overlays, measuring a see-through one, inverting it, the visibility score.
- `unmark/media.py`: every ffmpeg and ffprobe call: probing, reading frames, finding scene cuts, writing the output.
- `unmark/fill.py`: filling a solid overlay, with ProPainter on pieces of each scene, or with OpenCV.
- `tests/fake_propainter/`: a stand-in for ProPainter's script, same arguments and output, used by the tests.

Known shortcuts are marked in the code with `# shortcut:`; list them with `grep -rn "shortcut:" unmark`.
