# Setting up unmark on Windows

> **Warning: these steps have not been run on a Windows PC yet.** unmark was built and tested on a Mac only.
> The code was written to work on Windows, but nobody has run it there, and nobody has run it with the real
> ProPainter. Something will probably go wrong.
> When you finish, send the output listed in "Send this back", even if everything passes.

This guide takes the PC from nothing to a cleaned video. Part A is for everyone and takes about 15 to 30 minutes.
Part B is only needed for **solid** overlays (ones you cannot see through at all), and takes longer.

Rules for this guide:

- Do one step at a time. Check that you see the success message before going on.
- To paste a command, copy the text in the grey box, click in the PowerShell window, right-click (or press Ctrl+V),
  then press Enter.
- If a step fails, stop and send the full message you see.

This guide uses `C:\unmark` for unmark and `C:\ProPainter` for ProPainter. Keep both folders short and close to the
top of the drive; deep folders can hit the Windows limit on path length. Avoid cloud-synced folders (OneDrive,
Google Drive): they are slow and can lock files while syncing.

## Part A: unmark itself (everyone)

### Open PowerShell

1. Press the Windows key, type `PowerShell`, and press Enter. Do not choose "Run as administrator".
   Success: a window opens with a line like `PS C:\Users\...>`.

2. Check that winget (the Windows installer tool) is there:

   ```
   winget --version
   ```

   Success: you see a version number like `v1.9...`.
   If you see "not recognized", open the Microsoft Store, search for "App Installer", install or update it, then open
   a new PowerShell window.

### Install ffmpeg (the video tool)

unmark needs an ffmpeg that includes libx264 (the H.264 encoder) and is fairly recent. The `Gyan.FFmpeg` package is
the full build from gyan.dev, which includes libx264.

3. Install it:

   ```
   winget install --id Gyan.FFmpeg -e
   ```

   If it asks you to agree to the source terms, read them and type `Y` if you agree.
   Success: it ends with `Successfully installed`.

4. Close PowerShell and open a new PowerShell window, so it can find the new program.

5. Check it:

   ```
   ffmpeg -version
   ```

   Success: the first line starts with `ffmpeg version`.
   If you see "not recognized", go to "How unmark finds ffmpeg" at the bottom.

### Install uv (runs the Python part)

6. Install uv:

   ```
   winget install --id astral-sh.uv -e
   ```

   Success: it ends with `Successfully installed`.

7. Close PowerShell and open a new PowerShell window.

8. Check it:

   ```
   uv --version
   ```

   Success: you see `uv` and a version number.

### Get the unmark folder

The unmark project is private on GitHub. Use **one** of these two ways.

9. **Way 1: copy the folder.** You get the folder, or a zip file, from the person who sent you this guide.
   - If you got a zip file: right-click it in File Explorer, choose "Extract All...", and extract it to `C:\`. Rename
     the new folder to `unmark` if needed, so you have `C:\unmark`.
   - If the copy contains a folder called `.venv`, delete it. It comes from a Mac and will not work on Windows:

     ```
     Remove-Item -Recurse -Force C:\unmark\.venv
     ```

     Success: nothing is printed. (If it says the path does not exist, that is fine too.)

   **Way 2: git clone with your GitHub login.** This only works if your GitHub account has been given access to the
   project. First install git:

   ```
   winget install --id Git.Git -e
   ```

   Close PowerShell and open a new window. Then:

   ```
   git clone https://github.com/rajdeepmondaldotcom/unmark.git C:\unmark
   ```

   A browser window opens and asks you to sign in to GitHub. Sign in yourself; never type your password into a chat.
   Success: it ends with a line like `Resolving deltas: 100%`.

10. Go into the folder:

    ```
    cd C:\unmark
    ```

    Success: the line now starts with `PS C:\unmark>`.

11. Check it is the right folder:

    ```
    dir
    ```

    Success: you see `README.md`, `pyproject.toml`, `unmark`, `tests` and `docs` among the names.

### Install the Python parts

12. Install everything unmark needs:

    ```
    uv sync
    ```

    The first time, uv also downloads Python 3.12 by itself.
    Success: it ends with a line like `Installed ... packages`.

### Run the check

13. Run the doctor:

    ```
    uv run unmark doctor
    ```

    Success: the last line is

    ```
    doctor: all required checks passed
    ```

    You should see `[ok]` lines for `ffmpeg`, `ffprobe`, `filters`, `encoder` and `end-to-end run`.
    A `[warn] ProPainter: UNMARK_PROPAINTER is not set...` line is fine. It only means solid overlays will use the
    softer simple fill until you do Part B.
    If the `end-to-end run` line says `[FAIL]`, this ffmpeg is probably too old. Send the output.

### Clean your first video

14. Copy a video into a folder with plenty of free space, for example `C:\videos`. The cleaned copy can be bigger than
    the original, because it is saved at high quality.

15. Run unmark on it. Type `uv run unmark clean ` (with a space at the end), then drag the video from File Explorer
    into the PowerShell window, or hold Shift, right-click the file, choose "Copy as path", and paste it. Press Enter.

    ```
    uv run unmark clean "C:\videos\my-video.mp4"
    ```

    It reads the whole video, which takes a while. Then it opens a picture with numbered boxes around the overlays it
    found and asks which one should go. Type the number and press Enter.
    Success: you see `wrote C:\videos\my-video.clean.mp4 (... frames)` and a line
    `overlay outline visibility (1.0 = gone): input ..., output ...`. The README explains how to read it.
    Watch the cleaned video before using it.

## Part B: ProPainter (only for solid overlays)

ProPainter is an AI video filler. unmark uses it to fill in the area under a solid overlay. Without it, unmark uses a
simpler fill that looks softer. ProPainter needs an **NVIDIA** graphics card.

ProPainter's licence (the NTU S-Lab License 1.0) allows **non-commercial use only**. Do not use it for paid work
without permission from its authors.

ProPainter's own instructions use conda and Python 3.8. This guide uses uv and Python 3.11 instead, with a newer
torch. That combination has not been tried.

### Check the graphics card

16. Check the NVIDIA driver:

    ```
    nvidia-smi
    ```

    Success: a table with your card's name, a `Driver Version` and a `CUDA Version`. The `CUDA Version` must be
    **12.8 or higher**. If it is lower, or the command is "not recognized", update the driver with the NVIDIA App (or
    GeForce Experience) first, restart the PC, and try again.

### Get ProPainter

17. Install git, if you did not already in step 9:

    ```
    winget install --id Git.Git -e
    ```

    Close PowerShell and open a new window.

18. Download ProPainter's code and pin it to the exact version unmark was written against:

    ```
    git clone https://github.com/sczhou/ProPainter.git C:\ProPainter
    ```

    ```
    cd C:\ProPainter
    ```

    ```
    git checkout 326782529a63d997cab80ab21439b92ed422b3d3
    ```

    Success: the last command says `HEAD is now at 3267825`.

### Give ProPainter its own Python

unmark looks for ProPainter's Python in a folder called `.venv` inside `C:\ProPainter`.

19. Make that folder:

    ```
    uv venv --python 3.11 .venv
    ```

    Success: it ends with `Activate with: .venv\Scripts\activate`. You do not need to activate it.

20. Install torch, the AI library, in the version built for NVIDIA cards (CUDA 12.8). This is a large download
    (several GB):

    ```
    uv pip install --python .venv\Scripts\python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    ```

    Success: it ends with a list of installed packages, including `+ torch==2.11.0+cu128`.

21. Install the rest of ProPainter's packages, from its `requirements.txt`:

    ```
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    ```

    Success: it ends with a list of installed packages.

22. Check that torch sees the card:

    ```
    .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

    Success: `2.11.0+cu128 True`.
    If it says `False`, or the version has no `+cu128`, stop and send the output.

### Tell unmark where ProPainter is

23. Set this for good (it is kept after a restart):

    ```
    setx UNMARK_PROPAINTER C:\ProPainter
    ```

    Success: `SUCCESS: Specified value was saved.`

24. Close PowerShell and open a new window (the setting only reaches new windows). Go back to unmark:

    ```
    cd C:\unmark
    ```

### Run the graphics card check

25. Run:

    ```
    uv run unmark doctor --gpu
    ```

    The first time, ProPainter downloads its model files (about 200 MB: `ProPainter.pth` 158 MB,
    `recurrent_flow_completion.pth` 20 MB and `raft-things.pth` 21 MB) into `C:\ProPainter\weights`. That only happens
    once.
    Success: the last line is `doctor: all required checks passed`, and you see

    - `[ok] ProPainter: C:\ProPainter, run by ...`
    - `[ok] graphics card: torch 2.11.0+cu128 sees <your card's name>, ... GB`
    - `[ok] ProPainter sample: 60 frames of 256x128 filled in ... s, ... dB inside the block, card memory in use peaked
      at ... MiB`

    From now on, `uv run unmark clean` uses ProPainter for solid overlays by itself.

## Send this back

Whether things worked or not, run these in PowerShell from `C:\unmark`, then select all the text from the first command
down to the last line, copy it, and send it to the person who sent you this guide:

```
cd C:\unmark
```

```
uv run unmark doctor --gpu
```

```
nvidia-smi
```

If you did not do Part B, run `uv run unmark doctor` instead of `uv run unmark doctor --gpu`, and skip `nvidia-smi`.

If a `clean` run failed, also send the full output of that run with `--debug` added at the end, for example:

```
uv run unmark clean "C:\videos\my-video.mp4" --debug
```

None of these print any password or key.

## How unmark finds ffmpeg

unmark picks the ffmpeg program in this order:

1. The path in the setting `UNMARK_FFMPEG` (and `UNMARK_FFPROBE` for its partner program `ffprobe`), if you set one.
2. The first `ffmpeg` that Windows finds on its search path (PATH). This is what winget sets up.

If `unmark doctor` uses the wrong ffmpeg, or cannot find it, find where `ffmpeg.exe` and `ffprobe.exe` are on the PC
and set both for good, for example:

```
setx UNMARK_FFMPEG C:\ffmpeg\bin\ffmpeg.exe
```

```
setx UNMARK_FFPROBE C:\ffmpeg\bin\ffprobe.exe
```

(These paths are an example. Use the real ones.) Close PowerShell, open a new window, and run `uv run unmark doctor`
again. The `[ok] ffmpeg:` line shows which program it used.

## Windows notes

- **ProPainter's Python somewhere else.** If you made ProPainter's Python in another place than `C:\ProPainter\.venv`,
  set `UNMARK_PROPAINTER_PYTHON` to the full path of that `python.exe` with `setx`, the same way as above.
- **Out of memory on the graphics card.** If a `clean` run ends with an error that mentions `CUDA out of memory`, send
  the output. Closing games and other programs that use the card can help.
- **Antivirus.** Windows Security scans every new file. unmark and ProPainter write many temporary files, so this can
  slow them down, and in rare cases it can lock a file while it is being written. If you see errors about a file being
  "in use" or "access denied", tell the person who sent you this guide. Adding an exclusion is a security decision
  for the PC's owner, not something to do on your own.
- **Long paths.** If you see an error that mentions a path being too long, move the folders closer to the top of the
  drive first (like `C:\unmark`). Changing the Windows setting for long paths needs an administrator, so ask the PC's
  owner to decide.
