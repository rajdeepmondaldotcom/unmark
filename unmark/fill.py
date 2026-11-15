"""Invent the pixels a solid overlay hides, from the surrounding video: ProPainter when set up, else OpenCV."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np

GROW = 2  # pixels the mask is widened by at full strength, to cover the overlay's own soft edge
FEATHER = 4  # pixels beyond that over which the fill fades into the untouched picture
RADIUS = 5  # cv2.inpaint neighbourhood
PIECE = 400  # most frames sent to ProPainter at once; bounds memory however long the video is
OVERLAP = 10  # frames shared by consecutive pieces of one scene
SHORTEST = 2  # ProPainter needs one pair of frames for its motion; shorter scenes are filled by OpenCV


def _edge(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pixels to replace and, per pixel, how much of the fill to take (1 near the mask, falling to 0)."""
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.clip((GROW + FEATHER - d) / FEATHER, 0, 1)
    return alpha > 0, alpha[..., None]


def _opencv(piece: list[np.ndarray], hole: np.ndarray) -> list[np.ndarray]:
    return [cv2.inpaint(c, hole.astype(np.uint8), RADIUS, cv2.INPAINT_TELEA) for c in piece]


def checkout_python(checkout: Path) -> str:
    """The Python that runs the ProPainter checkout, raising a plain error if the folder or the Python is wrong."""
    if not (checkout / "inference_propainter.py").is_file():
        raise RuntimeError(f"UNMARK_PROPAINTER is {checkout}, which is not a ProPainter checkout (it has no "
                           "inference_propainter.py)")
    python = os.environ.get("UNMARK_PROPAINTER_PYTHON") or str(
        checkout / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    if not Path(python).is_file():
        raise RuntimeError(f"no Python found at {python}; create the checkout's .venv or set UNMARK_PROPAINTER_PYTHON "
                           "to the Python that has ProPainter's packages")
    return python


def _propainter(piece: list[np.ndarray], hole: np.ndarray, checkout: Path) -> list[np.ndarray]:
    """Run the ProPainter checkout's script on piece; returns its filled frames."""
    script, python = checkout / "inference_propainter.py", checkout_python(checkout)
    with tempfile.TemporaryDirectory(prefix="unmark-fill-") as tmp:
        tmp = Path(tmp)
        (tmp / "frames").mkdir()
        for i, crop in enumerate(piece):
            cv2.imwrite(str(tmp / "frames" / f"{i:04d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp / "mask.png"), hole.astype(np.uint8) * 255)
        h, w = hole.shape
        # the mask is already grown here, so ProPainter is told not to grow it again
        p = subprocess.run([python, str(script), "--video", str(tmp / "frames"), "--mask", str(tmp / "mask.png"),
                            "--output", str(tmp / "out"), "--width", str(w), "--height", str(h),
                            "--mask_dilation", "0", "--save_frames", "--fp16"],
                           check=False, cwd=checkout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise RuntimeError(f"ProPainter failed ({p.returncode}): {p.stdout[-2000:]}")
        # the script saves into <output>/<name of the input folder>/frames/0000.png, 0001.png, ...
        done = sorted((tmp / "out" / "frames" / "frames").glob("*.png"))
        if len(done) != len(piece):
            raise RuntimeError(f"ProPainter returned {len(done)} frames for {len(piece)}: {p.stdout[-2000:]}")
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in done]


def fill(crops: Iterable[np.ndarray], mask: np.ndarray, cuts: Iterable[int] = ()) -> Iterator[np.ndarray]:
    """Yield each RGB crop with the pixels under mask replaced; pixels outside the grown, feathered mask are kept."""
    hole, alpha = _edge(mask)
    checkout = os.environ.get("UNMARK_PROPAINTER")
    cuts = set(cuts)

    def run(piece: list[np.ndarray]) -> list[np.ndarray]:
        done = _propainter(piece, hole, Path(checkout)) if checkout and len(piece) >= SHORTEST else _opencv(piece, hole)
        return [np.where(hole[..., None], np.rint(alpha * f + (1 - alpha) * c), c).astype(np.uint8)
                for c, f in zip(piece, done)]

    def faded(piece: list[np.ndarray], tail: list[np.ndarray]) -> list[np.ndarray]:
        """piece filled, its first frames faded in from tail, the previous piece's fill of the same frames."""
        done = run(piece)
        for j, old in enumerate(tail):
            w = (j + 1) / (len(tail) + 1)
            done[j] = np.rint((1 - w) * old + w * done[j]).astype(np.uint8)
        return done

    piece: list[np.ndarray] = []
    tail: list[np.ndarray] = []
    for i, crop in enumerate(crops):
        if piece and i in cuts:
            yield from faded(piece, tail)
            piece, tail = [], []
        elif len(piece) == PIECE:
            done = faded(piece, tail)
            yield from done[:-OVERLAP]
            piece, tail = piece[-OVERLAP:], done[-OVERLAP:]
        piece.append(crop)
    if piece:
        yield from faded(piece, tail)
