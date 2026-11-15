"""`unmark doctor`, through cli.main, against the real ffmpeg on this machine and the stand-in ProPainter."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from unmark import cli

FAKE = Path(__file__).parent / "fake_propainter"


def test_doctor_passes_here_and_runs_ffmpeg_end_to_end(capsys, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)

    assert cli.main(["doctor"]) == 0

    said = capsys.readouterr().out
    print(said)
    assert "[ok] ffmpeg:" in said and "[ok] end-to-end run:" in said and "[FAIL]" not in said


def test_doctor_fails_in_plain_words_when_ffmpeg_is_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("UNMARK_FFMPEG", str(tmp_path / "no-such-ffmpeg"))

    assert cli.main(["doctor"]) == 1

    said = capsys.readouterr().out
    print(said)
    assert "[FAIL] ffmpeg:" in said and "was not found" in said and "Traceback" not in said
    assert said.rstrip().endswith("check(s) failed")


def test_doctor_gpu_fills_the_sample_with_the_stand_in(capsys, monkeypatch):
    monkeypatch.setenv("UNMARK_PROPAINTER", str(FAKE))
    monkeypatch.setenv("UNMARK_PROPAINTER_PYTHON", sys.executable)  # has no torch

    assert cli.main(["doctor", "--gpu"]) == 1

    said = capsys.readouterr().out
    print(said)
    assert "[ok] ProPainter:" in said and "Traceback" not in said
    assert re.search(r"\[FAIL\] graphics card: torch is not installed", said)
    assert re.search(r"\[ok\] ProPainter sample: 60 frames .* in [\d.]+ s, [\d.]+ dB inside the block", said)


def test_doctor_gpu_without_propainter_says_how_to_set_it(capsys, monkeypatch):
    monkeypatch.delenv("UNMARK_PROPAINTER", raising=False)

    assert cli.main(["doctor", "--gpu"]) == 1

    said = capsys.readouterr().out
    assert "[FAIL] graphics card: --gpu needs ProPainter; set UNMARK_PROPAINTER" in said


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in nvidia-smi is a shell script")
def test_doctor_gpu_reports_the_peak_card_memory_from_nvidia_smi(capsys, monkeypatch, tmp_path):
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 512\necho 1234\n")  # two cards
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("UNMARK_PROPAINTER", str(FAKE))
    monkeypatch.setenv("UNMARK_PROPAINTER_PYTHON", sys.executable)

    cli.main(["doctor", "--gpu"])

    assert "card memory in use peaked at 1234 MiB" in capsys.readouterr().out
