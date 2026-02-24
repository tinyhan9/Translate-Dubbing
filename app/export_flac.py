from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run_cmd


def export_flac(
    input_wav: Path,
    output_flac: Path,
    sr: int = 22050,
    mono: bool = True,
    audio_bitrate: str = "256k",
) -> None:
    output_flac.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-ar",
        str(sr),
        "-ac",
        "1" if mono else "2",
        "-sample_fmt",
        "s16",
        "-c:a",
        "flac",
        "-b:a",
        audio_bitrate,
        str(output_flac),
    ]
    run_cmd(cmd, f"Failed to export FLAC {output_flac}")
