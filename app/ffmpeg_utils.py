from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import AppError


class AudioProcessError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=5)


def run_cmd(cmd: list[str], fail_msg: str, code: int = 5) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        if code == 4:
            raise AppError(f"{fail_msg}: {exc}", code=4) from exc
        raise AudioProcessError(f"{fail_msg}: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if code == 4:
            raise AppError(f"{fail_msg}: {detail}", code=4)
        raise AudioProcessError(f"{fail_msg}: {detail}")


def probe_duration_ms(audio_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        raise AudioProcessError(f"Failed to probe duration for {audio_path}: {exc}") from exc
    if proc.returncode != 0:
        raise AudioProcessError(f"ffprobe failed for {audio_path}: {(proc.stderr or proc.stdout).strip()}")
    try:
        seconds = float((proc.stdout or "0").strip())
    except Exception as exc:
        raise AudioProcessError(f"Unable to parse duration from ffprobe for {audio_path}") from exc
    return int(round(seconds * 1000))


def normalize_to_wav(input_path: Path, output_wav: Path, sr: int = 16000, channels: int = 1) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd, f"Failed to normalize audio {input_path}")


def normalize_to_wav_16k_mono(input_path: Path, output_wav: Path, sr: int = 16000) -> None:
    normalize_to_wav(input_path, output_wav, sr=sr, channels=1)


def concat_wavs(inputs: list[Path], output_wav: Path, sr: int = 16000, channels: int = 1) -> None:
    if not inputs:
        raise AudioProcessError("concat_wavs requires at least one input file")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_wav.with_suffix(".concat.txt")
    lines = []
    for path in inputs:
        escaped = str(path.resolve()).replace("'", "''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd, f"Failed to concat wav files to {output_wav}")
    list_path.unlink(missing_ok=True)


def trim_wav(input_wav: Path, target_ms: int, output_wav: Path, sr: int = 16000, channels: int = 1) -> None:
    seconds = max(target_ms / 1000.0, 0.001)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-t",
        f"{seconds:.3f}",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd, f"Failed to trim wav {input_wav}")


def cut_wav_segment(
    input_wav: Path,
    start_ms: int,
    duration_ms: int,
    output_wav: Path,
    sr: int = 16000,
    channels: int = 1,
) -> None:
    start_s = max(start_ms / 1000.0, 0.0)
    dur_s = max(duration_ms / 1000.0, 0.001)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_wav),
        "-t",
        f"{dur_s:.3f}",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd, f"Failed to cut wav segment from {input_wav}")
