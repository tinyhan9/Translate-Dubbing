from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run_cmd


def create_silence_wav(output_wav: Path, duration_ms: int, sr: int = 16000, channels: int = 1) -> None:
    seconds = max(duration_ms / 1000.0, 0.0)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    channel_layout = "stereo" if channels == 2 else "mono"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sr}:cl={channel_layout}",
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
    run_cmd(cmd, f"Failed to create silence wav: {output_wav}")


def create_noise_bed_wav(
    ref_wav: Path,
    output_wav: Path,
    duration_ms: int,
    sr: int = 16000,
    channels: int = 1,
    level_db: float = -35.0,
) -> None:
    seconds = max(duration_ms / 1000.0, 0.0)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    # Loop and attenuate reference voice as low-level bed.
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(ref_wav),
        "-t",
        f"{seconds:.3f}",
        "-af",
        f"volume={level_db}dB",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd, f"Failed to create noise bed wav: {output_wav}")
