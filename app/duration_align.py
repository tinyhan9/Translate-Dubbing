from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import AudioProcessError, concat_wavs, normalize_to_wav, probe_duration_ms, run_cmd, trim_wav
from .noise_bed import create_noise_bed_wav, create_silence_wav


def build_atempo_filter_chain(speed_factor: float) -> str:
    if speed_factor <= 0:
        raise AudioProcessError(f"Invalid speed factor: {speed_factor}")
    factors: list[float] = []
    remain = float(speed_factor)
    while remain > 2.0:
        factors.append(2.0)
        remain /= 2.0
    while remain < 0.5:
        factors.append(0.5)
        remain /= 0.5
    factors.append(remain)
    return ",".join([f"atempo={f:.6f}" for f in factors])


def _pad_to_target(
    input_wav: Path,
    target_ms: int,
    output_wav: Path,
    ref_wav: Path,
    gap_fill_mode: str,
    sr: int,
    channels: int,
) -> int:
    now_ms = probe_duration_ms(input_wav)
    missing = max(target_ms - now_ms, 0)
    if missing == 0:
        normalize_to_wav(input_wav, output_wav, sr=sr, channels=channels)
        return 0

    pad_path = output_wav.with_suffix(".pad.wav")
    if gap_fill_mode == "noise_bed":
        create_noise_bed_wav(ref_wav, pad_path, missing, sr=sr, channels=channels)
    else:
        create_silence_wav(pad_path, missing, sr=sr, channels=channels)
    concat_wavs([input_wav, pad_path], output_wav, sr=sr, channels=channels)
    pad_path.unlink(missing_ok=True)
    return missing


def align_segment_duration(
    input_wav: Path,
    target_duration_ms: int,
    output_wav: Path,
    ref_wav: Path,
    gap_fill_mode: str = "noise_bed",
    sr: int = 16000,
    channels: int = 1,
    tolerance_ms: int = 10,
) -> dict:
    if target_duration_ms <= 0:
        raise AudioProcessError(f"Invalid target_duration_ms: {target_duration_ms}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    normalized_input = output_wav.with_suffix(".src.wav")
    normalize_to_wav(input_wav, normalized_input, sr=sr, channels=channels)

    current_ms = probe_duration_ms(normalized_input)
    action: dict = {
        "target_ms": target_duration_ms,
        "source_ms": current_ms,
        "stretched": False,
        "stretch_ratio": 1.0,
        "padded_ms": 0,
        "trimmed_ms": 0,
        "fallback_trim": False,
        "final_ms": None,
    }

    working = normalized_input
    if current_ms > target_duration_ms + tolerance_ms:
        speed = current_ms / target_duration_ms
        action["stretched"] = True
        action["stretch_ratio"] = speed
        filter_chain = build_atempo_filter_chain(speed)
        stretched = output_wav.with_suffix(".stretch.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(working),
            "-filter:a",
            filter_chain,
            "-ar",
            str(sr),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(stretched),
        ]
        run_cmd(cmd, f"Failed to stretch audio for duration alignment: {working}")
        working = stretched

    now_ms = probe_duration_ms(working)
    if now_ms > target_duration_ms + tolerance_ms:
        trimmed = output_wav.with_suffix(".trim.wav")
        trim_wav(working, target_duration_ms, trimmed, sr=sr, channels=channels)
        action["trimmed_ms"] = max(now_ms - target_duration_ms, 0)
        action["fallback_trim"] = True
        working = trimmed
        now_ms = probe_duration_ms(working)

    if now_ms < target_duration_ms - tolerance_ms:
        padded_out = output_wav.with_suffix(".paded.wav")
        padded_ms = _pad_to_target(working, target_duration_ms, padded_out, ref_wav, gap_fill_mode, sr=sr, channels=channels)
        action["padded_ms"] = padded_ms
        working = padded_out
        now_ms = probe_duration_ms(working)

    # Final correction guard.
    if now_ms > target_duration_ms + tolerance_ms:
        corrected = output_wav.with_suffix(".corr_trim.wav")
        trim_wav(working, target_duration_ms, corrected, sr=sr, channels=channels)
        action["trimmed_ms"] += max(now_ms - target_duration_ms, 0)
        action["fallback_trim"] = True
        working = corrected
    elif now_ms < target_duration_ms - tolerance_ms:
        corrected = output_wav.with_suffix(".corr_pad.wav")
        action["padded_ms"] += _pad_to_target(
            working,
            target_duration_ms,
            corrected,
            ref_wav,
            gap_fill_mode,
            sr=sr,
            channels=channels,
        )
        working = corrected

    normalize_to_wav(working, output_wav, sr=sr, channels=channels)
    final_ms = probe_duration_ms(output_wav)
    action["final_ms"] = final_ms
    action["duration_error_ms"] = final_ms - target_duration_ms
    if abs(action["duration_error_ms"]) > tolerance_ms:
        raise AudioProcessError(
            f"Segment duration alignment failed: target={target_duration_ms}ms final={final_ms}ms (>{tolerance_ms}ms)"
        )
    return action
