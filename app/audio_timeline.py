from __future__ import annotations

import tempfile
from pathlib import Path

from .ffmpeg_utils import AudioProcessError, concat_wavs, probe_duration_ms, trim_wav
from .models import SRTTimeline
from .noise_bed import create_noise_bed_wav, create_silence_wav


def _create_gap(
    out_wav: Path,
    duration_ms: int,
    ref_wav: Path,
    gap_fill_mode: str,
    sr: int = 16000,
    channels: int = 1,
) -> None:
    if duration_ms <= 0:
        raise AudioProcessError(f"Invalid gap duration: {duration_ms}")
    if gap_fill_mode == "noise_bed":
        create_noise_bed_wav(ref_wav, out_wav, duration_ms, sr=sr, channels=channels)
    else:
        create_silence_wav(out_wav, duration_ms, sr=sr, channels=channels)


def build_timeline_wav(
    timeline: SRTTimeline,
    segment_wavs: list[Path],
    out_wav: Path,
    ref_wav: Path,
    gap_fill_mode: str = "noise_bed",
    sr: int = 16000,
    channels: int = 1,
    tolerance_ms: int = 10,
) -> dict:
    if len(timeline.segments) != len(segment_wavs):
        raise AudioProcessError(
            f"segment_wavs length mismatch: expected {len(timeline.segments)}, got {len(segment_wavs)}"
        )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    gaps: list[dict] = []
    cursor = 0

    with tempfile.TemporaryDirectory(prefix="timeline_parts_") as td:
        td_root = Path(td)
        for idx, (seg, seg_wav) in enumerate(zip(timeline.segments, segment_wavs), start=1):
            if seg.start_ms > cursor:
                gap_ms = seg.start_ms - cursor
                gap_path = td_root / f"{idx:05d}.gap.wav"
                _create_gap(gap_path, gap_ms, ref_wav, gap_fill_mode, sr=sr, channels=channels)
                part_paths.append(gap_path)
                gaps.append({"type": "between", "start_ms": cursor, "end_ms": seg.start_ms, "duration_ms": gap_ms})
            part_paths.append(seg_wav)
            cursor = seg.end_ms

        if timeline.expected_total_ms > cursor:
            gap_ms = timeline.expected_total_ms - cursor
            gap_path = td_root / "99999.final_gap.wav"
            _create_gap(gap_path, gap_ms, ref_wav, gap_fill_mode, sr=sr, channels=channels)
            part_paths.append(gap_path)
            gaps.append(
                {"type": "tail", "start_ms": cursor, "end_ms": timeline.expected_total_ms, "duration_ms": gap_ms}
            )

        if not part_paths:
            raise AudioProcessError("No timeline parts to concatenate")
        concat_wavs(part_paths, out_wav, sr=sr, channels=channels)

        actual_ms = probe_duration_ms(out_wav)
        if actual_ms > timeline.expected_total_ms + tolerance_ms:
            trimmed = out_wav.with_suffix(".timeline_trim.wav")
            trim_wav(out_wav, timeline.expected_total_ms, trimmed, sr=sr, channels=channels)
            trimmed.replace(out_wav)
            actual_ms = probe_duration_ms(out_wav)
        elif actual_ms < timeline.expected_total_ms - tolerance_ms:
            gap_path = td_root / "99998.correction_gap.wav"
            _create_gap(
                gap_path,
                timeline.expected_total_ms - actual_ms,
                ref_wav,
                gap_fill_mode,
                sr=sr,
                channels=channels,
            )
            corrected = out_wav.with_suffix(".timeline_pad.wav")
            concat_wavs([out_wav, gap_path], corrected, sr=sr, channels=channels)
            corrected.replace(out_wav)
            actual_ms = probe_duration_ms(out_wav)

    if abs(actual_ms - timeline.expected_total_ms) > tolerance_ms:
        raise AudioProcessError(
            f"Final timeline duration mismatch: expected={timeline.expected_total_ms}ms actual={actual_ms}ms"
        )
    return {
        "expected_total_ms": timeline.expected_total_ms,
        "actual_total_ms": actual_ms,
        "duration_error_ms": actual_ms - timeline.expected_total_ms,
        "gaps": gaps,
    }
