from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioSrtPair:
    audio_path: Path
    srt_path: Path

    @property
    def stem(self) -> str:
        return self.audio_path.stem


@dataclass(frozen=True)
class SRTSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class SRTTimeline:
    segments: list[SRTSegment]
    expected_total_ms: int

