from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import SRTSegment


@dataclass(frozen=True)
class SegmentBatch:
    segments: list[SRTSegment]

    @property
    def total_ms(self) -> int:
        return sum(seg.duration_ms for seg in self.segments)

    @property
    def text(self) -> str:
        # Keep short explicit boundary markers for better prosody separation.
        return " . ".join([seg.text.strip() for seg in self.segments if seg.text.strip()])


def build_segment_batches(
    segments: Sequence[SRTSegment],
    max_items: int = 4,
    max_chars: int = 180,
    max_total_ms: int = 12_000,
) -> list[SegmentBatch]:
    if max_items <= 1:
        return [SegmentBatch(segments=[seg]) for seg in segments]

    batches: list[SegmentBatch] = []
    current: list[SRTSegment] = []
    current_chars = 0
    current_ms = 0

    for seg in segments:
        seg_text = seg.text.strip()
        seg_chars = len(seg_text)
        seg_ms = seg.duration_ms

        if not current:
            current = [seg]
            current_chars = seg_chars
            current_ms = seg_ms
            continue

        exceeds = (
            len(current) + 1 > max_items
            or current_chars + 3 + seg_chars > max_chars
            or current_ms + seg_ms > max_total_ms
        )
        if exceeds:
            batches.append(SegmentBatch(segments=current))
            current = [seg]
            current_chars = seg_chars
            current_ms = seg_ms
        else:
            current.append(seg)
            current_chars += 3 + seg_chars
            current_ms += seg_ms

    if current:
        batches.append(SegmentBatch(segments=current))
    return batches

