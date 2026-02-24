from __future__ import annotations

import re
from pathlib import Path

from .errors import AppError
from .models import SRTSegment, SRTTimeline


class SRTParseError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=3)


TIME_RE = re.compile(
    r"^\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


def _time_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def _ms_to_time(ms_total: int) -> str:
    ms_total = max(0, int(ms_total))
    hh = ms_total // 3_600_000
    mm = (ms_total % 3_600_000) // 60_000
    ss = (ms_total % 60_000) // 1_000
    ms = ms_total % 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _normalize_time_line(line: str) -> str:
    fixed = line.strip()
    fixed = fixed.replace("—>", "-->")
    fixed = re.sub(r"(?<!-)->", "-->", fixed)
    fixed = re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", fixed)
    return fixed


def _parse_srt_text(raw: str, multiline_joiner: str, allow_overlap: bool, path: Path) -> SRTTimeline:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw.strip()) if block.strip()]
    segments: list[SRTSegment] = []
    prev_end = -1

    for bidx, block in enumerate(blocks, start=1):
        lines = [line.rstrip("\n\r") for line in block.splitlines() if line.strip() != ""]
        if len(lines) < 3:
            raise SRTParseError(f"Invalid SRT block #{bidx} in {path}")

        try:
            index = int(lines[0].strip())
        except Exception as exc:
            raise SRTParseError(f"Invalid subtitle index in block #{bidx} ({path})") from exc

        mt = TIME_RE.match(lines[1])
        if not mt:
            raise SRTParseError(f"Invalid time range in block #{bidx} ({path})")
        start_ms = _time_to_ms(mt.group(1), mt.group(2), mt.group(3), mt.group(4))
        end_ms = _time_to_ms(mt.group(5), mt.group(6), mt.group(7), mt.group(8))

        if start_ms >= end_ms:
            raise SRTParseError(f"SRT has start >= end at block #{bidx} ({path})")

        if segments and start_ms < segments[-1].start_ms:
            raise SRTParseError(f"SRT is not time-ordered at block #{bidx} ({path})")

        if (not allow_overlap) and prev_end > -1 and start_ms < prev_end:
            raise SRTParseError(f"SRT has overlapping segments at block #{bidx} ({path})")

        text = multiline_joiner.join(lines[2:])
        segments.append(
            SRTSegment(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )
        prev_end = max(prev_end, end_ms)

    if not segments:
        raise SRTParseError(f"Empty SRT file: {path}")

    expected_total_ms = max(s.end_ms for s in segments)
    return SRTTimeline(segments=segments, expected_total_ms=expected_total_ms)


def _repair_format_only(raw: str, multiline_joiner: str, path: Path) -> str:
    """Minimal auto-fix: only repair format-broken blocks and keep valid logic unchanged."""
    blocks = [block for block in re.split(r"\n\s*\n", raw.strip()) if block.strip()]
    repaired: list[tuple[int, int, str]] = []

    for block in blocks:
        lines = [line.rstrip("\n\r") for line in block.splitlines() if line.strip() != ""]
        if not lines:
            continue

        mt = None
        timing_idx = -1
        for idx, line in enumerate(lines):
            cand = TIME_RE.match(line.strip()) or TIME_RE.match(_normalize_time_line(line))
            if cand:
                mt = cand
                timing_idx = idx
                break

        if mt is None:
            continue

        start_ms = _time_to_ms(mt.group(1), mt.group(2), mt.group(3), mt.group(4))
        end_ms = _time_to_ms(mt.group(5), mt.group(6), mt.group(7), mt.group(8))
        if start_ms >= end_ms:
            continue

        text_lines = [line.strip() for line in lines[timing_idx + 1 :] if line.strip()]
        if not text_lines:
            # Drop malformed empty-text blocks only.
            continue

        repaired.append((start_ms, end_ms, multiline_joiner.join(text_lines)))

    if not repaired:
        raise SRTParseError(f"Invalid SRT and no valid blocks after format repair: {path}")

    out_lines: list[str] = []
    for idx, (start_ms, end_ms, text) in enumerate(repaired, start=1):
        out_lines.append(str(idx))
        out_lines.append(f"{_ms_to_time(start_ms)} --> {_ms_to_time(end_ms)}")
        out_lines.append(text)
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


def parse_srt_file(path: Path, multiline_joiner: str = " ", allow_overlap: bool = False) -> SRTTimeline:
    if not path.exists() or not path.is_file():
        raise SRTParseError(f"SRT file not found: {path}")

    raw = path.read_text(encoding="utf-8-sig")
    try:
        return _parse_srt_text(raw, multiline_joiner, allow_overlap, path)
    except SRTParseError as exc:
        # Keep original behavior for logical timeline errors; only repair format issues.
        msg = str(exc)
        format_errors = (
            "Invalid SRT block",
            "Invalid subtitle index",
            "Invalid time range",
        )
        if not any(token in msg for token in format_errors):
            raise

        repaired_raw = _repair_format_only(raw, multiline_joiner, path)
        timeline = _parse_srt_text(repaired_raw, multiline_joiner, allow_overlap, path)
        if repaired_raw.strip() != raw.strip():
            path.write_text(repaired_raw, encoding="utf-8", newline="\n")
        return timeline
