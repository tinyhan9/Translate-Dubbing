from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".ts", ".mov", ".avi", ".webm", ".m4v"}
EN_SHORTEN_STOPWORDS = {
    "a",
    "an",
    "the",
    "just",
    "really",
    "very",
    "basically",
    "actually",
    "kind",
    "sort",
    "of",
    "that",
    "then",
    "well",
}
EN_COMPACT_REPLACEMENTS = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "will not": "won't",
    "i am": "i'm",
    "you are": "you're",
    "we are": "we're",
    "they are": "they're",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "in order to": "to",
    "for example": "e.g.",
}
EN_SPELLING_CORRECTIONS = {
    "helo": "hello",
    "wil": "will",
    "instal": "install",
    "adition": "addition",
    "smal": "small",
    "litle": "little",
    "normaly": "normally",
    "seting": "setting",
    "isue": "issue",
    "suport": "support",
    "becuse": "because",
    "teh": "the",
    "lok": "look",
    "chosee": "choose",
    "ill": "i'll",
    "im": "i'm",
}
ZH_FILLERS = ["呃", "啊", "那个", "就是", "然后", "嗯", "嘛", "哼", "哈"]
EN_FILLERS = ["uh", "um", "erm", "hmm"]
PAUSE_PUNCT = r"[，,、；;：:。\.…]"
REMOVE_PUNCT = r"[\"'“”‘’（）()\[\]【】《》<>`·]"
ACRONYM_MAP = {
    "api": "API",
    "ai": "AI",
    "llm": "LLM",
    "gemini": "GEMINI",
    "cpu": "CPU",
    "gpu": "GPU",
    "ui": "UI",
    "ux": "UX",
}
TERM_NORMALIZE_MAP = {
    "open ai": "OpenAI",
    "chat gpt": "ChatGPT",
    "chatgpt": "ChatGPT",
    "stable diffusion": "Stable Diffusion",
    "胡定你": "houdini",
    "胡迪你": "houdini",
    "hudini": "houdini",
    "胡定尼": "houdini",
    "虎迪里": "houdini",
    "针": "帧",
    "泰利": "tiny",
    "tyler": "tiny",
    "valum": "vellum",
}


@dataclass
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str


class WorkflowError(RuntimeError):
    pass


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def format_srt_time(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hh = milliseconds // 3_600_000
    mm = (milliseconds % 3_600_000) // 60_000
    ss = (milliseconds % 60_000) // 1000
    ms = milliseconds % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def parse_srt_time(raw: str) -> float:
    hh, mm, rest = raw.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


SRT_TIME_RANGE_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*(?:-->|->)\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*$"
)


def _normalize_srt_time_token(raw: str) -> str:
    return raw.strip().replace(".", ",")


def _normalize_srt_timing_line(raw: str) -> str:
    fixed = raw.strip()
    fixed = fixed.replace("鈥?", "-->")
    fixed = re.sub(r"(?<!-)->", "-->", fixed)
    mt = SRT_TIME_RANGE_RE.match(fixed)
    if not mt:
        return fixed
    start = _normalize_srt_time_token(mt.group(1))
    end = _normalize_srt_time_token(mt.group(2))
    return f"{start} --> {end}"


def _parse_srt_text_strict(text: str, path: Path) -> list[SubtitleEntry]:
    blocks = re.split(r"\n\s*\n", text.strip())
    entries: list[SubtitleEntry] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise WorkflowError(f"Invalid SRT block in {path}")

        try:
            index = int(lines[0].strip())
        except Exception as exc:
            raise WorkflowError(f"Invalid subtitle index in {path}") from exc

        timing = _normalize_srt_timing_line(lines[1])
        mt = SRT_TIME_RANGE_RE.match(timing)
        if not mt:
            raise WorkflowError(f"Invalid timing in {path}")

        start_raw = _normalize_srt_time_token(mt.group(1))
        end_raw = _normalize_srt_time_token(mt.group(2))
        try:
            start = parse_srt_time(start_raw)
            end = parse_srt_time(end_raw)
        except Exception as exc:
            raise WorkflowError(f"Invalid time range in {path}") from exc

        content = "\n".join(lines[2:]).strip()
        if not content:
            raise WorkflowError(f"Invalid SRT block in {path}")

        entries.append(SubtitleEntry(index=index, start=start, end=end, text=content))
    return entries


def _repair_srt_format_only(raw: str, path: Path) -> tuple[list[SubtitleEntry], str]:
    blocks = [block for block in re.split(r"\n\s*\n", raw.strip()) if block.strip()]
    repaired: list[SubtitleEntry] = []

    for block in blocks:
        lines = [line.rstrip("\n\r") for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timing_idx = -1
        timing_line = ""
        for idx, line in enumerate(lines):
            candidate = _normalize_srt_timing_line(line)
            if SRT_TIME_RANGE_RE.match(candidate):
                timing_idx = idx
                timing_line = candidate
                break
        if timing_idx < 0:
            continue

        mt = SRT_TIME_RANGE_RE.match(timing_line)
        if mt is None:
            continue

        start_raw = _normalize_srt_time_token(mt.group(1))
        end_raw = _normalize_srt_time_token(mt.group(2))
        try:
            start = parse_srt_time(start_raw)
            end = parse_srt_time(end_raw)
        except Exception:
            continue
        if start >= end:
            continue

        text_lines = [line.strip() for line in lines[timing_idx + 1 :] if line.strip()]
        if not text_lines:
            # Keep minimal-repair scope: drop malformed empty-text blocks only.
            continue

        repaired.append(SubtitleEntry(index=0, start=start, end=end, text="\n".join(text_lines)))

    if not repaired:
        raise WorkflowError(f"Invalid SRT and no valid blocks after format repair: {path}")

    repaired = normalize_indices(repaired)
    rebuilt: list[str] = []
    for entry in repaired:
        rebuilt.append(str(entry.index))
        rebuilt.append(f"{format_srt_time(entry.start)} --> {format_srt_time(entry.end)}")
        rebuilt.append(entry.text)
        rebuilt.append("")
    repaired_text = "\n".join(rebuilt).strip() + "\n"
    return repaired, repaired_text


def write_srt(entries: Sequence[SubtitleEntry], path: Path, bilingual: bool = False, en_lines: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, entry in enumerate(entries, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_time(entry.start)} --> {format_srt_time(entry.end)}\n")
            if bilingual:
                if en_lines is None:
                    raise ValueError("en_lines is required for bilingual output")
                en_text = en_lines[idx - 1] if idx - 1 < len(en_lines) else ""
                f.write(f"{entry.text}\n{en_text}\n\n")
            else:
                f.write(f"{entry.text}\n\n")


def parse_srt(path: Path) -> list[SubtitleEntry]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return _parse_srt_text_strict(text, path)
    except WorkflowError as exc:
        msg = str(exc)
        format_errors = (
            "Invalid SRT block",
            "Invalid subtitle index",
            "Invalid timing",
            "Invalid time range",
        )
        if not any(token in msg for token in format_errors):
            raise

        repaired_entries, repaired_text = _repair_srt_format_only(text, path)
        if repaired_text.strip() != text.strip():
            path.write_text(repaired_text, encoding="utf-8", newline="\n")
        return repaired_entries


def list_media_files(root: Path) -> list[Path]:
    start_dir = (root / "start").resolve()
    if not start_dir.exists() or not start_dir.is_dir():
        return []

    files: list[Path] = []
    for path in start_dir.rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in SUPPORTED_EXTENSIONS or path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ):
            files.append(path.resolve())
    return files


def sort_paths_by_mtime_desc(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)


def resolve_audio_file(root: Path, requested: str | None) -> Path:
    start_dir = (root / "start").resolve()
    media_files = list_media_files(root)
    if not media_files:
        raise WorkflowError(f"No audio/video files found in start directory: {start_dir}")

    if requested:
        requested_path = (root / requested).resolve()
        if requested_path.exists() and (
            requested_path.suffix.lower() in SUPPORTED_EXTENSIONS
            or requested_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ):
            try:
                requested_path.relative_to(start_dir)
            except Exception as exc:
                raise WorkflowError(f"Requested file must be under start directory: {start_dir}") from exc
            return requested_path

        req_name = Path(requested).name.lower()
        matches = [p for p in media_files if p.name.lower() == req_name]
        if not matches:
            raise WorkflowError(f"Requested audio/video file not found: {requested}")
        return sort_paths_by_mtime_desc(matches)[0]

    return max(media_files, key=lambda x: x.stat().st_mtime)


def resolve_audio_queue(root: Path, requested: str | None) -> list[Path]:
    if requested:
        return [resolve_audio_file(root, requested)]

    start_dir = (root / "start").resolve()
    media_files = list_media_files(root)
    if not media_files:
        raise WorkflowError(f"No audio/video files found in start directory: {start_dir}")
    return sort_paths_by_mtime_desc(media_files)


def extract_mp3_from_video(video_file: Path, root: Path, log_file: Path) -> Path:
    output_mp3 = (root / f"{video_file.stem}.mp3").resolve()
    try:
        if output_mp3.exists() and output_mp3.stat().st_mtime >= video_file.stat().st_mtime:
            append_log(log_file, f"Video audio reuse: {video_file.name} -> {output_mp3.name}")
            append_log(log_file, "extract_audio progress: 100.0%")
            return output_mp3
    except Exception:
        pass

    append_log(log_file, f"Video detected, extracting MP3: {video_file.name} -> {output_mp3.name}")
    total_seconds = probe_duration_seconds(video_file)
    if total_seconds > 0:
        append_log(log_file, f"extract_audio progress: 0.0% (0.0/{total_seconds:.1f}s)")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_mp3),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        raise WorkflowError(f"Failed to run ffmpeg for video extraction: {exc}") from exc

    last_bucket = -1
    last_line = ""
    if proc.stdout is not None:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            last_line = line
            if not line.startswith("out_time_ms="):
                continue
            if total_seconds <= 0:
                continue
            try:
                out_us = int(line.split("=", 1)[1].strip())
            except Exception:
                continue
            out_seconds = max(out_us / 1_000_000.0, 0.0)
            pct = min(99.0, (out_seconds / total_seconds) * 100.0)
            bucket = int(pct // 2)  # log every 2%
            if bucket > last_bucket:
                last_bucket = bucket
                append_log(
                    log_file,
                    f"extract_audio progress: {pct:.1f}% ({out_seconds:.1f}/{total_seconds:.1f}s)",
                )
    return_code = proc.wait()

    if return_code != 0 or not output_mp3.exists():
        detail = last_line.strip()
        raise WorkflowError(f"Failed to extract MP3 from video {video_file.name}: {detail}")
    append_log(log_file, "extract_audio progress: 100.0%")
    return output_mp3


def probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return 0.0
    if proc.returncode != 0:
        return 0.0
    raw = (proc.stdout or "").strip()
    try:
        return max(float(raw), 0.0)
    except Exception:
        return 0.0


class WhisperProgressCapture:
    # Example line: [00:00.000 --> 00:03.520] text...
    _SEGMENT_RE = re.compile(
        r"\[(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2})\.(\d{3})\]"
    )

    def __init__(self, task: str, total_seconds: float, log_file: Path) -> None:
        self.task = task
        self.total_seconds = total_seconds
        self.log_file = log_file
        self._buf = ""
        self._last_percent_bucket = -1

    @staticmethod
    def _to_seconds(hh: str, mm: str, ms: str) -> float:
        return int(hh) * 60 + int(mm) + int(ms) / 1000.0

    def _emit_line(self, line: str) -> None:
        # Keep original console behavior.
        try:
            sys.__stdout__.write(line + "\n")
            sys.__stdout__.flush()
        except Exception:
            pass
        m = self._SEGMENT_RE.search(line)
        if not m or self.total_seconds <= 0:
            return
        end_sec = self._to_seconds(m.group(4), m.group(5), m.group(6))
        pct = min(99.0, (end_sec / self.total_seconds) * 100.0)
        bucket = int(pct // 2)  # log every 2%
        if bucket > self._last_percent_bucket:
            self._last_percent_bucket = bucket
            append_log(
                self.log_file,
                f"{self.task} progress: {pct:.1f}% ({end_sec:.1f}/{self.total_seconds:.1f}s)",
            )

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._emit_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit_line(self._buf.strip())
            self._buf = ""


def release_runtime(model, log_file: Path) -> None:
    try:
        if model is not None:
            del model
    except Exception:
        pass

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    append_log(log_file, "Runtime cleanup finished: model released, cache/memory cleanup attempted")


def open_cmd_progress_window(log_file: Path) -> None:
    if os.name != "nt":
        return
    owner_pid = os.getpid()
    monitor_cmd = log_file.with_suffix(".monitor.cmd")
    monitor_cmd.parent.mkdir(parents=True, exist_ok=True)
    monitor_cmd.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                f"set \"LOG={str(log_file.resolve())}\"",
                f"set \"OWNER_PID={owner_pid}\"",
                "title Subtitle Workflow Progress Monitor",
                ":wait_log",
                "set \"ALIVE=0\"",
                "for /f \"tokens=2 delims=,\" %%a in ('tasklist /FI \"PID eq %OWNER_PID%\" /FO CSV /NH') do set \"ALIVE=1\"",
                "if \"%ALIVE%\"==\"0\" goto done",
                "if exist \"%LOG%\" goto loop",
                "echo Waiting for log file...",
                "timeout /t 1 /nobreak >nul",
                "goto wait_log",
                ":loop",
                "set \"ALIVE=0\"",
                "for /f \"tokens=2 delims=,\" %%a in ('tasklist /FI \"PID eq %OWNER_PID%\" /FO CSV /NH') do set \"ALIVE=1\"",
                "if \"%ALIVE%\"==\"0\" goto done",
                "cls",
                "echo Subtitle Workflow Progress Monitor",
                "echo Log: %LOG%",
                "echo ----------------------------------------",
                "type \"%LOG%\"",
                "echo ----------------------------------------",
                "echo Refresh every 1 second. Press Ctrl+C to close.",
                "timeout /t 1 /nobreak >nul",
                "goto loop",
                ":done",
                "exit /b 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(["cmd", "/c", str(monitor_cmd.resolve())], creationflags=flags)


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def estimate_zh_line_length(text: str) -> float:
    chinese = count_chinese_chars(text)
    other = len(re.sub(r"[\u4e00-\u9fff]", "", text))
    return chinese + other * 0.5


def normalize_terms(text: str) -> str:
    normalized = text
    for wrong, right in TERM_NORMALIZE_MAP.items():
        normalized = re.sub(re.escape(wrong), right, normalized, flags=re.IGNORECASE)

    for low, up in ACRONYM_MAP.items():
        normalized = re.sub(rf"\b{re.escape(low)}\b", up, normalized, flags=re.IGNORECASE)

    nonhuman_context = r"(模型|系统|工具|软件|程序|节点|引擎|平台|框架|插件|服务|模块|接口|AI|API|LLM|GEMINI)"
    normalized = re.sub(rf"{nonhuman_context}([^\n]{{0,8}}?)(他|她|它)", lambda m: f"{m.group(1)}{m.group(2)}TA", normalized)
    normalized = normalized.replace("它", "TA")
    return normalized


def strip_fillers(text: str) -> str:
    cleaned = text
    for filler in ZH_FILLERS:
        cleaned = cleaned.replace(filler, " ")
    cleaned = cleaned.replace("呢", "")
    cleaned = re.sub(r"\b(?:" + "|".join(EN_FILLERS) + r")\b", " ", cleaned, flags=re.IGNORECASE)
    # Deduplicate short stutter-like repeats only when Chinese text is present.
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        cleaned = re.sub(r"([\u4e00-\u9fffA-Za-z]{1,3})(?:\s*\1){1,}", r"\1", cleaned)
    return cleaned


def clean_zh_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(PAUSE_PUNCT, "§", normalized)
    normalized = re.sub(REMOVE_PUNCT, "", normalized)
    normalized = strip_fillers(normalized)
    normalized = normalize_terms(normalized)
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9?!§\s]", "", normalized)
    normalized = re.sub(r"\s*§\s*", "§", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("§", "  ")
    normalized = re.sub(r" {3,}", "  ", normalized)
    normalized = normalized.strip()
    return normalized


def clean_en_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = strip_fillers(normalized)
    normalized = normalize_terms(normalized)
    normalized = re.sub(PAUSE_PUNCT, " ", normalized)
    normalized = re.sub(REMOVE_PUNCT, "", normalized)
    normalized = re.sub(r"[^A-Za-z0-9?!\s\-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = spell_correct_en_text(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _apply_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def spell_correct_en_text(text: str) -> str:
    if not text:
        return text
    out: list[str] = []
    for token in text.split():
        m = re.match(r"^([A-Za-z]+)([?!]*)$", token)
        if not m:
            out.append(token)
            continue
        word = m.group(1)
        suffix = m.group(2)
        corrected = EN_SPELLING_CORRECTIONS.get(word.lower())
        if corrected is None:
            out.append(token)
            continue
        out.append(_apply_case(word, corrected) + suffix)
    return " ".join(out)


def compress_en_for_timing(text: str, max_chars: int) -> str:
    max_chars = max(12, max_chars)
    compact = clean_en_text(text)
    if len(compact) <= max_chars:
        return compact

    shortened = compact
    for src, dst in EN_COMPACT_REPLACEMENTS.items():
        shortened = re.sub(rf"\b{re.escape(src)}\b", dst, shortened, flags=re.IGNORECASE)
    shortened = re.sub(r"\s+", " ", shortened).strip()

    words = shortened.split()
    if not words:
        return shortened

    filtered = [w for w in words if w.lower() not in EN_SHORTEN_STOPWORDS]
    if filtered:
        shortened = " ".join(filtered)

    shortened = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", shortened, flags=re.IGNORECASE)
    shortened = re.sub(r"\s+", " ", shortened).strip()

    # Do not hard-cut text. Keep semantic completeness and whole words.
    return shortened if shortened else compact


def choose_limit(text: str, max_zh_chars: int, max_en_chars: int) -> int:
    zh_count = count_chinese_chars(text)
    return max_zh_chars if zh_count >= max(1, len(text) // 4) else max_en_chars


def line_metric(text: str, limit: int) -> float:
    if limit <= 18:
        return estimate_zh_line_length(text)
    return float(len(text))


def hard_split_phrase(phrase: str, limit: int) -> list[str]:
    if not phrase:
        return []
    if limit <= 18:
        parts: list[str] = []
        current = ""
        for ch in phrase:
            candidate = f"{current}{ch}"
            if current and estimate_zh_line_length(candidate) > limit:
                parts.append(current.strip())
                current = ch
            else:
                current = candidate
        if current.strip():
            parts.append(current.strip())
        return parts

    words = phrase.split(" ")
    parts = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > limit:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_text_to_chunks(text: str, max_zh_chars: int = 18, max_en_chars: int = 42) -> list[str]:
    if not text:
        return []

    limit = choose_limit(text, max_zh_chars, max_en_chars)
    phrases = [p.strip() for p in text.split("  ") if p.strip()]
    if not phrases:
        phrases = [text.strip()]

    chunks: list[str] = []
    current = ""

    for phrase in phrases:
        candidate = phrase if not current else f"{current}  {phrase}"
        if current and line_metric(candidate, limit) > limit:
            chunks.append(current)
            current = phrase
        else:
            current = candidate

        if line_metric(current, limit) > limit:
            oversize = current
            current = ""
            pieces = hard_split_phrase(oversize, limit)
            chunks.extend(piece for piece in pieces[:-1] if piece)
            if pieces:
                current = pieces[-1]

    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        if line_metric(chunk, limit) <= limit:
            final_chunks.append(chunk.strip())
        else:
            final_chunks.extend(piece.strip() for piece in hard_split_phrase(chunk, limit) if piece.strip())
    return final_chunks


def split_entry_to_length(entry: SubtitleEntry, max_zh_chars: int = 18, max_en_chars: int = 42) -> list[SubtitleEntry]:
    text = entry.text.replace("\n", " ").strip()
    if not text:
        return []

    chunks = split_text_to_chunks(text, max_zh_chars=max_zh_chars, max_en_chars=max_en_chars)
    if len(chunks) == 1:
        return [SubtitleEntry(index=entry.index, start=entry.start, end=entry.end, text=chunks[0])]

    duration = max(entry.end - entry.start, 0.001)
    weights = [max(len(re.sub(r"\s+", "", chunk)), 1) for chunk in chunks]
    total = sum(weights)
    acc = 0
    out: list[SubtitleEntry] = []

    for idx, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        chunk_start = entry.start + duration * (acc / total)
        acc += weight
        chunk_end = entry.end if idx == len(chunks) else entry.start + duration * (acc / total)
        out.append(SubtitleEntry(index=0, start=chunk_start, end=chunk_end, text=chunk))
    return out


def normalize_indices(entries: Sequence[SubtitleEntry]) -> list[SubtitleEntry]:
    return [
        SubtitleEntry(index=i, start=entry.start, end=entry.end, text=entry.text)
        for i, entry in enumerate(entries, start=1)
    ]


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _split_en_words_evenly(text: str, parts: int) -> list[str]:
    compact = clean_en_text(text)
    if parts <= 1:
        return [compact]
    words = compact.split()
    if not words:
        return ["" for _ in range(parts)]

    chunks: list[str] = []
    start = 0
    remaining_words = len(words)
    remaining_parts = parts
    while remaining_parts > 0:
        take = max(1, math.ceil(remaining_words / remaining_parts))
        end = min(start + take, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        start = end
        remaining_words = max(0, len(words) - start)
        remaining_parts -= 1

    while len(chunks) < parts:
        chunks.append("")

    # Reduce awkward splits: avoid dangling stopwords at line tail and tiny chunks.
    for i in range(len(chunks) - 1):
        cur_words = chunks[i].split()
        next_words = chunks[i + 1].split()
        if not cur_words or not next_words:
            continue
        if cur_words[-1].lower() in EN_SHORTEN_STOPWORDS:
            moved = cur_words.pop()
            next_words.insert(0, moved)
            chunks[i] = " ".join(cur_words).strip()
            chunks[i + 1] = " ".join(next_words).strip()

    for i in range(len(chunks) - 1):
        cur_words = chunks[i].split()
        next_words = chunks[i + 1].split()
        if len(cur_words) >= 2 or not next_words:
            continue
        # Borrow one word from next chunk to avoid one-word fragments.
        cur_words.append(next_words.pop(0))
        chunks[i] = " ".join(cur_words).strip()
        chunks[i + 1] = " ".join(next_words).strip()

    return chunks


def align_english_to_zh(zh_entries: Sequence[SubtitleEntry], en_segments: Sequence[dict]) -> list[str]:
    if not zh_entries:
        return []
    if not en_segments:
        return ["" for _ in zh_entries]

    matched: list[tuple[int, str]] = []
    for zh in zh_entries:
        overlaps: list[tuple[float, int, str]] = []
        for seg_idx, seg in enumerate(en_segments):
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            ov = overlap_seconds(zh.start, zh.end, seg_start, seg_end)
            if ov > 0:
                overlaps.append((ov, seg_idx, str(seg.get("text", "")).strip()))

        if overlaps:
            overlaps.sort(key=lambda x: x[0], reverse=True)
            best = overlaps[0]
            matched.append((best[1], best[2]))
        else:
            mid = (zh.start + zh.end) / 2
            nearest_idx, nearest = min(
                enumerate(en_segments),
                key=lambda p: abs(((float(p[1].get("start", 0.0)) + float(p[1].get("end", 0.0))) / 2) - mid),
            )
            merged = str(nearest.get("text", "")).strip()
            matched.append((nearest_idx, merged))

    lines: list[str] = []
    i = 0
    while i < len(zh_entries):
        seg_idx, seg_text = matched[i]
        j = i + 1
        while j < len(zh_entries) and matched[j][0] == seg_idx:
            j += 1

        group_size = j - i
        pieces = _split_en_words_evenly(seg_text, group_size)
        for k in range(group_size):
            zh = zh_entries[i + k]
            duration = max(zh.end - zh.start, 0.4)
            # Keep English concise so speaking rhythm can better fit the source timeline.
            en_char_budget = max(28, min(72, int(duration * 20)))
            text_piece = pieces[k] if k < len(pieces) else seg_text
            merged = compress_en_for_timing(text_piece, max_chars=en_char_budget)
            lines.append(merged)
        i = j

    return lines


def entries_from_whisper_segments(segments: Sequence[dict]) -> list[SubtitleEntry]:
    entries: list[SubtitleEntry] = []
    for i, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        entries.append(
            SubtitleEntry(
                index=i,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=text,
            )
        )
    return entries


def clean_and_split_zh_entries(raw_entries: Sequence[SubtitleEntry], max_zh_chars: int = 18) -> list[SubtitleEntry]:
    cleaned: list[SubtitleEntry] = []
    for entry in raw_entries:
        text = clean_zh_text(entry.text)
        if not text:
            continue
        candidate = SubtitleEntry(index=entry.index, start=entry.start, end=entry.end, text=text)
        cleaned.extend(split_entry_to_length(candidate, max_zh_chars=max_zh_chars, max_en_chars=42))
    return normalize_indices(cleaned)


def summarize_transcript(full_text: str) -> str:
    normalized = re.sub(r"\s+", " ", full_text).strip()
    pieces = [p.strip() for p in re.split(r"[。！？?!\n]", normalized) if p.strip()]

    topic = pieces[0] if pieces else "未识别到明确主题"
    methods = [p for p in pieces if any(k in p for k in ["通过", "使用", "方法", "步骤", "流程"])][:3]
    steps = [p for p in pieces if any(k in p for k in ["首先", "然后", "接着", "最后"])][:4]
    conclusion = pieces[-1] if len(pieces) > 1 else topic

    lines = [
        "主题",
        f"- {topic}",
        "方法",
    ]

    if methods:
        lines.extend([f"- {m}" for m in methods])
    else:
        lines.append("- 未提取到明确方法")

    lines.append("关键步骤")
    if steps:
        lines.extend([f"- {s}" for s in steps])
    else:
        lines.append("- 未提取到明确步骤")

    lines.extend([
        "结论",
        f"- {conclusion}",
    ])
    return "\n".join(lines) + "\n"


def extract_terms(text: str) -> list[str]:
    terms = set()
    for hit in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b", text):
        terms.add(normalize_terms(hit))
    for hit in re.findall(r"[\u4e00-\u9fff]{2,8}(?:模型|系统|节点|插件|算法|接口)", text):
        terms.add(hit)
    return sorted(terms)


def validate_zh_entries(entries: Sequence[SubtitleEntry]) -> bool:
    for entry in entries:
        text = entry.text
        if "\n" in text:
            return False
        if count_chinese_chars(text) > 18:
            return False
        if re.search(r"[，,、；;：:。\.…\"'“”‘’（）()\[\]【】《》<>`·]", text):
            return False
        if not text:
            return False
    return True


def verify_alignment(zh_entries: Sequence[SubtitleEntry], en_entries: Sequence[SubtitleEntry], bi_entries: Sequence[SubtitleEntry]) -> bool:
    if len(zh_entries) != len(en_entries) or len(zh_entries) != len(bi_entries):
        return False

    for zh, en, bi in zip(zh_entries, en_entries, bi_entries):
        if not math.isclose(zh.start, en.start, abs_tol=0.02) or not math.isclose(zh.end, en.end, abs_tol=0.02):
            return False
        if not math.isclose(zh.start, bi.start, abs_tol=0.02) or not math.isclose(zh.end, bi.end, abs_tol=0.02):
            return False
        if "\n" not in bi.text:
            return False
    return True


def save_raw_outputs(raw_dir: Path, base_name: str, result: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    segments = entries_from_whisper_segments(result.get("segments", []))

    txt_path = raw_dir / f"{base_name}.raw.txt"
    json_path = raw_dir / f"{base_name}.raw.json"
    srt_path = raw_dir / f"{base_name}.raw.srt"

    txt_path.write_text(str(result.get("text", "")).strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(normalize_indices(segments), srt_path)


def run_whisper_with_retry(
    model,
    audio: Path,
    *,
    task: str,
    language: str | None,
    log_file: Path,
    audio_duration_seconds: float | None = None,
) -> dict:
    total_seconds = audio_duration_seconds if audio_duration_seconds is not None else probe_duration_seconds(audio)
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            append_log(log_file, f"{task} started (attempt {attempt})")
            append_log(log_file, f"{task} progress: 0.0%")
            capture = WhisperProgressCapture(task=task, total_seconds=total_seconds, log_file=log_file)
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result = model.transcribe(
                    str(audio),
                    task=task,
                    language=language,
                    verbose=True,
                    fp16=False,
                    condition_on_previous_text=False,
                )
            capture.flush()
            append_log(log_file, f"{task} progress: 100.0%")
            return result
        except Exception as exc:  # pragma: no cover - depends on runtime model state
            last_error = exc
            append_log(log_file, f"{task} attempt {attempt} failed: {exc}")
    assert last_error is not None
    raise WorkflowError(f"Whisper {task} failed after retry: {last_error}")


def choose_model_name(model_dir: Path, model_size: str) -> str:
    direct_model = model_dir / f"{model_size}.pt"
    if direct_model.exists():
        return str(direct_model)
    return model_size


def ensure_srt_parseable(path: Path) -> None:
    _ = parse_srt(path)


def generate_subtitles(
    root: Path,
    requested_audio: str | None,
    model_size: str,
    model_dir: Path,
    out_dir: Path,
    raw_dir: Path,
    log_file: Path,
) -> dict[str, Path]:
    source_file = resolve_audio_file(root, requested_audio)
    audio_file = source_file
    if source_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        audio_file = extract_mp3_from_video(source_file, root, log_file)
    base_name = audio_file.stem
    model = None
    audio_duration_seconds = probe_duration_seconds(audio_file)
    if audio_duration_seconds > 0:
        append_log(log_file, f"Audio duration: {audio_duration_seconds:.1f}s")

    try:
        import whisper
    except Exception as exc:  # pragma: no cover - import depends on environment
        raise WorkflowError(f"Failed to import whisper: {exc}") from exc

    model_name = choose_model_name(model_dir, model_size)
    try:
        model = whisper.load_model(model_name, download_root=str(model_dir))
    except Exception as exc:  # pragma: no cover - runtime model loading
        raise WorkflowError(f"Failed to load whisper model {model_name}: {exc}") from exc

    try:
        transcribe_result = run_whisper_with_retry(
            model,
            audio_file,
            task="transcribe",
            language="zh",
            log_file=log_file,
            audio_duration_seconds=audio_duration_seconds,
        )
        save_raw_outputs(raw_dir, base_name, transcribe_result)

        raw_entries = entries_from_whisper_segments(transcribe_result.get("segments", []))
        zh_entries = clean_and_split_zh_entries(raw_entries, max_zh_chars=18)
        if not zh_entries:
            raise WorkflowError("No subtitle entries remained after cleaning")

        if not validate_zh_entries(zh_entries):
            # Automatic repair path: re-clean every line and re-split.
            repaired = clean_and_split_zh_entries(normalize_indices(zh_entries), max_zh_chars=18)
            zh_entries = repaired
            if not validate_zh_entries(zh_entries):
                raise WorkflowError("zh subtitle validation failed after auto-repair")

        media_out_dir = out_dir / base_name
        media_out_dir.mkdir(parents=True, exist_ok=True)
        zh_path = media_out_dir / f"{base_name}.zh.srt"
        write_srt(zh_entries, zh_path)

        full_text = str(transcribe_result.get("text", ""))
        summary_path = raw_dir.parent / f"{base_name}.summary.txt"
        terms_path = raw_dir.parent / f"{base_name}.terms.txt"
        summary_path.write_text(summarize_transcript(full_text), encoding="utf-8")
        terms = extract_terms(full_text)
        terms_path.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")

        en_path = media_out_dir / f"{base_name}.en.srt"
        bi_path = media_out_dir / f"{base_name}.bi.srt"

        en_entries: list[SubtitleEntry] = []
        en_lines: list[str] = []

        try:
            translate_result = run_whisper_with_retry(
                model,
                audio_file,
                task="translate",
                language="zh",
                log_file=log_file,
                audio_duration_seconds=audio_duration_seconds,
            )
            en_segments = translate_result.get("segments", [])
            en_lines = align_english_to_zh(zh_entries, en_segments)
            en_entries = [
                SubtitleEntry(index=entry.index, start=entry.start, end=entry.end, text=en_lines[idx])
                for idx, entry in enumerate(zh_entries)
            ]

            write_srt(en_entries, en_path)
            write_srt(zh_entries, bi_path, bilingual=True, en_lines=en_lines)

            bi_entries = parse_srt(bi_path)
            # Rebuild comparable bi entries from parser preserving timeline/text format.
            if not verify_alignment(zh_entries, en_entries, bi_entries):
                append_log(log_file, "Alignment validation failed once, rebuilding en/bi outputs")
                en_lines = align_english_to_zh(zh_entries, en_segments)
                en_entries = [
                    SubtitleEntry(index=entry.index, start=entry.start, end=entry.end, text=en_lines[idx])
                    for idx, entry in enumerate(zh_entries)
                ]
                write_srt(en_entries, en_path)
                write_srt(zh_entries, bi_path, bilingual=True, en_lines=en_lines)
        except Exception as exc:
            append_log(log_file, f"Translation stage failed, kept zh only: {exc}")

        ensure_srt_parseable(zh_path)
        if en_path.exists():
            ensure_srt_parseable(en_path)
        if bi_path.exists():
            ensure_srt_parseable(bi_path)

        outputs = {
            "audio": audio_file,
            "zh": zh_path,
            "summary": summary_path,
            "terms": terms_path,
        }
        if en_path.exists():
            outputs["en"] = en_path
        if bi_path.exists():
            outputs["bi"] = bi_path
        return outputs
    finally:
        release_runtime(model, log_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subtitle workflow assistant")
    parser.add_argument("audio", nargs="?", help="Audio file name or path")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--model-size", default="large-v3", help="Whisper model size")
    parser.add_argument("--model-dir", default="models/whisper", help="Local whisper model directory")
    parser.add_argument("--out-dir", default="output", help="Directory for final subtitle outputs")
    parser.add_argument("--raw-dir", default=".transcripts/raw", help="Directory for raw transcription outputs")
    parser.add_argument("--log-file", default=".transcripts/error.log", help="Error log path")
    parser.add_argument("--open-progress-window", action="store_true", default=True, help="Open a CMD progress monitor window")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_dir = (root / args.model_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    raw_dir = (root / args.raw_dir).resolve()
    log_file = (root / args.log_file).resolve()
    if args.open_progress_window:
        open_cmd_progress_window(log_file)

    try:
        queue = resolve_audio_queue(root, args.audio)
        append_log(log_file, f"Queue prepared: {len(queue)} media file(s)")

        failures: list[tuple[Path, str]] = []
        for idx, media in enumerate(queue, start=1):
            append_log(log_file, f"Queue item start [{idx}/{len(queue)}]: {media}")
            try:
                _ = generate_subtitles(
                    root=root,
                    requested_audio=str(media),
                    model_size=args.model_size,
                    model_dir=model_dir,
                    out_dir=out_dir,
                    raw_dir=raw_dir,
                    log_file=log_file,
                )
                append_log(log_file, f"Queue item done [{idx}/{len(queue)}]: {media.name}")
            except WorkflowError as exc:
                failures.append((media, str(exc)))
                append_log(log_file, f"Queue item failed [{idx}/{len(queue)}]: {media.name} | {exc}")

        if failures:
            first_path, first_error = failures[0]
            raise WorkflowError(
                f"Queued run failed for {len(failures)} file(s). "
                f"First failure: {first_path.name} | {first_error}"
            )
        return 0
    except WorkflowError as exc:
        append_log(log_file, str(exc))
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
