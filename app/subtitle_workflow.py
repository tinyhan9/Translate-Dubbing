from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

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
DEFAULT_ONLINE_ASR_CONFIG = "config/online_asr.json"
DEFAULT_ONLINE_TRANSLATE_CONFIG = "config/online_translate.json"
DOUBAO_AUC_API = "https://openspeech.bytedance.com/api/v1/auc"


@dataclass
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str


class WorkflowError(RuntimeError):
    pass


class NetworkWorkflowError(WorkflowError):
    pass


@dataclass(frozen=True)
class OnlineASRConfig:
    app_id: str
    access_token: str
    secret_key: str
    cluster: str
    api_version: str = "v3_flash"
    recognize_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    resource_id: str = "volc.bigasr.auc_turbo"
    request_timeout_seconds: int = 300
    max_audio_size_mb: int = 100
    service_url: str = DOUBAO_AUC_API
    upload_enabled: bool = True
    upload_endpoint: str = "https://transfer.sh"
    upload_timeout_seconds: int = 180
    poll_interval_seconds: float = 2.0
    max_wait_seconds: int = 1200
    language: str | None = None
    channel: int = 1
    sample_rate: int = 16000
    codec: str = "raw"
    speech_noise_threshold: float = 0.8


@dataclass(frozen=True)
class OnlineTranslateConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 90
    batch_size: int = 20


@dataclass(frozen=True)
class RuntimeBackends:
    asr_mode: str
    translate_mode: str


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


def write_backend_report(
    media_out_dir: Path,
    base_name: str,
    *,
    asr_mode: str,
    translate_mode: str,
) -> Path:
    media_out_dir.mkdir(parents=True, exist_ok=True)
    asr_label = "在线识别" if asr_mode == "online" else "本地识别"
    translate_label = "在线翻译" if translate_mode == "online" else "本地翻译"
    report_path = media_out_dir / f"{base_name}.backend.txt"
    content = "\n".join(
        [
            "字幕流程方式说明",
            f"字幕识别最终方式: {asr_label}",
            f"字幕翻译最终方式: {translate_label}",
            "",
        ]
    )
    report_path.write_text(content, encoding="utf-8", newline="\n")
    return report_path


def read_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise WorkflowError(f"Config file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise WorkflowError(f"Invalid JSON config: {config_path} | {exc}") from exc
    if not isinstance(raw, dict):
        raise WorkflowError(f"Config root must be object: {config_path}")
    return raw


def load_online_asr_config(config_path: Path) -> OnlineASRConfig:
    raw = read_json_config(config_path)
    cfg = raw.get("doubao_asr", raw)
    if not isinstance(cfg, dict):
        raise WorkflowError(f"Invalid doubao_asr config object: {config_path}")

    def _need(key: str) -> str:
        value = str(cfg.get(key, "")).strip()
        if not value:
            raise WorkflowError(f"Missing required ASR config key: {key} ({config_path})")
        return value

    upload_cfg = cfg.get("upload", {})
    if not isinstance(upload_cfg, dict):
        upload_cfg = {}

    language_raw = str(cfg.get("language", "")).strip()
    return OnlineASRConfig(
        app_id=_need("app_id"),
        access_token=_need("access_token"),
        secret_key=_need("secret_key"),
        cluster=str(cfg.get("cluster", "volcengine_input_common")).strip() or "volcengine_input_common",
        api_version=str(cfg.get("api_version", "v3_flash")).strip() or "v3_flash",
        recognize_url=(
            str(
                cfg.get(
                    "recognize_url",
                    "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
                )
            ).strip()
            or "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
        ),
        resource_id=str(cfg.get("resource_id", "volc.bigasr.auc_turbo")).strip() or "volc.bigasr.auc_turbo",
        request_timeout_seconds=max(int(cfg.get("request_timeout_seconds", 300)), 10),
        max_audio_size_mb=max(int(cfg.get("max_audio_size_mb", 100)), 1),
        service_url=str(cfg.get("service_url", DOUBAO_AUC_API)).strip() or DOUBAO_AUC_API,
        upload_enabled=bool(upload_cfg.get("enabled", True)),
        upload_endpoint=str(upload_cfg.get("endpoint", "https://transfer.sh")).strip() or "https://transfer.sh",
        upload_timeout_seconds=max(int(upload_cfg.get("timeout_seconds", 180)), 10),
        poll_interval_seconds=max(float(cfg.get("poll_interval_seconds", 2.0)), 0.2),
        max_wait_seconds=max(int(cfg.get("max_wait_seconds", 1200)), 30),
        language=language_raw or None,
        channel=max(int(cfg.get("channel", 1)), 1),
        sample_rate=max(int(cfg.get("sample_rate", 16000)), 8000),
        codec=str(cfg.get("codec", "raw")).strip() or "raw",
        speech_noise_threshold=float(cfg.get("speech_noise_threshold", 0.8)),
    )


def load_online_translate_config(config_path: Path) -> OnlineTranslateConfig:
    raw = read_json_config(config_path)
    cfg = raw.get("rightcode_translate", raw)
    if not isinstance(cfg, dict):
        raise WorkflowError(f"Invalid rightcode_translate config object: {config_path}")

    def _need(key: str) -> str:
        value = str(cfg.get(key, "")).strip()
        if not value:
            raise WorkflowError(f"Missing required translate config key: {key} ({config_path})")
        return value

    return OnlineTranslateConfig(
        api_key=_need("api_key"),
        base_url=_need("base_url"),
        model=_need("model"),
        timeout_seconds=max(int(cfg.get("timeout_seconds", 90)), 10),
        batch_size=max(int(cfg.get("batch_size", 20)), 1),
    )


def choose_runtime_mode_interactive(timeout_seconds: int = 5) -> str:
    print(
        "请选择识别方式：1=在线大模型识别+翻译，2=本地模型识别。"
        f"{timeout_seconds}秒内不输入将默认选择1。",
        flush=True,
    )
    print("请输入 1 或 2: ", end="", flush=True)

    answer_q: queue.Queue[str] = queue.Queue(maxsize=1)

    def _read_input() -> None:
        try:
            value = input().strip()
        except EOFError:
            return
        if value not in {"1", "2"}:
            return
        try:
            answer_q.put_nowait(value)
        except queue.Full:
            pass

    worker = threading.Thread(target=_read_input, daemon=True)
    worker.start()

    try:
        value = answer_q.get(timeout=timeout_seconds).strip()
    except queue.Empty:
        print("", flush=True)
        return "online"

    return "local" if value == "2" else "online"


def _is_network_exception(exc: BaseException) -> bool:
    if isinstance(exc, (NetworkWorkflowError, TimeoutError, socket.timeout, ConnectionError)):
        return True
    if isinstance(exc, urlerror.URLError):
        return True
    if isinstance(exc, OSError) and exc.errno in {101, 110, 111, 113}:
        return True
    return False


def _http_json_request(
    *,
    url: str,
    payload: dict,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = urlrequest.Request(url=url, data=body, headers=req_headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code >= 500:
            raise NetworkWorkflowError(f"HTTP {exc.code} on {url}: {detail}") from exc
        raise WorkflowError(f"HTTP {exc.code} on {url}: {detail}") from exc
    except Exception as exc:
        if _is_network_exception(exc):
            raise NetworkWorkflowError(f"Network error on {url}: {exc}") from exc
        raise

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise WorkflowError(f"Invalid JSON response from {url}: {raw[:500]}") from exc
    if not isinstance(parsed, dict):
        raise WorkflowError(f"Invalid response object from {url}")
    return parsed


def _upload_audio_to_temp_url(audio_file: Path, cfg: OnlineASRConfig, log_file: Path) -> str:
    if not cfg.upload_enabled:
        raise WorkflowError("Online ASR upload is disabled in config; cannot build publicly reachable audio URL")

    quoted_name = urlparse.quote(audio_file.name)
    target = f"{cfg.upload_endpoint.rstrip('/')}/{quoted_name}"
    append_log(log_file, f"Online ASR upload start: {audio_file.name}")

    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            str(cfg.upload_timeout_seconds),
            "--upload-file",
            str(audio_file),
            target,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            uploaded = proc.stdout.strip()
            if uploaded.startswith("http://") or uploaded.startswith("https://"):
                append_log(log_file, f"Online ASR upload done: {uploaded}")
                return uploaded

    try:
        data = audio_file.read_bytes()
    except Exception as exc:
        raise WorkflowError(f"Failed to read audio for upload: {audio_file} | {exc}") from exc

    req = urlrequest.Request(url=target, data=data, method="PUT")
    req.add_header("Max-Days", "1")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urlrequest.urlopen(req, timeout=cfg.upload_timeout_seconds) as resp:
            uploaded = resp.read().decode("utf-8", errors="replace").strip()
    except Exception as exc:
        if _is_network_exception(exc):
            raise NetworkWorkflowError(f"Failed to upload audio for online ASR: {exc}") from exc
        raise WorkflowError(f"Failed to upload audio for online ASR: {exc}") from exc

    if not (uploaded.startswith("http://") or uploaded.startswith("https://")):
        raise WorkflowError(f"Upload endpoint did not return URL: {uploaded[:200]}")
    append_log(log_file, f"Online ASR upload done: {uploaded}")
    return uploaded


def _extract_doubao_utterances(raw: dict) -> list[dict]:
    utterances = raw.get("utterances")
    if isinstance(utterances, list):
        return [u for u in utterances if isinstance(u, dict)]

    result = raw.get("result")
    if isinstance(result, dict):
        inner = result.get("utterances")
        if isinstance(inner, list):
            return [u for u in inner if isinstance(u, dict)]
    if isinstance(result, list):
        merged: list[dict] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            inner = item.get("utterances")
            if isinstance(inner, list):
                merged.extend(u for u in inner if isinstance(u, dict))
        if merged:
            return merged
    return []


def _to_seconds_from_maybe_ms(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if abs(value) > 1000:
        return value / 1000.0
    if isinstance(raw, int):
        return value / 1000.0
    return value


def _build_transcribe_result_from_doubao(raw_query: dict) -> dict:
    utterances = _extract_doubao_utterances(raw_query)
    segments: list[dict] = []
    for idx, utt in enumerate(utterances, start=1):
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        start = _to_seconds_from_maybe_ms(utt.get("start_time", utt.get("start", 0)))
        end = _to_seconds_from_maybe_ms(utt.get("end_time", utt.get("end", start)))
        if end <= start:
            end = start + 0.4
        segments.append({"id": idx, "start": start, "end": end, "text": text})

    full_text = " ".join(seg["text"] for seg in segments).strip()
    if not full_text:
        result = raw_query.get("result")
        if isinstance(result, dict):
            full_text = str(result.get("text", "")).strip()
    if not full_text:
        full_text = str(raw_query.get("text", "")).strip()
    if not segments and full_text:
        segments.append({"id": 1, "start": 0.0, "end": 2.0, "text": full_text})
    return {"text": full_text, "segments": segments, "raw_query": raw_query}


def _run_doubao_v3_flash_once(audio_file: Path, cfg: OnlineASRConfig, log_file: Path) -> dict:
    try:
        audio_bytes = audio_file.read_bytes()
    except Exception as exc:
        raise WorkflowError(f"Failed to read audio file: {audio_file} | {exc}") from exc
    append_log(log_file, "transcribe progress: 12.0%")

    if len(audio_bytes) > cfg.max_audio_size_mb * 1024 * 1024:
        raise WorkflowError(
            f"Audio file too large for v3 flash ({len(audio_bytes)} bytes), "
            f"max={cfg.max_audio_size_mb}MB"
        )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    append_log(log_file, "transcribe progress: 28.0%")
    payload = {
        "user": {"uid": cfg.app_id},
        "audio": {"data": audio_b64},
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_ddc": True,
            "show_utterances": True,
            "result_type": "single",
        },
    }

    req = urlrequest.Request(
        url=cfg.recognize_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Api-App-Key": cfg.app_id,
            "X-Api-Access-Key": cfg.access_token,
            "X-Api-Resource-Id": cfg.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=cfg.request_timeout_seconds) as resp:
            status_header = str(resp.headers.get("X-Api-Status-Code", "")).strip()
            message_header = str(resp.headers.get("X-Api-Message", "")).strip()
            raw_text = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code >= 500:
            raise NetworkWorkflowError(f"Doubao v3 flash HTTP {exc.code}: {detail}") from exc
        raise WorkflowError(f"Doubao v3 flash HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        if _is_network_exception(exc):
            raise NetworkWorkflowError(f"Doubao v3 flash network error: {exc}") from exc
        raise WorkflowError(f"Doubao v3 flash request failed: {exc}") from exc

    append_log(log_file, "transcribe progress: 72.0%")
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise WorkflowError(f"Doubao v3 flash invalid JSON response: {raw_text[:400]}") from exc
    if not isinstance(parsed, dict):
        raise WorkflowError(f"Doubao v3 flash invalid response object: {raw_text[:400]}")

    if status_header and status_header != "20000000":
        raise WorkflowError(
            f"Doubao v3 flash status error: code={status_header}, message={message_header}, body={parsed}"
        )
    return _build_transcribe_result_from_doubao(parsed)


def _run_doubao_v1_submit_query_once(audio_file: Path, cfg: OnlineASRConfig, log_file: Path) -> dict:
    audio_url = _upload_audio_to_temp_url(audio_file, cfg, log_file)
    submit_payload = {
        "appid": cfg.app_id,
        "token": cfg.access_token,
        "cluster": cfg.cluster,
        "audio": {
            "url": audio_url,
        },
        "request": {
            "model_name": "bigmodel",
            "show_utterances": True,
            "enable_itn": True,
            "enable_ddc": True,
            "result_type": "single",
        },
        "additions": {
            "speech_noise_threshold": cfg.speech_noise_threshold,
        },
    }
    if cfg.language:
        submit_payload["additions"]["language"] = cfg.language

    submit_resp = _http_json_request(
        url=f"{cfg.service_url.rstrip('/')}/submit",
        payload=submit_payload,
        timeout_seconds=cfg.upload_timeout_seconds,
    )
    submit_code = int(submit_resp.get("code", -1))
    if submit_code != 1000:
        raise WorkflowError(f"Doubao ASR submit failed: {submit_resp}")

    task_id = str(submit_resp.get("id", "")).strip()
    if not task_id:
        raise WorkflowError(f"Doubao ASR submit returned empty task id: {submit_resp}")

    append_log(log_file, "transcribe progress: 5.0%")
    query_payload = {
        "appid": cfg.app_id,
        "token": cfg.access_token,
        "cluster": cfg.cluster,
        "id": task_id,
    }

    started = time.monotonic()
    last_bucket = -1
    while True:
        query_resp = _http_json_request(
            url=f"{cfg.service_url.rstrip('/')}/query",
            payload=query_payload,
            timeout_seconds=cfg.upload_timeout_seconds,
        )
        code = int(query_resp.get("code", -1))
        if code == 1000:
            return _build_transcribe_result_from_doubao(query_resp)
        if code < 2000:
            raise WorkflowError(f"Doubao ASR query failed: {query_resp}")

        elapsed = time.monotonic() - started
        if elapsed > cfg.max_wait_seconds:
            raise NetworkWorkflowError("Doubao ASR query timed out")
        pct = min(95.0, 5.0 + (elapsed / cfg.max_wait_seconds) * 90.0)
        bucket = int(pct // 2)
        if bucket > last_bucket:
            last_bucket = bucket
            append_log(log_file, f"transcribe progress: {pct:.1f}%")
        time.sleep(cfg.poll_interval_seconds)


def run_doubao_asr_with_retry(
    *,
    audio_file: Path,
    cfg: OnlineASRConfig,
    log_file: Path,
) -> dict:
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            append_log(log_file, f"transcribe started (attempt {attempt})")
            append_log(log_file, "transcribe progress: 0.0%")
            if cfg.api_version == "v1_submit_query":
                result = _run_doubao_v1_submit_query_once(audio_file, cfg, log_file)
            else:
                result = _run_doubao_v3_flash_once(audio_file, cfg, log_file)
            append_log(log_file, "transcribe progress: 100.0%")
            return result
        except Exception as exc:
            last_error = exc
            append_log(log_file, f"transcribe attempt {attempt} failed: {exc}")
            if attempt == 2:
                if _is_network_exception(exc):
                    raise NetworkWorkflowError(f"Doubao ASR failed after retry: {exc}") from exc
                raise WorkflowError(f"Doubao ASR failed after retry: {exc}") from exc
    assert last_error is not None
    raise WorkflowError(f"Doubao ASR failed: {last_error}")


def _extract_json_array_from_text(raw_text: str) -> list[str]:
    text = raw_text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    mt = re.search(r"\[[\s\S]*\]", text)
    if not mt:
        raise WorkflowError(f"Translate response is not JSON array: {text[:300]}")
    try:
        parsed = json.loads(mt.group(0))
    except Exception as exc:
        raise WorkflowError(f"Failed to parse translate JSON array: {text[:300]}") from exc
    if not isinstance(parsed, list):
        raise WorkflowError("Translate response JSON is not an array")
    return [str(x) for x in parsed]


def _rightcode_translate_batch(
    zh_lines: Sequence[str],
    cfg: OnlineTranslateConfig,
) -> list[str]:
    prompt = (
        "你是字幕翻译器。请把输入的中文数组逐条翻译为简短自然英文。"
        "保持数组长度一致、顺序一致。仅返回JSON数组，不要任何额外文本。\n"
        f"输入: {json.dumps(list(zh_lines), ensure_ascii=False)}"
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.2,
    }
    resp = _http_json_request(
        url=f"{cfg.base_url.rstrip('/')}/chat/completions",
        payload=payload,
        timeout_seconds=cfg.timeout_seconds,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    choices = resp.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise WorkflowError(f"Invalid translate response: {resp}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    raw_content = str(message.get("content", "")).strip()
    translated = _extract_json_array_from_text(raw_content)
    if len(translated) != len(zh_lines):
        # Fallback: translate line-by-line to guarantee one-to-one alignment.
        repaired: list[str] = []
        for line in zh_lines:
            single_payload = {
                "model": cfg.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"把这句中文翻成简短英文，只输出英文：{line}",
                    }
                ],
                "stream": False,
                "temperature": 0.2,
            }
            single_resp = _http_json_request(
                url=f"{cfg.base_url.rstrip('/')}/chat/completions",
                payload=single_payload,
                timeout_seconds=cfg.timeout_seconds,
                headers={"Authorization": f"Bearer {cfg.api_key}"},
            )
            single_choices = single_resp.get("choices", [])
            if not isinstance(single_choices, list) or not single_choices:
                repaired.append("")
                continue
            one = single_choices[0] if isinstance(single_choices[0], dict) else {}
            one_msg = one.get("message", {}) if isinstance(one, dict) else {}
            one_text = str(one_msg.get("content", "")).strip()
            one_text = re.sub(r"^['\"`\s]+|['\"`\s]+$", "", one_text)
            repaired.append(one_text)
        translated = repaired
    return translated


def translate_entries_with_rightcode(
    zh_entries: Sequence[SubtitleEntry],
    cfg: OnlineTranslateConfig,
    log_file: Path,
) -> list[str]:
    append_log(log_file, "translate started (attempt 1)")
    append_log(log_file, "translate progress: 0.0%")
    if not zh_entries:
        append_log(log_file, "translate progress: 100.0%")
        return []

    out: list[str] = []
    total = len(zh_entries)
    for offset in range(0, total, cfg.batch_size):
        batch = list(zh_entries[offset : offset + cfg.batch_size])
        zh_lines = [item.text for item in batch]
        translated = _rightcode_translate_batch(zh_lines, cfg)
        for item, raw_en in zip(batch, translated):
            duration = max(item.end - item.start, 0.4)
            en_char_budget = max(28, min(72, int(duration * 20)))
            out.append(compress_en_for_timing(raw_en, max_chars=en_char_budget))

        done = min(total, offset + len(batch))
        pct = min(99.0, (done / total) * 100.0)
        append_log(log_file, f"translate progress: {pct:.1f}%")
    append_log(log_file, "translate progress: 100.0%")
    return out


def translate_entries_with_local_whisper(
    *,
    audio_file: Path,
    zh_entries: Sequence[SubtitleEntry],
    model_size: str,
    model_dir: Path,
    log_file: Path,
    audio_duration_seconds: float,
) -> list[str]:
    if not zh_entries:
        append_log(log_file, "translate progress: 100.0%")
        return []

    append_log(log_file, "Switching to local translate backend (whisper)")
    model = None
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - import depends on environment
        raise WorkflowError(f"Failed to import whisper for local translate fallback: {exc}") from exc

    model_name = choose_model_name(model_dir, model_size)
    try:
        model = whisper.load_model(model_name, download_root=str(model_dir))
    except Exception as exc:  # pragma: no cover - runtime model loading
        raise WorkflowError(f"Failed to load whisper model {model_name} for local translate fallback: {exc}") from exc

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
        return align_english_to_zh(zh_entries, en_segments)
    finally:
        release_runtime(model, log_file)


def generate_subtitles_local(
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
        for stale in (en_path, bi_path):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

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
        backend_report_path = write_backend_report(
            media_out_dir,
            base_name,
            asr_mode="local",
            translate_mode="local",
        )

        outputs = {
            "audio": audio_file,
            "zh": zh_path,
            "summary": summary_path,
            "terms": terms_path,
            "backend_report": backend_report_path,
        }
        if en_path.exists():
            outputs["en"] = en_path
        if bi_path.exists():
            outputs["bi"] = bi_path
        return outputs
    finally:
        release_runtime(model, log_file)


def generate_subtitles_online(
    root: Path,
    requested_audio: str | None,
    model_size: str,
    model_dir: Path,
    out_dir: Path,
    raw_dir: Path,
    log_file: Path,
    asr_cfg: OnlineASRConfig,
    translate_cfg: OnlineTranslateConfig,
) -> dict[str, Path]:
    source_file = resolve_audio_file(root, requested_audio)
    audio_file = source_file
    if source_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        audio_file = extract_mp3_from_video(source_file, root, log_file)
    base_name = audio_file.stem
    audio_duration_seconds = probe_duration_seconds(audio_file)
    if audio_duration_seconds > 0:
        append_log(log_file, f"Audio duration: {audio_duration_seconds:.1f}s")

    try:
        transcribe_result = run_doubao_asr_with_retry(
            audio_file=audio_file,
            cfg=asr_cfg,
            log_file=log_file,
        )
        save_raw_outputs(raw_dir, base_name, transcribe_result)

        raw_entries = entries_from_whisper_segments(transcribe_result.get("segments", []))
        zh_entries = clean_and_split_zh_entries(raw_entries, max_zh_chars=18)
        if not zh_entries:
            raise WorkflowError("No subtitle entries remained after online cleaning")

        if not validate_zh_entries(zh_entries):
            repaired = clean_and_split_zh_entries(normalize_indices(zh_entries), max_zh_chars=18)
            zh_entries = repaired
            if not validate_zh_entries(zh_entries):
                raise WorkflowError("zh subtitle validation failed after online auto-repair")

        media_out_dir = out_dir / base_name
        media_out_dir.mkdir(parents=True, exist_ok=True)
        zh_path = media_out_dir / f"{base_name}.zh.srt"
        en_path = media_out_dir / f"{base_name}.en.srt"
        bi_path = media_out_dir / f"{base_name}.bi.srt"
        for stale in (en_path, bi_path):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

        write_srt(zh_entries, zh_path)

        full_text = str(transcribe_result.get("text", ""))
        summary_path = raw_dir.parent / f"{base_name}.summary.txt"
        terms_path = raw_dir.parent / f"{base_name}.terms.txt"
        summary_path.write_text(summarize_transcript(full_text), encoding="utf-8")
        terms = extract_terms(full_text)
        terms_path.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")

        translate_mode = "online"
        try:
            en_lines = translate_entries_with_rightcode(zh_entries, translate_cfg, log_file)
            en_entries = [
                SubtitleEntry(index=item.index, start=item.start, end=item.end, text=en_lines[idx])
                for idx, item in enumerate(zh_entries)
            ]
            write_srt(en_entries, en_path)
            write_srt(zh_entries, bi_path, bilingual=True, en_lines=en_lines)

            bi_entries = parse_srt(bi_path)
            if not verify_alignment(zh_entries, en_entries, bi_entries):
                append_log(log_file, "Alignment validation failed once, rebuilding online en/bi outputs")
                en_lines = translate_entries_with_rightcode(zh_entries, translate_cfg, log_file)
                en_entries = [
                    SubtitleEntry(index=item.index, start=item.start, end=item.end, text=en_lines[idx])
                    for idx, item in enumerate(zh_entries)
                ]
                write_srt(en_entries, en_path)
                write_srt(zh_entries, bi_path, bilingual=True, en_lines=en_lines)
        except Exception as exc:
            append_log(log_file, f"Online translation failed, switching to local translation: {exc}")
            try:
                en_lines = translate_entries_with_local_whisper(
                    audio_file=audio_file,
                    zh_entries=zh_entries,
                    model_size=model_size,
                    model_dir=model_dir,
                    log_file=log_file,
                    audio_duration_seconds=audio_duration_seconds,
                )
                en_entries = [
                    SubtitleEntry(index=item.index, start=item.start, end=item.end, text=en_lines[idx])
                    for idx, item in enumerate(zh_entries)
                ]
                write_srt(en_entries, en_path)
                write_srt(zh_entries, bi_path, bilingual=True, en_lines=en_lines)
                translate_mode = "local"

                bi_entries = parse_srt(bi_path)
                if not verify_alignment(zh_entries, en_entries, bi_entries):
                    append_log(log_file, "Alignment validation failed after local translate fallback")
            except Exception as fallback_exc:
                append_log(log_file, f"Local translate fallback failed, kept zh only: {fallback_exc}")

        ensure_srt_parseable(zh_path)
        if en_path.exists():
            ensure_srt_parseable(en_path)
        if bi_path.exists():
            ensure_srt_parseable(bi_path)
        backend_report_path = write_backend_report(
            media_out_dir,
            base_name,
            asr_mode="online",
            translate_mode=translate_mode,
        )

        outputs = {
            "audio": audio_file,
            "zh": zh_path,
            "summary": summary_path,
            "terms": terms_path,
            "backend_report": backend_report_path,
        }
        if en_path.exists():
            outputs["en"] = en_path
        if bi_path.exists():
            outputs["bi"] = bi_path
        return outputs
    finally:
        release_runtime(None, log_file)


def run_single_media_with_backends(
    *,
    root: Path,
    media: Path,
    model_size: str,
    model_dir: Path,
    out_dir: Path,
    raw_dir: Path,
    log_file: Path,
    backends: RuntimeBackends,
    asr_cfg: OnlineASRConfig | None,
    translate_cfg: OnlineTranslateConfig | None,
) -> tuple[dict[str, Path], RuntimeBackends]:
    if backends.asr_mode == "online":
        if asr_cfg is None or translate_cfg is None:
            raise WorkflowError("Online mode selected but online config is missing")
        for attempt in (1, 2):
            try:
                outputs = generate_subtitles_online(
                    root=root,
                    requested_audio=str(media),
                    model_size=model_size,
                    model_dir=model_dir,
                    out_dir=out_dir,
                    raw_dir=raw_dir,
                    log_file=log_file,
                    asr_cfg=asr_cfg,
                    translate_cfg=translate_cfg,
                )
                return outputs, backends
            except WorkflowError as exc:
                append_log(log_file, f"Online mode failed attempt {attempt}: {exc}")
                if attempt == 1 and _is_network_exception(exc):
                    append_log(log_file, "Retry online mode once because of network failure")
                    continue
                append_log(log_file, "Switching to local mode after online ASR failure")
                break
        local_backends = RuntimeBackends(asr_mode="local", translate_mode="local")
        outputs = generate_subtitles_local(
            root=root,
            requested_audio=str(media),
            model_size=model_size,
            model_dir=model_dir,
            out_dir=out_dir,
            raw_dir=raw_dir,
            log_file=log_file,
        )
        return outputs, local_backends

    outputs = generate_subtitles_local(
        root=root,
        requested_audio=str(media),
        model_size=model_size,
        model_dir=model_dir,
        out_dir=out_dir,
        raw_dir=raw_dir,
        log_file=log_file,
    )
    return outputs, backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subtitle workflow assistant")
    parser.add_argument("audio", nargs="?", help="Audio file name or path")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--mode",
        choices=["auto", "online", "local"],
        default="auto",
        help="Subtitle backend mode: auto asks at startup, online uses network APIs, local uses whisper",
    )
    parser.add_argument("--model-size", default="large-v3", help="Whisper model size")
    parser.add_argument("--model-dir", default="models/whisper", help="Local whisper model directory")
    parser.add_argument(
        "--online-asr-config",
        default=DEFAULT_ONLINE_ASR_CONFIG,
        help="Online ASR config JSON path",
    )
    parser.add_argument(
        "--online-translate-config",
        default=DEFAULT_ONLINE_TRANSLATE_CONFIG,
        help="Online translate config JSON path",
    )
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
        if args.mode == "auto":
            selected_mode = choose_runtime_mode_interactive(timeout_seconds=5)
        elif args.mode == "online":
            selected_mode = "online"
        else:
            selected_mode = "local"

        backends = RuntimeBackends(asr_mode=selected_mode, translate_mode=selected_mode)
        append_log(log_file, f"Runtime mode selected: {backends.asr_mode}")

        asr_cfg: OnlineASRConfig | None = None
        translate_cfg: OnlineTranslateConfig | None = None
        if backends.asr_mode == "online":
            asr_cfg = load_online_asr_config((root / args.online_asr_config).resolve())
            translate_cfg = load_online_translate_config((root / args.online_translate_config).resolve())
            append_log(log_file, "Online ASR/translate config loaded")

        queue = resolve_audio_queue(root, args.audio)
        append_log(log_file, f"Queue prepared: {len(queue)} media file(s)")

        failures: list[tuple[Path, str]] = []
        for idx, media in enumerate(queue, start=1):
            append_log(log_file, f"Queue item start [{idx}/{len(queue)}]: {media}")
            try:
                _, backends = run_single_media_with_backends(
                    root=root,
                    media=media,
                    model_size=args.model_size,
                    model_dir=model_dir,
                    out_dir=out_dir,
                    raw_dir=raw_dir,
                    log_file=log_file,
                    backends=backends,
                    asr_cfg=asr_cfg,
                    translate_cfg=translate_cfg,
                )
                append_log(
                    log_file,
                    f"Queue item done [{idx}/{len(queue)}]: {media.name} | backend={backends.asr_mode}",
                )
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
