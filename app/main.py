from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .audio_timeline import build_timeline_wav
from .batching import build_segment_batches
from .duration_align import align_segment_duration
from .errors import AppError
from .export_flac import export_flac
from .ffmpeg_utils import AudioProcessError, cut_wav_segment, normalize_to_wav, probe_duration_ms
from .file_discovery import (
    DiscoveryError,
    build_manifest_srt_map,
    collect_srt_files_for_srt_only,
    collect_audio_files,
    discover_reference_wav,
    pair_audio_to_srt,
    resolve_output_flac,
    stem_from_srt,
)
from .srt_parser import SRTParseError, parse_srt_file
from .tts_adapter_indexTTS2 import IndexTTS2Adapter, TTSError
from .models import AudioSrtPair


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AGENTS2 SRT-to-FLAC pipeline")
    p.add_argument("--project_root", default=".", help="Root for recursive file discovery")
    p.add_argument("--audio", action="append", help="Audio file path or glob pattern (repeatable)")
    p.add_argument("--audio_dir", help="Directory to scan audio files")
    p.add_argument("--manifest", help="JSON list of {audio, srt}")
    p.add_argument("--srt", help="Single SRT override (only valid with one audio)")
    p.add_argument("--ref_wav", help="Reference audio path override (must be under ./ref-audio)")
    p.add_argument("--model_dir", default="models/indexTTS2", help="Local indexTTS2 model folder")
    p.add_argument("--tts_cmd_template", help="Custom command template for indexTTS2 invocation")
    p.add_argument("--out_dir", help="Output directory (default: <project_root>/output)")
    p.add_argument("--report_dir", help="Directory for report json files")
    p.add_argument("--sr", type=int, default=22050, help="Output sample rate")
    p.add_argument("--mono", action="store_true", default=False, help="Output mono audio (default: stereo)")
    p.add_argument("--audio_bitrate", default="320k", help="Output audio bitrate (default: 320k)")
    p.add_argument("--gap_fill_mode", choices=["zero", "noise_bed"], default="zero")
    p.add_argument("--mock_tts", action="store_true", help="Use internal mock TTS for testing")
    p.add_argument("--batch_items", type=int, default=1, help="Max subtitle items per TTS batch (1 disables batching)")
    p.add_argument("--batch_chars", type=int, default=420, help="Max text chars per TTS batch")
    p.add_argument("--batch_ms", type=int, default=30000, help="Max target duration per TTS batch (ms)")
    p.add_argument("--log_file", help="Progress log file path (default auto when --open_progress_window)")
    p.add_argument("--open_progress_window", action="store_true", default=True, help="Open a separate CMD window to tail progress log")
    p.add_argument("--quiet", action="store_true", help="Disable stdout progress logs (file logs still enabled)")
    return p


def _write_report(report_dir: Path, stem: str, payload: dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{stem}.report.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


class _ProgressLogger:
    def __init__(self, enabled: bool, log_file: Path | None) -> None:
        self.enabled = enabled
        self.log_file = log_file
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        if self.enabled:
            print(line, flush=True)
        if self.log_file is not None:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _format_seconds(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _open_progress_window(log_file: Path, logger: _ProgressLogger) -> None:
    if os.name != "nt":
        logger.log("open_progress_window is only supported on Windows.")
        return
    owner_pid = os.getpid()
    log_path = str(log_file.resolve())
    monitor_cmd = log_file.with_suffix(".monitor.cmd")
    monitor_cmd.parent.mkdir(parents=True, exist_ok=True)
    monitor_cmd.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                f"set \"LOG={log_path}\"",
                f"set \"OWNER_PID={owner_pid}\"",
                "title AGENTS2 Progress Monitor",
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
                "echo AGENTS2 Progress Monitor",
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
    try:
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            ["cmd", "/c", str(monitor_cmd.resolve())],
            creationflags=flags,
        )
        logger.log(f"Progress CMD window opened. log={log_file}")
    except Exception as exc:
        logger.log(f"Failed to open progress window: {exc}")


def _release_runtime_memory(progress: _ProgressLogger) -> None:
    gc.collect()
    cleaned = ["gc"]
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
                cleaned.append("cuda_cache")
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
                cleaned.append("cuda_ipc")
            except Exception:
                pass
    except Exception:
        pass
    progress.log(f"Runtime cleanup finished: {', '.join(cleaned)}")


def run_pipeline(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    audio_dir = Path(args.audio_dir).resolve() if args.audio_dir else None
    manifest = Path(args.manifest).resolve() if args.manifest else None
    srt_override = Path(args.srt).resolve() if args.srt else None
    ref_override = Path(args.ref_wav).resolve() if args.ref_wav else None
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_root / "output").resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else (out_dir / "reports")
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (project_root / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    log_file = Path(args.log_file).resolve() if args.log_file else None
    if args.open_progress_window and log_file is None:
        log_file = (report_dir / "run.progress.log").resolve()
    progress = _ProgressLogger(enabled=not args.quiet, log_file=log_file)
    if args.open_progress_window and log_file is not None:
        _open_progress_window(log_file, progress)

    try:
        run_start = time.perf_counter()
        use_audio_mode = bool(args.audio or args.audio_dir or args.manifest)
        if use_audio_mode:
            audios = collect_audio_files(project_root, audio=args.audio, audio_dir=audio_dir, manifest=manifest)
            manifest_map = build_manifest_srt_map(manifest, project_root)
            pairs = pair_audio_to_srt(audios, project_root, srt_override=srt_override, manifest_srt_map=manifest_map)
        else:
            srt_files = collect_srt_files_for_srt_only(project_root, srt_override)
            pairs = []
            for srt in srt_files:
                stem = stem_from_srt(srt)
                pseudo_audio = project_root / f"{stem}.mp3"
                pairs.append(AudioSrtPair(audio_path=pseudo_audio, srt_path=srt))

        ref_wav = discover_reference_wav(project_root, ref_override)
        ref_wav_for_tts = ref_wav
        ref_tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        progress.log(
            f"Start pipeline mode={'audio+srt' if use_audio_mode else 'srt_only'} "
            f"tasks={len(pairs)} model_dir={model_dir} ref_wav={ref_wav.name}"
        )
        adapter = IndexTTS2Adapter(
            model_dir=model_dir,
            sr=args.sr,
            mono=args.mono,
            audio_bitrate=args.audio_bitrate,
            cmd_template=args.tts_cmd_template,
            mock_tts=args.mock_tts,
        )

        try:
            if ref_wav.suffix.lower() != ".wav":
                ref_tmp_dir = tempfile.TemporaryDirectory(prefix="agents2_ref_")
                converted_ref = Path(ref_tmp_dir.name) / f"{ref_wav.stem}.wav"
                normalize_to_wav(ref_wav, converted_ref, sr=args.sr, channels=1 if args.mono else 2)
                ref_wav_for_tts = converted_ref
                progress.log(f"Reference audio normalized to WAV: {ref_wav.name} -> {ref_wav_for_tts.name}")

            for pair_idx, pair in enumerate(pairs, start=1):
                pair_start = time.perf_counter()
                timeline = parse_srt_file(pair.srt_path)
                pair_total_segments = len(timeline.segments)
                report_payload: dict = {
                    "audio_input": str(pair.audio_path) if use_audio_mode else None,
                    "srt_input": str(pair.srt_path),
                    "ref_wav": str(ref_wav),
                    "gap_fill_mode": args.gap_fill_mode,
                    "expected_total_ms": timeline.expected_total_ms,
                    "segments": [],
                    "mode": "audio+srt" if use_audio_mode else "srt_only",
                    "batching": {
                        "batch_items": args.batch_items,
                        "batch_chars": args.batch_chars,
                        "batch_ms": args.batch_ms,
                    },
                    "output_audio": {
                        "sample_rate": args.sr,
                        "channels": 1 if args.mono else 2,
                        "audio_bitrate": args.audio_bitrate,
                    },
                }
                progress.log(
                    f"[{pair_idx}/{len(pairs)}] {pair.srt_path.name} "
                    f"segments={pair_total_segments} expected_total_ms={timeline.expected_total_ms}"
                )
                progress.log(f"[{pair.stem}] tts_to_flac progress: 0.0% stage=start")

                with tempfile.TemporaryDirectory(prefix=f"agents2_{pair.stem}_") as td:
                    td_root = Path(td)
                    aligned_paths: list[Path] = []
                    batches = build_segment_batches(
                        timeline.segments,
                        max_items=args.batch_items,
                        max_chars=args.batch_chars,
                        max_total_ms=args.batch_ms,
                    )
                    report_payload["batching"]["batch_count"] = len(batches)
                    pair_done = 0
                    for bidx, batch in enumerate(batches, start=1):
                        batch_start = time.perf_counter()
                        batch_raw = td_root / f"batch_{bidx:05d}.tts.wav"
                        batch_aligned = td_root / f"batch_{bidx:05d}.aligned.wav"
                        adapter.synthesize(batch.text, ref_wav_for_tts, batch_raw)
                        batch_align = align_segment_duration(
                            batch_raw,
                            batch.total_ms,
                            batch_aligned,
                            ref_wav=ref_wav_for_tts,
                            gap_fill_mode=args.gap_fill_mode,
                            sr=args.sr,
                            channels=1 if args.mono else 2,
                            tolerance_ms=10,
                        )

                        local_start = 0
                        for seg in batch.segments:
                            seg_aligned = td_root / f"{seg.index:05d}.aligned.wav"
                            cut_wav_segment(
                                batch_aligned,
                                start_ms=local_start,
                                duration_ms=seg.duration_ms,
                                output_wav=seg_aligned,
                                sr=args.sr,
                                channels=1 if args.mono else 2,
                            )
                            local_start += seg.duration_ms
                            aligned_paths.append(seg_aligned)
                            report_payload["segments"].append(
                                {
                                    "index": seg.index,
                                    "start_ms": seg.start_ms,
                                    "end_ms": seg.end_ms,
                                    "target_ms": seg.duration_ms,
                                    "batch_id": bidx,
                                    "batch_size": len(batch.segments),
                                    **batch_align,
                                }
                            )
                        pair_done += len(batch.segments)
                        elapsed_pair = time.perf_counter() - pair_start
                        avg_per_seg = elapsed_pair / max(pair_done, 1)
                        eta = avg_per_seg * max(pair_total_segments - pair_done, 0)
                        pct = (pair_done / pair_total_segments * 100.0) if pair_total_segments else 100.0
                        progress.log(
                            f"[{pair.stem}] batch {bidx}/{len(batches)} done "
                            f"segments={pair_done}/{pair_total_segments} "
                            f"pct={pct:.1f}% batch_time={_format_seconds(time.perf_counter() - batch_start)} "
                            f"eta={_format_seconds(eta)}"
                        )
                        tts_to_flac_pct = 90.0 * (pair_done / pair_total_segments) if pair_total_segments else 90.0
                        progress.log(
                            f"[{pair.stem}] tts_to_flac progress: {tts_to_flac_pct:.1f}% "
                            "stage=tts_align"
                        )

                    timeline_wav = td_root / f"{pair.stem}.timeline.wav"
                    timeline_info = build_timeline_wav(
                        timeline=timeline,
                        segment_wavs=aligned_paths,
                        out_wav=timeline_wav,
                        ref_wav=ref_wav_for_tts,
                        gap_fill_mode=args.gap_fill_mode,
                        sr=args.sr,
                        channels=1 if args.mono else 2,
                        tolerance_ms=10,
                    )
                    report_payload["timeline"] = timeline_info
                    progress.log(f"[{pair.stem}] tts_to_flac progress: 95.0% stage=timeline")

                    out_flac = resolve_output_flac(pair.audio_path, out_dir, pair.srt_path)
                    export_flac(
                        timeline_wav,
                        out_flac,
                        sr=args.sr,
                        mono=args.mono,
                        audio_bitrate=args.audio_bitrate,
                    )
                    final_ms = probe_duration_ms(out_flac)
                    report_payload["output_flac"] = str(out_flac)
                    report_payload["output_total_ms"] = final_ms
                    report_payload["output_error_ms"] = final_ms - timeline.expected_total_ms
                    if abs(report_payload["output_error_ms"]) > 10:
                        raise AudioProcessError(
                            f"Output duration mismatch for {out_flac.name}: "
                            f"expected {timeline.expected_total_ms}ms, got {final_ms}ms"
                        )
                    progress.log(f"[{pair.stem}] tts_to_flac progress: 100.0% stage=flac_export")

                _write_report(report_dir, pair.stem, report_payload)
                progress.log(
                    f"[{pair_idx}/{len(pairs)}] {pair.stem} complete "
                    f"output={out_flac} duration_error_ms={report_payload['output_error_ms']} "
                    f"elapsed={_format_seconds(time.perf_counter() - pair_start)}"
                )
        finally:
            adapter.close()
            _release_runtime_memory(progress)
            if ref_tmp_dir is not None:
                ref_tmp_dir.cleanup()
        progress.log(f"All tasks completed in {_format_seconds(time.perf_counter() - run_start)}")
        return 0
    except DiscoveryError as exc:
        raise SystemExit(exc.code) from exc
    except SRTParseError as exc:
        raise SystemExit(exc.code) from exc
    except TTSError as exc:
        raise SystemExit(exc.code) from exc
    except AudioProcessError as exc:
        raise SystemExit(exc.code) from exc
    except AppError as exc:
        raise SystemExit(exc.code) from exc


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())

