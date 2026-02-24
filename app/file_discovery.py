from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import AppError
from .models import AudioSrtPair

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".ts", ".mov", ".avi", ".webm", ".m4v"}
SUPPORTED_MEDIA_EXTS = SUPPORTED_AUDIO_EXTS | SUPPORTED_VIDEO_EXTS


class DiscoveryError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=2)


@dataclass(frozen=True)
class ManifestEntry:
    audio_path: Path
    srt_path: Path | None


def stem_from_srt(srt_path: Path) -> str:
    name = srt_path.name
    lower = name.lower()
    if lower.endswith(".en.srt"):
        return name[: -len(".en.srt")]
    return srt_path.stem


def _ensure_exists(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise DiscoveryError(f"{label} not found: {path}")
    return path


def _dedupe_keep_order(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(path.resolve())
    return out


def _expand_audio_inputs(project_root: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        if raw.exists() and raw.is_file() and raw.suffix.lower() in SUPPORTED_MEDIA_EXTS:
            found.append(raw.resolve())
            continue

        # Support glob patterns both absolute and project-root relative.
        if raw.is_absolute():
            parent = raw.parent if raw.parent != Path("") else project_root
            matches = list(parent.glob(raw.name))
        else:
            matches = list(project_root.glob(pattern))

        for match in matches:
            if match.is_file() and match.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                found.append(match.resolve())
    return found


def load_manifest_entries(manifest_path: Path, project_root: Path) -> list[ManifestEntry]:
    manifest_file = _ensure_exists(manifest_path.resolve(), "manifest")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiscoveryError(f"Invalid manifest JSON: {manifest_file}") from exc

    if not isinstance(payload, list):
        raise DiscoveryError("Manifest must be a JSON array")

    entries: list[ManifestEntry] = []
    for i, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise DiscoveryError(f"Manifest row {i} must be an object")
        audio_raw = item.get("audio")
        srt_raw = item.get("srt")
        if not isinstance(audio_raw, str) or not audio_raw.strip():
            raise DiscoveryError(f"Manifest row {i} missing audio")
        audio_path = (project_root / audio_raw).resolve() if not Path(audio_raw).is_absolute() else Path(audio_raw).resolve()
        _ensure_exists(audio_path, f"manifest audio row {i}")
        if audio_path.suffix.lower() not in SUPPORTED_MEDIA_EXTS:
            raise DiscoveryError(f"Manifest audio row {i} is not a supported media file: {audio_path}")

        srt_path: Path | None = None
        if srt_raw is not None:
            if not isinstance(srt_raw, str) or not srt_raw.strip():
                raise DiscoveryError(f"Manifest row {i} has invalid srt path")
            candidate = (project_root / srt_raw).resolve() if not Path(srt_raw).is_absolute() else Path(srt_raw).resolve()
            _ensure_exists(candidate, f"manifest srt row {i}")
            srt_path = candidate
        entries.append(ManifestEntry(audio_path=audio_path, srt_path=srt_path))
    return entries


def collect_audio_files(
    project_root: Path,
    audio: list[str] | None,
    audio_dir: Path | None,
    manifest: Path | None,
) -> list[Path]:
    out: list[Path] = []
    if audio:
        out.extend(_expand_audio_inputs(project_root, audio))

    if audio_dir:
        root = audio_dir.resolve() if audio_dir.is_absolute() else (project_root / audio_dir).resolve()
        if not root.exists() or not root.is_dir():
            raise DiscoveryError(f"audio_dir not found: {root}")
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                out.append(item.resolve())

    if manifest:
        manifest_entries = load_manifest_entries(manifest, project_root)
        out.extend([entry.audio_path for entry in manifest_entries])

    # Default mode: read root-level MP3 files when no explicit input is provided.
    if not audio and not audio_dir and not manifest:
        for item in sorted(project_root.iterdir()):
            if item.is_file() and item.suffix.lower() == ".mp3":
                out.append(item.resolve())

    out = _dedupe_keep_order(out)
    if not out:
        raise DiscoveryError("No input audio files found. Provide --audio, --audio_dir, or --manifest.")
    return out


def build_manifest_srt_map(manifest: Path | None, project_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not manifest:
        return mapping
    for entry in load_manifest_entries(manifest, project_root):
        if entry.srt_path is not None:
            mapping[str(entry.audio_path.resolve()).lower()] = entry.srt_path.resolve()
    return mapping


def _find_srt_candidates_by_filename(project_root: Path, filename_lower: str) -> list[Path]:
    candidates: list[Path] = []
    for item in project_root.rglob("*"):
        if item.is_file() and item.name.lower() == filename_lower:
            candidates.append(item.resolve())
    return sorted(candidates, key=lambda p: str(p).lower())


def pair_audio_to_srt(
    audio_files: list[Path],
    project_root: Path,
    srt_override: Path | None = None,
    manifest_srt_map: dict[str, Path] | None = None,
) -> list[AudioSrtPair]:
    manifest_srt_map = manifest_srt_map or {}
    if srt_override is not None and len(audio_files) != 1:
        raise DiscoveryError("--srt override can only be used with exactly one input audio")

    pairs: list[AudioSrtPair] = []
    for audio in audio_files:
        resolved_audio = audio.resolve()
        key = str(resolved_audio).lower()
        if key in manifest_srt_map:
            srt = manifest_srt_map[key]
            _ensure_exists(srt, "manifest srt")
            pairs.append(AudioSrtPair(audio_path=resolved_audio, srt_path=srt.resolve()))
            continue

        if srt_override is not None:
            resolved_srt = srt_override.resolve()
            _ensure_exists(resolved_srt, "srt override")
            pairs.append(AudioSrtPair(audio_path=resolved_audio, srt_path=resolved_srt))
            continue

        stem = resolved_audio.stem
        preferred_filenames = [f"{stem}.en.srt".lower(), f"{stem}.srt".lower()]
        selected: Path | None = None
        for filename in preferred_filenames:
            candidates = _find_srt_candidates_by_filename(project_root, filename)
            if len(candidates) > 1:
                lines = "\n".join([f"- {p}" for p in candidates])
                raise DiscoveryError(
                    f"Multiple SRT files match audio {resolved_audio.name} with name {filename}. "
                    f"Use --srt or --manifest to disambiguate:\n{lines}"
                )
            if len(candidates) == 1:
                selected = candidates[0]
                break

        if selected is None:
            raise DiscoveryError(f"SRT not found for audio: {resolved_audio.name}. Expected {stem}.en.srt or {stem}.srt")
        pairs.append(AudioSrtPair(audio_path=resolved_audio, srt_path=selected))
    return pairs


def discover_reference_wav(project_root: Path, ref_wav: Path | None) -> Path:
    ref_root = (project_root / "ref-audio").resolve()

    if ref_wav is not None:
        resolved = ref_wav.resolve()
        _ensure_exists(resolved, "ref_wav")
        try:
            resolved.relative_to(ref_root)
        except ValueError as exc:
            raise DiscoveryError(f"ref_wav must be under {ref_root}: {resolved}") from exc
        if resolved.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            raise DiscoveryError(
                f"ref_wav must be an audio file with one of {sorted(SUPPORTED_AUDIO_EXTS)}: {resolved}"
            )
        return resolved

    if not ref_root.exists() or not ref_root.is_dir():
        raise DiscoveryError(f"Reference audio directory not found: {ref_root}")

    hits: list[Path] = []
    for item in ref_root.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            hits.append(item.resolve())

    if not hits:
        raise DiscoveryError(
            f"No reference audio found under {ref_root}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTS))}"
        )

    ext_rank = {".wav": 0, ".flac": 1, ".mp3": 2, ".m4a": 3, ".aac": 4, ".ogg": 5}

    def _rank(path: Path) -> tuple[int, int, str]:
        tiny_first = 0 if path.stem.lower() == "tiny" else 1
        return (tiny_first, ext_rank.get(path.suffix.lower(), 99), path.name.lower())

    return sorted(hits, key=_rank)[0]


def resolve_output_flac(audio_path: Path, out_dir: Path | None, srt_path: Path | None = None) -> Path:
    if srt_path is not None:
        stem = stem_from_srt(srt_path)
        if out_dir is None:
            return srt_path.with_name(f"{stem}.flac")
        out_root = out_dir.resolve()
        srt_parent = srt_path.resolve().parent
        try:
            srt_parent.relative_to(out_root)
            return srt_parent / f"{stem}.flac"
        except ValueError:
            return out_root / f"{stem}.flac"

    if out_dir is None:
        return audio_path.with_suffix(".flac")
    return out_dir.resolve() / f"{audio_path.stem}.flac"


def collect_srt_files_for_srt_only(project_root: Path, srt_override: Path | None) -> list[Path]:
    if srt_override is not None:
        resolved = srt_override.resolve()
        _ensure_exists(resolved, "srt")
        if resolved.suffix.lower() != ".srt":
            raise DiscoveryError(f"--srt must point to a .srt file: {resolved}")
        return [resolved]

    default_dir = (project_root / "output").resolve()
    if not default_dir.exists() or not default_dir.is_dir():
        raise DiscoveryError("No output directory found under project_root. Provide --srt explicitly.")

    srt_files = [item.resolve() for item in default_dir.rglob("*.en.srt") if item.is_file()]
    srt_files = sorted(srt_files, key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
    if not srt_files:
        raise DiscoveryError("No .en.srt found in project_root/output. Provide --srt explicitly.")
    return srt_files
