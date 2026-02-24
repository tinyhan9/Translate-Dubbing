from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AppError
from .ffmpeg_utils import normalize_to_wav


class TTSError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=4)


def candidate_worker_configs(prefer_fast: bool = True) -> list[dict[str, bool]]:
    if prefer_fast:
        return [
            {"use_fp16": True, "use_cuda_kernel": True, "use_deepspeed": False},
            {"use_fp16": False, "use_cuda_kernel": False, "use_deepspeed": False},
        ]
    return [{"use_fp16": False, "use_cuda_kernel": False, "use_deepspeed": False}]


class _PersistentV2Worker:
    def __init__(
        self,
        runtime_dir: Path,
        cfg_path: Path,
        model_dir: Path,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        shared_model_root: Path | None = None,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.cfg_path = cfg_path
        self.model_dir = model_dir
        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed
        self.shared_model_root = shared_model_root
        self._proc: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._start()

    def _worker_script(self) -> str:
        return (
            "import json,sys,traceback\n"
            "from indextts.infer_v2 import IndexTTS2\n"
            f"tts=IndexTTS2(cfg_path=r'{self.cfg_path}', model_dir=r'{self.model_dir}', "
            f"use_fp16={str(self.use_fp16)}, use_cuda_kernel={str(self.use_cuda_kernel)}, "
            f"use_deepspeed={str(self.use_deepspeed)})\n"
            "print(json.dumps({'event':'ready'}), flush=True)\n"
            "for raw in sys.stdin:\n"
            "    line=raw.strip()\n"
            "    if not line:\n"
            "        continue\n"
            "    try:\n"
            "        req=json.loads(line)\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'ok':False,'id':None,'error':f'invalid_json:{e}'}), flush=True)\n"
            "        continue\n"
            "    if req.get('cmd') == 'quit':\n"
            "        print(json.dumps({'event':'bye'}), flush=True)\n"
            "        break\n"
            "    rid=req.get('id')\n"
            "    try:\n"
            "        tts.infer(spk_audio_prompt=req['ref_wav'], text=req['text'], output_path=req['out_wav'], verbose=False)\n"
            "        print(json.dumps({'ok':True,'id':rid}), flush=True)\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'ok':False,'id':rid,'error':str(e)}), flush=True)\n"
        )

    def _resolve_python_cmd(self) -> list[str]:
        venv_candidates = [
            self.runtime_dir / ".venv" / "Scripts" / "python.exe",
            self.runtime_dir / ".venv" / "bin" / "python",
        ]
        for cand in venv_candidates:
            if cand.exists():
                return [str(cand.resolve())]
        if sys.executable:
            return [sys.executable]
        if os.name == "nt":
            return ["py", "-3"]
        return ["python3"]

    def _start(self) -> None:
        env = os.environ.copy()
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.runtime_dir}{os.pathsep}{old_pp}" if old_pp else str(self.runtime_dir)
        if self.shared_model_root is not None:
            env["INDEXTTS_SHARED_MODEL_ROOT"] = str(self.shared_model_root)
        env.setdefault("HF_HUB_CACHE", str((self.runtime_dir.parent / "hf_cache").resolve()))
        env.setdefault("PYTHONUTF8", "1")
        cmd = [*self._resolve_python_cmd(), "-u", "-c", self._worker_script()]
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.runtime_dir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            raise TTSError(f"Failed to start IndexTTS2 worker process: {exc}") from exc

        ready = self._read_until_json(expect_ready=True)
        if ready.get("event") != "ready":
            raise TTSError(f"IndexTTS2 worker did not become ready: {ready}")

    def _read_until_json(self, expect_id: int | None = None, expect_ready: bool = False) -> dict[str, Any]:
        if not self._proc or not self._proc.stdout:
            raise TTSError("IndexTTS2 worker is not running")
        last_line = ""
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                code = self._proc.poll()
                raise TTSError(f"IndexTTS2 worker exited unexpectedly (code={code}). Last output: {last_line}")
            stripped = line.strip()
            if not stripped:
                continue
            last_line = stripped
            try:
                payload = json.loads(stripped)
            except Exception:
                # Ignore non-JSON logs produced by runtime libs.
                continue

            if expect_ready:
                return payload
            if expect_id is None:
                return payload
            if payload.get("id") == expect_id:
                return payload

    def synthesize(self, text: str, ref_wav: Path, out_wav: Path) -> None:
        if not self._proc or not self._proc.stdin:
            raise TTSError("IndexTTS2 worker is not running")
        self._request_id += 1
        rid = self._request_id
        req = {
            "id": rid,
            "text": text,
            "ref_wav": str(ref_wav.resolve()),
            "out_wav": str(out_wav.resolve()),
        }
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise TTSError(f"Failed to send request to IndexTTS2 worker: {exc}") from exc

        resp = self._read_until_json(expect_id=rid)
        if not resp.get("ok"):
            raise TTSError(f"IndexTTS2 worker inference failed: {resp.get('error', 'unknown error')}")

    def close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None


class IndexTTS2Adapter:
    def __init__(
        self,
        model_dir: Path,
        sr: int = 22050,
        mono: bool = True,
        audio_bitrate: str = "256k",
        cmd_template: str | None = None,
        mock_tts: bool = False,
        prefer_fast: bool = True,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.sr = sr
        self.mono = mono
        self.audio_bitrate = audio_bitrate
        self.cmd_template = cmd_template or os.getenv("INDEXTTS2_CMD_TEMPLATE")
        self.mock_tts = mock_tts
        self.prefer_fast = prefer_fast
        self._worker: _PersistentV2Worker | None = None
        self._worker_config: dict[str, bool] | None = None
        if not self.mock_tts and (not self.model_dir.exists() or not self.model_dir.is_dir()):
            raise TTSError(f"indexTTS2 model_dir not found: {self.model_dir}")

    def _resolve_cfg_and_model_dir(self) -> tuple[Path, Path]:
        # Layout A: model_dir is project root with checkpoints/ subfolder.
        a_cfg = self.model_dir / "checkpoints" / "config.yaml"
        if a_cfg.exists():
            return a_cfg, (self.model_dir / "checkpoints")

        # Layout B: model_dir itself is the checkpoints folder.
        b_cfg = self.model_dir / "config.yaml"
        if b_cfg.exists():
            return b_cfg, self.model_dir

        # Layout C: indexTTS runtime dir with shared ComfyUI model assets.
        c_dir = (self.model_dir.parent / "IndexTTS" / "IndexTTS-2").resolve()
        c_cfg = c_dir / "config.yaml"
        if c_cfg.exists():
            return c_cfg, c_dir

        raise TTSError(
            f"Cannot locate config.yaml under model_dir={self.model_dir}. "
            "Expected either <model_dir>/checkpoints/config.yaml, <model_dir>/config.yaml, "
            "or <project_root>/IndexTTS/IndexTTS-2/config.yaml."
        )

    def _find_runtime_dir(self) -> Path | None:
        candidates: list[Path] = []
        env_runtime = os.getenv("INDEXTTS_RUNTIME_DIR")
        if env_runtime:
            candidates.append(Path(env_runtime).resolve())
        cwd = Path.cwd().resolve()
        parent = self.model_dir.parent.resolve()
        grandparent = parent.parent.resolve()
        candidates.extend(
            [
                (self.model_dir / "indexTTS_runtime").resolve(),
                (parent / "indexTTS_runtime").resolve(),
                (parent / "indexTTS2_runtime").resolve(),
                (parent / "indexTTS2").resolve(),
                (grandparent / "indexTTS2").resolve(),
                (cwd / "indexTTS2").resolve(),
            ]
        )

        for cand in candidates:
            if (cand / "indextts" / "infer_v2.py").exists():
                return cand
        return None

    def _run(self, cmd: list[str], fail_msg: str) -> None:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:
            raise TTSError(f"{fail_msg}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise TTSError(f"{fail_msg}: {detail}")

    def _generate_mock_tts(self, text: str, out_wav: Path) -> None:
        duration_ms = max(250, min(5000, len(text.strip()) * 70))
        seconds = duration_ms / 1000.0
        samples = int(seconds * self.sr)
        t = np.linspace(0, seconds, samples, endpoint=False)
        freq = 220.0 + (len(text) % 7) * 30.0
        signal = 0.18 * np.sin(2 * math.pi * freq * t)
        pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(pcm.tobytes())

    def _candidate_scripts(self) -> list[Path]:
        candidates: list[Path] = []
        names = [
            "inference.py",
            "infer.py",
            "tts.py",
            "app.py",
            "run.py",
        ]
        for name in names:
            p = self.model_dir / name
            if p.exists():
                candidates.append(p)
        return candidates

    def _has_official_v2_layout(self) -> bool:
        runtime = self._find_runtime_dir()
        if runtime is None:
            return False
        try:
            self._resolve_cfg_and_model_dir()
        except TTSError:
            return False
        return True

    def _resolve_shared_model_root(self, ckpt: Path) -> Path | None:
        candidates = [
            ckpt.parent,
            ckpt.parent.parent,
            (Path.cwd() / "IndexTTS").resolve(),
        ]
        required = ["w2v-bert-2.0", "MaskGCT", "campplus", "bigvgan_v2_22khz_80band_256x"]
        for cand in candidates:
            if cand.exists() and all((cand / name).exists() for name in required):
                return cand.resolve()
        return None

    def _start_worker_with_configs(self, runtime: Path, cfg: Path, ckpt: Path, configs: list[dict[str, bool]]) -> None:
        shared_model_root = self._resolve_shared_model_root(ckpt)
        last_error: Exception | None = None
        for c in configs:
            try:
                self._worker = _PersistentV2Worker(
                    runtime_dir=runtime,
                    cfg_path=cfg,
                    model_dir=ckpt,
                    use_fp16=c["use_fp16"],
                    use_cuda_kernel=c["use_cuda_kernel"],
                    use_deepspeed=c["use_deepspeed"],
                    shared_model_root=shared_model_root,
                )
                self._worker_config = c
                return
            except Exception as exc:
                last_error = exc
                self._worker = None
                continue
        raise TTSError(f"Failed to start IndexTTS2 worker in all modes: {last_error}")

    def _run_official_v2(self, text: str, ref_wav: Path, out_wav: Path) -> None:
        runtime = self._find_runtime_dir()
        if runtime is None:
            raise TTSError("Cannot find IndexTTS runtime code (missing indextts/infer_v2.py).")
        cfg, ckpt = self._resolve_cfg_and_model_dir()
        if self._worker is None:
            self._start_worker_with_configs(runtime, cfg, ckpt, candidate_worker_configs(prefer_fast=self.prefer_fast))
        try:
            assert self._worker is not None
            self._worker.synthesize(text, ref_wav, out_wav)
        except Exception as exc:
            fast_mode = bool(self._worker_config and (self._worker_config.get("use_fp16") or self._worker_config.get("use_cuda_kernel")))
            if self.prefer_fast and fast_mode:
                # Auto fallback: retry once using safe config.
                self.close()
                self._start_worker_with_configs(runtime, cfg, ckpt, candidate_worker_configs(prefer_fast=False))
                assert self._worker is not None
                self._worker.synthesize(text, ref_wav, out_wav)
            else:
                raise TTSError(f"IndexTTS2 worker inference failed: {exc}") from exc
        if not out_wav.exists() or out_wav.stat().st_size == 0:
            raise TTSError(f"IndexTTS2 worker did not produce output wav: {out_wav}")

    def _try_commands(self, script: Path, text: str, ref_wav: Path, out_wav: Path) -> None:
        commands = [
            [
                "py",
                "-3",
                str(script),
                "--text",
                text,
                "--ref_wav",
                str(ref_wav),
                "--out_wav",
                str(out_wav),
                "--model_dir",
                str(self.model_dir),
            ],
            [
                "py",
                "-3",
                str(script),
                "--text",
                text,
                "--speaker_wav",
                str(ref_wav),
                "--output",
                str(out_wav),
            ],
            [
                "py",
                "-3",
                str(script),
                "--prompt",
                text,
                "--reference_audio",
                str(ref_wav),
                "--save_path",
                str(out_wav),
            ],
        ]

        errors: list[str] = []
        for cmd in commands:
            try:
                self._run(cmd, f"indexTTS2 inference failed with script {script.name}")
            except TTSError as exc:
                errors.append(str(exc))
                continue
            if out_wav.exists() and out_wav.stat().st_size > 0:
                return
        joined = "\n".join([f"- {e}" for e in errors]) if errors else "- no command succeeded"
        raise TTSError(f"No known indexTTS2 CLI args matched for {script}:\n{joined}")

    def synthesize(self, text: str, ref_wav: Path, out_wav: Path) -> Path:
        out_wav = out_wav.resolve()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if not text.strip():
            text = " "

        if self.mock_tts:
            self._generate_mock_tts(text, out_wav)
        elif self.cmd_template:
            cmd_text = self.cmd_template.format(
                model_dir=str(self.model_dir),
                text=text,
                ref_wav=str(ref_wav),
                out_wav=str(out_wav),
            )
            proc = subprocess.run(
                cmd_text,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise TTSError(f"indexTTS2 command template failed: {detail}")
            if not out_wav.exists():
                raise TTSError(f"indexTTS2 command template did not produce output wav: {out_wav}")
        else:
            if self._has_official_v2_layout():
                self._run_official_v2(text, ref_wav, out_wav)
            else:
                scripts = self._candidate_scripts()
                if not scripts:
                    raise TTSError(
                        f"Cannot find indexTTS2 entry script in {self.model_dir}. "
                        "Provide --tts_cmd_template or set INDEXTTS2_CMD_TEMPLATE."
                    )
                last_error: TTSError | None = None
                for script in scripts:
                    try:
                        self._try_commands(script, text, ref_wav, out_wav)
                        break
                    except TTSError as exc:
                        last_error = exc
                if not out_wav.exists():
                    raise last_error or TTSError("indexTTS2 inference failed")

        channels = 1 if self.mono else 2
        compressed = out_wav.with_suffix(".internal.flac")
        self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(out_wav),
                "-ar",
                str(self.sr),
                "-ac",
                str(channels),
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-b:a",
                self.audio_bitrate,
                str(compressed),
            ],
            "indexTTS2 internal transcode failed",
        )
        normalized = out_wav.with_suffix(".16k.wav")
        normalize_to_wav(compressed, normalized, sr=self.sr, channels=channels)
        normalized.replace(out_wav)
        compressed.unlink(missing_ok=True)
        return out_wav

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
