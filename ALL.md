# ALL Workflow (AGENT1 + AGENT2 Serial)

## 1) Goal
For each media file in `start/`, run in strict order:
1. `AGENTS.md` (generate `output/<name>/<name>.zh.srt/.en.srt/.bi.srt`)
2. `AGENTS2.md` (read `output/<name>/<name>.en.srt`, generate `output/<name>/<name>.flac`)

## 2) Input Scope
- Only scan `project_root/start/`.
- Supported media:
  - Audio: `mp3 wav m4a aac flac ogg`
  - Video: `mp4 mkv ts mov avi webm m4v`

## 3) Queue Rules
- Multiple files must run serially, never in parallel.
- Per file order is fixed: `AGENT1 -> AGENT2`.
- If one file fails, stop the whole queue.
- If a video is used, extracted MP3 must be written to project root: `./<video_stem>.mp3`.
- De-dup rule: if same-folder same-stem video and audio both exist, process video only and skip that audio entry.

## 4) CMD Template
```bat
@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

for /r ".\start" %%F in (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mkv *.ts *.mov *.avi *.webm *.m4v) do (
  set "EXT=%%~xF"
  set "STEM=%%~nF"
  set "DIR=%%~dpF"
  set "SKIP=0"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.mp4"  set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.mp4"  set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.mp4"  set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.mp4"  set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.mp4"  set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.mp4"  set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.mkv"  set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.mkv"  set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.mkv"  set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.mkv"  set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.mkv"  set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.mkv"  set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.ts"   set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.ts"   set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.ts"   set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.ts"   set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.ts"   set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.ts"   set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.mov"  set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.mov"  set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.mov"  set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.mov"  set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.mov"  set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.mov"  set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.avi"  set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.avi"  set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.avi"  set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.avi"  set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.avi"  set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.avi"  set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.webm" set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.webm" set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.webm" set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.webm" set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.webm" set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.webm" set "SKIP=1"

  if /I "!EXT!"==".mp3"  if exist "!DIR!!STEM!.m4v"  set "SKIP=1"
  if /I "!EXT!"==".wav"  if exist "!DIR!!STEM!.m4v"  set "SKIP=1"
  if /I "!EXT!"==".m4a"  if exist "!DIR!!STEM!.m4v"  set "SKIP=1"
  if /I "!EXT!"==".aac"  if exist "!DIR!!STEM!.m4v"  set "SKIP=1"
  if /I "!EXT!"==".flac" if exist "!DIR!!STEM!.m4v"  set "SKIP=1"
  if /I "!EXT!"==".ogg"  if exist "!DIR!!STEM!.m4v"  set "SKIP=1"

  if "!SKIP!"=="1" (
    echo [SKIP] Duplicate audio with same-stem video: %%~fF
  ) else (
    echo ==================================================
    echo [QUEUE] Processing: %%~fF

    py -3 -m app.subtitle_workflow "%%~fF" --root . --model-dir models/whisper --out-dir output --raw-dir .transcripts/raw --log-file .transcripts/error.log --open-progress-window
    if errorlevel 1 (
      echo [ERROR] AGENT1 failed: %%~fF
      exit /b 1
    )

    py -3 -m app --project_root . --srt "output/%%~nF/%%~nF.en.srt" --model_dir models/indexTTS2 --open_progress_window
    if errorlevel 1 (
      echo [ERROR] AGENT2 failed: output/%%~nF/%%~nF.en.srt
      exit /b 1
    )
  )
)

echo [OK] ALL queue completed.
exit /b 0
```

## 5) Outputs
- Subtitles: `output/<name>/<name>.zh.srt`, `output/<name>/<name>.en.srt`, `output/<name>/<name>.bi.srt`
- Audio: `output/<name>/<name>.flac`
