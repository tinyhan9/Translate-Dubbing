@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "HAS_FILES=0"

for /r ".\start" %%F in (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mkv *.ts *.mov *.avi *.webm *.m4v) do (
  set "HAS_FILES=1"
  set "EXT=%%~xF"
  set "STEM=%%~nF"
  set "DIR=%%~dpF"
  set "SKIP=0"

  if /I "!EXT!"==".mp3"  call :has_same_stem_video "!DIR!" "!STEM!" SKIP
  if /I "!EXT!"==".wav"  call :has_same_stem_video "!DIR!" "!STEM!" SKIP
  if /I "!EXT!"==".m4a"  call :has_same_stem_video "!DIR!" "!STEM!" SKIP
  if /I "!EXT!"==".aac"  call :has_same_stem_video "!DIR!" "!STEM!" SKIP
  if /I "!EXT!"==".flac" call :has_same_stem_video "!DIR!" "!STEM!" SKIP
  if /I "!EXT!"==".ogg"  call :has_same_stem_video "!DIR!" "!STEM!" SKIP

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

    py -3 -m app --project_root . --srt "output/%%~nF/%%~nF.en.srt" --model_dir models/indexTTS2 --gap_fill_mode zero --sr 22050 --audio_bitrate 256k --open_progress_window
    if errorlevel 1 (
      echo [ERROR] AGENT2 failed: output/%%~nF/%%~nF.en.srt
      exit /b 1
    )
  )
)

if "%HAS_FILES%"=="0" (
  echo [ERROR] No media files found under .\start
  exit /b 1
)

echo [OK] ALL queue completed.
exit /b 0

:has_same_stem_video
set "P_DIR=%~1"
set "P_STEM=%~2"
set "P_OUT_VAR=%~3"

if exist "%P_DIR%%P_STEM%.mp4"  set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.mkv"  set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.ts"   set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.mov"  set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.avi"  set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.webm" set "%P_OUT_VAR%=1"
if exist "%P_DIR%%P_STEM%.m4v"  set "%P_OUT_VAR%=1"

exit /b 0


