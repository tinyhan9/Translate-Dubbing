@echo off
setlocal
cd /d "%~dp0"

py -3 -m app --project_root . --model_dir models/indexTTS2 --gap_fill_mode zero --sr 22050 --audio_bitrate 320k --open_progress_window
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo [ERROR] AGENT2 failed with exit code %CODE%.
) else (
  echo [OK] AGENT2 completed.
)

exit /b %CODE%



