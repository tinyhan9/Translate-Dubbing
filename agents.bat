@echo off
setlocal
cd /d "%~dp0"

py -3 -m app.subtitle_workflow --root . --model-dir models/whisper --out-dir output --raw-dir .transcripts/raw --log-file .transcripts/error.log --open-progress-window
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo [ERROR] AGENT failed with exit code %CODE%.
) else (
  echo [OK] AGENT completed.
)

exit /b %CODE%
