@echo off
setlocal
cd /d "%~dp0"

set "FFMPEG_BIN=%~dp0tools\ffmpeg\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" (
  set "PATH=%FFMPEG_BIN%;%PATH%"
)

set "PYTHON_EXE=%~dp0runtime\python\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

"%PYTHON_EXE%" -m app.subtitle_workflow --root . --model-dir models/whisper --online-asr-config config/online_asr.json --online-translate-config config/online_translate.json --out-dir output --raw-dir .transcripts/raw --log-file .transcripts/error.log --open-progress-window
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo [ERROR] AGENT failed with exit code %CODE%.
) else (
  echo [OK] AGENT completed.
)

exit /b %CODE%
