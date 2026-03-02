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

"%PYTHON_EXE%" -m app --project_root . --model_dir models/indexTTS2 --gap_fill_mode zero --sr 22050 --audio_bitrate 320k --open_progress_window
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo [ERROR] AGENT2 failed with exit code %CODE%.
) else (
  echo [OK] AGENT2 completed.
)

exit /b %CODE%



