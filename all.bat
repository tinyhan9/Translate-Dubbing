@echo off
setlocal EnableDelayedExpansion
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
    call :now_iso ALL_MD_START_TS

    call :now_iso AGENTS_MD_START_TS

    "!PYTHON_EXE!" -m app.subtitle_workflow "%%~fF" --root . --model-dir models/whisper --out-dir output --raw-dir .transcripts/raw --log-file .transcripts/error.log --open-progress-window
    if errorlevel 1 (
      echo [ERROR] AGENT1 failed: %%~fF
      exit /b 1
    )
    call :now_iso AGENTS_MD_END_TS

    call :now_iso AGENTS2_MD_START_TS

    "!PYTHON_EXE!" -m app --project_root . --srt "output/%%~nF/%%~nF.en.srt" --model_dir models/indexTTS2 --gap_fill_mode zero --sr 22050 --audio_bitrate 320k --open_progress_window
    if errorlevel 1 (
      echo [ERROR] AGENT2 failed: output/%%~nF/%%~nF.en.srt
      exit /b 1
    )
    call :now_iso AGENTS2_MD_END_TS

    call :now_iso ALL_MD_END_TS
    call :write_backend_timing "output\%%~nF\%%~nF.backend.txt" "!ALL_MD_START_TS!" "!ALL_MD_END_TS!" "!AGENTS_MD_START_TS!" "!AGENTS_MD_END_TS!" "!AGENTS2_MD_START_TS!" "!AGENTS2_MD_END_TS!"
  )
)

if "%HAS_FILES%"=="0" (
  echo [ERROR] No media files found under .\start
  exit /b 1
)

echo [OK] ALL queue completed.
exit /b 0

:now_iso
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-ddTHH:mm:ss\")"') do set "%~1=%%I"
exit /b 0

:write_backend_timing
set "BACKEND_FILE=%~1"
set "ALL_START=%~2"
set "ALL_END=%~3"
set "AGENT1_START=%~4"
set "AGENT1_END=%~5"
set "AGENT2_START=%~6"
set "AGENT2_END=%~7"

if not exist "%BACKEND_FILE%" (
  > "%BACKEND_FILE%" echo Subtitle Workflow Backend Report
)

>> "%BACKEND_FILE%" echo.
>> "%BACKEND_FILE%" echo Execution Timings
>> "%BACKEND_FILE%" echo ALL.md start: %ALL_START%
>> "%BACKEND_FILE%" echo ALL.md end: %ALL_END%
>> "%BACKEND_FILE%" echo AGENTS.md start: %AGENT1_START%
>> "%BACKEND_FILE%" echo AGENTS.md end: %AGENT1_END%
>> "%BACKEND_FILE%" echo AGENTS2.md start: %AGENT2_START%
>> "%BACKEND_FILE%" echo AGENTS2.md end: %AGENT2_END%
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



