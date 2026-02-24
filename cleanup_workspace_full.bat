@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [CLEAN] 1/4 Remove root-level MP3 files...
for %%F in (*.mp3) do (
  if exist "%%~fF" (
    del /f /q "%%~fF"
    echo   deleted: %%~nxF
  )
)

echo [CLEAN] 2/4 Clean output directory (keep output\reports)...
if exist "output\" (
  for /f "delims=" %%I in ('dir /b /a "output"') do (
    if /I not "%%I"=="reports" (
      if exist "output\%%I\" (
        rmdir /s /q "output\%%I"
        echo   removed dir: output\%%I
      ) else (
        del /f /q "output\%%I"
        echo   removed file: output\%%I
      )
    )
  )
) else (
  echo   output folder not found, skipped.
)

echo [CLEAN] 3/4 Clean .transcripts cache files...
if exist ".transcripts\" (
  for /r ".transcripts" %%F in (*) do (
    if exist "%%~fF" (
      del /f /q "%%~fF"
      echo   removed cache: %%~fF
    )
  )
  for /f "delims=" %%D in ('dir /s /b /ad ".transcripts" ^| sort /R') do (
    if /I not "%%~fD"=="%CD%\.transcripts" rmdir "%%~fD" 2>nul
  )
) else (
  echo   .transcripts folder not found, skipped.
)

echo [CLEAN] 4/4 Clean output\reports runtime cache files (keep *.report.json)...
if exist "output\reports\" (
  for /r "output\reports" %%F in (*) do (
    if /I not "%%~xF"==".json" (
      del /f /q "%%~fF"
      echo   removed report-cache: %%~fF
    ) else (
      echo %%~nxF | findstr /I /R "\.report\.json$" >nul
      if errorlevel 1 (
        del /f /q "%%~fF"
        echo   removed report-cache: %%~fF
      )
    )
  )
  for /f "delims=" %%D in ('dir /s /b /ad "output\reports" ^| sort /R') do (
    if /I not "%%~fD"=="%CD%\output\reports" rmdir "%%~fD" 2>nul
  )
) else (
  echo   output\reports folder not found, skipped.
)

echo [DONE] Workspace cleanup complete.
exit /b 0
