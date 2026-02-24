@echo off
setlocal
cd /d "%~dp0"

if not exist "start\" (
  echo [SKIP] start folder not found.
  exit /b 0
)

echo [CLEAN] Deleting all files under start\ ...
for /r "start" %%F in (*) do (
  if exist "%%~fF" (
    del /f /q "%%~fF"
    echo   deleted: %%~fF
  )
)

echo [DONE] All files under start\ have been deleted.
exit /b 0
