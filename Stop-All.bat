@echo off
rem Beendet Strata (Engine + Server) und die vLLM-Fassade.
call "%~dp0settings.cmd"
powershell -NoProfile -Command "foreach($p in %STRATA_PORT%,%FACADE_PORT%){ Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }; Get-Process strata,strata-vision -ErrorAction SilentlyContinue | Stop-Process -Force"
if /i not "%~1"=="quiet" (
  echo Strata und vLLM-Fassade beendet.
  timeout /t 3 >nul
) else (
  timeout /t 3 >nul
)
