@echo off
rem Beendet Strata, vLLM-Fassade, Open WebUI und SearXNG (das Dashboard laeuft weiter).
call "%~dp0Stop-All.bat" quiet
powershell -NoProfile -Command "foreach($p in 3000,8888){ Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"
echo Strata, vLLM-Fassade, Open WebUI und SearXNG beendet.
timeout /t 3 >nul
