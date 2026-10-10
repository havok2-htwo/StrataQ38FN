@echo off
rem Startet Strata + SearXNG + Open WebUI. Was schon laeuft, bleibt unangetastet
rem (ein laufendes Modell wird also NICHT neu geladen).
rem Modell (nur wenn Strata noch nicht laeuft): Start-Alles.bat [IQ4_XS|Q4_K_XL|Coder]
rem Modell wechseln: direkt Start-<Modell>.bat
call "%~dp0settings.cmd"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=IQ4_XS"
if not exist "%~dp0Start-%MODEL%.bat" (
  echo Unbekanntes Modell "%MODEL%". Moeglich: IQ4_XS, Q4_K_XL, Coder
  pause
  exit /b 1
)

call :start_once 8888 "SearXNG (Port 8888)" "X:\dev\SearXNG\Start-SearXNG.bat"
call :start_once 3000 "Open WebUI (Port 3000)" "X:\dev\OpenWebUI\Start-OpenWebUI.bat"

call :listening %STRATA_PORT%
if errorlevel 1 (
  call "%~dp0Start-%MODEL%.bat"
) else (
  echo Strata laeuft bereits - bleibt so.
)

echo.
echo   Open WebUI:  https://192.168.0.200:3000/   (am PC: https://localhost:3000/)
echo   Beenden:     Stop-Alles.bat
timeout /t 8 >nul
exit /b

:listening
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %1 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
exit /b %errorlevel%

:start_once
rem ohne ( )-Block, weil die Namen selbst Klammern enthalten
call :listening %1
if not errorlevel 1 goto :already_running
echo Starte %~2 ...
start "%~2" /min cmd /c ""%~3""
exit /b
:already_running
echo %~2 laeuft bereits.
exit /b
