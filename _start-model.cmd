@echo off
rem intern: %1 = Config-Datei (in Strata\), %2 = Strata-Modellname, %3 = Anzeigename
call "%~dp0settings.cmd"
call "%~dp0Stop-All.bat" quiet
echo Starte %~3 ...
start "Strata %~3 (Port %STRATA_PORT%)" cmd /c ""%~dp0_run-strata.cmd" "%~dp0Strata\%~1""
start "vLLM-Fassade %~3 (Port %FACADE_PORT%)" cmd /c ""%~dp0_run-facade.cmd" "%~dp0Strata\%~1" "%~2""
rem Dashboard (Port 8082) nur starten, wenn es nicht schon laeuft - es ueberlebt Modellwechsel
powershell -NoProfile -Command "if (-not (Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction SilentlyContinue)) { exit 1 }"
if errorlevel 1 start "Strata Dashboard (Port 8082)" /min cmd /c ""%~dp0Start-Dashboard.bat""
echo.
echo   Das Laden dauert ca. 1-2 Minuten (Fortschritt im Strata-Fenster).
echo   Strata (OpenAI/Anthropic):  http://IP:%STRATA_PORT%/v1
echo   vLLM-Fassade (genesis3):    http://IP:%FACADE_PORT%/v1   Modellname: %~2
echo   Web-Chat:                   http://IP:%STRATA_PORT%/
echo   Dashboard:                  http://IP:8082/
echo   API-Key:                    siehe api_key.txt
echo   Beenden: Stop-All.bat (oder beide Fenster schliessen)
timeout /t 8 >nul
