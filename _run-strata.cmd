@echo off
rem intern: startet den Strata-Server mit der Config %1
call "%~dp0settings.cmd"
set /p KEY=<"%~dp0api_key.txt"
cd /d "%~dp0Strata"
".venv\Scripts\python.exe" serve\server.py --engine strata --config "%~1" --port %STRATA_PORT% --host %BIND_HOST% --api-key %KEY%
if errorlevel 1 pause
