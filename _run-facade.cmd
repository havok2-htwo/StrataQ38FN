@echo off
rem intern: startet die vLLM-Fassade vor Strata; %1 = Config, %2 = Strata-Modellname
call "%~dp0settings.cmd"
set /p KEY=<"%~dp0api_key.txt"
set "NAMES=%~2"
if not "%GENESIS3_MODEL%"=="" set "NAMES=%GENESIS3_MODEL% %~2"
cd /d "%~dp0vllm_facade"
"%~dp0Strata\.venv\Scripts\python.exe" vllm_facade.py --upstream http://127.0.0.1:%STRATA_PORT% --host %BIND_HOST% --port %FACADE_PORT% --strata-root "%~dp0Strata" --strata-config "%~1" --served-model-name %NAMES% --api-key %KEY% --upstream-api-key %KEY%
if errorlevel 1 pause
