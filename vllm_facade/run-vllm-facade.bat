@echo off
rem vLLM-Fassade vor Strata (IQ4_XS). Strata selbst muss schon laufen (run-unsloth-ud-iq4_xs.bat, Port 8080).
rem Weitere Flags werden angehaengt und gewinnen, z.B.:  run-vllm-facade.bat --api-key geheim --reasoning-output reasoning
setlocal

rem === genesis3: hier den Modellnamen eintragen, den genesis3 bisher an vLLM schickt (model=...). ==========
rem     Leer lassen = nur Strata's Modell-ID. Mit Namen: Antworten tragen diesen Namen, die Strata-ID bleibt Alias.
set "GENESIS3_MODEL="

set "PORT=8000"
set "UPSTREAM=http://127.0.0.1:8080"
rem Client-Schluessel (vLLM-Semantik, nur /v1/*):   set "VLLM_API_KEY=..."
rem Schluessel fuer Strata, falls dort api_key gesetzt ist:   set "STRATA_API_KEY=..."

set "PY=X:\dev\StrataQ38FN\Strata\.venv\Scripts\python.exe"
set "STRATA_ROOT=X:\dev\StrataQ38FN\Strata"
set "CONFIG=%STRATA_ROOT%\strata-unsloth-ud-iq4_xs.json"
set "STRATA_MODEL=qwen3.8-flash-next-unsloth-ud-iq4_xs"

set "NAMES=%STRATA_MODEL%"
if not "%GENESIS3_MODEL%"=="" set "NAMES=%GENESIS3_MODEL% %STRATA_MODEL%"

"%PY%" "%~dp0vllm_facade.py" --upstream %UPSTREAM% --host 0.0.0.0 --port %PORT% ^
  --strata-root "%STRATA_ROOT%" --strata-config "%CONFIG%" --served-model-name %NAMES% %*
