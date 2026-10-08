@echo off
rem Strata Monitor (Dashboard) im LAN: http://<IP dieses PCs>:8082/  - zeigt nur Zahlen, keine Prompts/Antworten
cd /d "%~dp0dashboard"
"%~dp0Strata\.venv\Scripts\python.exe" dashboard.py --port 8082 --host 0.0.0.0
if errorlevel 1 pause
