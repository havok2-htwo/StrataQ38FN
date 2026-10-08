@echo off
rem ===== Gemeinsame Einstellungen fuer alle Start-Skripte =====
rem Ports (Firewall gibt 8080-8082 frei)
set "STRATA_PORT=8080"
set "FACADE_PORT=8081"
rem 0.0.0.0 = im LAN erreichbar (nur mit API-Key!), 127.0.0.1 = nur dieser PC
set "BIND_HOST=0.0.0.0"
rem Modellname, den genesis3 an vLLM schickt (leer = nur der Strata-Modellname)
set "GENESIS3_MODEL="
rem Der API-Key steht in api_key.txt
