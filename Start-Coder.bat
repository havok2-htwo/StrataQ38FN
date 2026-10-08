@echo off
rem ISTA Coder (256/512 Experten, fast alles im VRAM): nur Code/Englisch, 165-230 tok/s, 128K Kontext
call "%~dp0_start-model.cmd" strata-coder-iq1_m.json qwen3.8-flash-next-coder-iq1_m "Coder"
