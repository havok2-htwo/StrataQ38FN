@echo off
rem Unsloth UD-Q4_K_XL (~5 bit): etwas bessere Qualitaet, 85-95 tok/s, 64K Kontext. Braucht viel RAM: andere Programme schliessen.
call "%~dp0_start-model.cmd" strata-unsloth-ud-q4_k_xl.json qwen3.8-flash-next-unsloth-ud-q4_k_xl "UD-Q4_K_XL"
