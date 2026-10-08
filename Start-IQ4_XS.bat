@echo off
rem Unsloth UD-IQ4_XS (~4 bit): Allround, Deutsch, 130-165 tok/s, 128K Kontext
call "%~dp0_start-model.cmd" strata-unsloth-ud-iq4_xs.json qwen3.8-flash-next-unsloth-ud-iq4_xs "UD-IQ4_XS"
