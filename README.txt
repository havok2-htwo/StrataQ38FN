Strata - Qwen3.8-Flash-Next local (RTX 5090 + 64 GB)
====================================================

Start (double-click). Only ONE model runs at a time; a start stops the previous one.

  Start-IQ4_XS.bat    Unsloth UD-IQ4_XS (~4 bit)  - all-round/German, 130-165 tok/s, 128K context   (recommended)
  Start-Q4_K_XL.bat   Unsloth UD-Q4_K_XL (~5 bit) - slightly better quality, 85-95 tok/s, 64K context
                      (needs ~45 GB RAM for experts: close the browser/other large programs first)
  Start-Coder.bat     ISTA Coder (half the experts) - code/English only, 165-230 tok/s, 128K context

  Stop-All.bat        stops the model + vLLM facade

  Start-Alles.bat     Strata + SearXNG (web search) + Open WebUI (https://192.168.0.200:3000/).
                      Starts only what is not running yet - a loaded model stays.
                      Without an argument UD-IQ4_XS; otherwise e.g. "Start-Alles.bat Coder".
  Stop-Alles.bat      stops Strata, facade, Open WebUI and SearXNG (dashboard stays)

Each start opens two windows: "Strata ..." (the model, loading ~1-2 min) and "vLLM-Fassade ...".
Closing a window = that part is stopped.

Addresses (IP of this PC: 192.168.0.200), API key is in api_key.txt (header: Authorization: Bearer <key>):
  Web chat                    http://192.168.0.200:8080/
  OpenAI-compatible           http://192.168.0.200:8080/v1          (model name does not matter)
  Anthropic-compatible        http://192.168.0.200:8080/v1/messages
  vLLM facade (genesis3)      http://192.168.0.200:8081/v1          (model name = that of the started model,
                                                                      see /v1/models, or GENESIS3_MODEL)

Settings (ports, LAN/local, genesis3 model name): settings.cmd
Model configs (flags): Strata\strata-*.json  (originals from setup: *.json.orig)
Logs: Strata\strata-<model>.log
