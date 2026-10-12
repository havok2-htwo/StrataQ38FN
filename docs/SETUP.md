# Setup

This guide rebuilds the environment on a fresh Windows PC. Depending on the connection it takes several hours,
because the models are about 260 GB in total.

## 0. Requirements

| | Minimum | tested |
|---|---|---|
| Operating system | Windows 10/11, 64 bit | Windows 11 Pro |
| GPU | NVIDIA with 24 GB+ VRAM (the configs are tuned for 32 GB) | RTX 5090, driver 610.88 |
| RAM | 64 GB | 64 GB |
| SSD | fast NVMe, ~300 GB free for models + ~60 GB for packs | WD SN850X |
| Software | Git, current NVIDIA driver; Strata installs Python 3.12 itself if needed | Python 3.12.4 |

> **Paths:** The configs in `strata-configs/` contain absolute paths
> (`X:\dev\StrataQ38FN\...` and `C:\StrataQ38FN-models\...`). If you use other paths, adjust them in step 5.
> The start scripts themselves work relative to their folder.

## 1. Clone the repo

```powershell
git clone https://github.com/havok2-htwo/StrataQ38FN.git X:\dev\StrataQ38FN
cd X:\dev\StrataQ38FN
```

## 2. Clone the Strata engine

The engine is a separate clone in the subfolder `Strata\` (listed in `.gitignore`):

```powershell
git clone https://github.com/Niko1221/Strata.git Strata
git -C Strata checkout 82f46a8    # tested revision; newer revisions usually work too
```

## 3. Strata setup for each model

Run `Strata\START-HERE.bat`. On the first run it creates `Strata\.venv`, installs the Python packages, builds
or downloads the engine, downloads the model, builds the pack under `Strata-data\packs\` and the MTP draft under
`Strata-data\mtp\`.

For the other models run `Strata\SETUP.bat` (= `START-HERE.bat --setup`). Choices in the setup:

| Run | Model | Size | Context in setup | Download |
|---|---|---|---|---|
| 1 | Qwen3.8-Flash-Next (Unsloth) | UD-IQ4_XS | any (overwritten in step 5) | 94 GB |
| 2 | Qwen3.8-Flash-Next (Unsloth) | UD-Q4_K_XL | any | 111 GB |
| 3 | Qwen3.8-Flash-Next Coder | IQ1_M | any | 58 GB |

Images: off. Parallelism: one request after the other (batching is broken or slower on this configuration).

If you only need the recommended model, run 1 is enough; the other two start scripts will then not work.

Alternatively, `download_q4kxl.py` downloads the UD-Q4_K_XL shards resumably and with SHA-256 verification (pinned
revision); `download_hf.py REPO REVISION TARGETDIR` downloads any Hugging Face repo.

## 4. Put the GGUF files on the fastest NVMe

During prompt processing the engine reads parts of the GGUF directly from the SSD. On a slower SSD (here: Crucial P3 on
`X:`) with `--ple-io mmap` there were hangs ("stalls" at "layer 1 of the prompt chunk"). Since the models are on the
fastest NVMe and `--ple-io direct` is set, there have been none.

Move the shards of each model into these folders:

```
C:\StrataQ38FN-models\unsloth-UD-IQ4_XS\    Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1..3}-of-00003.gguf
C:\StrataQ38FN-models\unsloth-UD-Q4_K_XL\   Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf
C:\StrataQ38FN-models\coder-IQ1_M\          Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1..2}-of-00002.gguf
```

## 5. Install the tuned configs

The setup has written `Strata\strata-<model>.json`. Back these up and replace them with the templates:

```powershell
foreach ($f in Get-ChildItem strata-configs\strata-*.json -Exclude *setup-original*) {
  $dst = "Strata\$($f.Name)"
  if (Test-Path $dst) { Copy-Item $dst "$dst.orig" }
  Copy-Item $f.FullName $dst
}
```

Then check and, if needed, adjust in each `Strata\strata-*.json`:

- `exe`, `cwd`, `tokenizer`, `log`, `lib_dirs`, `--pack`, `--mtp`, `--expert-profile`: paths under `X:\dev\StrataQ38FN`
- `--native` / `--ple-gguf`: paths under `C:\StrataQ38FN-models`
- `--vram-reserve-mib`: 1000 at 32 GB. If the card drives a monitor and the display freezes, raise it to 1500.

`strata-configs/strata-unsloth-ud-iq4_xs.setup-original.json` shows, for comparison, what the setup originally generated.

## 6. Create the API key

All services require a bearer key. It is stored in `api_key.txt` (one line, listed in `.gitignore`):

```powershell
$k = -join ((48..57 + 65..90 + 97..122) | Get-Random -Count 42 | ForEach-Object { [char]$_ })
Set-Content -NoNewline -Encoding ascii api_key.txt $k
```

## 7. Settings

`settings.cmd`:

| Variable | Default | Meaning |
|---|---|---|
| `STRATA_PORT` | 8080 | Strata (web chat, OpenAI, Anthropic) |
| `FACADE_PORT` | 8081 | vLLM facade |
| `BIND_HOST` | `0.0.0.0` | `0.0.0.0` = reachable on the LAN, `127.0.0.1` = this PC only |
| `GENESIS3_MODEL` | empty | additional model name the facade accepts (the name the client sends) |

The dashboard (port 8082) calculates power and cost from `dashboard\settings.json`; the values can also be changed on
the page (electricity price €/kWh, measured idle power at the wall socket, GPU idle, CPU surcharge).

## 8. Firewall (only for LAN access)

As administrator, only for the private network profile:

```powershell
New-NetFirewallRule -DisplayName "StrataQ38FN 8080-8082" -Direction Inbound -Protocol TCP `
  -LocalPort 8080-8082 -Action Allow -Profile Private
```

## 9. Page file (recommended)

The scarce resource is the **Windows commit** (RAM + page file), not the physical RAM. strata.exe uses
82–88 GB private bytes. With a system-managed page file only 2–7 GB of commit remained free during operation; the
event log showed Event 26 "Nicht genügend virtueller Speicher" (out of virtual memory) twice.

Recommendation: a **fixed** page file on the NVMe, e.g. 64 GB initial = maximum size
(Control Panel → System → Advanced system settings → Performance → Advanced → Virtual memory),
then reboot.

Only then are these additional flags in the IQ4_XS config safe to run:

```
--max-context 262144                          (instead of 131072, measured without speed loss)
--prompt-cache-every 8192 --prompt-cache 12   (finer checkpoints, faster partial reuse)
```

## 10. Start and verify

Double-click `Start-IQ4_XS.bat`. The windows "Strata …" (loading takes 1–2 minutes) and
"vLLM-Fassade …" open, and the dashboard opens minimized.

```powershell
$k = Get-Content api_key.txt
curl.exe -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $k"
curl.exe -s http://127.0.0.1:8081/v1/models -H "Authorization: Bearer $k"
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $k" `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"Sag Hallo.\"}],\"max_tokens\":50}'
```

Then open `http://<PC>:8082/` in the browser; after the first response, decode should be at ~140–150 tok/s.

Facade tests (no GPU, do not use ports 8000/8080):

```powershell
Strata\.venv\Scripts\python.exe vllm_facade\test_facade.py
```

## 11. Connect clients

| Client | Base URL | Model name |
|---|---|---|
| OpenAI-compatible (Open WebUI, Continue, …) | `http://<PC>:8080/v1` | any |
| Anthropic-compatible (e.g. Claude-Code-like tools) | `http://<PC>:8080` (`/v1/messages`) | any |
| vLLM clients (genesis3) | `http://<PC>:8081/v1` | Strata model ID (see `/v1/models`) or `GENESIS3_MODEL` |

Details and known gaps of the facade: [vllm_facade/README.md](../vllm_facade/README.md).

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Engine log reports `LOW` free VRAM, display freezes | Raise `--vram-reserve-mib` (1000 → 1500) |
| Engine hangs during prompt processing ("layer 1 of the prompt chunk") | Model on the fastest NVMe, `--ple-io direct` |
| Crash / Event 26 "Nicht genügend virtueller Speicher" (out of virtual memory) | Commit full: fixed page file (step 9), reduce context or `--ple-row-cache` |
| Q4_K_XL does not start or is very slow | Needs ~45 GB RAM for experts: close the browser and other large programs |
| Crash with two concurrent requests with `parallel` | Known MTP bug ("unsupported native MMVQ GGML type"): omit `parallel` |
| Port in use, `Stop-All.bat` does not help | `Get-NetTCPConnection -LocalPort 8080 -State Listen` (German `netstat` shows "ABHÖREN", not "LISTENING") |
| 401 Unauthorized | Header `Authorization: Bearer <api_key.txt>` missing or wrong |

Logs: `Strata\strata-<model>.log` (engine), window "vLLM-Fassade …" (requests), `dashboard\dashboard.log`.

## 13. Own measurements (optional)

`agent_bench.py` starts a config variant on port 8090, plays back an agentic load (long main context,
alternating sub-agent, long decodes) and appends the result to `agent_results.jsonl`. The live server on 8080
is not touched, but the RAM/commit reserve must be enough for both (usually run `Stop-All.bat` first).

```powershell
Strata\.venv\Scripts\python.exe agent_bench.py kvres --add kv-resident=32768 --env STRATA_KV_PREFETCH=1
Strata\.venv\Scripts\python.exe agent_bench.py ctx262k --set max-context=262144 --workload longctx
Strata\.venv\Scripts\python.exe agent_bench.py pcie07 --decode-runs 3 --tune '{\"pcie_frac\": 0.7}'
```

Further workloads: `session`, `diverge`, `parallel`. The series that were run are in `series_*.sh`, the evaluation in the
[analysis report](../ANALYSE-Speicherhierarchie-XS.md), open levers in [HEBEL-Kandidaten-XS.md](../HEBEL-Kandidaten-XS.md).
