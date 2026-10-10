# StrataQ38FN

Lokaler Betrieb von **Qwen3.8-Flash-Next** mit der Engine [Strata](https://github.com/Niko1221/Strata) auf einem
Windows-PC mit **RTX 5090 (32 GB) + 64 GB RAM**, abgestimmt auf agentische Last (lange Kontexte, Haupt- und
Sub-Agent im Wechsel).

Dieses Repo enthält **nicht** die Engine und **nicht** die Modelle, sondern alles drumherum:

- Start-/Stop-Skripte für drei Modellvarianten (immer nur eines aktiv)
- getunte Strata-Configs (`strata-configs/`)
- eine **vLLM-kompatible Fassade** (`vllm_facade/`), damit vLLM-Clients (z. B. genesis3) ohne Änderung andocken
- ein **LAN-Dashboard** (`dashboard/`) mit Tempo, Caches, Speicher, Strom und Kosten
- Benchmark- und A/B-Werkzeuge samt Messdaten und Analyseberichten

**Linux mit 4× RTX 3090** (Setup, Batching bis 32 Slots, Vergleich mit vLLM): **[linux/README.md](linux/README.md)**

Einrichtung Schritt für Schritt: **[docs/EINRICHTUNG.md](docs/EINRICHTUNG.md)** · Änderungen: [CHANGELOG.md](CHANGELOG.md)

## Modelle

| Start | Modell | Qualität / Einsatz | Decode | Kontext |
|---|---|---|---|---|
| `Start-IQ4_XS.bat` **(empfohlen)** | Unsloth UD-IQ4_XS (~4 bit) | Allround, Deutsch | ~145 tok/s | 128K |
| `Start-Q4_K_XL.bat` | Unsloth UD-Q4_K_XL (~5 bit) | etwas bessere Qualität, braucht ~45 GB RAM | 85–95 tok/s | 64K |
| `Start-Coder.bat` | ISTA Coder IQ1_M (halbe Experten) | nur Code / Englisch | 165–230 tok/s | 128K |

`Stop-All.bat` beendet Modell und Fassade. Ein Start beendet automatisch das vorher laufende Modell.

## Dienste und Ports

| Port | Dienst | Adresse |
|---|---|---|
| 8080 | Strata: Web-Chat, OpenAI- und Anthropic-API | `http://<PC>:8080/`, `/v1`, `/v1/messages` |
| 8081 | vLLM-Fassade (für vLLM-Clients) | `http://<PC>:8081/v1` |
| 8082 | Dashboard (nur Zahlen, keine Prompts) | `http://<PC>:8082/` |
| 8090 | nur Benchmarks (`agent_bench.py`) | – |

Alle API-Aufrufe brauchen den Header `Authorization: Bearer <Key aus api_key.txt>`. Ports, LAN/lokal und den
Modellnamen für genesis3 stellt man in `settings.cmd` ein.

## Ergebnisse des Tunings (UD-IQ4_XS)

Gemessen am 2026-10-08 (Details: [ANALYSE-Speicherhierarchie-XS.md](ANALYSE-Speicherhierarchie-XS.md), Abschnitte 7–8):

| | Setup-Standard | getunt |
|---|---|---|
| Decode, kurzer Kontext | ~125 tok/s | **~145 tok/s** |
| Wechsel Haupt-Agent ↔ Sub-Agent | ~25 s Prefill | **~1,5 s** |
| SSD-Reads im Prompt-Pfad | ja (1,1 TB) | keine |

Die wichtigsten Flags: `--resident-experts`, `--conversation-cache-mib 6144 --conversation-cache-slots 3`,
`--adapt-async 1`, `--kv-resident 32768` + `STRATA_KV_PREFETCH=1`, `--vram-reserve-mib 1000`, `--ple-io direct`.

**Engpass ist der Windows-Commit, nicht das physische RAM.** 262K Kontext läuft ohne Tempoverlust, ist aber erst mit
fester Auslagerungsdatei betriebssicher (siehe Einrichtung, Schritt 9).

Sackgassen (gemessen, nicht wiederholen): Batching (`parallel: 2` + `--batch-mtp`) stürzt ab bzw. kostet −18 %;
`--vram-reserve-mib` < 800 meldet LOW; `pcie_frac` > 0,7, `--spec` ≠ 4 und `--spec-min-p` ≠ 0,5 sind langsamer.

## Verzeichnis

```
Start-*.bat, Stop-All.bat     Bedienung (Doppelklick)
_start-model.cmd, _run-*.cmd  intern: startet Strata + Fassade (+ Dashboard)
settings.cmd                  Ports, Bind-Adresse, genesis3-Modellname
strata-configs/               Strata-Configs (Vorlagen, werden nach Strata\ kopiert)
vllm_facade/                  vLLM-Fassade, eigene README und Tests
dashboard/                    LAN-Dashboard (dashboard.py, index.html, settings.json)
agent_bench.py                A/B-Runner mit agentischer Last (Port 8090)
bench_strata.py, run_variant.py, parallel_bench.py, stall_probe.py, trace_analyze.py
series_*.sh                   die gefahrenen Messserien
configs/, logs/, *.jsonl      Varianten, Engine-Logs und Ergebnisse der Messserien
ANALYSE-*.md, HEBEL-*.md      Abschlussbericht und offene Hebel
download_hf.py, download_q4kxl.py  wiederaufnehmbare Hugging-Face-Downloads
```

Nicht im Repo (per `.gitignore`): `Strata/` (Engine-Clone + venv), `Strata-data/` (Packs, MTP),
`api_key.txt`, Session-Dateien, Traces, Laufzeit-Logs, Dashboard-Zähler.

## Getestete Umgebung

Windows 11 Pro · AMD Ryzen 7 9800X3D · 64 GB RAM · NVIDIA RTX 5090 (Treiber 610.88) · Python 3.12.4 ·
Strata-Commit `82f46a8` · Modelle auf NVMe (WD SN850X, `C:\StrataQ38FN-models`).

## Lizenz

Strata steht unter eigener Lizenz (siehe Upstream-Repo). Die Modelle unterliegen den Lizenzen von Qwen bzw. Unsloth.
