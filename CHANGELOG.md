# Changelog

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unveröffentlicht]

### Geplant
- Messserie I (`series_i.sh`, siehe `HEBEL-Kandidaten-XS.md`): GPU-Kernel-Varianten, CPU-Pool, PCIe-Kopiermodus,
  Prefill-Chunk 16K/32K, Drafting, gelerntes Expert-Profil, Server-Optionen.
- Feste Auslagerungsdatei, danach `--max-context 262144` und `--prompt-cache-every 8192 --prompt-cache 12`.

## [0.4.0] – 2026-10-11

### Hinzugefügt
- **Linux-Betrieb auf 4× RTX 3090** (`linux/`, Bericht `linux/README.md`): Setup per `Strata/setup.sh` mit
  Layer-Split über alle 4 Karten, `start-strata.sh` (eigener systemd-Scope, Port 8090), Bench-Werkzeuge
  `bench_concurrency.py` (Last mit vielen Nutzern) und `bench_multiturn.py` (Agentenlast), Runner
  `run_strata_variant.sh`, alle gemessenen Configs und Ergebnisse.
- **Engine-Patches** für Strata (`linux/patches/`, bit-identisch zu solo getestet):
  mehr als 8 Batch-Slots bei Layer-Split (8 Zeilen pro Pipeline-Gruppe) und direkte Slot-Kopie GPU→GPU
  (65,6 → 34,0 ms pro Aufnahme). Upstream-Texte in `linux/upstream/` (noch nicht eingereicht).
- Empfohlene Linux-Config: Split `12,24,36` + `--trim-stage-weights`, 16 Slots, `--prefill 512`, MTP:
  ~121 tok/s für einen Nutzer, ~405 tok/s gesamt (kurze Prompts) bzw. ~294 tok/s (1,2K-Prompts);
  Agenten-Variante mit 262K Kontext, `--kv-resident 32768` und 24 GB Konversations-Cache.
- Vergleich mit vLLM 0.31 (Flash-Next W4A16 mit FP8-N-Gram-Tabelle per Patch, mit/ohne MTP; Qwen3.8-27B-FP8),
  Qualitäts-Set (`linux/eval/`: MMLU-Pro 240 + GSM8K 150): Strata UD-IQ4_XS 77,9 % MMLU-Pro vs. 27B-FP8 72,1 %
  vs. vLLM-W4A16 70,4 %; Auswertung aller 910 Flash-Next-Repos auf Hugging Face (`linux/quant-survey/`).

### Verworfen (gemessen)
- `STRATA_BATCH_DECODE_SHARE`: bei Layer-Split mit Pipeline-Gruppen wirkungslos (Prompt wird am Stück gelesen).
- Weniger Pipeline-Gruppen (2 statt 4), 2 Instanzen à 2 GPUs, mehr als 16 Slots, größere Prefill-Chunks.
- RAZOR-65B (50 % Experten gepruned) in vLLM: deutlich schlechtere Qualität (62,5 % MMLU-Pro).

## [0.3.0] – 2026-10-09

### Hinzugefügt
- Git-Repo mit README, Changelog und Einrichtungsanleitung (`docs/EINRICHTUNG.md`).
- `strata-configs/`: die drei Modell-Configs als Vorlagen (plus Setup-Original der IQ4_XS-Config zum Vergleich).

## [0.2.0] – 2026-10-08

### Hinzugefügt
- Dashboard (`dashboard/`, `Start-Dashboard.bat`, Port 8082): Tempo, Caches, VRAM/RAM/Commit, SSD-Traffic,
  geschätzte Leistung an der Steckdose, kWh, € und € pro 1M Tokens (Ein-/Ausgabe), Laufzeitanteile, 24-h-Hochrechnung,
  Reset mit Archiv. Startet mit jedem Modell mit und überlebt Modellwechsel.
- `agent_bench.py`: A/B-Runner mit agentischer Last (Workloads `agent`, `session`, `diverge`, `longctx`,
  `parallel`), misst Commit und physisches RAM mit.
- `trace_analyze.py` für Routing-Traces; Messserien E–H (`series_*.sh`, `configs/`, `logs/`, `agent_results.jsonl`).
- Abschlussbericht `ANALYSE-Speicherhierarchie-XS.md` und Hebelliste `HEBEL-Kandidaten-XS.md`.

### Geändert
- UD-IQ4_XS-Config getunt: `--resident-experts` (statt `--resident-budget-gib 40`), `--ple-row-cache 33554432`,
  `--conversation-cache-mib 6144 --conversation-cache-slots 3`, `--adapt-async 1`, `--vram-reserve-mib 1000`,
  `--kv-resident 32768`, `STRATA_KV_PREFETCH=1`. Decode ~125 → ~145 tok/s, Agenten-Wechsel ~25 s → ~1,5 s,
  keine SSD-Reads mehr im Prompt-Pfad.
- GGUF-Dateien von `X:` (Crucial P3) nach `C:\StrataQ38FN-models` (SN850X) verschoben, `--ple-io direct`.

### Behoben
- Engine-Hänger beim Prompt-Lesen („layer 1 of the prompt chunk", 9 Stück am 7./8.10.); seit dem Umzug keine mehr.

### Verworfen (gemessen)
- Batching (`parallel: 2` + `--batch-mtp`): Absturz bei gleichzeitigen Anfragen (MTP-Bug), sonst −18 %.
- `--vram-reserve-mib` < 800, `pcie_frac` > 0,7, `--spec` ≠ 4, `--spec-min-p` ≠ 0,5.
- `STRATA_CACHE_MESSAGE_BOUNDARY=1`: ohne Wirkung.

## [0.1.0] – 2026-10-06

### Hinzugefügt
- Strata-Setup für drei Modelle: Unsloth UD-IQ4_XS, Unsloth UD-Q4_K_XL, ISTA Coder IQ1_M.
- Start-Skripte (`Start-IQ4_XS.bat`, `Start-Q4_K_XL.bat`, `Start-Coder.bat`, `Stop-All.bat`), gemeinsame
  Einstellungen in `settings.cmd`, API-Key aus `api_key.txt`, LAN-Betrieb auf 8080/8081.
- vLLM-Fassade (`vllm_facade/`, Port 8081) für vLLM-Clients wie genesis3, mit Testsuite.
- Benchmarks `bench_strata.py`, `run_variant.py`, `parallel_bench.py`, `stall_probe.py`; Messserien A–D.
- Wiederaufnehmbare Downloads `download_hf.py`, `download_q4kxl.py`.
- `LIESMICH.txt` als Kurzbedienung.
