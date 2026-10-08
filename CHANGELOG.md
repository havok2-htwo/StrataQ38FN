# Changelog

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unveröffentlicht]

### Geplant
- Messserie I (`series_i.sh`, siehe `HEBEL-Kandidaten-XS.md`): GPU-Kernel-Varianten, CPU-Pool, PCIe-Kopiermodus,
  Prefill-Chunk 16K/32K, Drafting, gelerntes Expert-Profil, Server-Optionen.
- Feste Auslagerungsdatei, danach `--max-context 262144` und `--prompt-cache-every 8192 --prompt-cache 12`.

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
