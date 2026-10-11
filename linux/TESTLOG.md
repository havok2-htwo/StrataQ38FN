# Testprotokoll (chronologisch)

Jeder Versuch steht hier **bevor** der nächste beginnt: Datum, was getestet wurde, Ergebnis, Urteil, Rohdaten. Vor einem
neuen Test hier nachsehen, ob es ihn schon gab. Lastprofil, wenn nicht anders angegeben: `bench_concurrency.py`, je Nutzer
2 Anfragen, 512 Ausgabe-Tokens, Temperatur 0.7, Thinking aus; „kurz" ≈ 120, „lang" ≈ 1.230 Prompt-Tokens. Zahlen =
Gesamtdurchsatz in tok/s (Output) bzw. Decode pro Nutzer, TTFT in s. Alle Strata-Läufe: UD-IQ4_XS, 4× RTX 3090.

Urteile: ✅ übernommen · ➖ kein messbarer Effekt · ❌ schlechter / verworfen · 🔬 nur Erkenntnis

## 2026-10-10

| # | Test | Ergebnis | Urteil | Rohdaten |
|---|---|---|---|---|
| 1 | Strata Setup-Standard, solo (kein `parallel`), mit/ohne `--resident-experts` | 121 tok/s single, ~88 gesamt; ohne Resident-Modus 60 GB RAM → oomd-Kill | ✅ `--resident-experts` Pflicht | `strata-A-solo*` |
| 2 | `parallel 8` (Upstream-Limit), Split auto | ~177 kurz/lang, 31 tok/s pro Nutzer | 🔬 Basis | `strata-B-batch8` |
| 3 | Engine-Patch 1: 16/32 Slots (8 Zeilen pro Gruppe) | 16 Slots bit-identisch, Batch-Decode 603 tok/s; über HTTP nur ~195 | ✅ Patch (HTTP-Limit = Prefill) | `batch_test_16/32.txt`, `strata-C/D-*` |
| 4 | Prefill-Chunks 256/512/768/1024 + Decode-Anteil 1/2 | 512 am besten (~246–250 lang); Decode-Anteil wirkungslos (s. #21) | ✅ `--prefill 512` | `strata-E*`, `strata-F*` |
| 5 | Slots 12/16/20/24/28 (kurze Prompts) | Plateau ~330–361, 12 Slots beste Latenz | 🔬 | `strata-H*-short` |
| 6 | Nutzer 24–40 bei 12 Slots | flach 346–358, Wartezeit steigt | 🔬 | `strata-U-b12-users` |
| 7 | 3 GPUs (`--trim-stage-weights`, Split 16/32) | 117 single / 266 Batch | 🔬 4. Karte nur unter Last nützlich | `strata-T3-*` |
| 8 | UD-Q4_K_XL | 97 % Experten im VRAM, 100 single / 241 Batch | 🔬 IQ4_XS bleibt Target | `strata-K-*` |
| 9 | MTP aus (`--spec 4` ohne `--mtp`) | 81 statt 117 tok/s | ✅ MTP an (+45 %) | `strata-M-nomtp` |
| 10 | Prefill über Länge 2K–28K: Chunks 512 / auto / auto:32768 | 512: 1.247→2.239 tok/s, bestes Gesamtbild | ✅ 512 | `strata-P-*` |
| 11 | vLLM 0.29 Flash-Next W4A16 (EP, FP8-Tabelle per Patch), Prefix-Cache an/aus | 65–75 single, ~420–450 Plateau, max ~21 gleichzeitig | 🔬 | `vllm-w4a16-*` |
| 12 | vLLM 0.31 Flash-Next, Prefix-Cache an | 79 single, ~400 lang / ~710 kurz, Cap ~22 | 🔬 | `vllm031-flashnext-pc-*` |
| 13 | vLLM 0.31 Flash-Next mit MTP bei 64 bzw. 32 Seqs | OOM bzw. kein KV-Platz | ❌ | `vllm031_mtp*.log` |
| 14 | vLLM 0.31 Prefill über Länge | ~3.500 tok/s konstant | 🔬 | `vllm031-P-*` |
| 15 | vLLM 0.31 27B-FP8-Preset | 64 single, 409 lang / 838 kurz (64 Nutzer), ~100 GB RAM | 🔬 | `vllm031-27b-fp8-*` |

## 2026-10-11

| # | Test | Ergebnis | Urteil | Rohdaten |
|---|---|---|---|---|
| 16 | vLLM 0.31 Flash-Next MTP mit 8 Seqs, 1.024 Batch-Tokens, 0.93 | **136 single**, 490 kurz (8 Nutzer), 65K KV | 🔬 MTP geht nur mit wenig Parallelität | `vllm031-flashnext-mtp2-s8-*` |
| 17 | Qualitäts-Set (MMLU-Pro 240 / GSM8K 150) | Strata IQ4_XS 77,9/94,0 · 27B 72,1/95,3 · vLLM-W4A16 70,4/94,7 · RAZOR 62,5/92,7 | 🔬 | `eval/eval_results.jsonl` |
| 18 | RAZOR-65B (50 % Experten gepruned, NVFP4) in vLLM | 111 single, ~412 lang; Qualität stark gefallen, Deutsch fehlerhaft | ❌ Modell gelöscht | `vllm031-razor65-*` |
| 19 | Hub-Erhebung 910 Flash-Next-Repos | kein kleinerer vLLM-tauglicher Quant ohne Qualitätsverlust | 🔬 | `quant-survey/` |
| 20 | 16 Slots in 2 Gruppen; 12 Slots in 2 Gruppen | 266–272 / 264–270 kurz, 200 lang | ❌ halbe Karten idle | `strata-Q2-*` |
| 21 | Decode-Anteil 4.0 | 366–375 / 259 | ➖ im Pipeline-Modus wirkungslos (`generate.cpp:9706` upstream) | `strata-Q3-*` |
| 22 | 2 Instanzen à 2 GPUs (Round-Robin) | 152–182 kurz, 100 lang, 90 single | ❌ | `strata-Q1-dp2-*` |
| 23 | **Split `12,24,36` + `--trim-stage-weights`** | **405 kurz / 288 lang** (+14/+16 %), keine Puffer-Leihe | ✅ | `strata-T16-trim-*` |
| 24 | dito 20 / 24 Slots | 409/288, 416/293 | ➖ Plateau | `strata-T20/T24-*` |
| 25 | dito Chunks 1.024 / 2.048 | 406/281, 403/290 | ➖ | `strata-T16-pf*` |
| 26 | Multi-Turn 8/16 Agenten, Trim-Config 32K vs. Agenten-Config (262K, `--kv-resident`, 24 GB Konv.-Cache) | TTFT Folge-Turn 7,1/10,4 s → **4,8/5,0 s**, 16 Agenten 43 → 73 tok/s | ✅ Agenten-Config | `multiturn.jsonl` |
| 27 | Multi-Turn gegen vLLM 0.31 (+20 GB RAM-KV-Stufe) | 8: 89 tok/s, 6,4 s; 16: 77 tok/s, **32 s** | 🔬 Strata vorne bei Agentenlast | `multiturn.jsonl` |
| 28 | Engine-Patch 2: direkte Slot-Kopie GPU→GPU | Kopie 65,6 → 34,0 ms, 405/294, Agenten-TTFT 4,5 s; bit-identisch | ✅ (klein) | `exact_direct_*`, `strata-T16-direct-*` |
| 29 | Profiling Batch-Fenster (`STRATA_VERIFY_PROFILE`) | GPU 5,1–6,1 ms von ~6,95 ms pro Schritt; Karte 4 +0,73 ms Head | 🔬 Overhead klein | `prof_batch16*` |
| 30 | Split `12,24,37` / `13,25,37` | 399/289, 397/282 | ➖ | `strata-T16-split*` |
| 31 | `--ple-io ram` | **417 / 297**, 1,2K-Prompt 821 statt ~1.100 ms | ✅ | `strata-T16-pleram-*` |
| 32 | Überlappende Aufnahme, Prototyp (`STRATA_PIPE_OVERLAP_SPIKE`) | 413/294 (kein Gewinn), **TTFT p50 2,03 → 0,57 s** | 🔬 nur Latenz-Hebel | `strata-S16-spike-*`, `patches/experimental/` |
| 33 | Prefill-Timing kurzer Prompts (`STRATA_PREFILL_TIMING`) | 154 Tok.: 78–87 ms pro Karte, 31 % „dequant" = Experten-Kopie für MMQ | 🔬 | `prefill_timing.txt`, `upstream/ANALYSIS-short-prompt-moe.md` |
| 34 | `STRATA_PF_FUSED=1` | keine Änderung (greift erst ab 1.024-Token-Chunks) | ➖ | `prefill_timing_fused.txt` |

## Analysen (ohne GPU)

- `upstream/ANALYSIS-batching-levers.md`: 4 Engine-Hebel (überlappende Aufnahme, MTP im Batch, Fenster-Overhead,
  Quick-Wins) mit Aufwand und Gegenprüfung.
- `upstream/DESIGN-overlap.md`: vollständiges Design der überlappenden Aufnahme (6 Meilensteine).
- `upstream/ANALYSIS-short-prompt-moe.md`: Kurz-Prompt-Prefill, Optionen A (Decode-Kernel für ≤ 8 Zeilen),
  B (MMQ direkt auf den Cache-Slots), C (Fused-Kernels).

## Offen / nächste Schritte

- Variante B (MMQ direkt auf gepaddeten Cache-Slots): Design läuft, danach Umsetzung + Exaktheit + Messung.
- Sichere überlappende Aufnahme (Latenz).
- Geparkt: dynamisches Umschalten Pipeline ↔ Expert-Parallel (siehe Gedächtnis-/Ideennotiz im README-Abschnitt).

## Anhang: alle Messläufe (automatisch aus den JSONL-Dateien)

### `results/sweep.jsonl`

| Label | Nutzer: Gesamt-tok/s (TTFT p50) |
|---|---|
| `vllm-w4a16-ep4-nomtp` | 1: 72.2 (0.99 s), 2: 121.6 (0.57 s), 4: 210.3 (1.46 s), 1: 65.0 (1.34 s), 2: 115.1 (0.58 s), 4: 197.2 (1.45 s), 8: 269.3 (2.67 s), 16: 375.8 (2.42 s), 32: 384.2 (7.2 s), 48: 418.4 (29.54 s), 60: 400.1 (50.93 s), 96: 414.2 (87.52 s) |
| `vllm-w4a16-ep4-nopc` | 1: 74.9 (0.37 s), 8: 255.9 (1.81 s), 16: 374.7 (2.26 s), 24: 419.8 (3.4 s), 32: 418.1 (6.62 s), 48: 452.2 (27.14 s), 64: 442.0 (48.21 s) |
| `strata-A-solo` | 1: 91.5 (1.5 s), 2: 87.1 (7.37 s), 4: 87.4 (18.94 s), 8: 87.6 (41.64 s), 16: 89.7 (85.95 s) |
| `strata-B-batch8` | 1: 85.9 (1.69 s), 2: 106.2 (2.51 s), 4: 147.9 (3.32 s), 8: 177.0 (5.03 s), 12: 167.8 (10.84 s), 16: 170.8 (11.74 s), 32: 174.4 (35.66 s), 64: 175.8 (65.32 s) |
| `strata-C-batch16` | 1: 84.8 (1.7 s), 8: 171.0 (5.94 s), 16: 198.4 (9.26 s), 24: 193.6 (19.38 s), 32: 192.8 (20.26 s), 64: 194.9 (46.04 s) |
| `strata-D-batch32` | 16: 179.9 (10.61 s), 32: 191.4 (8.41 s), 48: 191.5 (21.26 s), 64: 192.1 (27.28 s), 96: 191.9 (37.75 s) |
| `strata-E1-b16-pf512-share1` | 16: 246.3 (4.58 s), 32: 240.8 (15.1 s) |
| `strata-E2-b16-pf256-share1` | 16: 228.7 (5.42 s), 32: 226.5 (17.78 s) |
| `strata-E3-b16-prefillhelp` | 16: 193.6 (7.17 s), 32: 193.4 (20.26 s) |
| `strata-F1-b16-pf1024-share1` | 16: 226.4 (6.65 s), 32: 221.9 (12.89 s) |
| `strata-F2-b16-pf512-share2` | 16: 246.0 (4.58 s), 32: 250.6 (12.25 s) |
| `strata-F3-b16-pf768-share1` | 16: 245.3 (4.56 s), 32: 238.6 (15.62 s) |
| `strata-G16-long` | 1: 95.6 (1.15 s), 2: 117.6 (1.12 s), 4: 164.4 (2.28 s), 8: 209.4 (4.52 s), 16: 248.4 (4.57 s), 24: 237.7 (11.02 s), 32: 233.9 (11.07 s), 64: 246.0 (46.48 s) |
| `strata-G16-short` | 1: 103.7 (0.53 s), 8: 281.0 (2.0 s), 16: 357.5 (2.6 s), 24: 364.6 (6.55 s), 32: 364.5 (20.27 s), 64: 354.3 (29.53 s) |
| `strata-G32-short` | 16: 318.8 (2.73 s), 32: 342.4 (2.98 s), 48: 330.0 (7.63 s), 64: 340.9 (12.86 s), 96: 344.7 (14.22 s) |
| `strata-H12-short` | 12: 361.1 (1.6 s), 18: 335.9 (2.62 s) |
| `strata-H16-short` | 16: 347.9 (2.07 s), 24: 358.1 (6.31 s) |
| `strata-H20-short` | 20: 335.9 (2.97 s), 30: 326.6 (6.51 s) |
| `strata-H24-short` | 24: 331.2 (2.63 s), 36: 328.3 (7.43 s) |
| `strata-H28-short` | 28: 328.0 (3.16 s), 42: 328.3 (8.21 s) |
| `strata-A-solo-mtp` | 1: 87.2 (1.5 s) |
| `vllm031-27b-fp8-p1000` | 1: 59.1 (0.76 s), 4: 130.7 (2.62 s), 12: 241.0 (4.3 s), 16: 283.1 (4.34 s), 32: 370.6 (5.13 s), 64: 409.1 (9.2 s) |
| `vllm031-27b-fp8-p100` | 1: 63.5 (0.13 s), 4: 203.7 (0.46 s), 12: 382.0 (1.24 s), 16: 484.9 (1.25 s), 32: 705.2 (2.25 s), 64: 838.1 (3.35 s) |
| `vllm031-flashnext-pc-p1000` | 1: 75.0 (0.38 s), 8: 238.2 (1.85 s), 12: 365.9 (1.9 s), 16: 385.8 (1.95 s), 24: 416.0 (4.16 s), 32: 397.5 (22.27 s), 48: 395.0 (33.44 s), 64: 398.2 (49.53 s) |
| `vllm031-flashnext-pc-p100` | 1: 78.0 (0.15 s), 8: 413.5 (0.56 s), 12: 499.3 (0.61 s), 16: 621.8 (0.68 s), 24: 590.8 (0.88 s), 32: 706.4 (0.95 s), 48: 683.2 (16.21 s), 64: 710.0 (31.46 s) |
| `strata-K-q4kxl-solo` | 1: 70.0 (2.26 s) |
| `strata-K-q4kxl-b12-short` | 12: 241.0 (4.12 s), 24: 233.7 (11.02 s) |
| `strata-K-q4kxl-b12-long` | 12: 149.8 (9.32 s) |
| `strata-T3-solo` | 1: 95.8 (1.1 s) |
| `strata-T3-b12-short` | 12: 265.9 (2.48 s), 18: 253.0 (6.26 s) |
| `strata-M-nomtp` | 1: 64.6 (1.42 s) |
| `strata-U-b12-users` | 24: 347.3 (4.17 s), 28: 346.0 (5.06 s), 32: 357.3 (9.11 s), 36: 358.2 (15.81 s), 40: 351.2 (17.2 s) |
| `strata-P-pf512-pf1600` | 1: 17.1 (1.56 s) |
| `strata-P-pf512-pf6500` | 1: 7.3 (4.06 s) |
| `strata-P-pf512-pf13000` | 1: 4.2 (7.28 s) |
| `strata-P-pf512-pf24000` | 1: 2.4 (12.73 s) |
| `strata-P-auto-pf1600` | 1: 11.6 (2.44 s) |
| `strata-P-auto-pf6500` | 1: 5.0 (6.16 s) |
| `strata-P-auto-pf13000` | 1: 3.7 (8.35 s) |
| `strata-P-auto-pf24000` | 1: 2.5 (12.45 s) |
| `strata-P-auto32k-pf1600` | 1: 11.5 (2.46 s) |
| `strata-P-auto32k-pf6500` | 1: 4.7 (6.46 s) |
| `strata-P-auto32k-pf13000` | 1: 2.7 (11.36 s) |
| `strata-P-auto32k-pf24000` | 1: 1.5 (20.84 s) |
| `vllm031-P-pf1600` | 1: 33.6 (0.56 s) |
| `vllm031-P-pf6500` | 1: 12.2 (2.22 s) |
| `vllm031-P-pf13000` | 1: 6.7 (4.41 s) |
| `vllm031-P-pf24000` | 1: 3.8 (8.13 s) |
| `vllm031-flashnext-mtp2-s8-p1000` | 1: 115.8 (0.66 s), 4: 268.9 (0.81 s), 8: 224.9 (1.53 s), 12: 234.6 (14.15 s) |
| `vllm031-flashnext-mtp2-s8-p100` | 1: 132.7 (0.15 s), 4: 346.4 (0.3 s), 8: 490.3 (0.3 s), 12: 486.1 (0.91 s) |
| `strata-eval-iq4xs` | 1: 10.8 (1.33 s) |
| `vllm031-razor65-mtp2-s12-p1000` | 1: 102.7 (0.36 s), 4: 263.9 (0.58 s), 8: 345.7 (0.76 s), 12: 411.7 (1.11 s), 24: 416.2 (15.04 s) |
| `vllm031-razor65-mtp2-s12-p100` | 1: 110.2 (0.15 s), 4: 304.1 (0.22 s), 8: 445.5 (0.37 s), 12: 536.1 (0.29 s), 24: 532.4 (11.49 s) |
| `strata-Q2-b16-groups2-short` | 16: 266.3 (1.59 s), 24: 271.6 (5.13 s) |
| `strata-Q2-b16-groups2-long` | 16: 199.5 (3.45 s) |
| `strata-Q2-b12-groups2-short` | 12: 269.8 (0.93 s), 24: 263.7 (2.91 s) |
| `strata-Q3-b12-share4-short` | 12: 366.4 (1.59 s), 24: 374.9 (4.13 s) |
| `strata-Q3-b12-share4-long` | 12: 258.9 (4.17 s) |
| `strata-Q1-dp2-short` | 1: 74.4 (1.11 s), 12: 152.5 (7.23 s), 24: 174.2 (7.17 s), 32: 182.3 (8.83 s) |
| `strata-Q1-dp2-long` | 16: 99.5 (25.39 s) |
| `strata-T12-trim-short` | 1: 108.7 (0.4 s), 12: 344.8 (2.03 s), 24: 366.2 (3.39 s) |
| `strata-T12-trim-long` | 12: 275.6 (4.35 s) |
| `strata-T16-trim-short` | 1: 110.6 (0.4 s), 16: 404.7 (2.32 s), 32: 389.1 (4.83 s) |
| `strata-T16-trim-long` | 16: 288.1 (5.37 s) |
| `strata-T20-trim-short` | 20: 409.4 (2.05 s) |
| `strata-T20-trim-long` | 20: 288.5 (5.35 s) |
| `strata-T24-trim-short` | 24: 416.5 (2.85 s) |
| `strata-T24-trim-long` | 24: 292.8 (5.34 s) |
| `strata-T16-pf1024-long` | 16: 280.7 (4.47 s) |
| `strata-T16-pf1024-short` | 16: 406.3 (2.33 s) |
| `strata-T16-pf2048-long` | 16: 290.5 (5.3 s) |
| `strata-T16-pf2048-short` | 16: 403.4 (2.15 s) |
| `strata-T16-trim-split-mt-warm` | 1: 11.6 (1.19 s) |
| `strata-X16-agent-trim-262k-mt-warm` | 1: 9.9 (1.41 s) |
| `strata-T16-direct-short` | 16: 405.3 (1.97 s), 24: 391.8 (3.37 s) |
| `strata-T16-direct-long` | 16: 294.5 (4.31 s) |
| `strata-X16-direct-mt-warm` | 1: 10.2 (1.43 s) |
| `strata-T16-split12-24-37-short` | 16: 399.2 (2.25 s) |
| `strata-T16-split12-24-37-long` | 16: 289.3 (4.47 s) |
| `strata-T16-split13-25-37-short` | 16: 397.3 (2.22 s) |
| `strata-T16-split13-25-37-long` | 16: 281.6 (4.27 s) |
| `strata-T16-pleram-short` | 16: 416.8 (2.03 s) |
| `strata-T16-pleram-long` | 16: 296.5 (4.32 s) |
| `strata-S16-spike-short` | 16: 412.9 (0.57 s) |
| `strata-S16-spike-long` | 16: 293.5 (1.18 s) |
| `strata-P16-timing` | 1: 6.1 (1.23 s) |
| `strata-P16-fused` | 1: 6.0 (1.23 s) |

### `results/multiturn.jsonl`

| Label | Nutzer | tok/s | TTFT Folge-Turn p50 |
|---|---|---|---|
| `strata-T16-trim-split-mt-u8` | 8 | 49.2 | 7.05 s |
| `strata-T16-trim-split-mt-u16` | 16 | 42.8 | 10.4 s |
| `strata-X16-agent-trim-262k-mt-u8` | 8 | 57.1 | 4.76 s |
| `strata-X16-agent-trim-262k-mt-u16` | 16 | 72.8 | 4.95 s |
| `vllm031-full-offload20-mt-u8` | 8 | 89.2 | 6.37 s |
| `vllm031-full-offload20-mt-u16` | 16 | 76.7 | 32.18 s |
| `strata-X16-direct-mt-u16` | 16 | 62.3 | 4.47 s |

### `eval/eval_results.jsonl`

| Label | MMLU-Pro | GSM8K |
|---|---|---|
| `vllm-full-w4a16-fp8ple` | 70.4 % | 94.7 % |
| `strata-ud-iq4xs` | 77.9 % | 94.0 % |
| `vllm-qwen3.8-27b-fp8` | 72.1 % | 95.3 % |
| `vllm-razor65-nvfp4` | 62.5 % | 92.7 % |
