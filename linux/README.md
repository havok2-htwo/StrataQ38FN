# StrataQ38FN unter Linux: 4× RTX 3090

Messungen vom 2026-10-10 auf dem Linux-Host „Galactica": 4× RTX 3090 (je 24 GB, PCIe effektiv ~13 GB/s pro Karte),
Threadripper 3990X, 128 GB RAM (Budget für Tests: 96 GB), Treiber 580, CUDA 13.0. Strata v0.1.41 (für sm_86 selbst
kompiliert), Modell **Unsloth UD-IQ4_XS** (Target), zum Vergleich UD-Q4_K_XL, vLLM 0.29/0.31 und Qwen3.8-27B-FP8.

## Kurzfassung

| | Strata UD-IQ4_XS | vLLM 0.31 Flash-Next W4A16 | vLLM 0.31 Qwen3.8-27B FP8 |
|---|---|---|---|
| Decode, 1 Nutzer | **~120 tok/s** (MTP) | 79 tok/s (MTP passt nicht in VRAM) | 64 tok/s |
| Gesamt, 1,2K-Prompts | ~250 tok/s (Plateau ab 12–16 Nutzern) | ~400 tok/s | 409 tok/s (64 Nutzer) |
| Gesamt, kurze Prompts | ~360 tok/s (Plateau ab 12 Nutzern) | ~710 tok/s | 838 tok/s (64 Nutzer) |
| Prefill | 1.250 (2K) → 2.240 tok/s (28K) | ~3.500 tok/s | bis ~3.100 tok/s |
| VRAM | ~88 GB (alle Experten resident) | ~86 GB | ~63 GB (Preset 0.68) |
| RAM (echt) | **~3–5 GB** | ~64 GB (FP8-N-Gram-Tabelle) | ~100 GB (80 GB KV-Offload) |
| Gleichzeitig dekodierend | 12 Slots (empfohlen) | ~22 (Block-Pool voll) | 64 |

Strata ist für **einzelne bis ~12 gleichzeitige Nutzer** die beste Wahl: doppelt so schnell pro Nutzer wie vLLM und
fast ohne RAM. vLLM liefert bei vielen Nutzern etwa den doppelten Gesamtdurchsatz und liest Prompts schneller ein.

## Einrichtung (Linux)

```bash
git clone git@github.com:havok2-htwo/StrataQ38FN.git /media/raid0/StrataQ38FN
cd /media/raid0/StrataQ38FN
git clone https://github.com/Niko1221/Strata.git Strata
git -C Strata am ../linux/patches/0001-*.patch        # optional: mehr als 8 Batch-Slots (siehe unten)
cd Strata
./setup.sh --yes --family unsloth --model UD-IQ4_XS --gpus all \
  --models-dir /media/raid0/Models/Strata-GGUF --no-start --no-browser --vision no
```

Das Setup kompiliert die Engine (kein Prebuilt für 0.1.41, ~10 min mit CUDA 13), lädt 94 GB und baut Pack und
MTP-Draft. Danach startet `linux/start-strata.sh` die empfohlene Konfiguration auf `http://127.0.0.1:8090/v1`
(8080 belegt hier Open WebUI). Für LAN: `HOST=0.0.0.0 API_KEY=<geheim> ./start-strata.sh`.

Zwei Fallen:
- **Ohne `--resident-experts` lädt Strata alle Experten (~60 GB) in den RAM**, obwohl alle 24.576 Experten in die vier
  Karten passen (Split 12/24/36) – das sprengt das Budget und wurde von `systemd-oomd` beendet. Mit dem Flag liegen
  0 GB Experten im RAM.
- Schwere Prozesse immer in einem eigenen systemd-Scope starten (`run_strata_variant.sh`, `start-strata.sh`): ein
  globaler OOM beendet sonst den ganzen Editor-Scope (ist einmal passiert).

## Empfohlene Konfiguration

`configs/G12-pf512-share2.json` = Setup-Config plus:

| Einstellung | Wert | Warum |
|---|---|---|
| `--resident-experts` | an | alle Experten im VRAM, RAM-Kopie entfällt |
| `"parallel"` | **12** | höchster Gesamtdurchsatz bei bester Latenz (s. u.) |
| `--prefill` | **512** | kurze Prompts 1,6× schneller (1.247 statt 795 tok/s bei 2K), bis 15K besser als `auto` |
| `STRATA_BATCH_DECODE_SHARE` | **2.0** | Slots dekodieren zwischen den Prefill-Chunks länger |
| MTP (`--mtp`, `--spec 4`) | an | +45 % Decode (117 statt 81 tok/s), 61 % Draft-Akzeptanz bei Temperatur 0.7 |

## Messergebnisse Strata

Lastprofil (`bench_concurrency.py`): N Nutzer senden je 2 gestreamte Anfragen nacheinander, eindeutiger Präfix (kein
Cache-Treffer), 512 Ausgabe-Tokens, Temperatur 0.7, Thinking aus. „Lang" = ~1.230 Prompt-Tokens, „kurz" = ~120.

**Batching-Varianten (Gesamt-tok/s):**

| Konfiguration | 1 Nutzer | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Setup-Standard, solo (lang) | 92 | 88 | 90 | – | – |
| `parallel 8` (lang) | 86 | 177 | 171 | 174 | 176 |
| 16 Slots, Chunks 512, Anteil 2.0 (lang) | 96 | 209 | 248 | 234 | 246 |
| 16 Slots, Chunks 512, Anteil 2.0 (kurz) | 104 | 281 | 358 | 365 | 354 |
| 32 Slots, Chunks 512, Anteil 2.0 (kurz) | – | – | 319 | 342 | 341 |

**Slot-Feinabstufung (kurze Prompts, Nutzer = Slots / 1,5× Slots):**

| Slots | Zeilen pro Fenster | tok/s | tok/s (1,5×) | pro Nutzer | Experten im VRAM |
|---|---|---|---|---|---|
| **12** | 3 | **361** | 336 | **34** | 100 % |
| 16 | 4 | 348 | 358 | 24 | ~100 % |
| 20 | 5 | 336 | 327 | 19 | 99 % |
| 24 | 6 | 331 | 328 | 15 | 98 % |
| 28 | 7 | 328 | 328 | 13 | 96 % |

Mehr Nutzer als Slots warten nur länger (12 Slots, 24–40 Nutzer: 346–358 tok/s, TTFT p95 37 → 72 s).

**Prefill über die Prompt-Länge (1 Anfrage):**

| Prompt | Chunks 512 | `auto` | `auto:32768` | vLLM 0.31 |
|---|---|---|---|---|
| ~2K | **1.247** | 795 | 790 | 3.471 |
| ~8K | **1.913** | 1.260 | 1.202 | 3.498 |
| ~15K | **2.120** | 1.848 | 1.359 | 3.500 |
| ~28K | 2.239 | **2.289** | 1.368 | 3.505 |

Der Prefill ist Strata's Engpass unter Last: Ein Prompt, der in einen Chunk passt, wird am Stück gelesen und hält in
der Zeit alle Batch-Slots an.

**Weitere Varianten:**

| Variante | 1 Nutzer | Batch (12 Slots, kurz) | Anmerkung |
|---|---|---|---|
| UD-IQ4_XS, 4 GPUs | 117–121 | 361 | Referenz |
| UD-IQ4_XS, **3 GPUs** (`--trim-stage-weights`, Split 16/32) | 117 | 266 | vierte Karte zählt nur unter Last |
| **UD-Q4_K_XL**, 4 GPUs | 100 | 241 (lang: 150) | 97 % der Experten im VRAM, 1,9 GB im RAM |
| UD-IQ4_XS ohne MTP | 81 | – | |

PCIe (alle Karten ~13 GB/s über Riser) spielt bei IQ4_XS und Q4_K_XL praktisch keine Rolle: Experten liegen im VRAM,
beim Decode wandert pro Fenster nur ein kleiner Übergabepuffer von Karte zu Karte.

## Mehr als 8 Batch-Slots (Engine-Patch)

Strata begrenzt Batch-Slots auf 8 (ein Verify-Fenster hat höchstens 8 Zeilen). Mit Layer-Split laufen die Slots aber
in Gruppen durch die Karten, und jedes Gruppenfenster trägt nur `Slots / Gruppen` Zeilen. Der Patch in
`patches/0001-*.patch` erlaubt deshalb bis zu 8 Zeilen **pro Gruppe** (bei 4 GPUs bis 32 Slots) und dimensioniert den
Übergabepuffer zwischen den Karten pro Slot. Geprüft mit Strata's eigenem Exaktheitstest (`tools/batch_test16.py`,
`batch_test32.py`): **16 bzw. 32 Slots liefern Token für Token dieselben Ausgaben wie solo**; Batch-Decode 16 Slots
603 tok/s gegenüber 345 tok/s mit 8. Gebaut als `Strata/engine/strata-g3`. Kandidat für einen Upstream-PR.

## vLLM zum Vergleich

Qwen3.8-Flash-Next läuft in vLLM nur als W4A16-Checkpoint (`VnimanieAI/Qwen3.8-Flash-Next-W4A16`, 67 GB Experten +
102 GB N-Gram-Tabelle in bf16) und braucht auf 4× 24 GB:
- `--enable-expert-parallel` (640 / TP4 ist nicht durch die Gruppengröße 128 teilbar; nebenbei Marlin-MoE statt Triton),
- die N-Gram-Tabelle als **FP8** in Pinned-RAM (~51 GB statt 102 GB): `vllm/patch031/sitecustomize.py` (für 0.31; ältere
  Fassung für 0.29 in `vllm/patch/`), Skala aus `vllm/ple_scan.py` (absmax 0,0894 → 1,995e-4). Der Patch pinnt exakt
  (Torch rundet Pinned-Speicher auf Zweierpotenzen auf) und lässt den Triton-Lookup über uint8-Sichten laufen, weil
  Triton auf Ampere kein e4m3 laden kann. Start: `vllm/run_vllm031_flashnext.sh`.

| vLLM Flash-Next | 1 Nutzer | 12 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 0.29, Prefix-Caching an (lang) | 65 | – | 376 | 384 | – |
| 0.29, Prefix-Caching aus (lang) | 75 | – | 375 | 418 | 442 |
| 0.31, Prefix-Caching an (lang) | 75 | 366 | 386 | 398 | 398 |
| 0.31, Prefix-Caching an (kurz) | 78 | 499 | 622 | 706 | 710 |

vLLM hält höchstens ~22 Anfragen gleichzeitig (der gemeinsame KV-/Mamba-Blockpool ist voll), MTP passt nicht in den
VRAM (OOM bei 0.90 Auslastung, bei 0.86 kein Platz mehr für den KV-Cache).

## Dateien

| Datei | Inhalt |
|---|---|
| `start-strata.sh` | empfohlene Konfiguration starten |
| `run_strata_variant.sh` | eine Config-Variante starten, Benchmark fahren, stoppen |
| `bench_concurrency.py` | Last-Benchmark (OpenAI-kompatibel, Streaming, TTFT/Decode/Gesamt) |
| `series_final.sh` | letzte Messserie (Q4_K_XL, 3 GPUs, MTP aus, Nutzer 24–40, Prefill) |
| `configs/` | alle gemessenen Strata-Configs |
| `results/sweep.jsonl`, `results/*.txt` | alle Messergebnisse |
| `patches/` | Strata-Engine-Patch für mehr als 8 Slots |
| `vllm/` | vLLM-Startskripte und PLE-FP8-Patches |
