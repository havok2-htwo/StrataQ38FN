# StrataQ38FN unter Linux: 4× RTX 3090

Messungen vom 2026-10-10/11 auf dem Linux-Host „Galactica": 4× RTX 3090 (je 24 GB, PCIe effektiv ~13 GB/s pro Karte),
Threadripper 3990X, 128 GB RAM (Budget für Tests: 96 GB), Treiber 580, CUDA 13.0. Strata v0.1.41 (für sm_86 selbst
kompiliert), Modell **Unsloth UD-IQ4_XS** (Target), zum Vergleich UD-Q4_K_XL, vLLM 0.29/0.31 und Qwen3.8-27B-FP8.

## Kurzfassung

| | Strata UD-IQ4_XS | vLLM 0.31 Flash-Next W4A16 | vLLM 0.31 Qwen3.8-27B FP8 |
|---|---|---|---|
| Decode, 1 Nutzer | **~121 tok/s** (MTP) | 79 tok/s; mit MTP 136 tok/s (dann max. 8 Nutzer) | 64 tok/s |
| Gesamt, 1,2K-Prompts | ~294 tok/s (16 Slots, Plateau) | ~400 tok/s | 409 tok/s (64 Nutzer) |
| Gesamt, kurze Prompts | ~405–416 tok/s (16–24 Slots) | ~710 tok/s | 838 tok/s (64 Nutzer) |
| Agentenlast, 16 Nutzer × 5 Turns (11K→28K Kontext): TTFT Folge-Turn | **~4,5–5 s** | ~32 s | – |
| Prefill | 1.250 (2K) → 2.240 tok/s (28K) | ~3.500 tok/s | bis ~3.100 tok/s |
| Qualität: MMLU-Pro (240) / GSM8K (150), Thinking aus | **77,9 %** / 94,0 % | 70,4 % / 94,7 % | 72,1 % / 95,3 % |
| VRAM | ~88 GB (alle Experten resident) | ~86 GB | ~63 GB (Preset 0.68) |
| RAM (echt) | **~3–5 GB** (Agenten-Config: + KV/Konv.-Cache) | ~64 GB (FP8-N-Gram-Tabelle) | ~100 GB (80 GB KV-Offload) |

Strata ist für **einzelne Nutzer und agentische Last** die beste Wahl: schnellster Single-Stream, höchste gemessene
Qualität, kurze Wartezeiten bei vielen Konversationen und fast kein RAM. vLLM liefert bei vielen gleichzeitigen
*neuen* Prompts mehr Gesamtdurchsatz, aber mit diesem W4A16-Checkpoint messbar schlechtere Qualität.

## Einrichtung (Linux)

```bash
git clone git@github.com:havok2-htwo/StrataQ38FN.git /media/raid0/StrataQ38FN
cd /media/raid0/StrataQ38FN
git clone https://github.com/Niko1221/Strata.git Strata
git -C Strata am ../linux/patches/*.patch            # >8 Batch-Slots + direkte Slot-Kopie (siehe unten); Engine danach neu bauen
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

`configs/T16-trim-split.json` (Durchsatz) bzw. `configs/X16-agent-direct.json` (Agentenlast) = Setup-Config plus:

| Einstellung | Wert | Warum |
|---|---|---|
| `--resident-experts` | an | alle Experten im VRAM, RAM-Kopie entfällt |
| `"layer_split"` + `--trim-stage-weights` | **`12,24,36`** + an | jede Karte lädt nur die Dense-Gewichte ihrer Layer und behält eigene Prompt-Puffer: keine Experten-Leihe pro Anfrage, **+14 % kurz / +16 % lang** |
| `"parallel"` | **16** (12–24) | Plateau ab 16; 24 Slots +2 %, alle Experten bleiben resident |
| `--prefill` | **512** | kurze Prompts 1,6× schneller als `auto` (1.247 statt 795 tok/s bei 2K), bis 15K besser |
| MTP (`--mtp`, `--spec 4`) | an | +45 % Decode (117 statt 81 tok/s), 61 % Draft-Akzeptanz bei Temperatur 0.7 |
| Engine | `engine/strata-g3b` | mit beiden Patches (>8 Slots, direkte Slot-Kopie) |
| Agentenlast zusätzlich | `--max-context 262144 --kv-resident 32768 --conversation-cache-mib 24576 --conversation-cache-slots 24 --prompt-cache 24 --prompt-cache-every 8192`, env `STRATA_KV_PREFETCH=1` | KV im RAM, Konversationen bleiben geparkt: Folge-Turns lesen nur den neuen Teil |

**Korrektur:** `STRATA_BATCH_DECODE_SHARE` (früher hier mit 2.0 empfohlen) ist bei Layer-Split mit Pipeline-Gruppen
**wirkungslos**. Der Code liest den Prompt dort am Stück (`generate.cpp:9706` upstream). Der damalige Gewinn kam allein von
`--prefill 512`.

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
603 tok/s gegenüber 345 tok/s mit 8. Gebaut als `Strata/engine/strata-g3`. Kandidat für einen Upstream-PR (`upstream/PR.md`).

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

Mit MTP (`SPEC=2`, `--max-num-seqs 8`, `--max-num-batched-tokens 1024`, Auslastung 0.93) passt es doch: **136 tok/s
single**, 490 tok/s bei 8 Nutzern (kurz), mittlere Akzeptanzlänge 2,2. Dann aber nur 65K KV-Tokens und max. 8 Nutzer.

vLLM hält ohne MTP höchstens ~22 Anfragen gleichzeitig (der gemeinsame KV-/Mamba-Blockpool ist voll), MTP passt nicht in den
VRAM bei 64 Sequenzen (OOM bei 0.90 Auslastung, bei 0.86 kein Platz mehr für den KV-Cache).

## Batching-Optimierung (Nacht 10./11.10.)

| Versuch | kurz | lang | Ergebnis |
|---|---|---|---|
| Referenz: 16 Slots, Split auto, Chunks 512 | 348–358 | 248 | |
| 16 Slots in 2 statt 4 Gruppen (8 Zeilen/Fenster) | 266–272 | 200 | ❌ halbe Karten idle |
| Decode-Anteil 4.0 | 366–375 | 259 | ➖ Rauschen (im Pipeline-Modus wirkungslos) |
| 2 Instanzen à 2 GPUs (Round-Robin) | 152–182 | 100 | ❌ halbe Experten im RAM, langsamer Prefill |
| **Split 12/24/36 + Trim, 16 Slots** | **405** | **288** | ✅ keine Puffer-Leihe mehr |
| … 20 / 24 Slots | 409 / 416 | 288 / 293 | ➖ Plateau |
| … Chunks 1.024 / 2.048 | 406 / 403 | 281 / 290 | ➖ |
| … + direkte Slot-Kopie (Patch 2) | 405 | 294 | ✅ klein (+2 %) |

Code-Analyse (4 Analysten + 4 Gegenprüfer über `generate.cpp`, `verify.cpp`, `server.py`): Der Rest der Lücke zu vLLM
liegt am blockierenden Prompt-Einlesen (Aufnahmen laufen einzeln und leeren die Pipeline) und am Fixkosten-Anteil pro
Batch-Fenster (~6,6 ms statt ~2 ms Bandbreitenlimit). Geschätzter Engine-Umbau: überlappende Aufnahme 4–10 Tage, MTP im
Batch bei Layer-Split 8–16 Tage, Fenster-Overhead 5–14 Tage. Realistisches Ziel danach ~450–550 / ~300–350 tok/s,
vLLM-Niveau eher nicht. Upstream-Texte dazu: `upstream/ISSUE.md`, `upstream/PR.md`.

## Agentenlast (Multi-Turn)

`bench_multiturn.py`: N Nutzer, je 5 Turns, Turn 1 mit ~8K-Dokument, jeder weitere Turn hängt Antwort + ~3K neue
Tokens an (Kontext 11K → 28K); die Nutzer laufen verschränkt wie Haupt- und Sub-Agenten.

| | 8 Nutzer: tok/s | 8: TTFT Folge-Turn | 16 Nutzer: tok/s | 16: TTFT Folge-Turn |
|---|---|---|---|---|
| Strata Trim-Config (32K, ohne Konv.-Cache) | 49 | 7,1 s | 43 | 10,4 s |
| **Strata Agenten-Config** (262K, KV im RAM, 24 GB Konv.-Cache) | 57 | **4,8 s** | 73 | **5,0 s** |
| Strata Agenten-Config + direkte Slot-Kopie | – | – | 62* | **4,5 s** |
| vLLM 0.31 (Prefix-Cache + 20 GB RAM-KV-Stufe) | **89** | 6,4 s | 77 | 32 s |

\* ohne vorgewärmten Konversations-Cache (der 73er-Lauf profitierte im ersten Turn vom vorherigen 8er-Lauf).

## Qualität (eigenes Set)

`eval/quick_eval.py`: 240 MMLU-Pro-Fragen (alle 14 Fachgebiete) + 150 GSM8K, greedy, Thinking aus, identisch für alle.

| Modell / Setup | MMLU-Pro | GSM8K |
|---|---|---|
| **Flash-Next UD-IQ4_XS (Strata)** | **77,9 %** | 94,0 % |
| Qwen3.8-27B FP8 (vLLM) | 72,1 % | 95,3 % |
| Flash-Next W4A16 + FP8-N-Gram-Tabelle (vLLM) | 70,4 % | 94,7 % |
| Flash-Next RAZOR-65B (50 % Experten gepruned, NVFP4, vLLM) | 62,5 % | 92,7 % |

Bei MMLU-Pro liegt nur Strata bei 23 Fragen richtig, nur vLLM-W4A16 bei 5 (signifikant). Ursache vermutlich der
Drittanbieter-W4A16-Checkpoint oder vLLMs Ampere-Pfad, nicht MTP (bei Greedy verlustfrei). RAZOR wurde danach verworfen.

## Direkte Slot-Kopie (Engine-Patch 2)

`copy_to_slot`/`copy_from_slot` kopieren den Slot-Zustand jetzt direkt auf der GPU (`cudaMemcpyAsync`, ein Sync pro
Karte) statt über ein Host-Abbild mit Sync pro Layer: 65,6 → 34,0 ms pro Aufnahme (reine Kopie ~8 ms), bit-identisch zum
Solo-Lauf (`batch_test16.py` 16/16, `batch_interleave_test.py` inkl. Folge-Turn aus dem Slot), zwei Reviews ohne Befund
außer einem Fehlerpfad (behoben). Abschaltbar mit `STRATA_SLOT_COPY_DIRECT=0`.

## Dateien

| Datei | Inhalt |
|---|---|
| `start-strata.sh` | empfohlene Konfiguration starten |
| `run_strata_variant.sh` | eine Config-Variante starten, Benchmark fahren, stoppen |
| `bench_concurrency.py` | Last-Benchmark (OpenAI-kompatibel, Streaming, TTFT/Decode/Gesamt) |
| `series_final.sh` | letzte Messserie (Q4_K_XL, 3 GPUs, MTP aus, Nutzer 24–40, Prefill) |
| `configs/` | alle gemessenen Strata-Configs |
| `results/sweep.jsonl`, `results/*.txt` | alle Messergebnisse |
| `patches/` | Strata-Engine-Patches: mehr als 8 Slots, direkte Slot-Kopie |
| `bench_multiturn.py` | Agenten-/Multi-Turn-Benchmark |
| `eval/` | Qualitäts-Set (MMLU-Pro + GSM8K) mit Antworten und Ergebnissen |
| `upstream/` | Issue- und PR-Text für Niko1221/Strata (noch nicht gepostet) |
| `quant-survey/` | Auswertung aller 910 Flash-Next-Repos auf Hugging Face |
| `vllm/` | vLLM-Startskripte und PLE-FP8-Patches |
