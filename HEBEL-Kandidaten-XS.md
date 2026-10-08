# Weitere Optimierungen: Hebel, an die wir noch nicht gedacht haben

Alle Prüfungen waren read-only. Die laufende Serie wurde nicht angefasst. Jede Gewinnzahl ohne Messbeleg ist eine **Schätzung**.

Der wichtigste neue Befund betrifft die Decode-Zeit. Der Teil „GPU-reach wait“ bleibt bei **7,5–9,1 ms pro Window**, auch wenn die CPU-Experten auf 3 ms fallen (engine-H13-res600.log:53-122). Die GPU-Seite ist also inzwischen eine Untergrenze in derselben Größe wie die CPU-Seite. Bisher haben wir nur die CPU- und PCIe-Seite optimiert.

## 1. Top-Hebel

| # | Hebel | Mechanismus | erwarteter Gewinn | Aufwand | Risiko | Testarm |
|---|---|---|---|---|---|---|
| 1 | **GPU-Decode-Kernels aus der gfx1151-Tabelle** (`STRATA_EXPERT_V2`, `TSUM`, `LFUSE`, `QFUSE`, `GDN_SPLIT`, `PLE_BATCH`, `MMVF_ROWS`, `ATTN_LANECELL`) | Diese Kernels sind laut Code bitgleich, werden aber nur für gfx1151 eingeschaltet (arch_defaults.cpp:14-37). Auf CUDA wurden sie nie getestet. `EXPERT_V2` passt genau auf dieses Pack: IQ3_S gate/up, IQ4_NL down, 2560/640 (iq_kernels.cu:3387-3399). | Schätzung 0 bis +4 % Decode, weil sie die GPU-Untergrenze senken. Kann auch negativ sein, da die Kernels auf AMD getunt sind. | config-only | niedrig, weil bitgleich | `--env STRATA_EXPERT_V2=1 --env STRATA_TSUM=1`, danach je ein Arm pro Variable |
| 2 | **Größere Prefill-Chunks** `--prefill auto:16384` / `auto:32768` | Jeder Chunk ab 1024 Tokens streamt alle Experten, die nicht im VRAM liegen (prefill.cpp:108-116). Bei 8192er Chunks heißt das 12 Komplett-Streams für 97K Tokens. Am Code steht eine 5090-Messung: 5.624 → 6.465 tok/s (generate.cpp:5126-5133). | Schätzung +10–25 % Prefill bei langen Prompts | config-only | Commit/VRAM; der Planer kann den Chunk still halbieren | `--set prefill=auto:16384 --workload longctx` |
| 3 | **Ursache des Kaltstarts messen** (`STRATA_PREFILL_TIMING=1`) | Der Prefill ist abhängig vom **Inhalt** langsam, nicht vom Neustart. Neuer Text läuft mit ~1200–1500 tok/s, bereits gelesener mit ~5800 (E3a: 91K in 75,6 s, Re-read 97K in 16,7 s, neuer 18K-Text wieder langsam). Verdacht: PLE-Zeilen-Lesezugriffe. Simuliert ~766K Misses bei ~13K reads/s, also ~20 ms pro 4-KB-Read. Die Zeile „PLE N ms“ steht in prefill.cpp:3343-3345. | Diagnose. Wenn bestätigt, wird der Weg zu „cold 91K 76 s → ~20 s“ realistisch (PLE-only-GGUF, siehe Abschnitt 4). | config-only | keins | `--env STRATA_PREFILL_TIMING=1` |
| 4 | **Gelerntes Expert-Profil** (`--expert-profile-save`, dazu das Coder-Profil oder ein Profil aus dem F6-Trace) | Nach jedem Neustart wärmt sich der Cache über mehrere Requests auf: Hit-Rate 82,0 → 87,0 → 93,6 → 95,7 % bei 68,5 → 100 → 126 → 142 tok/s (H13). Ein Profil mit `--expert-profile-save` wird bisher nie geschrieben, weil agent_bench mit taskkill /F beendet (run_variant.py:53-59). Deshalb braucht es `--expert-profile-save-every 1`. | Schätzung +10–30 % auf den ersten 2–5 Requests nach dem Start, Steady State etwa ±0 | config-only | niedrig; die Rechnung bleibt gleich | `--add expert-profile-save=… --add expert-profile-save-every=1 --workload session`, danach `--set expert-profile=<gelernt>` |
| 5 | **Reasoning-Loop-Bremse** `"reasoning_loop_recovery": "stop"` | Im Live-Log steht ein Loop mit 64.000 Tokens in 283 s und 99,8 % Suffix-Treffern, gefolgt von einem 64K-Re-read (strata-unsloth-ud-iq4_xs.log:1238/1243). Das sind ~7,7 % der Engine-Zeit der Session. Die Option ist standardmäßig aus (server.py:2202). | ~280 s pro Runaway-Loop (Schätzung), sonst 0 | config-only | False Positives bei repetitivem Code | `--cfg 'reasoning_loop_recovery="stop"'` (der Bench prüft nur Regressionen; der echte Gewinn zeigt sich erst im Live-Betrieb) |
| 6 | **CPU-Pool: SMT-Siblings, Worker-Zahl, Zen-5-Pfade** (`--pool-workers 6/11/15`, `STRATA_IQ_MT_MIN=1`, `STRATA_IQ_GATHER=1`, `STRATA_IQ_PREFETCH`) | Heute laufen Worker nur auf den geraden LPs (pool.cpp:139). Worker über die Kernzahl hinaus laufen ungepinnt (pool.cpp:457). Gruppen mit einem Token fallen auf ggml-AVX2 zurück (native_expert.cpp:83-85, 137-148; portabler Build, CMakeLists.txt:1095-1099). `IQ_GATHER` und `MT_MIN` wurden nur auf Zen 4 gemessen, dort -1 bis -5 % (iq_avx512.cpp:32-33, DETAILS.md:87-93). | Schätzung je 0–5 % Decode, Vorzeichen offen | config-only | pw15: 16 Spinner auf 16 LPs, Verlust möglich | siehe I11–I17 im Skript |
| 7 | **`--pcie-mode dma`** (optional mit `STRATA_DMA_BATCH=2`) oder `direct` | Heute legt `auto` einen Copy-Kernel in den Graph (generate.cpp:6201-6205). Laut Code kostet das ~1–3 % Decode auf IQ3_S. DMA würde die Kopie neben den VRAM-Hit-Kernel legen. | 1–3 % (Upstream-Messung), hier vielleicht etwas mehr (Schätzung) | config-only | Issue #31: Host hängt in cudaMemcpyAsync. Daher Soak-Lauf und Log nach „no progress“ durchsuchen. | `--add pcie-mode=dma --workload session` |
| 8 | **Deutsches Draft-Vocab** (`--mtp-draft-vocab`) | Gemessen: 10,67 % der Tokens eines deutschen Fachtexts liegen außerhalb des Draft-Subsets. Mit 563 zusätzlichen Tokens sind es 0,99 %. Bench: dec-de Akzeptanz 0,58 vs. dec-code 0,81. Bei Französisch ging es auf einer 5090 von 141 auf 158 tok/s (DETAILS.md:112-116). | Schätzung +4–8 % auf deutschen Antworten; en/Code 0–2 % | config-only plus CPU-Build der Datei | niedrig; CJK-Antworten bekommen dann keine Drafts mehr | Kontrollarm sofort: `--add mtp-draft-vocab=X:/…/data/draft_vocab_en.bin`. Der `de`-Arm braucht vorher Korpus und Build mit tools/draft_vocab.py. |
| 9 | **Tokenizer-Cache im Server** | server.py:2702-2706 tokenisiert bei jedem Turn den ganzen Prompt neu, mit reinem Python-BPE ohne Memo (strata_tokenizer.py:141-163). Offline gemessen: 106K Tokens in 0,324 s, mit Wort-Cache 0,094–0,122 s bei identischen IDs. Im Bench fallen bei 91–103K-Turns konstant 0,28–0,32 s Overhead zusätzlich zu read_ms an. | Gemessen offline: ~0,2–0,3 s TTFT pro Turn bei 100K | small-code | niedrig; ID-Parität mit chat_golden.json prüfen | nach dem Patch: `S1-tokcache --workload agent`, Metrik ttft − read_ms |
| 10 | **`STRATA_HC_Q8=1`** (Hyper-Connection-Projektionen als Q8_0 statt BF16) | ~1,26 GB BF16 pro Window werden zu ~0,67 GB Q8_0 (native_dense.cpp:270-291, fused_gr.cu:1647-1648). Laut UNSLOTH_Q4.md:257-260 schließt das auch einen Teil einer PPL-Lücke von 6–9 % (nur auf AMD gemessen). | Schätzung +1–2 % Decode, dazu möglicherweise Qualität | config-only | ändert Bits; kostet ~670 MiB VRAM (≈ -290 Slots) | `--env STRATA_HC_Q8=1`, Antwort-Korrektheit prüfen |

Weitere config-only-Arme mit kleinerer Erwartung stehen auch im Skript: `--short-read 0/16` (TTFT kurzer Turns, generate.cpp:608), `--lookup-chain 3` (generate.cpp:2094-2096), `--mtp-hnorm stream`, `--conversation-cache-slots 6` (in H12 waren die Slots die Grenze, nicht die MiB), `STRATA_Q6_PACKED=1` (legt eine **Kopie** an, ~+500 MiB laut native_mmvq.cu:1710-1738) und `--kv k8v4` für Long-Context und Commit.

## 2. OS- und Treiber-Einstellungen (nur mit deiner Aktion bzw. Freigabe, frühestens nach der Serie)

| Einstellung | Aktueller Stand (read-only geprüft) | Änderung | erwarteter Effekt |
|---|---|---|---|
| **VRAM-Verbraucher vom 5090 nehmen** | `display_active=Enabled`. Etwa 20 C+G-Prozesse liegen auf der 5090: Antigravity IDE, PowerToys, aquasuite, WebView2, PhoneExperienceHost, Meta/Virtual Desktop Monitor. memory.used 31996/32607 MiB. | Stufe A: diese Apps **vor dem Enginestart** schließen, denn die Engine bemisst den Cache beim Start. Stufe B: Monitor an die iGPU, iGPU im BIOS aktivieren. | Schätzung +300–800 MiB, also ~100–250 Slots und +1–3 % Decode. Gleicher Mechanismus wie kv-resident (+568 Slots, gemessen +6 %). |
| **Feste Auslagerungsdatei** auf C: | systemverwaltet (0/0), 38.912 MB, Commit-Limit 101,6 GiB. Event Id 26 „Nicht genügend virtueller Speicher“ am 07.10. 19:26 und 08.10. 04:53. | C:\pagefile.sys initial = max = 98304 MB, dann Neustart | kein tok/s-Gewinn. Rund **+58 GiB** Commit (korrigiert, nicht +64). Beseitigt die Ablehnungen während die Datei wächst. Voraussetzung für 262K + kv-resident. |
| **CUDA Sysmem Fallback Policy** = Prefer No Sysmem Fallback (strata.exe) | nicht lesbar ohne Admin; Treiber-Default erlaubt Fallback. Die Reserve-600/400-Arme melden „191 MiB bzw. 4 MiB VRAM free … LOW“. | NVIDIA-Systemsteuerung, Programmeinstellungen | Kein Speed-Hebel. Sichert ab, dass die vram-reserve-Ergebnisse gültig sind. Vorher read-only prüfen: steigt „Shared Usage“ (GPU Process Memory) während der Arme? |
| **Prozesspriorität AboveNormal** | strata.exe ist Normal; keine SetPriorityClass im Code; Popen ohne Flags (server.py:71-76) | `(Get-Process strata).PriorityClass='AboveNormal'` nach READY | Schätzung 0 % isoliert, mehr bei parallelem Tool-Load der Agents (nicht gemessen). Nicht während der Serie setzen. |
| **HAGS A/B** | HwSchMode=2 (an) | Reihenfolge 1/2/1/2 mit Neustarts | Schätzung ±3 % auf GPU-reach wait; Vorzeichen offen |
| **RAM 6400 MT/s, UCLK 1:1** | 6200 MT/s, Kit F5-6800J3445G32G | BIOS plus Stresstest ≥ 1 h | Schätzung +0,5–1,5 %, im Rauschen. Niedrige Priorität. |
| **Defender-Ausschlüsse** | Echtzeitschutz an; Ausschlüsse ohne Admin nicht lesbar | Ausschlüsse für strata.exe, Modell- und Pack-Ordner und den Session-Ordner | höchstens Sekunden bei Start und Restore, kein Decode-Effekt |
| **GPU-Interrupt-Affinität** weg von LP 0 | Pool-Worker 0 liegt auf LP 0 (generate.cpp:833-837) | MSI Affinity Policy auf LP 15, Neustart | Schätzung 0–2 %, spekulativ. Nur wenn der Prioritätstest Wirkung zeigt. |

## 3. Skript `series_i.sh`

Ein Hinweis dazu: series_h7.sh wartet ebenfalls auf `SERIES-H6-DONE` und belegt Port 8090. Wenn series_i nur auf H6 wartet, laufen zwei Serien gleichzeitig auf demselben Port. Deshalb wartet das Skript auf **SERIES-H7-DONE**.

```sh
#!/bin/sh
# Series I (2026-10-08): untested levers from the lever hunt (GPU kernels, CPU pool, PCIe mode, prefill chunk, drafting, profile, server).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --decode-runs 3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }
# H7 also waits on H6 and uses port 8090 -> chain behind H7, not H6
while ! grep -q SERIES-H7-DONE logs/series_h7.txt 2>/dev/null; do sleep 20; done

run I00-ref                                                         # reference
# --- GPU floor: gfx1151 bitwise kernels, never tried on CUDA (arch_defaults.cpp:14-37)
run I01-v2tsum    --env STRATA_EXPERT_V2=1 --env STRATA_TSUM=1      # grouped VRAM-hit kernel for IQ3_S/IQ4_NL (iq_kernels.cu:3387)
run I02-v2        --env STRATA_EXPERT_V2=1                          # V2 without TSUM
run I03-lfuse     --env STRATA_LFUSE=1                              # fewer launches, shared expert + KV append
run I04-qfuse     --env STRATA_QFUSE=1                              # q8_1 images written by producers
run I05-gdnsplit  --env STRATA_GDN_SPLIT=1                          # GDN step over 4 blocks per head
run I06-plebatch  --env STRATA_PLE_BATCH=1                          # verify-window PLE K/V at once (decode opt-in, verify.cpp:748)
run I07-mmvf      --env STRATA_MMVF_ROWS=1                          # bf16 multi-row GEMV
run I08-lanecell  --env STRATA_ATTN_LANECELL=1                      # decode attention, one cell per thread
run I09-q6p       --env STRATA_Q6_PACKED=1                          # packed Q6_K head (extra copy, ~+500 MiB -> watch slots)
run I10-hcq8      --env STRATA_HC_Q8=1                              # HC projections as Q8_0 (bit-changing, ~+670 MiB)
# --- CPU pool (Zen 5, SMT siblings idle today)
run I11-pw6       --add pool-workers=6                              # fewer barrier participants
run I12-pw11      --add pool-workers=11                             # 4 unpinned extra workers
run I13-pw15      --add pool-workers=15                             # 8 unpinned extra workers (may lose)
run I14-mtmin1    --env STRATA_IQ_MT_MIN=1                          # single-token groups via Strata AVX-512 instead of ggml AVX2
run I15-gather    --env STRATA_IQ_GATHER=1                          # AVX-512 gather (slower on Zen 4, untested on Zen 5)
run I16-pf4k      --env STRATA_IQ_PREFETCH=4096                     # prefetch distance (default 2048)
run I17-pf1k      --env STRATA_IQ_PREFETCH=1024
# --- PCIe experts: copy engine / direct instead of in-graph copy kernel (generate.cpp:6201-6205); grep log for 'no progress'
run I18-dma       --add pcie-mode=dma --workload session
run I19-dmab      --add pcie-mode=dma --env STRATA_DMA_BATCH=2 --workload session
run I20-direct    --add pcie-mode=direct
# --- prefill
run I21-pftiming  --env STRATA_PREFILL_TIMING=1                     # diagnostic: PLE ms of cold vs re-read (do not compare tok/s)
run I22-ref-long  --workload longctx                                # long-context reference
run I23-chunk16k  --set prefill=auto:16384 --workload longctx       # check 'prompt chunk auto:' line
run I24-chunk32k  --set prefill=auto:32768 --workload longctx
run I25-k8v4-long --set kv=k8v4 --workload longctx                  # commit/VRAM; check longctx answer
run I26-sr0       --add short-read=0                                # short tails via batched path
run I27-sr16      --add short-read=16
# --- drafting
run I28-vocab-en  --add mtp-draft-vocab=X:/dev/StrataQ38FN/Strata/data/draft_vocab_en.bin   # smaller head (control for a later de arm)
run I29-hnorm     --add mtp-hnorm=stream                            # draft norm per stream (acceptance probe)
run I30-chain3    --add lookup-chain=3                              # S becomes 7 automatically
# --- start-up warmth: expert profile
run I31-coderprof --set expert-profile=X:/dev/StrataQ38FN/Strata/data/expert-profile-coder.bin
run I32-profsave  --add expert-profile-save=X:/dev/StrataQ38FN/expert-profile-learned.bin --add expert-profile-save-every=1 --workload session
if [ -f expert-profile-learned.bin ]; then
run I33-proflearn --set expert-profile=X:/dev/StrataQ38FN/expert-profile-learned.bin   # compare hit rate of requests 1-3
fi
# --- server
run I34-loopstop  --cfg 'reasoning_loop_recovery="stop"'           # no-regression check only
run I35-slots6    --set conversation-cache-slots=6                  # expect neutral on this workload
run I36-ref-again                                                   # drift check
echo SERIES-I-DONE $(date +%H:%M:%S)
```

Lies bei I01–I10 nicht nur `dec_short_median`, sondern auch die „GPU-reach wait“ in den decode-timing-Zeilen. Bei I11–I17 zählt der „CPU“-Term. Bei I01–I08 lohnt ein Stichprobenvergleich der Token-IDs mit dem Referenzlauf, weil die Kernels als bitgleich gelten. Wenn die IDs abweichen, wäre das ein Fehler.

## 4. Geprüft und verworfen bzw. zurückgestellt

- **X8-mtp4s6** (`--spec 6 --mtp-max-t 4`): identisch zur aktuellen Konfiguration, weil Suffix-Drafts bereits S=6 mit MTP 4 setzen (generate.cpp:2088-2091).
- **SPEC-2/3/5** (Lookup-Fenster bis 8, WAYS 16): Der Bench löst kaum Suffix-Windows aus (~4 pro Request gegenüber ~67 in Produktion). Zuerst bräuchte agent_bench einen copy-lastigen Schritt (small-code), sonst sind falsch-negative Ergebnisse vorprogrammiert.
- **Profile-Fill-Fix über den base_-Guard (P7)**: Die Diagnose ist falsch, denn base_ ist gesetzt (expert_source.cpp:775). Die eigentliche Ursache: `read_ahead_enabled()` liefert auf Windows false (memory.cpp:115-117). Mit PrefetchVirtualMemory ist das ein lohnender small-code-Fix, der Startzeit spart (46 s → ~11–20 s, Schätzung), aber kein tok/s.
- **PLE-only-GGUF (P2), PLE-Cache über Neustarts (P4), frühe PLE-Reads (P5)**: zurückgestellt bis I21 zeigt, dass der PLE-Wait den Kaltstart dominiert. Für P2 zuerst `ple_reader_test --direct-only` laufen lassen (nur Disk).
- **Large Pages für das Complement (W1)**: Schon bei 25 GB gab es Error 1450 (engine-G1-coder.log). 37 GiB sind unrealistisch, und der Nutzen liegt bei maximal 1–6 %.
- **Größeres `--conversation-cache-mib` allein**: In H12 kamen alle 9 Evictions bei 3 Slots und nur ~0,75 GB parked. Die Slots sind die Grenze.
- **`STRATA_Q8_PACKED`**: zweite Kopie mit bis zu +2,9 GB VRAM (≈ -1250 Slots), netto wahrscheinlich negativ.
- **`--kv q4_0`**: +8–12 % Dokument-PPL (kv-q4 README:36-45).
- **`STRATA_RESIDENT_PIN=0`**: spart keinen Commit, verliert Zero-Copy und DMA, und das Complement kann ausgelagert werden.
- **`STRATA_IQ3S_MT1`, AVX2-Gather**: auf AVX-512-CPUs bzw. Nicht-Intel per Code deaktiviert (native_expert.cpp:93-94, expert_layout.cpp:210-229).
- **`--kv-grow`, `STRATA_EXCHANGE_ROTATE`**: inkompatibel mit `--resident-experts` (generate.cpp:3401-3408) bzw. mit ungleichen Blob-Größen der Layer 2, 4 und 30.
- **Multi-GPU- und HIP-only-Schalter** (`--layer-split`, `PEER_*`, `HIP_*`, `PF_GEMM`, …): Hier gibt es nur eine GPU, und sie ist CUDA.
- **`STRATA_PREFILL_EQUAL`**: Die Chunk-Zahl ändert sich für diese Prompts nicht (91K bleibt bei 12 Chunks).
- **`--ple-io ram` / `mmap`**: `ram` gibt es auf Windows nicht, `mmap` war der Pfad hinter allen 9 Stalls.
- **`STRATA_POOL_SPIN_US`, Timer-Auflösung**: Die Worker spinnen ohnehin durchgehend (pool.hpp:214), kein Sleep im Hot Path.
- **`--mtp-window`**: Der Draft kostet nur ~1–1,45 ms pro Window, auch bei 100K. Ein anderer Wert invalidiert außerdem Session-Dateien (generate.cpp:6355). Niedrige Priorität.
- **`STRATA_PF_FUSED`** auf IQ3: Upstream „etwa gleich“ (DETAILS.md:26-28). Erst sinnvoll gestapelt auf den Gewinner von I23/I24.
- **Nativer ggml-Build (P5)**: unter 1 %, weil VNNI nur die 5 Q8_0-down-Layer betrifft. Lohnt nur zusammen mit anderen Rebuilds.
- **CPU-Co-Compute im Prefill (P9)**: großer Code-Aufwand für höchstens 0,15–0,25 s, nicht bitgleich.
- **Coupled Draft (SPEC-9)**: nur relevant, wenn genesis3 mit temperature > 0 sendet. Der Bench sendet 0 (agent_bench.py:109). Zuerst den Client prüfen.
- **Kürzeres Reasoning (S5) und gemeinsamer Tools-Prefix für Sub-Agents (S4)**: plausibel und mit großem Hebel auf die Wall-Zeit, aber Workflow-Änderungen in genesis3. Der Bench kann sie nicht messen. Für S4 zuerst read-only messen, wie lang der gemeinsame Token-Prefix zwischen einem echten Sub- und Main-Request ist.

Dateien:
- X:\dev\StrataQ38FN\series_h4.sh (Stilvorlage)
- X:\dev\StrataQ38FN\series_h7.sh (Grund für die geänderte Warte-Bedingung)
- X:\dev\StrataQ38FN\Strata\strata-unsloth-ud-iq4_xs.json.optimized
- X:\dev\StrataQ38FN\run_variant.py:24-50 (`--set`/`--add`-Semantik)