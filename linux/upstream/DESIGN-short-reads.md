# Empfehlung: kürzere Prompt-Reads bei vielen Nutzern

*Nur gelesen und gerechnet, ohne GPU-Messung und ohne Änderungen am Code. Zeilennummern: `Strata-ll4-lanes` @ ee403737. P = `src/prefill/prefill.cpp`, G = `src/program/generate.cpp`.*

## Kurzfassung

1. **Zuerst Approach 3, Stufe T0, zusammen mit Approach 1 P1** (das sind dieselbe Idee) und dem kostenlosen `--short-read`-A/B-Test. Alles ist bitgenau, das Risiko ist überschaubar, Aufwand ca. 30–35 h. Das bringt bei 16 Nutzern ca. **+4–8 % Gesamtdurchsatz** und **−20 % bei p95/max TTFT**.
2. **Danach erst entscheiden**, und zwar anhand von Messungen. Zur Wahl stehen das Packen mehrerer Prompts in einen Chunk (Approach 2, als v0) oder kurze Deltas komplett in den Gruppenfenstern (Approach 1 P2). Nur diese beiden teilen sich den Expert-Sweep von ca. 4 × 17 ms pro Read, und nur sie können die Lücke zu vLLM (637 tok/s) schließen.
3. **Ehrliche Erwartung:** T0 + P1 allein erreichen vLLM bei 16 Nutzern mit kurzen Prompts **nicht** (ca. 555–585 statt 637 tok/s). Mit Packing bzw. P2 sind ca. 600–650 tok/s realistisch, aber unsicher.

## Geprüfte Aussagen im Code

| Aussage | Befund |
|---|---|
| Ein Batch-Fenster hat höchstens 8 Zeilen | `kVerifyMaxT = 8` (include/strata/kernels/verify_kernels.hpp:21) ✔ |
| Der GPU-gruppierte MoE-Pfad läuft erst ab T ≥ 1024 | `stream_all_min()` gibt standardmäßig 1024 zurück (P:172-176), `stream_all` (P:2561) ✔ |
| Pro Layer gibt es eine Host-Rundreise | `d2h_sync` der Routing-IDs (P:3353) und `cudaStreamSynchronize(m.cs)` (P:3407) ✔ |
| Beim Hand-off wartet die Stage auf den ganzen Chunk | `cudaEventSynchronize(hand_done)` (P:4458) ✔ |
| Ein Chunk mit ≤512 Tokens läuft seriell durch die 4 Stages | `single_chunk = n <= m.T` (P:2330) ✔ |
| Jede BGEN-Zeile leert zuerst die Pipeline | `pipe_drain()` für alles außer BSTOP/BYIELD (G:13231-13236) ✔ |
| Der Overlap-Gate lehnt ab, solange Gruppen laufen | `if (pipe_inflight()) … return false` (G:14847) ✔. Damit ist T0c eine echte Gate-Änderung, nicht nur Konfiguration. |
| Der Hand-off ist im Prod-Betrieb 16 Zeilen breit | `hrows = max(kVerifyMaxT, o.batch)`, also 16 (G:8663) ✔. Für P1 muss er auf 32 Zeilen. |
| Mehrere Zeilen pro Slot erzwingen den Commit-Graph | `batch_self_commits` verlangt `one_row_per_slot` (verify.cpp:228) ✔ |
| Der Checkpoint-Pool liefert manchmal neue Vektoren | `ckpt_pool.take()` (G:9740), im Log „new vectors“ (G:9810) ✔ |
| `--short-read` steht standardmäßig auf 64 | G:944 ✔ |

**Messwerte** (`linux/results/lanes/prefill.txt`):

| Prompt | Read | send→BADM |
|---|---|---|
| 170 Tokens, GPUs frei | 139–155 ms | 190–208 ms |
| 170 Tokens, 15 Slots dekodieren | 236–254 ms | 315–387 ms |

Unter Last kommen also **ca. 80–130 ms Verwaltungsaufwand außerhalb des Reads** dazu. Genau diesen Teil adressiert T0. Er ist größer, als Approach 3 annimmt (50–75 ms).

## Bewertung der drei Entwürfe

| | Gewinn bei 16 Nutzern, kurze Prompts | Aufwand | Risiko | Exaktheit |
|---|---|---|---|---|
| **A3 T0 (a–d)** | +3,5–7 % Gesamtdurchsatz, max TTFT 5,0 → ca. 4,0 s | 26–30 h | mittel (T0c/T0d berühren Capture und Lanes) | bitgenau |
| **A1 P1** (≡ T0d) | Prompt-Ende ins Gruppenfenster; spart 30–45 ms pro Admission | 20–24 h (überschneidet sich mit T0d) | mittel: Zuordnung der Ausgaben zu den Slots, VRAM für Graphen | bitgenau |
| **A3 T1** (MoE ohne Host-Sync) | im Leerlauf −20–28 ms pro Read, unter Last nur 5–15 ms | 16–24 h | mittel: Puffergrößen, GROUP_SORT muss exakt nachgebaut werden | nur bitgenau, wenn die Reihenfolge exakt reproduziert wird |
| **A2 Packing** | +5–12 % Gesamtdurchsatz; stark bei Bursts (Kaltstart 3–5 s → 1–1,4 s, Sub-Agenten ca. 3× schneller) | 80–100 h (v0 ca. 40 h) | hoch: Isolation zwischen Sessions, Teilfehler, Lanes | bitgenau erreichbar (Schalter pro Segment-T) |
| **A1 P2** (Deltas im Fenster) | Obergrenze 650–690 tok/s, optimistisch | 25–35 h zusätzlich | hoch: Protokoll (BADM vor dem ersten Token), Burst-Policy | bei W ≤ 64 bitgenau, bei W > 64 **nicht** |

**Anmerkungen:**

- **P1 und T0d sind dieselbe Änderung.** Beide verlegen die letzten 7 Prompt-Tokens plus die erste Token-Zeile in das nächste Fenster der eigenen Gruppe, statt ein Solo-Fenster zu fahren, für das alle Gruppen geleert werden. Das wird einmal gebaut, in der Variante von P1 (`batch_launch` mit Zeilenliste, Hand-off 32 Zeilen, vorab gecapturte Layouts).
- **Bei P2 ist die Prognose über 637 tok/s die schwächste Aussage.** Sie beruht auf Grenzkosten pro Zeile im Leerlauf (0,33 ms pro Zeile und Stage). Unter Last treffen zusätzliche Zeilen mehr verschiedene Experten. Außerdem liegt die Decode-Obergrenze bei 717 Zeilen/s, und P2 verbraucht genau diese Zeilen für Prompts. Realistisch ist eher **ca. 600–640 tok/s**.
- **Approach 2 lohnt sich vor allem bei Bursts.** Im gleichmäßigen Betrieb mit 16 Nutzern liegt die mittlere Packgröße nur bei ca. 1,5–2. Bei Sub-Agent-Bursts und beim Kaltstart (16 Prompts gleichzeitig) ist es der klare Gewinner.
- **T2** (Sub-Chunk-Pipelining) hilft nur im Leerlauf, ist nicht bitgenau und verbraucht ca. 60 % mehr GPU-Arbeit. Das lasse ich vorerst weg.

## Reihenfolge der Umsetzung

**Phase 0: Messung, ca. 2 h GPU**
1. Prod-Binary mit `STRATA_PREFILL_TIMING=1` und `STRATA_TRACE=1` laufen lassen: Reads mit 170, 512 und 1.2K Tokens, einmal im Leerlauf und einmal mit 15 dekodierenden Slots. Das soll bestätigen, ob sich eine Stage in ca. 17 ms Expert-Sweep, 13 ms Rechnen und 7 ms Verwaltung aufteilt. Davon hängt ab, ob sich T1 lohnt.
2. A/B-Test `--short-read` 16/32 gegen 64 mit `bench_multiturn` (Agenten-Mix). Das kostet keinen Code.

**Phase 1: exakt, mit niedrigem Risiko, ca. 12 h**
- **T0a:** Checkpoint-Pool vorwärmen. Mindestens 2 Reserve-Checkpoints, Nachfüllen asynchron. Spart 7–14 ms.
- **T0b:** Turn-Checkpoint nach BADM ziehen, mit Fence auf `gdn_state`, QSA-K/V, `ple_prev` und auf die Wiederverwendung des Prefix. Spart 5,5–20 ms.
- **T0c:** BGEN ohne Drain, wenn ein Dry-Run zeigt, dass keine Layouts gecapturt werden müssen (`layouts_ready`, `prefill_records_ready`). Dafür muss der Gate G:14847 Gruppen in Flight erlauben. Spart ca. 20 ms.

**Phase 2: T0d ≡ P1, ca. 20 h**
- Statt der Solo-Fenster aller Gruppen wartet die Admission nur noch auf die Zielgruppe. Das Prompt-Ende läuft als zusätzliche Zeilen im Gruppenfenster, BADM kommt mit der ersten Ausgabe.
- Die Variante mit 4 Layouts und aufgefüllten Zeilen spart VRAM. Auf CUDA1 sind nur ca. 657 MiB frei.

**Entscheidungspunkt nach Phase 2:** Wenn der gleichmäßige Betrieb bei ≥ 580 tok/s liegt und Bursts das eigentliche Problem sind, kommt **Approach 2 v0** (ca. 40 h: nur leere oder frische Slots, ohne Merge-First). Wenn kurze Agenten-Follow-ups dominieren, kommt **P2 mit W ≤ 64** (bitgenau). T1 nur dann, wenn Phase 0 mindestens 5 ms Verwaltung pro Stage zeigt.

## Erwartung bei 16 Nutzern mit kurzen Prompts (vLLM 637 tok/s, TTFT p50 0,62 s)

| Stand | Gesamtdurchsatz | TTFT p50 / p95 / max |
|---|---|---|
| heute live (1441e-lanes, 28K) | 522–562 tok/s | 0,35–0,37 / 4,5 / 5,0 s |
| + Phase 1 | ca. 535–570 | 0,34 / ca. 4,1 / ca. 4,6 s |
| + Phase 2 (T0d/P1) | ca. 555–585 | 0,31 / ca. 3,6 / ca. 4,0 s |
| + Packing v0 **oder** P2 | ca. 590–640 (unsicher) | p95/max beim Kaltstart deutlich besser (bis ca. 1,4 s) |

Die Zeit bis zum ersten Token ist heute schon besser als bei vLLM. Beim Gesamtdurchsatz liegt vLLM mit kurzen Prompts erst nach Phase 3 in Reichweite.

## Testplan (für jede Stufe)

1. **Exaktheit**, bei 1, 2, 4 und 16 Nutzern:
   - `batch_test16`-Dumps;
   - `lanes/cmp_m1a_exact`, `cmp_hold*`, `lanes_http_ident`;
   - `ovkv_ident.py`;
   - HTTP-Vergleich der Antworten 32/32.
   - Für Packing zusätzlich: ca. 200 Prompt-Sets, gepackt gegen seriell (K = 1–4, Längen 10–512), max |Logit-Differenz| = 0. Dazu ein Isolationstest: Ein Segment wird gestört, die anderen müssen bitgleich bleiben.
2. **Leistung:**
   - `ab_quick` ABBA ×2 mit 16 Nutzern, kurz und lang (je ca. 1 h);
   - TTFT bei 1, 2 und 4 Nutzern (darf 0,47 s nicht verschlechtern);
   - `prefill.txt` wiederholen, im Leerlauf und mit dec15.
3. **Interplay:** BYIELD bzw. BSTOP während eines Reads, Spec-Lanes (M1a/M1b), Slot-Park, KVS-Staging über 28K, VRAM-Reserve auf CUDA1.
4. **Soak** 2 h, danach Redeploy der schnellsten bitgenauen Version (ein Restart von `strata.service` ist erlaubt). Zwischen den Stufen bleibt der aktuelle Prod-Build (`strata-g3-1441e-lanes`) live.

**Gesamtaufwand bis zum Entscheidungspunkt:** ca. 35 h Code plus ca. 10 h Tests. Mit Phase 3 kommen ca. 40–60 h dazu.