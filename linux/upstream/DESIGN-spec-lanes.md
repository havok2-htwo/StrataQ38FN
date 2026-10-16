# Empfehlung: Batch-MTP für 2–4 Nutzer

## Entscheidung

**Wir bauen „Spec Lanes“, und zwar zuerst mit L = 2, also auf den Slots 0 und 4.** Bei 2–4 aktiven Nutzern läuft dabei jeder Nutzer als eigener, vollständiger Solo-MTP-Strom. Die Ströme wechseln sich auf den 4 Karten ab, statt wie heute im Batch ohne Spekulation zu laufen.

- **Warum dieser Weg:** Im Solo-Betrieb ist jede Karte etwa 75 % der Zeit untätig. Ein Fenster dauert ~18,5 ms; davon belegen die Stufen S1–S3 je ~4,35 ms und S4 ~5,9 ms, weil dort zusätzlich der Draft läuft. Ein zweiter Strom passt fast vollständig in diese Lücken.
- **Strukturell am saubersten:** Jede Lane hat einen eigenen `Verifier` mit eigenem Scratch, eigenem Hand-off und eigenem `final_R`. Damit entfallen die schwierigen Teile der beiden anderen Entwürfe:
  - Commit-Scratch, der überschrieben wird,
  - neue Row-Layouts,
  - die Grenze `kVerifyMaxT = 8`.
- **Exaktheit am stärksten:** Es laufen dieselben Graphen wie solo, also sind die Tokens per Konstruktion gleich den Solo-Spec-Tokens.
- **Bedingung:** Es lohnt sich nur, wenn M0 zeigt, dass der `pl_*`-Pfad auf CUDA/sm_86 funktioniert. Der Pfad ist heute auf HIP ausgelegt und lief bei uns noch nie in Produktion.
- **Plan B, falls M0 scheitert:** „Lead-Slot MTP“ (Entwurf 1). Er bringt bei 2 Nutzern fast denselben Gewinn, kostet 18–26 h, braucht aber einen Stash für den Commit-Scratch pro Gruppe, und das ist eine Korrektheitsfalle.
- **Verworfen:** PB-MTP (Entwurf 2). Es kostet 35–50 h und bringt nur bei 8 Nutzern mehr. Rechnet man mit der gemessenen Akzeptanz von ~0,61 statt der angenommenen 0,78, bleiben dort nur etwa +5 %.

Die zitierten Stellen habe ich nachgeprüft:
- `generate.cpp:4043-4047`: Auto ist auf CUDA immer aus.
- `generate.cpp:4063`: `"not with --batch slots"`.
- `generate.cpp:4177`: `mtp_first` braucht `batch == 0`.
- `generate.cpp:4283`: `kPipeWindowMib = 160`.
- `verify.hpp:269-306`: die `pl_*`-API.

**Lane-Modus braucht ein eigenes Gate.** Ein Flag-Flip reicht nicht.

## Was aus den anderen Entwürfen übernommen wird

1. **Kein Server-Umbau in v1.** `server.py:980-982` legt Nutzer 1–4 schon auf die Slots 0, 4, 8 und 12. Es kommt nur eine Zeile `INFO spec_lanes=2 lane_slots=0,4` dazu.
2. **Striktes Gate:**
   ```
   lanes_on = 1 ≤ active ≤ L && alle aktiven Slots ∈ {0,4}
              && keine Bild-Slots && keine Penalties
              && kein STRATA_SPEC_PROB / coupled
   ```
   In allen anderen Fällen läuft das heutige `pump()` unverändert. Umgeschaltet wird nur nach `pipe_drain()` bzw. `lane_drain()`.
3. **Eager-Commit beim Verdict.** `pl_commit_async(a+1)` wird auf allen 4 Stufen ohne Host-Sync in die FIFO der Karte gestellt. Die Begründung, warum das ohne Race geht, kommt aus PB-MTP: pro Verdict ein Host-Write, und der ist durch die Stream-Reihenfolge vor dem nächsten Fenster dieser Lane verbraucht.
4. **Fertig-Erkennung per Event pro Fenster.** Der Commit einer Lane darf den Hand-off der anderen nicht verzögern. Das ist dasselbe Problem wie `cudaStreamQuery` in `verify.cpp:3476`.
5. **Slot-Drafter aus dem `--batch-mtp`-Code wiederverwenden** (`generate.cpp:5262-5285`, `8611-8630`). Gate dafür ist `batch_mtp || is_lane(b)`.
   - Draft-K/V-Kopie bei Aufnahme (`generate.cpp:10925-10938`) und bei Rückgabe (`10975-10986`), `draft_first` (`15588-15596`).
   - `set_max_drafts(spec-1)`, `set_force_capture(true)`, `prepare_chain`. Das Draften läuft async über `chain_launch` / `chain_poll` (`mtp.cpp:1760-1840`).
6. **Drafter aktuell halten, solange das Gate aus ist:** T=1-Catch-up auf den Lane-Slots. So bricht die Akzeptanz nach einem Wechsel von Pump zu Lane nicht ein.
7. **VRAM bezahlen statt Expert-Loans zu riskieren.** Die Loans haben in TESTLOG #78/#79 etwa −5 % gekostet. Pro Lane und Karte kommen +160 MiB dazu, bei L = 2 also 320 MiB pro Karte. Finanziert wird das mit `--batch 14` (spart 0,44–0,68 GiB pro Karte) oder über die Reserve. Danach müssen die Loans neu gemessen werden.
8. **Später:** eine lastabhängige Draft-Tiefe (z. B. d=3 bei 2 Lanes, kleiner bei 4) und den Drafter auf CUDA0 verlegen (`mtp_first`, heute nur HIP und nur bei `batch==0`, `generate.cpp:4172-4182`).

## Erwartete Geschwindigkeit pro Nutzer (tok/s)

Kalibriert habe ich auf den Solo-Fenstern aus dem Prod-Log: 18,5 ms und 2,2–2,4 Tokens pro Fenster, Akzeptanz ~0,61–0,63 pro Draft. Dazu kommen eine Simulation pro Karte als FIFO und ein Abschlag von 5–12 % für Host-Polling und `pl_*`-Overhead. **Hinweis:** Die Log-Probe (512 Tokens in 4379 ms) ergibt 117 und nicht 125 tok/s. Die Ziele beziehen sich deshalb immer auf die Solo-Rate, die im selben Lauf gemessen wurde.

| Nutzer | heute | Spec Lanes L=2 | mit L=4 (später) | Plan B Lead-Slot |
|---|---|---|---|---|
| 1 | 125 | 125 (Solo, unverändert) | 125 | 125 |
| 2 | 77–86 | **100–115 (+25–40 %)** | 100–115 | 100–108 |
| 4 | 75 | 75 (Gate aus) | **85–95** (S4 ist der Engpass) | 82–88 |
| 8 | 56 | 56 (Pump) | 56 | 56 |
| 16 | 34 | 34 (Pump) | 34 | 34 |

- **Obergrenze bei 4 Lanes:** S4 ist bei ~5,9 ms pro Fenster der Engpass, das deckelt die Summe bei ~400 tok/s. Ab 5–6 Nutzern ist der Batch-Pump wieder gleich gut oder besser. Lanes über 4 lohnen sich deshalb nie.
- **Auf den Verkehr gewichtet:** Produktion läuft meist mit ~2 Strömen, das ergibt etwa **+30 % pro Strom**.

## Folgen für die Exaktheit

- **Gate aus** (env nicht gesetzt, mehr als L aktiv, Nicht-Lane-Slots): Der Codepfad ist identisch zu heute. 8 und 16 Nutzer bleiben bitgleich, `batch_test16` bleibt gültig.
- **Tokens in Lanes:** Es laufen dieselben T-Row-Solo-Graphen, derselbe Commit, die Exact-Match-Annahme (`generate.cpp:15284`) und Philox(seed, position) (`verify.cpp:2543-2553`). Damit gilt **Tokens = Solo-Spec-Tokens**, und die bisherigen Token-Referenzen bleiben gültig.
- **Zustand:** Der GDN/KV-Zustand eines Lane-Slots folgt künftig dem Solo-Spec-Pfad und nicht mehr dem 1-Row-Batch-Pfad. Das hat zwei Folgen:
  - Die Pfadabhängigkeit aus #59(4) wird für Sitzungen mit ≤ 2 Nutzern kleiner.
  - Sitzungen, die die Schwelle überschreiten (teils Pump, teils Lane), bekommen einen dritten numerischen Pfad.
- **Neu nötige Referenzen:**
  - fortgesetzte Multi-Turn-Dumps aus Lane-Slots
  - Dumps über den Wechsel Lane↔Pump
  - RESUME- und Park-Images aus Lane-Slots
  - HTTP-sha256-Sets bei 2 Nutzern
- **Mehr VRAM verschiebt die Expert-Caches.** Ein Near-Tie kann dadurch gegen alte Referenzen kippen. Für A/B-Läufe deshalb immer `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` setzen und die Caches per `--vram-reserve-mib` angleichen (`docs/MULTI_GPU.md:290-293`).

## Meilensteine

**M0: Machbarkeit von `pl_*` auf 3090 (kein Code, ~60 GPU-min, Restart nötig)**
- Prod-Config mit `--batch 0 --pipeline-windows 2`, 1 Nutzer, `STRATA_PIPELINE_DEBUG=1` + Trace. Eine explizite Anforderung wird auf CUDA nicht abgelehnt (`generate.cpp:4054-4070`).
- Messen:
  1. Greedy-Gleichheit gegenüber der seriellen Solo-Schleife, 160 Tokens, mit den Determinismus-Flags von oben.
  2. Fensterzeit und Commit-Zeit pro Stufe. Ziel ≈ 4,1 ms pro Stufe und ≈ 5,9 ms auf CUDA3 inklusive Draft.
  3. tok/s gegen die Solo-Rate desselben Laufs.
  4. VRAM-Zuwachs pro Karte (erwartet +160 MiB) und ob dabei Expert-Loans entstehen.
- Ohne GPU, parallel aus den Prod-Logs: Anteil der Slots mit `kv_mode==1` (Kontext über 32K) und die Akzeptanzverteilung.
- **Go**, wenn die Tokens gleich sind und keine Hänger auftreten. **No-go** führt zu Plan B; dessen M0 läuft mit `--batch-groups 1 --batch-mtp STRATA_BATCH_MTP_SERIAL=1` bei 2 Nutzern (~30 min).

**M1: Engine-Prototyp `STRATA_SPEC_LANES=2` (8–10 Agent-h + ~1 GPU-h)**
- Lane-Verifier, aufgebaut wie der `ver_b`-Block in `generate.cpp:8447-8600`:
  - `set_stream(stage_verifier(k).stream())`
  - `init` auf `bslot_ss[k][LS[j]]`
  - `capture_all` beim Start
- Zwei Lane-Drafter auf CUDA3, `lane_pump()` neben `pump()` (`generate.cpp:11671-11826`): Verdict, Emit mit EOS-, `max_new`- und Kontext-Prüfung pro Token, Eager-Commits auf 4 Stufen, `chain_launch`, `min_p`-Schnitt.
- Nur aktiv, wenn genau die Slots 0 und 4 aktiv sind. Bei jeder Aufnahme wird gedraint.
- **Testfenster misst:**
  - tok/s pro Nutzer bei 2 Nutzern mit `bench_multiturn.py`. Ziel **≥ 100** (heute 77–86).
  - Token-Gleichheit pro Slot gegen Solo-`GEN`, greedy und T=0.7 / seed 1234.
  - `batch_test16` mit ausgeschaltetem env, bitgleich.

**M2: produktionsreif (8–12 h + 2–3 GPU-h)**
- Draft-K/V bei Aufnahme und Rückgabe, T=1-Catch-up bei ausgeschaltetem Gate, Moduswechsel, `set_always_publish` / `apply_pending` wie im Pump (#463).
- VRAM-Finanzierung (`--batch 14` oder Reserve), Abbuchung bei `generate.cpp:5586-5590` / `5719-5724`.
- **Testfenster misst:**
  - HTTP-Bench mit 1/2/4/8/16 Nutzern. Erwartung: 1, 8 und 16 Nutzer ±0; 2 Nutzer +25–40 %.
  - Expert-Loans vorher und nachher.
  - HTTP-sha256 bei 2 Nutzern und neue Multi-Turn-Referenzen.
  - Soak von ≥ 30 min mit wechselnder Last von 1 bis 4 Nutzern.
- Deploy erst nach deinem „deploy ja“.

**M3, optional (6–10 h):** L=4 mit lastabhängiger Draft-Tiefe, danach den Drafter auf CUDA0 portieren. Das zielt auf 95–110 tok/s bei 4 Nutzern.

**Summe:** ~22–30 Agent-h + 4–6 GPU-h (ohne M3).

## Risiken

1. **`pl_*` auf sm_86 ist unerprobt** (Doorbells, Graph-Capture). M0 klärt das für ~1 h GPU-Zeit.
2. **VRAM:** 320 MiB pro Karte bei L=2. Ohne `--batch 14` drohen Loans und −3 bis −5 % auf allen Lastniveaus.
3. **Das Modell ist optimistisch:** ein einziger Host-Thread pollt beide Lanes, dazu kommen Phasenkollisionen und Takt-Effekte bei doppelter Kartenlast. Realistisch landet der Wert eher in der unteren Hälfte von 100–115.
4. **CUDA3-Konkurrenz zwischen Draft und dem S4-Fenster der anderen Lane.** Beides wird per Event serialisiert, denn paralleles Laufen kostete auf R9700 etwa 2× (`generate.cpp:4172-4176`).
5. **Zustandsübergabe Slot↔Lane ist ungetestet:** zusammen mit `STRATA_QSA_SLOTS`, `--kv-resident`-Streaming, Self-Commit und Slot-Park. `Verifier::init` bindet `ss_` einmalig (`verify.cpp:597`), eine Lane ist also fest an ihren Slot gebunden.
6. **Hänger bei der Prompt-Aufnahme (bis 27 s) bleiben.** Das ist der orthogonale Punkt 2, die Prompt-Aufnahme parallel zum Decode.
7. **Neue Zustandspfade:** Fortgesetzte Turns, die die Schwelle überschreiten, werden lastabhängig.

## Was zuerst

1. **Heute, ohne Code:** M0, dazu die Log-Auswertung für `kv_mode` und die Akzeptanz. Das braucht einen kurzen Neustart von `strata.service` mit Test-Config. Der Ausfall dauert ~60 min Messfenster; das kündige ich vorher an.
2. Bei Go: M1 auf einem Branch von `g3-1441-ll`. Die Kernänderungen liegen in `src/program/generate.cpp` (Gate, Setup, `lane_pump`) und in `src/core/verify.cpp` / `include/strata/core/verify.hpp` (Event-Done, gegebenenfalls Feinschliff an `pl_commit_async`).
3. Unabhängig davon steht weiter das `s28r`-Deploy (+4,9 / +3,6 %) aus; dafür brauche ich dein „deploy ja“.

Dateien:
- `/media/raid0/StrataQ38FN/Strata-1441-ll/src/program/generate.cpp`
- `/media/raid0/StrataQ38FN/Strata-1441-ll/src/core/verify.cpp`
- `/media/raid0/StrataQ38FN/Strata-1441-ll/include/strata/core/verify.hpp`
- `/media/raid0/StrataQ38FN/Strata-1441-ll/src/core/mtp.cpp`
- `/media/raid0/StrataQ38FN/Strata-143-prod/serve/server.py`
- `/media/raid0/StrataQ38FN/linux/configs/T16-1441b-prod262r-vision-cpu.json`