# Design: faire und parallele Prompt-Reads (Stufe 1 FAIR, Stufe 2 CONCURRENT)

*Nur am Code gelesen. Keine GPU-Messung und keine Codeänderung. Zeilennummern beziehen sich auf `Strata-ll6-reads` @ f60f4ea5 (= ll4-lanes, das Engine-Binary `strata-g3-1441f-lanes4` in Produktion).*
*Abkürzungen: G = `src/program/generate.cpp`, P = `src/prefill/prefill.cpp`, PH = `include/strata/prefill/prefill.hpp`, S = `serve/server.py` (byte-gleich mit `Strata-143-prod/serve/server.py`).*
*Hintergrund: `DESIGN-short-reads.md`, `results/lanes/prefill.txt`, `results/reads/`, TESTLOG #80-#85.*

## Kurzfassung

| | Stufe 1 FAIR (`STRATA_READ_YIELD=1`) | Stufe 2a CONCURRENT (`STRATA_READS=N`) |
|---|---|---|
| Was es tut | Ein laufender langer Read gibt an einer Chunk-Grenze nach, sobald ein kurzer Prompt wartet. Der Teil, der schon gelesen ist, wird in den eigenen Slot geparkt. Danach liest die Engine zuerst die kurzen Prompts und setzt den langen Read anschließend fort. | Bis zu N Reads (Standard 4) laufen gleichzeitig in **einem** gepipelineten Prefill-Lauf, reihum Chunk für Chunk. Jeder Read schreibt direkt in die Sessions seines eigenen Slots. Der letzte Schritt (Tail-Fenster + erstes Token) läuft weiterhin seriell über die Haupt-Session. |
| Protokoll | **Keine Änderung**: `BYIELD`/`YIELDED` (#656) gibt es schon, Produktion verwirft sie nur. | Opt-in und getaggt (`INFO reads=N`, `BPP`, `BRESUME`, `BCUR`). Ohne `INFO reads` läuft alles wie heute. |
| VRAM | **0** | **0** (die Slot-Sessions und Prompt-Puffer gibt es schon) |
| Host-RAM | 0 (Drafter-K/V läuft über vorhandene Lane- bzw. Slot-Drafter) | wenige MiB pro Read (PLE-Lookahead, `ReadCtx`) |
| Exakt | ja: gleiche Chunk-Grenzen, gleiche Cuts, byte-exakte Slot-Kopie | ja: gleiche Kernel, gleiche Chunks pro Read, keine geteilten Zustände zwischen Reads |
| Nutzen | TTFT eines kurzen Requests hinter einem kalten 100K-Read: **bis 27 s → ca. 0,6-1,6 s**. Der lange Read verliert ca. 0,5-0,8 s plus die Zeit der kurzen Reads. | Bursts aus kurzen Prompts: das Füllen und Leeren der 4-Stage-Pipeline fällt pro Read weg, die Host-Phasen überlappen. Erwartung: ca. 30-50 % weniger Fixkosten pro kurzem Read im Burst. **Muss gemessen werden.** |
| Aufwand | ca. 12-16 h Engine, 3-4 h Server, 6-8 h Tests | ca. 50-70 h |
| Risiko | niedrig bis mittel | hoch |

**Empfehlung:** Stufe 1 zuerst bauen, messen und ausrollen. Stufe 2a erst angehen, wenn nach Stufe 1 die Warteschlange bei Sub-Agent-Bursts messbar der Engpass bleibt. Stufe 2 nutzt einen Teil von Stufe 1 wieder (Hook an der Chunk-Grenze, Delta-Kopie, Park-Lambda).

---

## 1. Ausgangslage (belegt)

1. **„Ausweichen" gibt es schon (#656), wird aber in Produktion nie ausgeführt.**
   - Der Server schickt `BYIELD <slot>`, sobald ein wartender Prompt höchstens halb so lang ist wie der laufende (`_shorter_waiting`, S:1610-1613; Solo: `give_way` S:1790-1801; Batch: `ask()` S:1908-1913).
   - Die Engine kann den Read in den Slot parken (G:15618-15650) und später fortsetzen (G:14552-14559).
   - In Produktion fällt das an drei Stellen weg:
     - `batch_read_piece()` liefert 0, sobald `piped` gesetzt ist (`include/strata/program/batch_read.hpp:24`). Damit läuft der Read als ein einziger `sp.run`, und BYIELD wird nie geprüft (G:15595).
     - Der überlappte Read läuft als ein `sp.run` auf einem eigenen Thread (G:15443), und `scan_ctl_lines` **löscht** BYIELD (G:13198-13199).
     - Die Hauptschleife verwirft BYIELD zwischen zwei Reads (G:13668-13673).
   - Im Prod-Log steht kein einziges „gives way".
2. **Die Prefill-Schicht hält zwischen zwei `run()`-Aufrufen keinen Zustand.**
   - Die Session wird einmal bei `init` gebunden (P:1225) und in `run_impl` gelesen (P:2297).
   - Am Ende werden nur `ple_prev` zurückgeschrieben (P:4546).
   - `run()` leert die Kette (P:4677 `drain_pipeline`).
   - Es gibt **keine CUDA-Graphen** im Prompt-Pfad (kein `cudaGraph` in P).
   - Spätere Stages laufen ohnehin **chunkweise** (`next_->run_impl(tokens + c0, T, p0, …)` P:4479/4486). Ihr `prev` kommt jedes Mal aus `ss.ple_prev` (P:2326).
3. **Rope im Prompt-Pfad ist rein positionsbasiert** (`rope(…, p0, …)` P:2944/3007/3010, kein M-RoPE-Bezug in P). Text-Reads in eine Slot-Session brauchen also keinen `mrope_base`. Die Regionen-Logik bleibt bei `copy_to_slot` (G:11712-11714, 11736-11740).
4. **Die Slot-Sessions sind vollwertige Sessions** (G:5214-5246) mit eigenem resident-Ring und eigener Host-Kopie bei `--kv-resident`. Ihr Speicher ist schon belegt (0,28 GiB pro Slot und Karte).
5. **Pad-up schützt geparkte Slots schon**: Eine Gruppenzeile schreibt keinen Slot, der `cached` oder `partial` ist (G:12150-12156). Das braucht Stufe 2 für Slots, in die gerade gelesen wird.

---

## 2. Stufe 1: FAIR (`STRATA_READ_YIELD=1`, Server `STRATA_YIELD_SHORT=<Tokens>`)

### 2.1 Zustand pro Read und wo er liegt

| Zustand | während des Reads | wenn der Read ausweicht (bei Position q) | VRAM extra |
|---|---|---|---|
| GDN-/Conv-State, QSA-K/V `[0,q)`, Indexer, PLE-History, `ple_prev` aller Stages | Haupt-Sessions `ss`, `stages[k]->ss` | `copy_to_slot(ys, ids[0,q))` in die Slot-Sessions von `ys` (G:11707; `STRATA_SLOT_COPY_DIRECT=1`: D2D, `direct_slot_copy` G:11674) | 0 |
| Drafter-K/V `[0,q)` | Haupt-`mtp` (E-9, G:10209-10230) | Lane-Slot: `copy_to_slot` kopiert in `lane_mtp[j]` (G:11760-11776); beim Fortsetzen holt `copy_from_slot` es zurück (Lane-Giveback G:11833-11850). Kein Lane-Slot und kein `batch_mtp` (Prod): kein Slot-Zustand. Das ist wie heute, und Drafts ändern nie Tokens (G:11757-11758). | 0 |
| Checkpoints `[≤q]` (Root, periodisch, Message-Tail) | `checks` | `sl.checks` (G:15634-15637), beim Fortsetzen zurück in `checks` (G:14557-14558) | 0 (Host, schon heute) |
| Read-Fortschritt `pp_next_check`, `part_next[]`, `pp_tail_saved`, `resumed_from0` | Request-Lokale (G:14822-14832) | **neu:** in `BSlot` (`partial_next_check`, `partial_part_next[]`, `partial_tail_saved`) | 0 |
| Prompt-Puffer, Staging-Pool, Hand-off | `Prefill::Impl` je Karte | nicht zustandsbehaftet zwischen Läufen (§1.2) | 0 |

**VRAM pro Karte: 0.** Die freien Reserven bleiben bei CUDA0 ca. 1,1 GiB, CUDA1 ca. 1,18 GiB, CUDA2 ca. 2,08 GiB und CUDA3 ca. 1,12 GiB.

Ein geparkter Read belegt seinen Slot. Der Server hat ihn reserviert (`slot_busy`, `slot_claims`, S:1795-1798, bzw. den Admit-Slot), deshalb kann der Slot in der Zeit nicht dekodieren.

### 2.2 Suspend an der Chunk-Grenze

**E1. Prefill-Hook** (PH neben `should_stop` PH:157, P:2376):
```cpp
// PH
std::function<bool(int64_t next_pos)> should_yield;   // stage 0 only; checked before a chunk, after the first
int64_t yielded_at() const { return yielded_at_; }     // -1: ran to the end
// P, run_impl, loop head, right after should_stop:
if (should_yield && hand_in_ == nullptr && c0 > 0 && should_yield(pos0 + c0)) { yielded_at_ = pos0 + c0; break; }
```
- `break` statt `return`: Der normale Abschluss läuft weiter. Dazu gehören das Zurückschreiben von `ple_prev` (P:4546, `prev` steht schon auf dem Ende des letzten fertigen Chunks, P:2518), die Stream-Syncs und in `run()` `drain_pipeline` (P:4677). Danach haben alle Stages genau `[pos0, q)` geschrieben.
- Der PLE-Lookahead für Chunk `c0` (`ple_next` P:2509) läuft als Future aus. Sein Destruktor wartet (P:2372), und das Ergebnis wird nicht benutzt.
- `yielded_at_` wird bei jedem `run()` auf -1 zurückgesetzt.
- Ist der Hook nicht gesetzt, ist das Verhalten byte-identisch. Es ist reiner Host-Code, also bleibt die SASS identisch.

**E2. Die Regel „darf ausweichen" wird geteilt.** Der Positionsteil ist rein und darf auf jedem Thread laufen:
`yield_pos_ok(q) = b0 - q > max(C, o.short_read)` (wie G:15623).
Damit hat der Rest mehr als einen Chunk, und der fortgesetzte Lauf ist nicht `single_chunk` (P:2330). Er liest also genau die Chunks, die ein Lauf ohne Unterbrechung gelesen hätte.
Der Zustandsteil (`ys` gültig, `!bs[ys].active`, `admit_slot<0 || ys==admit_slot`, `req_imgs.empty()`, `prompt_cache>0`, G:15621-15622) wird **auf dem Main-Thread** beim Annehmen von BYIELD geprüft. Erst danach wird `ov_yield_slot` gesetzt. So liest der Read-Thread `bs` nie, und es gibt kein Data Race mit `pump`.

**E3. Überlappter Pfad** (`read_overlapped`, G:15361):
- `scan_ctl_lines` (G:13187-13203): Ist `STRATA_READ_YIELD` an, wird `BYIELD <s>` nicht gelöscht. Stattdessen setzt die Funktion nach der Zustandsprüfung `ov_yield_slot.store(s)`. Fällt die Prüfung durch, folgt ein Log „BYIELD s not taken" wie bei G:15652.
- `sp.should_yield = [&](int64_t q){ return ov_yield_slot.load() >= 0 && yield_pos_ok(q); }` wird vor dem Start des Threads gesetzt und nach `guard.finish()` zurückgesetzt.
- Nach `pipe_drain()` + `guard.end_now()` (G:15525-15527), wenn `sp.yielded_at() >= 0`, wird `park_yield(ys, q)` aufgerufen (E5).
  - Klappt das, endet die Funktion mit `e = "yield"`, Rückgabe false.
  - Scheitert die Kopie, wird **im selben `read_overlapped` mit `[q, b0)` weitergelesen**: neuer Thread, gleiche `overlap_prepare`-Vorbereitung. Exakt, weil q eine Chunk-Grenze relativ zu `a0` ist.

**E4. Gepipeter, nicht überlappter Pfad** (G:15595, z. B. Solo-GEN mit `--batch`, Gate verweigert):
- Dort wird derselbe Hook gesetzt. Er läuft auf dem Main-Thread innerhalb von `sp.run`.
- Er scannt `in_lines` unter `in_mu` wie G:15603-15617: BSTOP setzt `stop`, BYIELD wird nach der Zustandsprüfung gemerkt.
- Danach folgt `park_yield`. Scheitert sie, wird mit `sp.run(q, b0)` weitergelesen.

**E5. `park_yield(ys, q)`**: ein Lambda, das aus G:15618-15650 herausgezogen wird. Der alte Stückel-Pfad nutzt es auch. Erweiterungen:
1. `slot_park_put(ys, bs[ys].cached && !bs[ys].partial)` **vor** der Kopie. Sonst geht die alte Konversation des Slots verloren. Bei BADM wird sie heute geparkt (G:17098). Einen Teil-Read selbst parkt `slot_park_put` nie.
2. `BSlot` bekommt `partial_next_check`, `partial_part_next[]`, `partial_tail_saved`. Beim Fortsetzen (G:14556-14559) werden sie statt `resume + prompt_cache_every` übernommen (G:14827-14832). So bleibt das Layout der periodischen Checkpoints wie ohne Yield.
3. Lane-Slot: nach der Kopie `lane_dk[j] = q` statt `q-1` (G:11766). Es gibt kein erstes Draft, das die letzte Zelle schreibt.
4. Die Haupt-Session: `live.clear(); live_ok = false; main_slot = -1`. Sonst parkt der nächste Request die 100K-Kopie aus der Haupt-Session zusätzlich in den RAM-Cache (G:14516-14533, G:16946-16956). Der Slot hält sie ja schon.
5. `SLOT_CKPT_MOVE` (G:14341-14352): Checkpoints eines `partial`-Slots werden nur **kopiert** und nicht verschoben, damit der fortgesetzte Read seine Kette behält.

Der Rest ist vorhanden: `cancelled` → `YIELDED <ys> <q>` → `DONE … cancel` → `BADM <slot> 0` (G:15846-15858, G:17084, G:17087ff.).

**E6. Hauptschleife** (G:13668): Ein BYIELD, das zwischen zwei Reads ankommt, wird weiter verworfen. Der Server fragt beim nächsten PP erneut (siehe S1).

### 2.3 Resume

- Der Server schickt nach `YIELDED` erneut `BGEN <ys> …` mit demselben Prompt (S:1872-1883).
- Die Engine findet `ys` als `slot_source` mit `partial` (`scan_slot`) und ruft `copy_from_slot(ys)` auf (G:11781). Mit `STRATA_SLOT_COPY_DIRECT=1` ist das eine D2D-Kopie.
- Danach folgen `resumed_from0`, `checks` und (neu) Checkpoint-Fortschritt (G:14556-14559) und `mtp.kv_restore(q)` (G:14816).
- Die Cuts werden ab `resume=q` neu berechnet:
  - `turn_at` und `pin_at` hängen nicht von `resume` ab.
  - `root_at` nutzt `resumed_from0` (G:15741).
  - `message_checkpoint_boundary` sucht von `turn_at` abwärts bis `> resume` (`message_boundary.hpp:10-18`). Es findet also entweder **dieselbe** Grenze wie im ersten Teil, oder eine Grenze `≤ q`, die schon geschnitten und gecheckpointet ist (dann liefert es -1).
- Die Segmentgrenzen `[a, b)` oberhalb von q sind damit dieselben, und q liegt auf dem Chunk-Raster des Segments `[a0, b0)`.

**Optional, in derselben Serie und nur hinter dem Env-Schalter:** Delta-Kopie. Hält der Slot schon `[0,r)` dieses Requests (eigene Fortsetzung, oder ein zweiter Yield nach `YIELDS_MAX=2`), kopiert `park_yield` nur `[r,q)` plus die kleinen Zustände (GDN, PLE, Indexer-Tail). Dafür bekommen `conversation_kv_copy` und `direct_slot_copy` einen Parameter für die erste Zelle. Stufe 2 braucht ihn ohnehin.

### 2.4 Scheduling-Policy (Server, opt-in)

**S1.** `_shorter_waiting(plen)` (S:1610) mit `STRATA_YIELD_SHORT=Y` (z. B. 2048):
- Es wird ausgewichen, wenn
  - irgendein wartender Eintrag `new_est ≤ Y` hat, **und**
  - der laufende Read noch `self.progress` total−done `≥ STRATA_YIELD_MIN_LEFT` hat (Standard 4·Y, mindestens 2 Chunks), **und**
  - `yields < STRATA_YIELDS_MAX` gilt (Standard 2, heute fest S:1487).
- `new_est` = `plen` minus der längste Präfix-Treffer über `slot_held`. Das ist dieselbe Suche wie in `pick_slot` S:2075 und wird in `_take_control` (S:1497) einmal berechnet und im Eintrag gespeichert.
- Grund: Heute zählt `plen` auch die gecachte History. Eine kurze Folgefrage in einer 100K-Konversation gilt deshalb als „lang", und ein Sub-Agent mit 20K gecachtem System-Prompt ebenfalls.
- Grenze: Der RAM- und SSD-Cache der Engine ist für den Server unsichtbar. Dieser Teil der Schätzung bleibt pessimistisch, der Server weicht also eher zu selten aus.
- Ist `STRATA_YIELD_SHORT` nicht gesetzt, gilt die heutige Regel `e[0]*2 <= plen`.

**S2.** Bild-Requests (`image_line`, `_take_engine_alone`) lösen kein Ausweichen aus. Sie warten ohnehin, bis alle Slots frei sind.

**S3.** `ask()` (S:1908) sendet heute höchstens einmal pro BGEN (`asked`). Mit E3 merkt die Engine sich ein BYIELD bis zur nächsten passenden Chunk-Grenze. Einmal reicht also. Wird es nicht angenommen (Log „not taken"), darf `ask()` nach `STRATA_YIELD_RETRY_S` (Standard 2 s) erneut senden.

**S4.** Danach greift die bestehende Reihenfolge: Nach `YIELDED` gibt der Server die Control-Lines frei, `_take_control(after_epoch)` lässt die Wartenden vor (S:1872-1883), und `STRATA_ADMIT_SHORT_FIRST` sortiert sie. `ADMIT_MAX_WAIT_S` (S:297) verhindert, dass der lange Read verhungert. `YIELDS_MAX` begrenzt die Kosten pro langem Request.

**Prod-Drop-in später:** `STRATA_READ_YIELD=1` (Engine-Env in der Config) + `STRATA_YIELD_SHORT=2048` (Server). Beide Seiten sind unabhängig abwärtskompatibel:
- Eine alte Engine verwirft BYIELD wie heute.
- Ein alter Server schickt BYIELD nur bei `2× kürzer`, und die neue Engine nimmt das dann an. **Das ändert das Verhalten auch ohne den neuen Server.** Deshalb braucht es den Engine-Schalter.

### 2.5 Exaktheit (Stufe 1)

Behauptung: Für jeden Request sind Tokens und Slot-Zustand identisch zum Lauf ohne Yield. Nur die Zeitpunkte ändern sich.

1. **Gleiche Chunk-Zerlegung.** Ohne Yield liest Segment `[a0,b0)` die Chunks `a0 + kC`. Mit Yield bei `q = a0 + jC` liest der zweite Lauf ab q die Chunks `q + kC`, also dieselben. Die Restlänge `> C` schließt den `single_chunk`-Sonderweg aus. Spätere Segmente beginnen an denselben Cuts (§2.3).
2. **Gleicher Startzustand je Chunk.** Nach dem Drain steht jede Stage-Session exakt auf `[0,q)`.
   - `copy_to_slot` und `copy_from_slot` sind byte-exakt: Sie kopieren Checkpoint plus K/V, und TESTLOG #81 zeigt Park-Images byte-identisch.
   - `ple_prev` ist Teil des Checkpoints.
   - `kv_pending` (KV-Prefetch) ist pro Chunk lokal (P:2528).
   - Der Staging-Pool wird pro Chunk und Layer aus der Host-Kopie der Session neu befüllt (P:2950-2957).
3. **Keine Abhängigkeit vom zwischendurch gelaufenen kurzen Request.** Der kurze Request überschreibt nur die Haupt-Sessions und den Haupt-Drafter. Beide werden beim Resume komplett aus dem Slot geholt. Der Drafter beeinflusst keine Tokens (G:11757-11758). Lane-Slots bekommen ihn trotzdem exakt zurück.
4. **Der kurze Request** liest wie immer. Sein RESUME kann vom seriellen Lauf abweichen, zum Beispiel wenn er im seriellen Fall den Root-Checkpoint des langen Reads hätte nutzen können. Das ist dieselbe Resume-Invarianz, auf der der Konversations-Cache beruht: Cuts hängen nur von den Prompt-Tokens ab. Sie wird im Test über den Token-Vergleich abgedeckt und nicht über einen RESUME-Vergleich.
5. **Nicht exakt gleich bleibt** nur das Layout der periodischen Checkpoints, falls E5.2 fehlt. Deshalb ist E5.2 Pflicht.

### 2.6 Zusammenspiel

| Funktion | Verhalten mit Stufe 1 |
|---|---|
| Overlap (`STRATA_PIPE_OVERLAP(+_KVS)`) | Primärpfad (E3). Die Gruppen dekodieren bis zum Yield weiter. Danach `pipe_drain` wie am Ende jedes überlappten Reads. |
| `STRATA_PIPE_BYIELD_NODRAIN` | Bleibt. BYIELD löst keinen Drain aus (G:13642-13643). |
| Spec-Lanes (1-4 Nutzer) | BGEN leert die Lanes (G:13628-13630). Danach läuft der Read überlappt. BYIELD leert sie nicht. Lane-Drafter: E5.3. |
| Solo-Pipeline / Solo-GEN | E4. Der Server reserviert den Slot (`give_way`). `can` erlaubt jeden freien Slot (`admit_slot<0`). |
| Slot-Park / Konversations-Cache / SSD-Tier | E5.1 (alte Konversation parken), E5.4 (keine Doppel-Parkung). Teil-Reads kommen nie in den Cache. |
| `MSG_CKPT_TAIL=8192` | Cut-Invarianz siehe §2.3. Ein Tail-Checkpoint vor q liegt in `sl.checks`. |
| Vision | Bild-Requests weichen nie aus (`req_imgs.empty()`) und lösen auch kein Ausweichen aus (S2). |
| Pad-up | `partial`-Slots sind geschützt (G:12153). |
| Flush nach NaN (#879) | Setzt `partial=false` und leert `sl.checks` (G:14297-14300). Der Resume liest dann kalt. Das ist korrekt. |

### 2.7 Fehlerpfade

| Fall | Verhalten |
|---|---|
| `copy_to_slot` scheitert | `bs[ys].cached=false` (halb geschrieben), Log, **Weiterlesen ab q** im selben Aufruf (E3/E4). Der Server bekommt kein YIELDED, der Request läuft normal zu Ende. |
| BSTOP für den lesenden Slot vor dem Yield | `should_stop` wird vor `should_yield` geprüft, Stop gewinnt (bestehender Pfad). |
| Client bricht ab, während der Read geparkt ist | Der Server gibt die Reservierung frei. Der Slot bleibt `partial` und wird von der nächsten Admission überschrieben. Der Pad-up schützt ihn bis dahin. |
| CUDA-Fehler im Drain | Bestehender Fault-Pfad in `read_overlapped` (G:15514-15524), kein Park. |
| BYIELD für einen inzwischen aktiven Slot | Die Zustandsprüfung auf dem Main-Thread lehnt ab, Log „not taken". |
| Server alt, Engine neu | Yield nur bei `2× kürzer` (alte Regel). Funktioniert, ist aber gröber. |

### 2.8 Kosten pro Yield (100K-Read, 15 Slots dekodieren)

- Drain der Kette: Stage 0 beendet ihren Chunk, dazu kommen ≤3 Chunks in Flight. Bei 100K-Tiefe sind das ca. 0,3-0,8 s.
- Gruppen-Drain: ca. 19 ms.
- `copy_to_slot(q)` und später `copy_from_slot(q)`: je 80-170 ms bei 90-155K (Prod-Log). Mit Delta-Kopie fällt der zweite Yield auf fast 0.
- Kurzer Request: TTFT ≈ Drain + Kopie + eigener Read (0,3-0,4 s beside slots, `prefill.txt`) ≈ **0,6-1,6 s statt bis 27 s**.
- Langer Request: plus ca. 0,5-0,8 s pro Yield plus die Dauer der vorgezogenen kurzen Reads, höchstens `YIELDS_MAX` Mal.

### 2.9 Test-Knöpfe (Stufe 1)

| Knopf | Zweck |
|---|---|
| `STRATA_READ_YIELD=1` | Engine opt-in. Ist er nicht gesetzt, ist das Verhalten byte-identisch. |
| `STRATA_READ_YIELD_TEST_AT=k` | Engine nimmt ein BYIELD erst ab Chunk k des laufenden Segments an. Damit lassen sich Yields deterministisch an jeder Stelle setzen (k=1, mitten im Segment, letzter erlaubter Chunk). |
| `STRATA_YIELD_SHORT`, `STRATA_YIELD_MIN_LEFT`, `STRATA_YIELDS_MAX`, `STRATA_YIELD_RETRY_S` | Server-Policy. |
| `tools/batch_overlap_test.py --yield-at k` | Engine-Protokoll direkt: BGEN lang, BYIELD, BGEN kurz, BGEN lang erneut. Vergleich der Tokens mit dem Lauf ohne BYIELD und mit `SLOT_PARK_VERIFY`. |
| `tests/program/batch_read_test.cpp` + neuer `read_yield_test.cpp` | Reine Arithmetik: `yield_pos_ok`, Cut-Invarianz (`message_cut` über verschiedene `resume`). |

---

## 3. Stufe 2: CONCURRENT (`STRATA_READS=N`, N ≤ 4)

### 3.1 Entscheidung: Slot-Sessions statt zweiter Haupt-Session

| Option | VRAM pro zusätzlichem Read und Karte | Bewertung |
|---|---|---|
| (A) Pro Read eine eigene Haupt-Session plus eigener Verifier | Session ca. 0,28 GiB (`session_bytes`, G:5222) plus Verifier-Graphen. 3 zusätzliche Reads = ca. 0,85 GiB+ auf **jeder** Karte. Das sprengt CUDA0, CUDA1 und CUDA3 (ca. 1,1 GiB frei) praktisch. | verworfen |
| **(B) Jeder Read schreibt in die Sessions seines Slots** | **0** (`bslot_ss` existiert, G:5214-5246). Prompt-Puffer (0,68 GiB/Karte) und Staging-Pool werden reihum pro Chunk geteilt. | **gewählt** |

### 3.2 Aufbau (2a)

**P1. Prefill mit mehreren Segmenten** (P, PH):
```cpp
struct Seg { const int64_t* tok; int64_t n, pos0; const float* const* embd; core::SessionState* ss[kMaxStages];
             void* ctx; int64_t done; bool yielded; };
bool run_segments(Seg* s, int k, const Schedule& sched, std::string& err);   // run() = 1 Segment, Haupt-Sessions
```
- Die Chunk-Schleife von Stage 0 nimmt pro Iteration ein Segment aus dem Schedule:
  - setzt `m.ss = seg.ss[0]`,
  - lädt `prev` aus `seg.ss[0]->ple_prev` (wie P:2326),
  - schreibt `prev` nach dem Chunk sofort zurück (statt einmal am Ende P:4546).
- Beim Hand-off (P:4470-4475, nach `next_run_.get()` P:4467, also race-frei) wird zusätzlich `next_->m.ss = seg.ss[k+1]` gesetzt und `ctx` mitgegeben. Spätere Stages laufen ohnehin chunkweise, und ihr `prev` kommt aus ihrer Session (P:2326).
- Der PLE-Lookahead (P:2479-2512) wird pro Segment geführt: Er gilt dem nächsten Chunk **desselben** Segments, oder er wird verworfen, wenn der Schedule wechselt.
- `on_chunk`/`on_stage_chunk` bekommen `ctx`.
- `run()` ruft `run_segments` mit einem Segment auf die Haupt-Sessions auf: **Der Default-Pfad ist dann derselbe Ablauf mit denselben Argumenten.** SASS bleibt unverändert, weil nur Host-Code betroffen ist.

**P2. `ReadCtx` statt Request-Globals** in `on_chunk`/`on_stage_chunk`/`checkpoint_at` (G:10041-10297): ids, `pp_*`, `part_at`/`part_next`, Ziel-Checkpointliste (`bs[b].checks`), Ziel-Sessions pro Stage (Save G:10079/10086), Ziel-Drafter.
- Drafter: E-9 schreibt in `lane_mtp[j]` für Lane-Slots. Für andere Slots wird kein Drafter-K/V gebaut (Prod hat kein `batch_mtp`). Beim Finish baut der Haupt-Drafter nur die Tail-Zellen. Das betrifft nur Drafts, nicht die Tokens (G:11757-11758).
- Voraussetzung: E-9 aktiv (`STRATA_SPLIT_MTP_BATCH=1`). Sonst bräuchte der Lane-Drafter `reserve_prefill_records` (G:15333), und das kostet VRAM. Ohne E-9 gibt es keinen Concurrent-Modus.

**P3. Admission in drei Phasen** (Block G:13693-17224):
1. **prepare(slot)** auf dem Read-Thread zwischen zwei Läufen:
   - `slot_park_put` (Capture abwarten), `bs[slot].reading = true`. Der Pad-up-Schutz bei G:12153 prüft `reading` mit.
   - Präfix in **die Slot-Sessions**: eigener Slot hält den Präfix (keine Kopie; heute Slot→Haupt→Slot), RAM-/SSD-Cache-Restore in die Slot-Session (`conversation_checkpoint_restore` und `conversation_kv_restore` nehmen jede `SessionState`) oder kalt (`session_zero`).
   - v0 ohne `SLOT_CKPT_MOVE` und ohne Präfix aus **anderen** Slots: Solche Reads gehen in den Exklusiv-Modus (P5).
   - `RESUME` als `BRESUME <slot> n`.
2. **read**: Segmente `[resume, cut)` mit denselben Cuts wie heute (G:15790-15870). Ein Cut beendet das Segment: Checkpoint in `bs[slot].checks`, dann das nächste Segment desselben Reads einreihen.
3. **finish(slot)** seriell auf dem Main-Thread, Read-Thread idle:
   - `copy_from_slot(slot)`, dann die Tail-Fenster (≤ `--short-read`), `merge_at` und das erste Fenster auf den Haupt-Verifiern wie heute (G:15904-15920),
   - danach `copy_to_slot` **nur für die neuen Zellen** `[m, n)` (Delta-Kopie aus Stufe 1) plus GDN/PLE/Indexer-State,
   - `BCUR <slot>`, `REUSED`, `T`, `DONE`, `BADM <slot> c`.
   - Volle Kopien pro Read: kalt 1 (heute 1), eigene Fortsetzung 1 + Delta (heute 2). **Nie mehr als heute.**

**P4. Scheduler** (Read-Thread, `read_overlapped` erweitert):
- Aktive Reads ≤ N. Schedule pro Chunk: **kürzester Rest zuerst**, bei Gleichstand reihum. Ein langer Read bekommt mindestens 1 von N Chunks (Anti-Starvation). So überholen kurze Reads lange, ohne sie zu stoppen.
- `run_segments` endet bei einem Ereignis: Ein Segment erreicht seinen Cut oder sein Ende, ein neuer BGEN ist angenommen, BSTOP kommt für einen lesenden Slot, oder das Quantum Q ist abgelaufen (Standard 16 Chunks, nur damit neue Reads hineinkommen). Nach der Rückkehr laufen auf dem Read-Thread prepare/checkpoint, auf dem Main-Thread finish. Dann folgt der nächste Lauf.
- Der Main-Thread pumpt die Gruppen weiter (`pump(true)`). `BGEN` löst keinen Drain mehr aus, solange der Read-Thread lebt (G:13642: Ausnahme für BGEN bei `STRATA_READS`). Kopien mit `cudaDeviceSynchronize` laufen nie auf dem Pump-Thread.
- Pro Chunk-Wechsel fällt kein Pipeline-Drain an. Gedrained wird nur bei Ereignissen (≈ einmal pro Read statt einmal pro Read **und** Segment).

**P5. Gate pro Read.** Erfüllt ein Read das Gate nicht, läuft der Exklusiv-Modus wie heute, sobald die aktiven Reads fertig sind.
- `overlap_gate` (G:15254-15308) muss gelten.
- `overlap_kvs_check` (G:15202-15239) vergleicht Slot b gegen die aktiven Slots statt die Haupt-Session gegen alle.
- Nur Text, kein `pin=`, kein cvec, `prompt_cache` an, Lanes nicht live.
- Slot-Sessions mit gebundenen PLE-Gewichten (Prüfung `ple.ready()` wie P:2232/2244. Lane-Slots nutzen sie schon, G:5367).

**P6. Protokoll (opt-in, abwärtskompatibel):**
- Engine: `INFO reads=N` nur bei `STRATA_READS=N`. Ohne das sendet sie keine getaggten Zeilen.
- Neue Zeilen:
  - `BRESUME <slot> n`
  - `BPP <slot> done total ms rate`
  - `BCUR <slot>`: Die folgenden ungetaggten Zeilen bis `BADM <slot>` gehören diesem Slot. Das reicht, weil finish seriell ist und der Read-Thread dabei nichts druckt.
  - `BERR <slot> msg`
- Server (`Strata-ll6-reads/serve/server.py`, `STRATA_SERVER_READS=1` **und** `INFO reads`):
  - Der Router (S:1037-1042) leitet `BPP`/`BRESUME` per Slot in `slot_q`. `BCUR` setzt den Besitzer der ungetaggten Zeilen.
  - Der Lock wird für BGEN zu einem Semaphor mit N Plätzen. GEN, GENI, SAVE, RESTORE und VRAM nehmen alle N.
  - `BYIELD` wird im Concurrent-Modus nicht gesendet. Fairness macht der Scheduler.
  - `BSTOP <slot>` beendet auch einen laufenden Read: Segment raus, `BCUR`, `DONE cancel`, `BADM <slot> 0`.

### 3.3 Exaktheit (Stufe 2)

Jeder Read sieht pro Chunk:
- dieselben Token,
- dieselbe Position `p0`,
- dieselbe Chunk-Grenze (Segmente beginnen an denselben Cuts, Chunks laufen auf `C` relativ zum Segmentstart),
- denselben Session-Zustand (seine eigene Session, die nur er schreibt),
- dieselben Kernel mit denselben T.

Geteilt werden nur zustandslose Arbeitspuffer:
- Prompt-Puffer, pro Chunk neu beschrieben.
- Staging-Pool, pro Chunk und Layer aus der Host-Kopie der Session neu befüllt (P:2950-2957). Der Prefetch ist pro Chunk lokal (P:2528).
- Hand-off-Puffer, FIFO.

Die Experten-Tabellen sind während eines überlappten Reads eingefroren (Gate). Rope ist positionsbasiert (§1.3). Der Präfix wird byte-exakt wiederhergestellt, und finish ist der heutige Code auf einer byte-exakt kopierten Session.

**Offene Prüfpunkte im Code, bevor gebaut wird:**
- (a) Greift ein Prefill-Kernel auf **zwei** Chunks desselben `Prefill`-Objekts zurück? Zum Beispiel ein Indexer-Cache in `Impl`, der Daten des letzten Chunks wiederverwendet. Dazu wird P:2292-4631 nach `Impl`-Membern durchsucht, die zwischen Iterationen weiterleben.
- (b) Die `stats_` und `pt`-Marken sind nur Statistik.

### 3.4 Speicher

- **VRAM:** 0 pro zusätzlichem Read und Karte.
- **Host:** `ReadCtx` (ids, n·4 B), PLE-Lookahead-Puffer pro Segment (T × PLE-Breite, wenige MiB, pinned), Checkpoints wie heute.

### 3.5 Zusammenspiel (Stufe 2)

- Overlap ist Voraussetzung.
- Lanes: Concurrent nur, wenn die Lanes nicht live sind. Bei 1-4 Nutzern laufen ohnehin selten mehrere Reads gleichzeitig.
- Solo-Pipeline, Vision, GENI: exklusiv.
- Slot-Park: prepare parkt vor dem Überschreiben, wie heute BADM.
- Konversations-Cache: Restore direkt in die Slot-Session. Die Einfügereihenfolge in den Cache kann sich gegenüber dem seriellen Lauf ändern. Das betrifft nur spätere Resume-Punkte, nicht die Tokens (Resume-Invarianz wie §2.5.4).
- `MSG_CKPT_TAIL` ist ein Cut und beendet das Segment.
- Pad-up: `reading`-Slots sind geschützt (P3.1).

### 3.6 Fehlerpfade (Stufe 2)

| Fall | Verhalten |
|---|---|
| Kopie oder Restore in prepare scheitert | Slot `cached=false`, `BERR <slot>`, `BADM <slot> 0`. Die anderen Reads laufen weiter. |
| CUDA-Fehler in `run_segments` | Wie heute: der ganze Prozess (#224). Alle aktiven Reads bekommen ERR. |
| BSTOP während des Reads | Segment entfernen, Slot `cached=false`, `DONE cancel`. |
| Server ohne `STRATA_SERVER_READS` | Sendet nie mehrere BGEN auf einmal. Die Engine läuft dann mit genau einem aktiven Read = Stufe 1. |
| Watchdog | `progress_at` pro Chunk eines beliebigen Segments. |

### 3.7 2b (später, separat beweisen)

Der letzte Schritt läuft im Gruppenfenster des Slots (T0d/P1 aus `DESIGN-short-reads.md`) oder auf dem Lane-Verifier. Das entfernt das serielle finish und die volle Kopie Slot→Haupt. Es braucht aber den Nachweis, dass eine Batch-Fensterzeile bitgleich zum Solo-Fenster ist (QSA_SLOTS, ATTN_VBATCH, andere Layouts). TESTLOG #85 zeigt, dass schon `--short-read 32` nicht exakt ist. Deshalb ist 2b nicht Teil dieses Designs.

### 3.8 Test-Knöpfe (Stufe 2)

| Knopf | Zweck |
|---|---|
| `STRATA_READS=N` | Engine opt-in. |
| `STRATA_READS_QUANTUM=Q` | Chunks pro Lauf. |
| `STRATA_READS_ORDER=rr\|srf` | Reihum oder kürzester Rest zuerst. |
| `STRATA_READS_TEST_SHUFFLE=seed` | Zufällige Chunk-Reihenfolge zwischen den Segmenten. Ein reiner Exaktheitstest: Die Tokens müssen unabhängig vom Seed gleich sein. |
| `STRATA_SERVER_READS=1` | Server. |

---

## 4. Testplan (beide Stufen, auf 127.0.0.1:8091, eigenes Tier-Verzeichnis)

1. **Env aus:** byte-identisch gegenüber `Strata-ll4-lanes/build` (`batch_test16`, SASS-Vergleich mit `sasscmp_ws.py`).
2. **Stufe 1, Engine-Ebene:** `batch_overlap_test.py --yield-at k` für k ∈ {1, Mitte, letzter erlaubter Chunk}, 1 Karte und Split 12,25,37, Greedy sowie T=0,7 mit Seed. Tokens müssen ohne Yield gleich sein. `SLOT_PARK_VERIFY`-Images des fortgesetzten Slots müssen byte-gleich zum Lauf ohne Yield sein.
3. **Stufe 1, HTTP:** kalter 100K-Read (+ Antwort < 262.144, mit Tokenizer gemessen) plus 4 kurze Requests (170 / 1,2K Tokens) während des Reads. Alle Antworten werden mit dem Prod-Binary verglichen (Prod-Server und Prod-Engine auf 8091 als Referenzarm). Gemessen werden TTFT der kurzen Requests und Gesamtzeit des langen.
4. **Lanes:** 2-4 Nutzer, Yield und Resume auf einem Lane-Slot, T=0,7 mit Seed.
5. **`ovkv_ident.py`** (KV-Streaming + Overlap) mit `STRATA_READ_YIELD=1` und `STRATA_YIELD_SHORT=2048`.
6. **Stufe 2:** `STRATA_READS_TEST_SHUFFLE` über 3 Seeds, 4 gleichzeitige Reads (1× 100K, 3× kurz), Tokens = serieller Lauf; Sub-Agent-Burst (16 Prompts mit 1-3K) mit `bench_multiturn`, TTFT p50/p95 und Durchsatz.
7. **Fehler:** `STRATA_PIPE_OVERLAP_TEST_FAULT=throw|cuda` mit Yield in Flight, BSTOP auf einen geparkten und einen lesenden Slot, Client-Abbruch während der Read geparkt ist.

---

## 5. Risiken, nach Schwere geordnet

| # | Risiko | Stufe | Gegenmaßnahme |
|---|---|---|---|
| 1 | Ein `Impl`-Member trägt Zustand von Chunk zu Chunk und bricht beim Wechsel der Session (§3.3 a) | 2 | Audit vor dem Bau. Ein Shuffle-Test deckt das auf. |
| 2 | Der Main-Thread führt eine Kopie mit `cudaDeviceSynchronize` aus, während der Read-Thread läuft. Das verursacht Stalls, im schlimmsten Fall greift der Watchdog. | 2 | prepare und Kopien nur zwischen Läufen, finish nur bei idle Read-Thread (P4). |
| 3 | Protokoll-Routing: Eine ungetaggte Zeile landet beim falschen Request. | 2 | `BCUR`-Klammer. Der Read-Thread druckt im finish nichts. Ein Server-Test mit gemischten Zeilen. |
| 4 | Die Yield-Policy schadet dem Durchsatz, weil lange Reads zu oft gedrained und kopiert werden. | 1 | `YIELDS_MAX=2`, `YIELD_MIN_LEFT`, Delta-Kopie; A/B mit `bench_multiturn`. |
| 5 | Doppel-Parkung oder ein verlorener Slot-Cache durch den Yield (E5.1/E5.4) | 1 | Explizit im Design. Prüfung über die Log-Zeilen „not parked here" und die Cache-Belegung. |
| 6 | Checkpoint-Layout nach dem Resume weicht ab (E5.2 fehlt). Die Tokens bleiben exakt, aber ein späterer Resume-Punkt ändert sich. | 1 | Pflicht-Teil von E5. Test vergleicht die RESUME-Werte des zweiten Turns. |
| 7 | Der Server schätzt neue Tokens falsch (RAM-Cache unsichtbar). Es wird dann zu selten ausgewichen. | 1 | Konservativ ist ungefährlich. Optional später: Engine-Hinweis per `INFO`. |
| 8 | Lane-Drafter nach dem Resume ohne Zelle q-1. Das kostet nur Draft-Qualität. | 1 | E5.3. |
| 9 | Pad-up schreibt in einen `reading`-Slot | 2 | Erweiterung von G:12153 und ein Test mit `force_padup`. |
| 10 | PLE-Gewichte in Slot-Sessions nicht gebunden | 2 | Gate-Prüfung (P5). |

---

## 6. Umsetzungsreihenfolge (Commits, alle opt-in)

1. `prefill: should_yield hook at a chunk boundary (opt-in STRATA_READ_YIELD=1)`: E1 + Unit-Test.
2. `batch: overlapped and piped reads give way at a chunk boundary (opt-in STRATA_READ_YIELD=1)`: E2-E4, `park_yield`.
3. `batch: yield keeps the slot's parked conversation, checkpoint layout and lane drafter (opt-in STRATA_READ_YIELD=1)`: E5.
4. `serve: yield to short new-token requests (opt-in STRATA_YIELD_SHORT=N)`: S1-S3.
5. `core: conversation copies from a first cell (opt-in STRATA_READ_YIELD_DELTA=1)`: Delta-Kopie.
6. Messfenster Stufe 1, dann Entscheidung über Stufe 2.
7. Stufe 2: `prefill: run_segments` → `batch: ReadCtx` → `batch: three-phase admission` → `batch: concurrent read scheduler` → `serve: tagged reads (opt-in STRATA_SERVER_READS=1)`.
