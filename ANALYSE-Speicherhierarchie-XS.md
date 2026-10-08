# Strata / UD-IQ4_XS: Speicherhierarchie, Caching, Prefetch, KV und Batching (Abschlussbericht)

Alles hier ist read-only ermittelt. Der Live-Server wurde nicht angefasst. Ich habe keine HTTP-Requests gesendet und keine Datei geändert. Gemessene Zahlen stehen mit ihrer Quelle da. Schätzungen sind als **(Schätzung)** markiert. Die Log-Zeilen beziehen sich auf `Strata\strata-unsloth-ud-iq4_xs.log` ab Zeile 1171 (Start 05:22, 365–369 Requests).

---

## 1. Kurzfazit

- **Die Hierarchie ist schon weitgehend da, es fehlt am Tuning.** Alle 24576 Experten liegen in VRAM (9332 Slots, 21,03 GiB) oder im gepinnten RAM (15244 Experten, 34,40 GiB) (log 1187, 1198). Der Decode liest **nichts** von der SSD. Den adaptiven Tier (decayed LFU, Swaps RAM↔VRAM) gibt es ebenfalls. Auch ein RAM-Tier für geparkte Konversationen existiert, er ist nur abgeschaltet.
- **Die 1,1 TB SSD-Reads stammen aus dem Prompt-Pfad, nicht aus dem Decode.** Der Prompt-Pfad leiht sich für seine Buffer 1550 VRAM-Slots (3,49 GiB). `--resident-budget-gib` schaltet die RAM-Kopie dieser „lend region" ab (`expert_source.cpp:1749`). Darum werden diese Experten pro 8192-Token-Chunk und danach noch einmal beim Refill ungepuffert aus der GGUF gelesen. Die eingebaute Abhilfe ist `--resident-experts` oder ein kleiner Code-Fix.
- **Größter realer Hebel für agentische Last: die Prompt-Cache-Misses.** Wechseln Main-Agent (~90K) und Sub-Agent (~18K), wird der KV des jeweils anderen überschrieben. Mindestens 10 Rückkehr-Events haben **200 s Prefill** gekostet, rund 28 % aller Prefill-Zeit (log 1472–1842). `--conversation-cache-mib` würde das als reine Config-Änderung beheben, auf Windows ist es aber ungetestet.
- **Größter Decode-Hebel: der Churn des adaptiven Tiers.** Gemessen sind 20–24 Exchanges pro Verify-Window, also knapp am Cap von 96 pro Runde. Die Kosten liegen bei ~0,19 ms je Exchange und Window (Regression). Ein Engine-Kommentar nennt ~24 ms pro Runde (`generate.cpp:6853-6856`). Das sind **~4,6–6 ms von 23–25 ms pro Window**, und dieser Teil lässt sich per Config tunen.
- **Expert-Vorhersage („35 → 85") ist machbar, aber großer Code.** Der Prädiktor existiert (`RouterLookahead`), wärmt heute aber nur File-Pages und ist live inaktiv. Für dasselbe Modell ist auf einem anderen Tier eine Recall von ~0,5 dokumentiert (`docs/UNSLOTH_Q4.md:225`). Realistisch sind +8–15 % Decode **(Schätzung)**, und das erst nach einer Messung.
- **Batching:** Echtes Batched Decode gibt es (`"parallel": N` / `--batch N`), es ist aber aus. Ohne `--batch-mtp` ist es hier eher neutral bis negativ. Mit `--batch-mtp` könnte es nur bei echter Gleichzeitigkeit +20–45 % Aggregat bringen **(Schätzung)**, kostet aber dauerhaft ~3,4 GiB VRAM (~1500 Expert-Slots). Erst muss gemessen werden, ob sich die Requests überhaupt überlappen.
- **Größtes Risiko: Commit, nicht physisches RAM.** Vom Commit-Limit (101,6 GB) sind nur **2,6–5,8 GB frei**, gemessen über mehrere Stichproben. Der größte rückholbare Posten ist der PLE-Row-Cache mit **11,77 GiB** (das Flag zählt *Zeilen*, nicht Bytes). Jede neue RAM-Stufe (lend region, Parking, KV-Streaming, adapt-async) braucht vorher freigemachten Commit.
- **Die Stalls** (9 Stück, 7.–8.10.) traten alle mit dem Modell auf X: und `--ple-io mmap` auf, alle bei „layer 1 of the prompt chunk". Seit 05:22 (C: + direct) gab es keinen mehr. Die Ursache ist konfundiert, weil Laufwerk, PLE-Modus, Budget und Row-Cache gleichzeitig geändert wurden.

---

## 2. Ist-Zustand der Speicherhierarchie

### 2.1 Was liegt wo

| Ebene | Inhalt | Größe | Messwerte |
|---|---|---|---|
| **VRAM** (32,6 GB, 31,46 GB belegt) | Expert-Cache, PROFILE-Prefill, keine Eviction, adaptive Swaps | 9332 Slots = 21,03 GiB (log 1187) | 86,25 % der gerouteten Entries (Aggregat); die gedruckte „hit rate" lässt PCIe-Entries aus dem Nenner weg (`generate.cpp:10010-10022`) |
| | davon an den Prompt-Pfad verliehen | 1550 Slots = 3,49 GiB (log 1211) | während eines Prompts nur auf der SSD vorhanden |
| | KV int8 bei 131072 + Indexer + Draft-KV + GDN-State | ~2,0 GiB (berechnet, `layer.cpp:670-690`, `kv_q8.hpp:25-27`) | +131K Kontext kosten gemessen 840 Slots (engine-C3 vs C4) |
| | Dense/Pack 1,38 · Native Proj 2,91 · Q5_K Head 0,49 · MTP 0,96 · Draft Head 0,21 GiB | log 1174–1215 | — |
| | frei nach dem Laden | 682 MiB (log 1216) | Reserve 1200 MiB |
| **RAM** (63,6 GB) | Expert-Komplement, gepinnt (`cudaHostAllocMapped`) | 15244 Experten = 34,40 GiB (log 1198) | 7,89 % CPU-Pool + 5,86 % GPU-Zero-Copy über PCIe |
| | PLE-Row-Cache (8-way, Round-Robin, zero-filled) | 134.217.728 Zeilen × 94 B = **11,77 GiB Commit** (`ple_reader.cpp:43-49`) | Hit-Rate in serve nicht geloggt; ≥ ~87 % als Untergrenze aus ReadOperationCount abgeleitet **(Schätzung)** |
| | Prompt-Checkpoints (Running State) | 6 × 118 MB ≈ 0,71 GB (`conversation_state.cpp:106-131`) | nur Rewind-Punkte auf dem aktuellen Branch |
| | Conversation Cache (Parking) | **aus** (`generate.cpp:586`) | — |
| **SSD C:** (SN850X) | GGUF-Shards 2/3: PLE-Tabelle 28,8 GB (IQ4_NL, 90 B/Zeile) + Experten-File-Tier | — | PLE ≤ ~1,6 MB/s im Schnitt; Experten-Reads ~1,07 TB in 3,1 h, alle aus dem Prompt-Pfad |
| **SSD X:** (P3 QLC) | Pack, MTP, Profile (nur beim Start gelesen) | — | zur Laufzeit irrelevant |

**Decode-Leistung (live):** 130,9 tok/s im Aggregat (418.010 Tokens / 3194 s), p50 116,9 tok/s, 23,0 ms/Window, 3,01 Tokens/Window, MTP-Akzeptanz 0,761. Laut OLS-Fit (n=351, R² 0,84) bringt jeder Prozentpunkt VRAM-Hit bei fester Akzeptanz etwa **+2,75 tok/s** (Schätzung, Regressionsmodell).

**Prefill (live):** p50 3006 tok/s für Reads ab 4096 Tokens. 93,9 % der Prompt-Tokens werden aus dem Cache wiederverwendet.

### 2.2 Das „1,1 TB files read"-Rätsel

`files N blobs M MB read (the GGUF in place)` (`generate.cpp:10041-10044`) misst den Expert-File-Tier. Mit PLE hat dieser Zähler nichts zu tun.
- **Blobs** = `blob()`-Aufrufe für Experten, die weder im Komplement noch in einem Exchange liegen (`expert_source.cpp:2249`). In diesem Lauf zählt das nur den Refill der verliehenen Slots nach einem Prompt (`generate.cpp:8664-8676`).
- **MB** zählt zusätzlich `copy_blob` (`expert_source.cpp:1569`). Der Stager kopiert verliehene, also „transient" gewordene Experten in **jedem Chunk** (`prefill.cpp:1957-1965, 394`). Diese Kopien erhöhen nur die Bytes, nicht den Blob-Zähler.

Der Mechanismus:
1. `lend()` markiert die letzten k Slots als `kNotResident` (`generate.cpp:8732-8762`).
2. Mit gesetztem Budget gilt `lend_from_slot = -1` (`expert_source.cpp:1749`, Kommentar „A budget turns `lend` off", :1756). Die Experten dieser Slots haben deshalb **keine RAM-Kopie**.
3. Jeder 8192-Chunk liest den kompletten Lend-Satz (3,75 GB). Danach kommt der Refill mit weiteren 3,75 GB.

Nachgerechnet:
- 64.815 Tokens → 8 × 3,75 + 3,75 = 33,7 GB vorhergesagt, gemessen 33.726 MB (+1550 Blobs).
- 12.539 Tokens → 3 × 3.747 MB = 11.242 MB.
- Kurze Prompts leihen nur einen Teil der Slots (`generate.cpp:8727-8732`), daher die Blob-Deltas von 141 bis 2641.
- Prompts mit ≤ 40 Tokens lesen 0 MB (28 von 28).
- Etwa 22,6 GB sind der Startup-Prefill.
- Über alles: 0,72 MB pro gelesenem Prompt-Token, Korrelation +0,994 mit den gelesenen Tokens und +0,065 mit den generierten.

Ein zusätzliches Problem: Die Unbuffered-Heuristik (`expert_source.cpp:1105-1108`) rechnet mit 21,0 GiB „aus Files zu lesen" und meint damit alle VRAM-Experten. Tatsächlich wird nur der 3,5-GiB-Lend-Satz gelesen. Die Folge ist `FILE_FLAG_NO_BUFFERING` (log 1209), und das OS cacht nichts, sodass jeder Prompt erneut auf die NVMe geht.

### 2.3 Commit und RAM-Headroom (der eigentliche Engpass)

| Größe | Wert | Quelle |
|---|---|---|
| Commit-Limit | 101,6 GB | Win32_OperatingSystem |
| Commit frei | **2,6–5,8 GB** (schwankt) | mehrere Stichproben während der Analyse |
| Physisch „available" | 13–16 GB, **vollständig Standby**, 0 GB Free/Zero | Get-Counter |
| strata.exe Private Bytes | 82–88 GB | Get-Process |
| strata.exe Working Set | 31,7–37,8 GB (wird getrimmt) | Get-Process |
| Pagefile | system-managed, 38,9 GB alloziert, Peak 22,1 GB | Win32_PageFileUsage |

Zerlegung der Private Bytes **(Schätzung)**: 34,4 gepinnt + 11,8 PLE-Cache + ~30 GiB WDDM-Commit-Charge für VRAM-Allokationen + ~4–5 Sonstiges ≈ 82 GiB. Die WDDM-These ist **nicht verifiziert** und bräuchte VMMap/RAMMap. Fest steht: Beim Start waren vor dem Komplement 46,99 GiB Commit verfügbar. Der Abfall auf ~5,8 GiB passiert erst zur Laufzeit, und der Verursacher dieser Drift ist offen.

---

## 3. Antworten auf die Fragen

### (a) RAM-Cache für das, was von der SSD gestreamt wird

- **Existiert schon?** Ja, an drei Stellen:
  - Die Experten sind im Decode bereits komplett RAM/VRAM-resident.
  - Der PLE-Row-Cache existiert und ist eher **überdimensioniert**.
  - Die „lend region" im RAM existiert (`expert_source.cpp:1776-1790`), ist aber durch `--resident-budget-gib` abgeschaltet.
- **Was gestreamt wird:** nur der Lend-Satz im Prompt-Pfad (3,49 GiB, immer dieselben Experten) und PLE-Misses (wenige MB/s).
- **Was es realistisch bringt:**
  - Lend region im RAM (B6-Sweep, Modell auf X:): long4k-TTFT 3,4 s → 2,0 s (−41 %), keine 10–46-s-Ausreißer (logs_seriesB.txt).
  - **Auf C: ist der Gewinn kleiner.** Live erreicht der Prefill schon p50 3006 tok/s, ungefähr B6-Niveau. Pro kleinem oder mittlerem Request fallen 0,3–1 s Refill-Latenz weg, bei großen Prompts sind es < 15 % **(Schätzung)**. Die SSD-Reads sinken um ~1,1 TB pro Session.
- **Drei Wege:**
  1. `STRATA_UNBUFFERED_LOAD=0` (config-only): Der Lend-Satz bleibt im Windows-Standby-Cache, ohne Commit zu belegen. Risiken: Nach einer Standby-Eviction werden die Daten kalt per Page-Fault gelesen, und das kann langsamer sein als heute (`generate.cpp:3716-3717`). Außerdem schaltet es den RouterLookahead-Thread **ein** (`expert_source.hpp:577`, `generate.cpp:4342`).
  2. `--resident-experts` statt `--resident-budget-gib 40` (config-only): +3,49 GiB gepinnt. Der Startup-Check besteht (37,9 < 43 GiB), **aber** der Commit driftet zur Laufzeit. Es schaltet ebenfalls den Lookahead ein (set_unbuffered wird übersprungen, `generate.cpp:3721`). **Warnsignal:** B6 lag beim Decode in allen 4 Kategorien unter B7 (z. B. code 127 vs 161 tok/s, `bench_results.jsonl:98-100` vs 110-112). Beide liefen mit `--host-core first`, B6 hatte aber nur 184 MiB VRAM frei. Die Ursache ist ungeklärt.
  3. Code-Fix (klein): `recheck_unbuffered` rechnet nur mit den Lend-Bytes plus den Experten, die in keinem Tier liegen, oder die lend region wird auch unter Budget erlaubt (das Budget hat 5,6 GiB Luft). Dieser Fix schaltet den Lookahead **nicht** ein, weil diese Entscheidung schon bei `generate.cpp:4342` fällt.
- **PLE:** `--ple-row-cache 33554432` (2,94 GiB) statt 134M macht **8,8 GiB Commit frei**. Die Simulation des exakten Hashs auf 1,78M Tokens zeigt, dass 16M Zeilen schon einem unendlichen Cache entsprechen (74,6 %). Live sind in 3,1 h höchstens ~4,7M Unique Rows dazugekommen **(Schätzung)**. Der Speed-Effekt ist ungemessen, wahrscheinlich neutral. Lazy-Commit des Caches (PLE-4) ist **widerlegt**: Der gehashte Set-Index berührt schnell fast alle Pages.
- **Aufwand/Risiko:** config-only bzw. ein kleiner Code-Fix. Das Risiko ist Commit; deshalb zuerst den PLE-Cache verkleinern.

### (b) Feste Hot-Experts im VRAM

- **Existiert schon?** Ja. Die 9332 Slots sind nach Profil gerankt, und es gibt keine Demand-Eviction (`expert_cache.cpp:642`). Darauf sitzt der adaptive Tier (`--adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7`, `generate.cpp:552-555, 581, 6720-6851`). Der wirklich heiße Kern ist faktisch statisch. Der Churn trifft die mittelwarmen Experten.
- **Gefundene Probleme:**
  1. **Churn:** 20–24 Exchanges pro Window, nahe am Cap. Das Gedächtnis ist sehr kurz (Halbwertszeit ≲ 25 Tokens). Die Kosten liegen bei ~4,6–6 ms pro Window. Die Exchange-Rate fällt über die Session nicht (5,0 → 7,3–8,0 pro Token je 50-Request-Block). Die Swaps verfolgen also Request-Lokalität und korrigieren kein schlechtes Startprofil.
  2. **Profil:** Es ist generisch (`data/expert-profile.bin`, Stand 6.10.), und `expert_profile_save` ist nicht gesetzt. Jeder Restart beginnt kalt. Die Slot-Verteilung pro Layer (138–240) ist eingefroren, weil Swaps wegen der sized slots innerhalb eines Layers bleiben (`generate.cpp:3916-3939`). Ein Rebalancing ist schon als Tool vorhanden: `tools/make_profile.py --reorder` plus ein `--dump-routing`-Trace (`make_profile.py:8-25`).
  3. **`STRATA_EXCHANGE_ROTATE=1` wirkt hier nicht.** Das Pack hat 3 Blob-Größen (2,33/3,15/3,48 MB), und die Rotation verlangt eine einheitliche Größe (`expert_source.cpp:2097-2100`).
- **Was es realistisch bringt:**
  - Adaptive-Tier-Tuning (`STRATA_ADAPT_LAG=2`, `--adapt-async 1`, weniger oder seltenere Swaps): **+5–12 %**, Obergrenze ~+20 % bei vollständigem Verstecken **(Schätzung, Vorzeichen bei Decay/Swaps unklar)**.
  - Size-Class-Rotation (Code, mittel): spart ~228 MB host-memcpy pro Runde, **+8–18 %**. Das überlappt mit dem Tuning und ist nicht additiv **(Schätzung)**.
  - Gelerntes oder rebalanciertes Profil: +1–3 Punkte VRAM-Anteil, also +3–8 %, vor allem nach Restarts **(Schätzung)**.
  - `--vram-reserve-mib 1000`: ~+0,5 Punkte bzw. +1–1,5 %. Die Engine-Richtlinie empfiehlt ≥ ~512 MiB frei (`generate.cpp:7157`).
  - `pcie_frac` > 0,55 (gedeckelt durch `generate.cpp:1497-1499`, obwohl der Link 57,8 GB/s schafft): 0–5 %, über 0,75 ist das Vorzeichen unklar. `tools/calibrate.py` gibt es schon.
- **Grenze:** 95 % VRAM-Anteil bräuchte rund 50–60 % aller Experten im VRAM. Das geht auf 32 GB nicht.

### (c) Expert-Korrelation und proaktives Laden

- **Existiert schon?** Teilweise. `RouterLookahead` wendet den Router von Layer L+1 auf den Input von Layer L an (`expert_source.cpp:1423-1530`). Er wärmt aber nur OS-Pages des File-Tiers und ist live **aus**, weil der Tier unbuffered läuft (keine Startzeile „routing-aware prefetch"). Ko-Okkurrenz-Tabellen gibt es nicht, einen GPU-Prefetch auch nicht, und Routing-Traces liegen nicht auf der Platte.
- **Fachlich:** Innerhalb eines Layers ist die Frage hinfällig, weil der Router alle 10 Experten auf einmal liefert. Über Layer hinweg ist der beste Prädiktor der Router des nächsten Layers. Eine statische Tabelle „35 → 85" ist schwächer, lässt sich aber offline aus `--dump-routing` messen. Über Tokens hinweg ist klare Lokalität messbar: Der VRAM-Anteil liegt bei 79 % für kurze Antworten und bei 88 % ab 5000 Tokens.
- **Physik:** Ein Layer dauert ~0,5 ms. Etwa 5 Misses × 2,4 MB brauchen ~0,2 ms über PCIe, ein Layer Vorlauf würde also reichen. **Aber** der Host wartet laut Diag-Lauf eher auf die GPU (GPU-reach wait 8,3 ms gegenüber Host 6,7 ms pro Window). Der Gewinn ist deshalb kleiner, als die Miss-Kosten nahelegen.
- **Was es realistisch bringt:** bei Recall ~0,5 etwa **+8–15 %** Decode **(Schätzung)**. Der Prior von 0,5 stammt aus `docs/UNSLOTH_Q4.md:225` und galt dort für SSD-Misses auf UD-Q4_K_XL, nicht für VRAM-Misses.
  - Ein billigerer Teilgewinn ohne Vorhersage: den PCIe-Staging-Copy-Kernel auf einen Side-Stream verlegen. Er läuft heute seriell nach dem VRAM-Grouped-Kernel (`verify.cpp:1260-1270`). Das bringt **+4–8 %** bei kleinem Code-Aufwand **(Schätzung)**.
- **Aufwand:** Recall-Zähler klein → Prefetch groß. **Risiko:** PCIe-Konkurrenz mit den Swaps, ein Staging von 2×16×3,48 MB = 111 MB VRAM (~45 Slots), Synchronisation mit `host_res`. DMA-Mode hatte den Hang aus Issue #31 (`docs/DETAILS.md:1227`), deshalb lieber einen Copy-Kernel verwenden.

### (d) KV-Cache: genug Speicher? RAM? SSD? 3-Tier?

- **Genug Speicher?** Ja. 128K int8 kosten ~2,0 GiB VRAM und sind komplett vorab alloziert. Der KV ist **nicht** der Engpass. Die Kosten fallen bei den Expert-Slots an: 64K Kontext entsprechen ~420 Slots. 128K beibehalten, weil Prompts bis 101K und Outputs bis 64K vorkommen.
- **RAM genutzt?** Kaum. Es gibt nur 6 Running-State-Checkpoints (0,71 GB), und die gelten ausschließlich für den aktuellen Branch. Der RAM-Tier für mehrere Konversationen (`--conversation-cache-mib`) ist implementiert (`generate.cpp:6382-6500, 8255-8387`), aber **aus**. Selbst `setup.py:3822` empfiehlt ihn (8192 MiB).
- **Messbarer Schaden heute:** Mindestens 10 Rückkehr-Events, z. B. log 1807–1842 (Main 81K/92K/96K jeweils „0 reused" im Wechsel mit Sub 18K), kosten **200 s**. Weitere Fälle wie log:1842 und kurze Probes, die die Main-Konversation wegwischen, kommen dazu. Die 200 s sind daher eine **Untergrenze**.
- **SSD (X:):** Es gibt nur manuelle Session-Files (POST `/slots/0?action=save|restore`, braucht `slot_save_path`, nicht mit parallel nutzbar). Laut `DETAILS.md:865-867` lief die Engine dafür nur auf Linux, unter Windows ist das ungetestet. Ein automatisches Auslagern RAM → SSD gibt es nicht; verdrängte Einträge werden verworfen (`conversation_cache.hpp:244-255`).
- **Lohnt Laden statt Neu-Inferieren?** Ja, deutlich. Re-Prefill von 95K dauert 26–28 s (gemessen, log 1832). Restore aus RAM: ~2 GB in 0,3–1 s auf WDDM **(Schätzung)**. Aus der SSD: 1,7 GB in ~1–2,5 s **(Schätzung)**. Bei der Rückkehr bleibt der neue Suffix (Tool-Output, oft 1–7K Tokens) und kostet 0,5–4 s. Realistische TTFT-Verbesserung: **9–27 s → 0,5–4 s**.
- **Größen:** 15.264 B/Token positional (validiert gegen die Session-Datei, 0,02 % Abweichung). Ein geparktes 95K-Image mit 6 Checkpoints ist ~2,0–2,3 GB groß. 4 GiB Budget halten etwa **2** solche Konversationen.
- **3-Tier (VRAM/RAM/SSD):** Sinnvoll. VRAM und RAM gibt es, der SSD-Tier existiert als Dateiformat; es fehlt der Kleber (Spill bei Eviction, Disk-Index, Lookup bei Miss) mit mittlerem Code-Aufwand. Wegen der Fingerprints (`DETAILS.md:841`) bringt er über Tuning-Restarts hinweg wenig.
- **Risiken:**
  - Die Parking-Admission prüft **nur physisches RAM** (`conversation_memory.cpp:37-41`), nicht Commit.
  - Ein **fehlgeschlagenes Parken beendet die Engine** (`generate.cpp` ~8290-8293).
  - Restore-Fehler sind ebenfalls fatal (`generate.cpp:8364-8370`).
  - Ein Fix für die Admission (min(AvailPhys, AvailPageFile)) ist klein und sinnvoll.
- **KV aus dem VRAM schieben** (`--kv-resident 32768`): ~+500 Slots, +1–3 % **(Schätzung)**, kostet aber ~1,6–1,7 GiB gepinnten Commit und konkurriert mit dem Parking. `--kv k8v4`: +150 Slots, laut Doku 2–5 % langsamerer Prefill, netto ≈ 0.

### (e) Parallele Zugriffe und Batching

- **Heute:** strikt sequenziell. Ein FIFO-Lock (`serve/server.py:2912`), kein Time-Slicing.
- **Existiert Batching?** Ja, als echtes Multi-Sequence-Decode: ein CUDA-Graph über S Zeilen, dabei werden geteilte Experten und Dense-Gewichte einmal gelesen (`verify.cpp:2339-2432`). Opt-in per `"parallel": N` → `--batch N`.
- **Lohnt es sich hier?**
  - **Ohne `--batch-mtp`:** nein. Jede Slot-Zeile bekommt 1 Token pro Window statt 3,0 im Solo-Modus. Aggregat ~115–135 tok/s, also wie solo; pro Stream ~60 tok/s **(Schätzung)**.
  - **Mit `--batch-mtp`, 2 Slots:** +20–45 % Aggregat **nur bei echter Überlappung (Schätzung; Doku: +31–39 % auf einer RTX PRO 5000)**. Solo wird dafür dauerhaft −3–7 % langsamer, weil ~3,4 GiB (~1500 Slots) an die Slots gehen. Batching schaltet zusätzlich `--adapt-async`, `--pipeline-windows` und die /slots-API ab. Scheitert das Laden eines Slot-Drafters, **beendet sich die Engine** (`generate.cpp:3583-3590`). Auch `setup.py:3748-3763` empfiehlt für dieses Modell „one at a time".
- **Für das Agent-Muster** ist Parking (Punkt d) der bessere erste Schritt, weil es keinen VRAM kostet. Ob die Agents wirklich gleichzeitig laufen, ist **unbekannt**: `queue_s` existiert nur im Speicher (`server.py:2425, 2565, 2917`). Ein Timestamp und `queue_s` im vorhandenen Print (`server.py:368`) bzw. das Umleiten von stdout würden das klären.

---

## 4. Priorisierte Experiment-Liste

**Alle Experimente brauchen einen Restart oder HTTP-Zugriff auf den Server und damit dein OK in einem Wartungsfenster.** Eine zweite Engine passt nicht neben den Live-Server (VRAM 682 MiB frei, Commit ~5 GB frei).

### Mess-Protokoll (für alle A/B)

1. **Replay-Set fest einfrieren.** Request-Bodies aus der echten Agent-Last mitschneiden (aktuell liegen keine auf der Platte, deshalb zuerst E0). Daraus drei Szenarien:
   - **S1 Agent-Turn:** 90K wiederverwendeter Präfix + Suffix von 200 / 1000 / 9000 Tokens.
   - **S2 Ping-Pong:** A = 80K Main, B = 18K Sub mit anderem System-Prompt, 6 Turns abwechselnd, dazwischen eine 800-Token-Probe.
   - **S3 Decode:** 30K-Kontext, `temperature 0`, fester Seed, 2000 Output-Tokens.
   - Zusätzlich kalt: 64K- und 95K-Prompt.
2. **Warm-up:** pro Arm nach dem Start 2 Requests verwerfen (der Startup-Prefill steckt im ersten Request, log 1224).
3. **N ≥ 3 je Arm, verschränkt** (A/B/A/B, Restart pro Wechsel). Median und Spannweite berichten. Einzelläufe sind hier nachweislich zu verrauscht (Chat 26–150 tok/s über die Labels).
4. **Konstant halten:** Modell auf C:, `--host-core last`, VRAM frei nach dem Laden ≥ 500 MiB (vermeidet den Konfund aus B6).
5. **Metriken pro Request:** TTFT, Prompt-tok/s, Decode-tok/s, VRAM-Anteil der gerouteten Entries (hits/(hits+CPU+PCIe)), Exchanges pro Window, Delta von `files … MB/blobs`, mit `STRATA_DECODE_TIMING=1` den ms/window-Split.
6. **System, sekündlich:** Commit (`\Memory\Committed Bytes`, `Commit Limit`), FreePhysical, C:-Disk-Read-Bytes/s. Zusätzlich Commit frei nach 30+ min Betrieb, nicht nur nach dem Laden. `stall_probe.py` für Lücken > 150 ms.
7. **Abbruch:** Commit frei < 2 GB, VRAM frei < 400 MiB oder eine „shrinking the expert cache"-Zeile.

### Experimente (Reihenfolge nach Nutzen und Aufwand)

| # | Änderung relativ zur Live-Config | Metrik | Erwartung | Restart |
|---|---|---|---|---|
| **E0 Mess-Restart** (Pflicht zuerst) | env `STRATA_DECODE_TIMING=1`; Arg `--expert-profile-save X:\…\expert-profile-learned.bin` (nicht `data/expert-profile.bin` überschreiben); `--dump-routing X:\traces\route.bin` (~4,2 KB pro verifiziertem Token); stdout des Servers in eine Datei (für `queue_s`) | stage-ms auf C:, Window-Split, Überlappungsanteil (queue_s > 1 s), Routing-Trace | reine Diagnose; Verhalten unverändert (Trace-I/O ~0,6 MB/s) | ja |
| **E1 Commit freimachen** | `--ple-row-cache 33554432` (zusätzlicher Arm: 16777216) | Decode-/Prompt-tok/s, Private Bytes, Commit frei | **+8,8 GiB Commit** (gerechnet), Speed neutral (Schätzung) | ja |
| **E2 Conversation Parking** (nach E1) | `--conversation-cache-mib 4096 --conversation-cache-slots 2` (Min-Free bei 2560 lassen). Vorher `tools/parking_test.py` als Paritätstest | S2: reused/TTFT pro Turn, Log-Zeilen `conversation cache: parked/restored … ms`, `snapshot_bytes`, Commit | Rückkehr-TTFT 9–27 s → 0,5–4 s; ~150–250 s pro 3 h (Schätzung). Unter Windows ungetestet, ein Park-Fehler ist fatal | ja |
| **E3 Lend region im RAM** (nach E1) | A = live; B = `--resident-experts` statt `--resident-budget-gib 40`; C = B + env `STRATA_LOOKAHEAD=0`. Zusätzlicher Arm D = nur env `STRATA_UNBUFFERED_LOAD=0 STRATA_LOOKAHEAD=0` | Startzeile „1550 of the prompt path's … lendable slots keep their experts in RAM too", files-Delta ≈ 0, TTFT, **Decode muss gleich bleiben** | 0,3–1 s weniger pro Request, < 15 % bei großen Prompts, −1,1 TB SSD-Reads (Schätzung); C trennt den Lookahead-Effekt ab | ja |
| **E4 Adaptive Tier** | Arme: A default; B env `STRATA_ADAPT_LAG=2`; C `--adapt-async 1` (+~334 MB pinned); D `--adapt-swaps 48`; E `--adapt-decay 0.85`; F `--adapt-every 0` (Baseline ohne Adapt) | S3 Decode-tok/s, VRAM-Anteil, Exchanges/Window, Rest-ms (window − verify − commit − draft); C loggt ms pro Runde | +5–12 % beim besten Arm (Schätzung); F zeigt, was der Tier überhaupt bringt | ja, je Arm |
| **E5 pcie_frac** | kein Restart: per Request `strata_tune {"pcie_frac": x}` mit x ∈ {0.55, 0.7, 0.85, 1.0} (`tools/calibrate.py` um 0.85/1.0 erweitern, langer Agent-Prompt) | Decode-tok/s, PCIe-Anteil | 0–5 %, Vorzeichen über 0,75 offen | nein, **aber HTTP an den Live-Server**, also nur in einem ruhigen Fenster |
| **E6 VRAM für Experten** | `--vram-reserve-mib 1000`; separat `--kv-resident 32768` + env `STRATA_KV_PREFETCH=1` | Slot-Anzahl, VRAM frei nach einem 131K-Request, Decode, kalter 95K-Prefill, Commit | +0,5 Punkte bzw. +1–1,5 %; KV-Streaming +1–3 % für ~1,7 GiB Commit (Schätzung) | ja |
| **E7 Profil** | Start mit dem gelernten Profil aus E0, zusätzlich `make_profile.py --reorder` aus dem Trace | VRAM-Anteil der ersten 50 Requests, Exchanges/Token | +1–3 Punkte, vor allem nach Restarts (Schätzung) | ja |
| **E8 Batching** (nur wenn E0 echte Überlappung zeigt) | `"parallel": 2` + Arg `--batch-mtp`; Kontrollarm ohne `--batch-mtp` | `parallel_bench.py` N=1/2/4 agg_gen, Solo-Regression via `bench_strata.py`, Zeile `--batch 2: the slot sessions take X GiB` | +20–45 % nur bei Überlappung, Solo −3–7 % (Schätzung); die Engine kann beim Start abbrechen | ja |

**Code-Arbeiten (nach den Messungen):**

- **klein:**
  - PLE-`io_report` pro Request in serve (`ngram.cpp:529-544`)
  - Unbuffered-Heuristik korrigieren (`expert_source.cpp:1105-1108`)
  - lend region unter Budget
  - Commit-bewusste Parking-Admission (`conversation_memory.cpp:37-41`)
  - Router-ahead-Recall-Zähler auf VRAM-Misses
  - PCIe-Staging auf einen Side-Stream (`verify.cpp:1260-1270`)
  - Timer für die Adapt-Runde in serve
- **mittel:**
  - Size-Class-Exchange-Rotation
  - automatischer SSD-Tier für Konversationen auf X:
- **groß:** Cross-Layer-Prefetch in GPU-Staging (erst bei gemessener Recall ≥ 0,5).

---

## 5. Was man NICHT tun sollte

1. **Nicht zurück auf `--ple-io mmap` und nicht zurück mit dem Modell auf X:.** Alle 9 Stalls passierten in dieser Kombination bei Layer 1 (PLE). In mmap macht `gather_batch` 131.072 serielle Page-Faults ohne Prefetch (`ngram.cpp:522-523`); gemessen wurden ~275 Faults/s während eines Stalls (log 424–430).
2. **Keinen zusätzlichen gepinnten oder geparkten RAM ohne vorher freigemachten Commit.** Das betrifft `--resident-experts`, Parking, `--kv-resident`, `--adapt-async` und Batching-Slot-Kopien. Bei 2,6–5,8 GB Commit frei droht Commit-Erschöpfung für das ganze System. Parking prüft den Commit nicht, und ein Park-Fehler beendet die Engine.
3. **`--resident-budget-gib` nicht für einen Speed-Hebel halten.** 40 und 54 verhalten sich identisch (das Komplement hat 34,40 GiB). Der einzige praktische Effekt ist, dass die lend region abgeschaltet wird.
4. **`--vram-reserve-mib` nicht so weit senken, dass < ~400–500 MiB frei bleiben.** WDDM-Oversubscription kann katastrophal enden, und die Engine verkleinert dann ohnehin den Cache.
5. **Den PLE-Cache nicht blind unter ~4M Zeilen setzen**, und die 134M→32M-Änderung nicht ohne Messung (E1) live einspielen. Der Effekt ist plausibel neutral, aber nicht gemessen.
6. **Kein `"parallel"` ohne `--batch-mtp`.** Es kostet Solo-Speed und VRAM und bringt keinen Durchsatz.
7. **`STRATA_EXCHANGE_ROTATE=1` bringt bei diesem Pack nichts** (gemischte Blob-Größen).
8. **`STRATA_UNBUFFERED_LOAD=0` oder `--resident-experts` nicht ohne den Arm `STRATA_LOOKAHEAD=0` bewerten.** Beide schalten nebenbei den Lookahead-Thread ein, und der B6-Lauf zeigt eine ungeklärte Decode-Regression von 15–20 %.
9. **Keine Benchmarks gegen den Live-Server** und keine zweite Engine daneben (VRAM und Commit reichen nicht).
10. **Working-Set-Zahlen unter WDDM nicht als Speicherbuchhaltung verwenden** (WS < gepinntes Komplement). Commit-Zähler sind die verlässliche Größe.

---

## 6. Offene Fragen

1. **Commit-Drift:** Welcher Verbraucher wächst zur Laufzeit von ~12,6 auf 2,6–5,8 GiB frei? Und stimmt die WDDM-These (~30 GiB der Private Bytes)? Zu klären mit VMMap/RAMMap (read-only).
2. Ist die **blockierende Adapt-Runde** hinter Commit+Draft versteckt oder liegt sie auf dem kritischen Pfad? Davon hängt ab, ob E4 +2 % oder +20 % bringt. Wie verteilt sich ein Exchange auf D2H-Sync, H2D-Wait und memcpy?
3. **Router-ahead-Recall auf VRAM-Misses** bei diesem Modell mit Hyper-Connections. Ungemessen; Voraussetzung für den Prefetch.
4. **Echte Gleichzeitigkeit** der Agent-Requests (queue_s), also Parking oder Batching?
5. Laufen **Parking und Session-Files unter Windows/CUDA** korrekt (Greedy-Parität, Restore-Zeit)? Bisher nur auf Linux verifiziert.
6. **PLE-Hit-Rate, Füllstand und Stage-ms auf C:.** In serve nicht geloggt. Wie viel Decode-Potenzial steckt überhaupt noch in PLE?
7. **Ursache der B6-Decode-Regression:** Lookahead-Thread, 184 MiB VRAM frei oder Rauschen?
8. **Stall-Ursache:** mmap-PLE-Faults oder Expert-Transient-Reads von X:? Die Dumps `Strata\strata-stall-*.dmp` würden das mit cdb/WinDbg zeigen.
9. Sind die „reused 16384"-Re-Reads (log 1278, 1534, 2002, 2152, 2487, 2736) echte Compactions oder Prompt-Edits (Timestamps, Tool-Liste)? Wie lang ist der wirklich gemeinsame Präfix (Root-Checkpoint bzw. `--prompt-cache-every 8192`)?
10. Bleiben die Grenzkosten von 44 µs pro CPU-Entry und 20 µs pro PCIe-Entry stabil, wenn `pcie_frac` steigt? Die Regression ist kollinear.
---

## 7. Messergebnisse (2026-10-08, 09:15–10:52, Serien E/F2/G)

Harness: `agent_bench.py` (Port 8090, Engine pro Arm neu gestartet). Workload: 9 kurze Decodes (chat/code/de × 3), Main-Agent 91K kalt → +3K Suffix → Sub-Agent 18K → Rückkehr Main → Rückkehr Sub → Rückkehr Main → langer Decode. Rohdaten: `agent_results.jsonl`, Engine-Logs `logs/engine-<arm>.log`. Drift-Check A0 vs. A0-again: 125,2 vs. 125,3 tok/s, M2 25,3 vs. 25,5 s.

| Arm | Änderung | Decode kurz (Median) | Main-Rückkehr TTFT | Sub-Rückkehr | +3K Suffix | Kalt 91K | Commit frei min |
|---|---|---|---|---|---|---|---|
| A0 / A0-again | Live-Config | 125,2 / 125,3 | 25,3 / 25,5 s | 8,1 s | 3,05 s | 90 s | 5,9 GiB |
| E1a | `--ple-row-cache 33554432` | 124,8 | 25,4 s | 8,1 s | 3,05 s | 90 s | **14,8 GiB** |
| E1b | `--ple-row-cache 16777216` | 126,4 | 25,3 s | 8,0 s | 3,03 s | 90 s | 16,2 GiB |
| E2 | E1a + Parking 4 GiB | 125,9 | **2,6 s** | 2,3 s | 3,05 s | 91 s | 11,6 GiB |
| E3a | E1a + `--resident-experts` (Lookahead aus) | 126,2 | 17,0 s | 4,0 s | **1,87 s** | **76 s** | 11,6 GiB |
| E3b | wie E3a, Lookahead an | 124,3 | 17,0 s | 3,9 s | 1,84 s | 76 s | 11,4 GiB |
| F0 / F0-again / G2 | **Kombi**: E1a + resident-experts + Parking 6 GiB/3 Slots | 129,7 / 131,5 / 128,4 | **1,4–1,5 s** | 1,2 s | 1,8 s | 76 s | 7,8 GiB |
| F1 | Kombi + `STRATA_ADAPT_LAG=2` | 134,5 | 1,5 s | 1,2 s | 1,8 s | 75 s | 7,8 GiB |
| **F2 / G1 / G3** | **Kombi + `--adapt-async 1`** | **147,6 / 146,6 / 146,5** | 1,4–1,5 s | 1,2 s | 1,8 s | 76 s | 7,4 GiB |
| F3 | Kombi + `--adapt-swaps 48` | 131,4 | | | | | 7,9 GiB |
| F4 | Kombi + `--adapt-decay 0.85` | 133,6 | | | | | 7,8 GiB |
| F5 | Kombi + `--adapt-every 0` (ohne adaptive Schicht) | **101,5** (VRAM-Anteil 72 % statt 90 %) | | | | | 8,0 GiB |
| G4 | Async + `--adapt-swaps 192` | 146,0 | | | | | 7,0 GiB |
| G5 | Async + `--adapt-every 2` | 144,8 | | | | | 7,2 GiB |
| G6 | Async + `--adapt-decay 0.85` | 141,9 | | | | | 7,5 GiB |

Befunde:
- **Async-Tier: +13 % Decode**, in allen 9 Einzelmessungen über der Kombi; eine Runde = 76 ms zwischen den Windows statt blockierend. Weitere Tuning-Arme (Swaps, Frequenz, Decay) bringen mit Async nichts.
- **Die adaptive Schicht selbst ist +28 % wert** (F5). Das statische Profil allein deckt nur 72 % ab.
- **Parking funktioniert unter Windows**: Restore 41–91 ms, Park 52–340 ms. 4 GiB reichen nicht für Main (3,2 GB) + Sub (1,3 GB) → 6 GiB.
- **resident-experts**: Prefill-Pfad liest nicht mehr von der SSD; neue Suffixe ~3000 statt ~1400 tok/s; Decode unverändert. Lookahead neutral (B6-Regression nicht reproduziert).
- **PLE-Cache 134M → 32M Zeilen**: kein messbarer Speed-Unterschied, +8,8 GiB Commit.
- **Kalter Start**: Der erste lange Prompt nach einem Neustart läuft nur mit ~1000 tok/s (90 s für 91K), warm ~3800 tok/s.
- Der Wert "Decode bei langem Kontext" ist nicht aussagekräftig (das Modell beendet M4 nach 60–170 Tokens).

### Routing-Trace (F6, 11.003 Decode-Tokens × 48 Layer, `trace_analyze.py`)

| Vorhersage von Layer L+1 | Recall @20 | nur kalte Experten @20 |
|---|---|---|
| globale Häufigkeit | 11 % | 0 % |
| Paar-Korrelation L → L+1 | 33 % | 4 % |
| vorheriger Token, gleicher Layer | 42 % | 44 % |
| beides | 57 % | 45 % |

Zeitliche Wiederverwendung: 42 % der Experten vom vorigen Token, 60 % aus den letzten 4. Für 90 % der Aufrufe braucht ein Layer ~246 von 512 Experten. Fazit: Statische Paar-Korrelation hilft beim Prefetch nicht, weil die "100 %-Paare" nur Experten sind, die sowieso immer aktiv sind. Der nutzbare Hebel ist die zeitliche Lokalität, und die nutzt der Async-Tier bereits.

### Empfohlene Live-Config (bereit: `Strata/strata-unsloth-ud-iq4_xs.json.optimized`)

Änderungen gegenüber live: `--ple-row-cache 33554432`, `--resident-budget-gib 40` entfernt, `--resident-experts`, `--conversation-cache-mib 6144 --conversation-cache-slots 3`, `--adapt-async 1`.
Erwartung: Decode +17 % (125 → 147 tok/s), Main/Sub-Wechsel 25 s → 1,5 s, Suffix-Prefill ~40 % schneller, Commit frei ~7,4 statt 5,9 GiB. Startzeit ~85 s statt ~30 s.
Achtung: `--adapt-async` ist nicht bit-exakt von Lauf zu Lauf. Ein fehlgeschlagener Park- oder Restore-Vorgang beendet die Engine (siehe 3d). In ~10 Park/Restore-Vorgängen ist das nicht aufgetreten.

---

## 8. Messergebnisse Serie H (2026-10-08, 13:03–15:50), Basis: optimierte Config

Alle Arme laufen auf `Strata/strata-unsloth-ud-iq4_xs.json.optimized`. Die Referenz schwankt über den Tag zwischen 138,8 und 147 tok/s (H3/H20/H26/H50: 138,8 / 143,0 / 142,0 / 143,4). Unterschiede unter ~4 % sind deshalb Rauschen.

### KV-Cache auf SSD (Session-Dateien, `--slot-save-path`)
| | X: (P3) | C: (SN850X) |
|---|---|---|
| Save, 91K Tokens = 1,63 GB | 1,2–1,5 s | 0,9–1,0 s |
| Restore nach Engine-Neustart | 1,09 s | 0,82 s |
| erster Turn danach (3K neu) | 2,8 s | 2,4 s |
| zum Vergleich: kalt neu lesen | 76–81 s | 76–81 s |
Unter Windows funktioniert das. Gespeichert wird nur der tiefste Checkpoint plus das Ende (`conversation_file.cpp:1143`). Weicht der Kontext nach einem Restore in der Mitte ab, wird er von vorn gelesen.

### Teilweise Wiederverwendung (abweichender Kontext, ~94K)
| Szenario | 16K-Checkpoints (Default) | 8K-Checkpoints (`--prompt-cache-every 8192 --prompt-cache 12`) |
|---|---|---|
| letzte Nachricht ersetzt | 81.920 wiederverwendet, 3,1 s | 90.112, **1,8 s** |
| Abweichung bei 75 % | 65.536, 16,0 s | 65.536, 15,7 s |
| Abweichung bei 50 % | 32.768, 15,8 s | 40.960, 14,4 s |
`STRATA_CACHE_MESSAGE_BOUNDARY=1` löst bei Tool-Outputs über 1024 Tokens nicht aus und war wirkungslos. Die 8K-Checkpoints kosten Commit: Das Minimum frei sinkt von ~5,5 auf 3,65 GiB.

### Decode-Hebel
| Arm | Decode kurz | Befund |
|---|---|---|
| `pcie_frac` 0,7 / 0,85 / 1,0 | 139,6 / 133,8 / 128,1 | Default 0,55–0,7 bleibt |
| `--vram-reserve-mib` 1000 / 800 / 600 / 400 | 142,4 / 138,4 / 144,2 / 140,4 | frei laut Engine 597 / 402 / **191 LOW** / **4 LOW** MiB → **1000** empfohlen |
| **`--kv-resident 32768` + `STRATA_KV_PREFETCH=1`** | **147,6** | +568 Slots (9900), +6 %; Commit −1,1 GiB |
| `spec_min_p` 0,3 / 0,7 | 137,4 / 142,6 | 0,5 bleibt |
| `--spec` 3 / 5 / 6 | 136,7 / 139,5 / 134,2 | 4 bleibt |
| MTP-Drafts aus (`spec_min_p 1.0`; `--spec 0` verweigert ein natives Pack) | **94,0** | Speculative Decoding bringt **+53 %** |

### Kontextfenster (Nadel bei 40 %, Prompts 115K / 178K / 240K)
| Config | Slots | Decode kurz | Decode bei langem Kontext | Nadel | Commit frei min |
|---|---|---|---|---|---|
| 128K | 9332 | 141,9 | 126 (115K) | ✅ | 7,2 GiB |
| 192K | 8909 | 139,2 | 123 (178K) | ✅ | 3,4 GiB |
| 262K | 8484 | 132,7 | 123 (240K) | ✅ | 2,4 GiB |
| 192K + kv-resident | 9851 | 145,2 | 125 (178K) | ✅ | 2,8 GiB |
| **262K + kv-resident** | **9803** | **145,2** | **128 (240K)** | ✅ | **1,7 GiB** |
Mit KV-Streaming kostet 262K keine Geschwindigkeit. Die Grenze ist allein der Windows-Commit. Ohne feste Auslagerungsdatei ist das nicht betriebssicher, im Ereignisprotokoll steht zweimal Event 26 „Nicht genügend virtueller Speicher“ (07.10. 19:26, 08.10. 04:53).

### Batching
`"parallel": 2` + `--batch-mtp` stürzt beim zweiten gleichzeitigen Request ab, mit und ohne `--mtp-q4`: `strata batch: MTP admission for slot 0 failed: mtp: unsupported native MMVQ GGML type` (Bug, Logs `logs/engine-H10-batch2.log`, `engine-H41-batch2-nomtpq4.log`). Ohne gleichzeitige Requests (H11) ist es −18 % Decode, der kalte Prefill steigt von 76 auf 93 s, und der Commit fällt auf 3,1 GiB. Die Slot-Sessions kosten 3,77 GiB VRAM (−1828 Slots). → aus lassen.

### Empfohlene finale Config (`Strata/strata-unsloth-ud-iq4_xs.json.final`)
Live-Config +
`--ple-row-cache 33554432`, `--resident-experts` (statt `--resident-budget-gib 40`), `--conversation-cache-mib 6144 --conversation-cache-slots 3`, `--adapt-async 1`, `--vram-reserve-mib 1000`, `--kv-resident 32768`, env `STRATA_KV_PREFETCH=1`.
Erwartet laut den Messungen: Decode ~125 → ~145 tok/s, Agenten-Wechsel 25 s → 1,5 s, keine SSD-Reads im Prompt-Pfad, Commit frei ~6 GiB.
Mit fester Auslagerungsdatei zusätzlich: `--max-context 262144` und `--prompt-cache-every 8192 --prompt-cache 12`.
