# Entwurf: ein Mechanismus für 1–4 Nutzer (Solo-Pipeline + Spec Lanes, eine geteilte VRAM-Region)

Stand 2026-10-16, Basis `Strata-ll4-lanes` @ `ee403737` (= g3-1441-ll + ll3-ovkv + ll3-lanes M1a/M1b + Interplay-Fix).
Zeilenangaben beziehen sich auf diesen Stand. Es wurde keine GPU benutzt. Alle VRAM-Zahlen stammen aus den Boot-Logs
in `results/lanes` und `results/both` oder sind daraus abgeleitet (gekennzeichnet).

## 0. Kurzfassung

- **Modi** (nie zwei gleichzeitig, jeder Wechsel an einer Fenstergrenze mit Drain wie heute):
  - **1 Nutzer:** Pipelined Solo (M1a, `GEN` auf den Haupt-Sessions, 2 Fenster in Flug), wie heute.
  - **2–4 Nutzer:** eine Spec Lane pro Nutzer auf den Slots 0/4/8/12 (`L ≤ Lmax`, Default 4).
  - **≥ 5 Nutzer oder jeder Ablehnungsgrund:** der heutige Batch-Pump, unverändert.
- **Kernidee VRAM:** Ein Verifier besteht praktisch nur aus seiner Arena: `window up to 8 tokens, 100.0 MiB of device
  buffers` (verify.cpp:885, im Log 16-mal), gemessen 102–104 MiB pro Verifier-Objekt. Die Arenen werden **über Modi hinweg
  geteilt**:
  - Lane 0 rechnet in der Arena des Haupt-Verifiers der Stufe. Der ist im Lane-Modus untätig, weil kein Pump und kein `GEN` läuft.
  - Lane 1 rechnet in der Arena des M1a-Zweitverifiers (`ver_b`).
  - Lanes 2 und 3 bekommen eigene Arenen. Auf deren Platz liegen im Solo-Modus die GDN-Snapshots von M1a.
  - **Nicht geteilt** werden die Graphen (~31 MiB pro Lane und Karte), die Lane-Drafter (50 MiB K/V + ~13 MiB Graphen pro
    Lane, nur CUDA3) und die Verifier-Objekte selbst (~3 MiB).
- **Ergebnis** (abgeleitet, ±30 MiB), freier VRAM im Dauerbetrieb, CUDA0/1/2/3:

  | Variante | `--kv-resident 32768` | `--kv-resident 28672` |
  |---|---|---|
  | heute Prod (env aus) | 1229 / 1329 / 2199 / 1539 (gemessen) | – |
  | heute M1a+M1b (BOTH) | ~762 / 837 / 1713 / 971 | 991 / 1051 / 1963 / 1191 (gemessen) |
  | **neu, Lmax = 2** | **940 / 1032 / 1910 / 1174** | 1169 / 1246 / 2160 / 1394 |
  | **neu, Lmax = 3** | **866 / 961 / 1836 / 977** | 1095 / 1175 / 2086 / 1197 |
  | **neu, Lmax = 4** | 732 / 826 / 1702 / 780 (unter der Boot-Schwelle 768 auf CUDA0) | **961 / 1040 / 1952 / 1000** |

  - M1a und 2 Lanes passen damit **ohne** den KV-Schnitt auf 28K, mit mehr Reserve als BOTH-28k heute.
  - Lmax = 3 passt bei 32K.
  - Lmax = 4 braucht 28K. Bei 32K fällt die Engine beim Boot selbst auf 3 zurück und sagt es.
  - Experten bleiben 100 % resident: kein Loan, kein Paging. Zusätzlicher Host-RAM: ~30 MiB pinned bei L = 4.
- **Erwartung pro Nutzer** (Simulation, kalibriert auf die gemessenen 18,3 ms pro Lane-Fenster bei L = 2), in tok/s:

  | Nutzer | heute | neu |
  |---|---|---|
  | 1 | 126 | 141 (gemessen) |
  | 2 | 82 | 113–121 (gemessen) |
  | 3 | ~72 | 100–112 |
  | 4 | 71 | 85–93 |

  Ab 3 Lanes ist S4 (CUDA3: Kopf + Drafter) zu 95–100 % ausgelastet.
- **Server:** keine Änderung nötig. Die Nutzer 1–4 landen schon auf 0/4/8/12 (`server.py:2075-2091`), ein einzelner geht auf `GEN`.

## 1. Was heute da ist (und wiederverwendet wird)

| Teil | Wo | Was es tut |
|---|---|---|
| M1a `STRATA_SOLO_PIPELINE=1` | generate.cpp:4083-4120, 8593-8620 (Platzprüfung), 8755-8800 (`ver_b`), 8889-8920 (GDN-Snapshots), 8951-8966 (`pl_prepare`, **lazy** beim ersten Request), 15565-15600 (Gate pro Request); solo_pipeline.hpp | Zweiter Verifier pro Stufe (`ver_b`, max_t 8), 2 Snapshots des GDN-Zustands pro Frontstufe, Haupt-Verifier mit `set_pl_mode` (verify.cpp:3851). |
| M1b `STRATA_SPEC_LANES=2` | generate.cpp:4332-4352 (asked, `kLaneVerMib = 200`), 5309-5410 (Lane-Drafter), 9000-9210 (Lane-Verifier, Boot-Capture), 11252-11302 (Catch-up), 11435-11455 (Drafter-K/V bei Aufnahme), 11503-11530 (Rückgabe), 12380-12780 (Lane-Engine), 13198-13208 (Hauptschleife); spec_lanes.hpp | Zwei Lanes auf den Slots 0 und 4, je ein Verifier pro Stufe (eigene Arena), eigener Drafter auf CUDA3, Eager-Commit, FIFO auf S4 (`lanes_serial3`, generate.cpp:12581-12606). |
| Gate | generate.cpp:12452-12468 → spec_lanes.hpp:43-52 | Startet nur, wenn **genau** L Lane-Slots aktiv sind (`STRATA_SPEC_LANES_ONE`: ≥ 1) und kein anderer Slot läuft. |
| Overlap × Lanes | generate.cpp:14852; Commit ee403737 | Kein überlappter Read bei laufenden Lanes, keine Lanes bei laufendem Read (`lanes_may_begin`). |

Gemessen (TESTLOG #82, `results/both`):

| Modus | Kosten pro Karte (CUDA0/1/2/3, MiB) | Ergebnis |
|---|---|---|
| M1a | Boot 153 / 166 / 162 / 104, dazu lazy beim ersten `GEN` 58 / 60 / 58 / 68 an Graphen | 1 Nutzer +12,8 % / +10,4 % |
| M1b (L = 2) | 256 / 266 / 266 / 396 | 2 Nutzer +45 % / +38 % |
| beide, 28K | – | 1 Nutzer 141, 2 Nutzer 113, 4 Nutzer 71 tok/s; frei 991 / 1051 / 1963 / 1191 |

Ein Lane-Fenster dauert bei L = 2 im Mittel 4089 ms / 223 = 18,3 ms (lc2-Log). Pro Fenster kommen 2,17–2,28 Tokens heraus.

## 2. Modi, Gate und Wechsel

### 2.1 Zustände

```
          GEN (Server: allein)                     BGEN / Slots aktiv
  IDLE ───────────────────────► SOLO_PL (M1a)   ┌──────────────► PUMP (heute)
   ▲                               │  STOP       │                  │   ▲
   │                               ▼             │       lanes_start│   │lanes_break / Zeile
   └──────── DONE ─────────── (pl drain) ───────┘                  ▼   │
                                                               LANES(n), 1 ≤ n ≤ Lmax
```

- **SOLO_PL** gibt es nur im `GEN`-Handler: die heutige pipelined Solo-Schleife, unverändert. Sie drainiert an ihrem Ende,
  wie heute (generate.cpp:8967ff., INVARIANT).
- **PUMP und LANES** leben in der Batch-Schleife (generate.cpp:13196ff.). Eine Zeile `GEN` kommt nur, wenn die Lanes gedraint
  sind: Jede Zeile außer `BSTOP` und `BYIELD` drainiert sie (generate.cpp:13219-13221).
- Damit ist **Exklusivität per Konstruktion** gegeben, und das ist die Voraussetzung für die Arena-Teilung in §4:
  - Der Haupt-Verifier rechnet nie, während eine Lane lebt.
  - `ver_b` rechnet nur in SOLO_PL.

### 2.2 Gate (verallgemeinert, spec_lanes.hpp)

```
lane_ok(b)     = Slot b aktiv ∧ b ∈ {0, GS, 2GS, 3GS}[0..Lmax) ∧ keine Bilder ∧ keine Penalties ∧ kein logit_bias
others         = ein aktiver Slot außerhalb der Lane-Slots
n              = # aktive Lane-Slots
lanes_start    = ¬others ∧ alle aktiven Lane-Slots lane_ok ∧ n_min ≤ n ≤ Lmax ∧ lanes_may_begin()
lanes_keep     = ¬others ∧ alle aktiven Lane-Slots lane_ok ∧ 1 ≤ n            (n ≤ Lmax folgt aus den Slots)
```

- `n_min` ist per Default 2. Mit `STRATA_SPEC_LANES_ONE=1` ist es 1.
- Mit `lanes_keep` statt eines `n == L`-Gates **endet keine Lane-Phase, nur weil ein Nutzer fertig ist**:
  - Von 4 auf 3 und von 3 auf 2 Nutzer: Die Lane des fertigen Slots wird `on = false`. Ihr Commit landet und ihr Park startet, wenn die
    Commits gelandet sind (generalisiertes `lane_park`, generate.cpp:12700-12709). Die anderen Lanes laufen ohne Drain weiter.
  - Von 2 auf 1: Die letzte Lane läuft allein weiter, mit serieller Spec-Geschwindigkeit (~118–126 tok/s statt ~81 im
    1-Zeilen-Pump). Der Server zieht den Nutzer per `_may_go_solo` ohnehin zurück auf `GEN` (`server.py:1596-1616`, 1982-2011):
    - Sein `BSTOP` endet die Lane am nächsten Verdict, ohne Drain.
    - Das folgende `GEN` drainiert und kopiert den Slot zurück.
    - Die Drafter-Rückgabe übernimmt generate.cpp:11503-11530.
    - Danach läuft SOLO_PL mit 141 tok/s.
  - `STRATA_SPEC_LANES_KEEP=0` stellt das M1b-Verhalten exakt her (Drain zum Pump, sobald `n < n_min`).

### 2.3 Wechselregeln (alle an Fenstergrenzen)

| Ereignis | heute | neu |
|---|---|---|
| 1 → 2 Nutzer | Der Server STOPpt den `GEN`, dann BGEN A mit Präfix (Slot 0) und BGEN B (Slot 4). Die pl-Schleife drainiert. | unverändert. Sobald beide Admissions durch sind, ist `lanes_start` wahr: erst `pipe_drain()`, dann `lane_begin()` (13205-13207). |
| n → n+1 (n ≥ 2) | Die BGEN-Zeile drainiert die Lanes, dann Admission, dann Pump (L = 2: Gate aus bei 3). | BGEN drainiert wie heute, dann Admission, dann `lanes_start` mit n+1 Lanes. Das Drain kostet ≤ 1 Fensterzeit (~20 ms). |
| n → n−1 (n ≥ 2) | Gate bricht, Drain, Pump | **kein Drain** (`lanes_keep`), siehe 2.2 |
| 2 → 1 | Drain, Pump, Server → `GEN` | die Lane läuft allein bis `BSTOP`, dann `GEN` (M1a) |
| ≥ 5 Nutzer oder ein Nicht-Lane-Slot aktiv | – | Die BGEN-Zeile drainiert die Lanes, `lanes_start` ist falsch, also Pump. Kommt man zurück auf ≤ 4, startet die nächste Lane-Phase, sobald nur noch Lane-Slots aktiv sind. |
| Bild-, Penalty- oder Bias-Request auf einem Lane-Slot | – | `lane_ok = false`, also Pump für alle, solange er läuft (wie heute) |
| `BSTOP` | endet am nächsten Verdict | unverändert |
| `BYIELD` mit Overlap | verworfen | unverändert |
| überlappter Prompt-Read | `lanes_may_begin` verweigert; der Read verweigert bei laufenden Lanes (ee403737) | unverändert |
| Experten-Swap oder `ar_live` falsch | `lanes_may_begin` verweigert | unverändert, für alle Lmax Lanes |

Ein Hysterese-Timer ist nicht nötig, weil jeder Wechsel durch eine Zeile oder ein BDONE ausgelöst wird. Flattern ist
höchstens so häufig wie Admissions.

## 3. Slot- und Lane-Zuordnung

- Lane j gehört zu Slot `j·GS` mit `GS = batch/groups = 16/4 = 4`, also zu den Slots 0, 4, 8 und 12.
  `spec_lanes::lane_slots(batch, groups, L, out)` (spec_lanes.hpp:29-35) kann das schon für L ≤ groups.
  `Lmax > groups` wird beim Boot auf `groups` gekappt.
- Lane j und Stufe st:
  - Verifier gebunden an `bslot_ss[st][j·GS]` (`Verifier::init` bindet `ss_` einmalig, verify.cpp:600)
  - `set_rope_base(Slot-Region)` (verify.hpp:291)
  - `set_always_publish(true)`
  - eigener `SplitDrive lane_sd[j]`
- Lane j und ihr Drafter: gebunden an die Slot-Session der letzten Stufe, PLE-History an die der ersten (generate.cpp:5386-5398).
- Feste Zuordnung von Lane zu Arena-Platz (§4): Lane j nutzt Platz j. Dabei ist Platz 0 die Arena des Haupt-Verifiers und Platz 1 die von `ver_b`.
- Der Server füllt die Slots in der Reihenfolge 0, 4, 8, 12, 1, 5, … (`slot_order`, server.py:981). `pick_slot` nimmt mit
  Gruppen zuerst Löcher unter belegten Slots derselben Gruppe, dann diese Reihenfolge (server.py:2081-2091). Bei ≤ 4 Nutzern
  gibt es keine Löcher, also landen sie auf 0/4/8/12.
  - Ausnahme: Ein Lane-Slot ist gerade `releasing` (BDONE-Wartezeit, Park). Dann kann Nutzer 4 auf Slot 1 landen.
  - Folge: `others = true`, Pump, bis Slot 1 fertig ist. Das ist harmlos. Gezählt wird es im Log als `lanes wait (slot 1 active)`.

## 4. Die geteilte VRAM-Region

### 4.1 Was eine Arena enthält und warum sie teilbar ist

Der Arena-Inhalt wird in `Verifier::init` geschnitten (`carve`, verify.cpp:698-776).

| Klasse | Felder | Lebensdauer |
|---|---|---|
| (a) Konstanten | `one_ = 1` (verify.cpp:770-776) | einmal bei `init` geschrieben |
| (b) Fenster bis Commit | `qkv_L_`, `h_L_`, `gate_L_`, `beta_L_`, `idx_raw_L_`, `tail_snap_`, `hist_snap_`, `commit_`, `R_` (letzte Stufe: Quelle des Drafters) | Inhalt eines Fensters, den sein Commit bzw. die Chain danach liest. Bei M1a liest ihn der nachgespielte Commit (generate.cpp:15560-15563). |
| (c) Scratch | alles andere, darunter `staging_` (16 × max_blob, verify.cpp:723, ≈ 37 MiB bei ~2,3 MiB pro Expert-Blob) | nur innerhalb eines Fensters |

Daraus folgt:

- **Zwei Verifier können eine Arena teilen, wenn nie ein Fenster des einen zwischen Fenster und Commit bzw. Chain des anderen
  fällt.** Das gilt für Verifier verschiedener Modi (§2.1).
- Für zwei Lanes gilt es **nicht**: Sie verschränken sich auf jeder Stufe. Lanes brauchen also paarweise verschiedene Arenen.
- Ebenso brauchen die zwei M1a-Paritäten verschiedene Arenen.
- Die Konstanten aus (a) sind bei identischem `carve` (gleiches `max_t`, gleiche Geometrie) an gleicher Stelle mit gleichem Wert. Bedingung: `max_t`
  ist überall `max(spec, min(batch, 8)) = 8`. Das gilt für den Haupt-Verifier, für `ver_b` (generate.cpp:8770) und für die
  Lanes (`lane_max_t`, generate.cpp:9025).
- Ein Init auf einer **fremden** Arena darf sie nicht nullen. Es schreibt nur die Konstanten, also dieselben Bytes.
- Ob es wirklich keinen weiteren Inhalt gibt, der über eine Modusgrenze lebt, prüft der Poison-Knopf (§10) per Test. Das ist das Hauptrisiko (§11, Nr. 1).

### 4.2 Layout pro Karte

`A` = Arena-Größe (100,0 MiB, aus `Verifier::arena_bytes()`), auf 2 MiB ausgerichtet. `S` = 2 × `gdn_snapshot_bytes` der
Stufe (CUDA0 60, CUDA1 64, CUDA2 60, CUDA3 0 MiB gemessen). `k = max(1, Lmax−1)`.

```
Platz 0  = Arena des Haupt-Verifiers der Stufe (existiert schon; NICHT in der Region)
Region   = [ Platz 1 : A ][ Platz 2 : A ][ Platz 3 : A ]        (k Plätze)
           Platz 1: ver_b (SOLO_PL) | Lane 1 (LANES)
           Platz 2: GDN-Snapshots buf[0], buf[1] (SOLO_PL, S ≤ A) | Lane 2 (LANES)
           Platz 3: –                                            | Lane 3 (LANES)
Lmax = 2:  [ Platz 1 : A ][ Snapshots : S ]                      (Snapshots hinter Platz 1)
Größe      R = k·A + max(0, S − (k−1)·A)
```

Nicht in der Region liegen:

- die Verifier-Objekte, ~3 MiB pro Stück (Streams, `qcnt_`, `slot_off_d_`, `d_spec_`, Events)
- alle Graphen
- die Lane-Drafter
- die pinned Hand-offs (Host)

### 4.3 Bilanz pro Karte (MiB)

Annahmen:

- Graphen pro Lane und Karte: Gl = 31 (CUDA1: 32). Gemessen sind 62/64 für 2 Lanes, jeder Capture kostet ~4–8 MiB (lc2-Log, Zeilen 128-222).
- M1a-Graphen: Gm = 58 / 60 / 58 / 68. Die werden heute erst beim ersten `GEN` lazy captured, neu beim Boot (§5).
- Lane-Drafter: 50 MiB plus ~13 MiB Chain-Graphen, nur auf CUDA3.

Kosten:

```
Kosten(c) = R(c) + 3·(Lmax+1) + Gm(c) + Gl(c)·Lmax + [c=3] · 63·Lmax
```

| Lmax | Kosten CUDA0/1/2/3 | heute M1a+M1b (L=2) zum Vergleich |
|---|---|---|
| 2 | 289 / 297 / 289 / 365 | 467 / 492 / 486 / 568 |
| 3 | 363 / 368 / 363 / 562 | – |
| 4 | 497 / 503 / 497 / 759 | – |

Freier VRAM im Dauerbetrieb:

- Basis ist die gemessene Prod-Basis bei 32K: 1229 / 1329 / 2199 / 1539 MiB, nach dem ersten überlappten Read (`engine-both-lc1-BOTH-prod.run.log`).
- Bei 28K kommen **abgeleitet** +229 / +214 / +250 / +220 dazu. Das ist BOTH-28k gemessen gegen BOTH bei 32K aus den Einzelkosten. Direkt gemessen ist es nicht.

| Lmax | 32K: frei CUDA0/1/2/3 | 28K: frei CUDA0/1/2/3 |
|---|---|---|
| (heute Prod, aus) | 1229 / 1329 / 2199 / 1539 | – |
| (heute BOTH) | ~762 / 837 / 1713 / 971 | 991 / 1051 / 1963 / 1191 (gemessen) |
| 2 | 940 / 1032 / 1910 / 1174 | 1169 / 1246 / 2160 / 1394 |
| 3 | 866 / 961 / 1836 / 977 | 1095 / 1175 / 2086 / 1197 |
| 4 | **732** / 826 / 1702 / **780** | 961 / 1040 / 1952 / 1000 |

**Boot-Schwelle:** Nach allen Boot-Captures muss jede Karte ≥ `kVramSafeMib` (512, generate.cpp:1571) + 256 = **768 MiB** frei haben.

- Die 256 MiB decken lazy Batch-Layout-Graphen und Overlap-Vorbereitung. Laut Soak #74 schwankt der Bedarf um ±100 MiB.
- Reicht es nicht, reduziert die Engine `Lmax` schrittweise (4 → 3 → 2 → nur M1a → aus) und meldet es pro Karte.
- Bei 32K landet `STRATA_SPEC_LANES=4` deshalb automatisch bei 3.

**Wahl von `--kv-resident`:**

- Lmax = 2 oder 3 brauchen keinen KV-Schnitt.
- Lmax = 4 braucht 28672. Das kostet bei Kontexten über 28K mehr RAM-Lesezugriffe beim K/V-Streaming. Für die 170K-Chats (#73) muss das gemessen werden.

**Kein Einfluss auf die Experten:**

- Die Reserve ersetzt `kLaneMib` und `kPipeWindowMib` in der Cache-Größenrechnung (generate.cpp:4331, 4345-4346, 5713-5714, 5845-5856, 6330-6340), und zwar pro Karte mit `Kosten(c)`.
- Bei 100 % residenten Experten (heute 56 767 MiB Cache, alle Slots) nimmt sie nur Luft aus dem Rest.
- Würde die Reserve die Residenz unter 100 % drücken, reduziert der Boot erst `Lmax`. Er leiht nie.

**Host-RAM:**

- Pinned Staging pro Verifier: `h_ymiss_` T·K·N·4 plus `h_x_`, `h_ple_` und `plan` (verify.cpp:657-686). Das sind ~1,5–2 MiB pro Verifier.
- Lane-Hand-offs: 8 Zeilen × `handoff_floats` pro Lane und Grenze, unter 1 MiB.
- Bei L = 4 kommen 16 Lane-Verifier zusammen: ~30 MiB pinned, vorher (L = 2) ~15 MiB.
- Kein zusätzlicher Conversation-Cache-Bedarf.

### 4.4 Optional: schlanke Arenen

Nur für die Plätze 2 und 3, die nicht mit Batch-Verifiern geteilt werden. Default aus.

Zwei Varianten:

- `max_t = spec + 1` statt 8.
- `staging_` weglassen, wenn die Stufe all-resident ist (der Zero-Doorbell-Graph liest es nicht, `lanes_may_begin` verweigert ohnehin außerhalb von all-resident).

Das spart grob 25–60 MiB pro Platz und würde Lmax = 4 bei 32K über die Schwelle heben. Nötig ist aber ein Bit-Paritätsbeweis,
dass `max_t` nichts an den Fenster-Numeriken ändert. Deshalb kommt das erst nach M2b, als `STRATA_LC_LEAN=1`.

## 5. Datenstrukturen und Code

**verify.hpp / verify.cpp** (nur Host-Code, die Kernel bleiben SASS-identisch):

- `void set_external_arena(void* base, size_t bytes)` wird vor `init` aufgerufen. Dann gilt für `init`:
  - `count.used ≤ bytes` prüfen, sonst Fehler.
  - Den Schnitt auf `base` setzen, **nicht** nullen (der Besitzer hat beim Allokieren genullt), und `one_` schreiben.
  - Die Region besitzt den Speicher, der Destruktor gibt `arena_` nur frei, wenn sie eigen ist (verify.cpp:560).
- `size_t arena_bytes() const` und `void* arena() const` als Accessoren. Damit bekommt Lane 0 die Arena des Haupt-Verifiers.
- `bool poison_scratch(cudaStream_t, uint8_t pattern)`: setzt alle geschnittenen Felder außer `one_` per memset auf das Muster (Test-Knopf, §10).
- `carve_signature()`, ein Hash der Offsets: Besitzer und Mitnutzer müssen gleich sein, sonst bleiben die Lanes aus.

**spec_lanes.hpp:**

- `env_lanes` (21-24) akzeptiert "2", "3" und "4". Der Wert ist `Lmax`; −1 bleibt „nicht unterstützt“.
- `gate(...)` (43-52) wird zu `gate_start(lanes, L, others, n_min)` und `gate_keep(lanes, L, others)`.
- neu `region_bytes(Lmax, A, S, align)` und `room_need(...)`, header-only und unit-getestet.

**generate.cpp:**

- `struct LcRegion { int dev; void* base; size_t bytes; size_t slot_off[4]; size_t snap_off; }`, einmal pro Stufe, nach den
  Caches allokiert, genullt und mit 2 MiB ausgerichtet. RAII-Halter mit einer Lebensdauer über alle Verifier hinaus (Reihenfolge der Destruktoren).
- `lane_slot[2]` (5312), `lane_v[2]` (9008), `lane_sd[2]`, `lane_R[2]`, `lane_cuR[2][2]`, `lane_cu_ev`, `lane_cu_rd`, `lane_ch_ev[2]` (9010-9022), `lane_dk[2]`
  usw. (11254) und `Lane lanes[2]` (12402) werden zu `[kLanesMax = 4]` und laufen über `Lmax`.
- Zeitpunkt der Arena-Zuordnung:
  - `ver_b`-Init (8755-8800) bekommt `set_external_arena(region.slot(1))`.
  - `PlSnap.buf[0/1]` (8896-8915) kommt aus `region.snap_off` statt aus `cudaMalloc`.
  - Lane-Init (9094-9133): Lane 0 bekommt `stage_ver(st).arena()`, Lane 1 `region.slot(1)`, die Lanes 2 und 3 `region.slot(2|3)`.
- **Nur ein Satz Verifier-Objekte pro Rolle:** `ver_b` und Lane 1 bleiben getrennte Objekte (anderes `ss_`, andere Graphen) über derselben Arena.
- `kLiveMax = 32` (verify.cpp:380) reicht: 4 Haupt- + 4 `ver_b`- + 16 Lane-Verifier = 24.
- Wenn nur M1a an ist (`STRATA_SPEC_LANES` nicht gesetzt), hat die Region nur Platz 1 und die Snapshots. Das ist byte-gleich zur heutigen M1a-Allokation, nur an einer Adresse.

## 6. Graph-Capture-Plan (alles beim Boot)

Reihenfolge nach den Caches und den Haupt-Verifiern. Jeder Capture synchronisiert seinen Stream, und es ist nichts in Flug.

1. Die Region pro Karte allokieren und die Platzprüfung machen. Bei Fehlschlag Lmax reduzieren (§4.3).
2. **M1a:** `ver_b` auf Platz 1, `capture_all(o.spec)` für beide Paritäten. Der Haupt-Verifier kurz `set_pl_mode(true)`,
   Capture (nur `exec_pl_[1]`: bei all-resident teilen T ≥ 2 `exec_`, verify.hpp `pl_exec`), dann `set_pl_mode(false)`.
   Dazu `mtp.prepare_prefill` und `mtp.prepare_chain`.
   - Das ist der heutige `pl_prepare` (8951-8966), nur vorgezogen.
   - Dann ist die Zahl „frei nach dem Laden“ die echte Dauerzahl, und kein Capture fällt in den Betrieb.
3. **Lanes j = 0..Lmax−1:** `init` mit der fremden Arena, `capture_all(o.spec)` (wie 9166-9186), dann `prepare_prefill` und
   `prepare_chain` des Drafters j.
4. Die freien MiB pro Karte nach allem prüfen (≥ 768). Fällt eine Karte darunter, werden Lanes von oben abgebaut: Graphen und
   Verifier freigeben, Lmax−1, erneut messen. Das ist ein Boot-Pfad, keine Laufzeitlogik.

Regeln:

- **Keine lazy Captures** für Lane- und M1a-Graphen im Betrieb.
  - `lane_prepare` (12502-12530) wird zur reinen Prüfung, ob schon captured ist.
  - `pl_prepare` beim ersten Request wird zum No-op, wenn der Boot es getan hat.
- Grund: Ein Capture während des Betriebs würde in eine Arena schreiben, die ein anderer Modus gerade hält. Außerdem soll der
  VRAM-Verbrauch nach dem Boot flach sein.
- Die Batch-Layout-Graphen des Pumps (`ensure_batch_layouts`, 14917) bleiben lazy wie heute. Sie gehören dem Haupt-Verifier,
  laufen nur im Pump-Modus und liegen in seinem `arena_b_`, das nicht geteilt wird.
- Die Graphen backen die Zeiger auf Arena-Plätze ein. Die Plätze sind ab dem Boot fix und wandern nie.
- **Re-Capture-Fälle**, wie heute: `refresh_ar` bzw. ein Experten-Swap macht eine all-resident-Stufe kurz nicht-resident.
  - Die Lanes starten dann nicht (`lanes_may_begin`), M1a läuft seriell (`why_serial` → `out_of_vram`).
  - Es wird nichts neu captured, denn die Graphen gelten wieder, sobald `ar_live()` wahr ist.

## 7. Drafter und S4 (CUDA3) bei bis zu 4 Lanes

- **Ein Drafter pro Lane**, wie M1b:
  - eigene Draft-K/V (inhaltstragend, daher nicht geteilt)
  - geteilte Gewichte des Solo-Drafters
  - 50 MiB + ~13 MiB Graphen
- **Die K/V aktuell halten:**
  - Catch-up aus jedem Pump-Fenster für aktive Lane-Slots (generate.cpp:11272-11302, 11654-11657, 12256-12257).
  - K/V-Kopie bei der Admission (11435-11455) und Rückgabe an den Solo-Drafter bei `GEN`-Fortsetzung (11503-11530).
  - Mit 4 Lanes sind es bis zu 4 T=1-Prefills pro Pump-Runde statt 2. Bei 2 Lanes kostete das bei 16 Nutzern ±0
    (NOSTART gegen OFF: 526,1 / 524,3 und 412,0 / 412,0 tok/s, `ab-lnnost-*`). Für 4 Lanes ist das im Testfenster zu prüfen.
  - Ausweichknopf: `STRATA_SPEC_LANES_CATCHUP_MAX=n` hält nur die ersten n Lane-Slots aktuell. Die anderen starten „stale“: Ihre
    Lane läuft ohne Drafts, bis der Drafter wieder auf Stand ist. Die Tokens bleiben exakt, nur die Akzeptanz sinkt.
- **S4-Serialisierung**, als Verallgemeinerung von `lanes_serial3` (12581-12606, 12669-12674):
  - Ein globales FIFO über alle S4-Arbeit: Lane-Fenster auf `lane_stream[last]` und Chains auf den Drafter-Streams.
  - Ein einziges Event `s4_last` reicht: Jeder S4-Launch auf einem anderen Stream als der vorige wartet auf `s4_last` und
    zeichnet es danach neu auf (transitive Ordnung).
  - Das ersetzt `lane_s3_ev` und `lane_ch_ev[1-j]`.
  - Höchstens ein Lane-Fenster in Flug auf der letzten Stufe (heute 12586-12589: „die andere Lane“, neu: „irgendeine andere“).
  - Lanes, die vor S4 warten, kommen in der Reihenfolge ihres Stufe-2-Endes dran.
  - Paralleles Laufen kostete auf R9700 etwa 2× (generate.cpp:4221-4224). Deshalb bleibt seriell der Default.
  - Option `STRATA_LC_S4_POLICY=chain`: Ist eine Chain startbereit, geht sie vor ein wartendes Fenster einer anderen Lane.
    Die Chain liegt auf dem kritischen Pfad zum nächsten Stufe-0-Start, das Fenster nur auf dem eigenen Verdict.
    Erst messen, dann vielleicht Default.
- **Modell** (Discrete-Event-Simulation; S1–S3 je 4,1 ms, S4 = 3,5 ms Fenster + 2,4 ms Chain, 0,2 ms Host; kalibriert auf 18,3 ms bei L = 2):

  | L | ms pro Fenster und Lane | S4 belegt | tok/s pro Lane (2,2–2,3 Tok./Fenster) | Summe |
  |---|---|---|---|---|
  | 1 | 18,4 | 32 % | 120–125 | 122 |
  | 2 | 18,4 | 64 % | 120–125 (gemessen 113–121) | 244 |
  | 3 | 18,6 (ideal) | 95 % | ideal 118–124, realistisch **100–112** | ~330 |
  | 4 | 23,6 | 100 % | ideal 93–97, realistisch **85–93** | ~350 |

  - Realistisch heißt: Phasenkollisionen, Host-Polling für 4 Lanes auf einem Thread und Takt-Effekte, mit 5–12 % Abschlag wie im DESIGN-spec-lanes.
  - Zum Vergleich der Pump bei 4 Nutzern: 71 pro Nutzer, 245–283 in Summe.
- **Spätere Hebel (nicht in diesem Entwurf):**
  - Lastabhängige Draft-Tiefe. Die bricht „Tokens = serielle Solo-Spec“ per Konstruktion, weil andere Fenster entstehen. Siehe §9.
  - Gebündelte Chains für mehrere Lanes: Die Gewichte werden einmal gelesen, braucht aber Synchronisation der Lanes.
  - Den Drafter auf CUDA0 verlegen (`mtp_first`, generate.cpp:4225-4229, nur bei `batch == 0`). Lohnt nur zusammen mit einem anderen Split,
    sonst wandert der Engpass nach S1.

## 8. Server

- **Keine Änderung nötig.**
  - 1 Nutzer → `GEN` (M1a).
  - Ab 2 Nutzern → STOP, dann BGEN mit Präfix in Slot 0 und weitere BGEN in 4, 8, 12 (`slot_order`, `pick_slot`).
  - Ein allein gebliebener Slot → `_may_go_solo`, bis zu `SOLO_AGAIN_MAX = 2` Mal (server.py:1596).
  - Danach bleibt er im Slot und läuft dort als einzelne Lane (`lanes_keep`), also nicht mehr im 1-Zeilen-Pump.
- Optional sagt die Engine `INFO spec_lanes=<Lmax> lane_slots=0,4,8,12`. Der Server parst INFO als Dictionary und ignoriert
  unbekannte Schlüssel, wie bei `spec_lanes=2` heute.
- Eine mögliche spätere Server-Verbesserung, nicht nötig: Bei `releasing` eines Lane-Slots kurz warten statt Slot 1 zu
  nehmen. Das ist erst sinnvoll, wenn die Logs `lanes wait (slot 1 active)` oft zeigen.

## 9. Exaktheit pro Modus

1. **env nicht gesetzt:**
   - Region, Lane- und M1a-Objekte werden nicht angelegt.
   - Alle neuen Zweige hängen an `lanes_on` bzw. `solo_pl`.
   - Die Reservenrechnung ist 0, also bleiben die Cache-Größen identisch.
   - Ergebnis: byte-identisch zu heute. Nachweis: `batch_test16` env-aus gegen die Prod-Binärdatei und SASS-Vergleich mit `sasscmp_ws.py` gegen `Strata-ll3-lanes/build`.
2. **Pump bei gesetzter env:** Der Pump rechnet im Haupt-Verifier, dessen Arena zwischendurch Lane 0 benutzt hat.
   - Byte-Identität gilt genau dann, wenn kein Arena-Inhalt eine Modusgrenze überlebt (§4.1, Klasse a).
   - Nachweis:
     - (i) `batch_test16` mit `STRATA_SPEC_LANES=4` + `NOSTART` (die Arena wird nie fremd benutzt) gegen Prod
     - (ii) ein Lauf, in dem Lane-Phasen und Pump-Phasen wechseln, mit `STRATA_LC_POISON=1`. Slots, die nie Lane waren, müssen byte-gleich zu env-aus sein.
   - Der Batch-Pfad selbst (Graphen, Kernels, `arena_b_`) ist unverändert.
3. **1 Nutzer (M1a):**
   - Dieselben Graphen und Kernels wie heute. Nur `ver_b`-Arena und Snapshots liegen an anderen Adressen (2 MiB ausgerichtet,
     also kein anderer Ausrichtungspfad).
   - Pipelined gleich serielles Solo per Rollback und Commit-Replay (generate.cpp:15556-15563). Gemessen: 11/11 und FORCE_MISS 5/5 identisch.
   - Neu ist nur, dass Lane 1 dieselbe Arena benutzt. Abgesichert durch den Poison-Lauf mit `GEN` nach Lane-Phasen.
4. **2–4 Lanes:** Jede Lane verwendet:
   - dieselben T-Zeilen-Solo-Graphen auf ihren Slot-Sessions
   - dieselbe Exact-Match-Annahme
   - Philox(seed, position) (verify.cpp:2553, 3310)
   - Eager-Commit genau der emittierten Zeilen (`spec_lanes::verdict`)

   Daraus folgt:
   - Für einen Slot, der nur Admission und dann Lanes durchläuft, sind die Tokens per Konstruktion die der seriellen Solo-Spec.
     Belegt für L = 2 (HOLD 14/14).
   - Mehr Lanes ändern daran nichts. Eine Lane liest nur ihre Session, ihre Arena und ihre Hand-offs. Andere Lanes teilen nur
     den Stream (Reihenfolge), den Pool-Service (pro Lane eigenes `lane_sd`) und die Drafter-Gewichte (nur gelesen).
   - Draft-Tiefe und `spec_min_p` bleiben die des seriellen Pfads, damit die Fenster gleich sind.
5. **Wechsel Pump ↔ Lane und Solo ↔ Slot:**
   - Der Zustand eines Slots ist dann teils über den Pump-Pfad (1 Zeile pro Slot), teils über den Lane-Pfad entstanden. Das ist
     die bekannte Pfadabhängigkeit (#59), wie bei M1b.
   - Die Tokens sind dann „gleich bis auf Near-Ties“, nicht per Konstruktion.
   - Neue Referenzen sind nötig: sha256-Sets bei 2, 3 und 4 Nutzern, Multi-Turn über Lane-Grenzen, Park- und RESUME-Images aus Lane-Slots.
6. **n → n−1 ohne Drain:** Es entsteht kein neuer Pfad, die übrigen Lanes rechnen unverändert weiter.

## 10. Fehler und Fallbacks

| Fall | Verhalten |
|---|---|
| Region oder Platz passen beim Boot nicht | Lmax 4 → 3 → 2 → nur M1a → aus, pro Karte gemeldet. Die Reserve wird nachgezogen: Was nicht gebraucht wird, bleibt frei, nicht im Cache. Nie ein Loan. |
| Die Reserve würde Experten aus dem VRAM drängen | Lmax vor der Cache-Rechnung senken. Die Experten bleiben 100 % resident (Prüfung wie `lanes_asked`, generate.cpp:4340). |
| Capture schlägt beim Boot fehl | Der betroffene Modus ist aus (gemeldet), der Rest läuft. Kein Retry im Betrieb. |
| `carve_signature` ungleich (max_t oder Geometrie) | Lanes aus, gemeldet. |
| Experten nicht resident im Betrieb (Swap, Shrink) | `lanes_may_begin` und `why_serial` wie heute: Pump bzw. serielles Solo. |
| CUDA-Fehler in einer Lane | `lane_die`, also `ERR` und Engine-Ende wie im Pump (#267: alle Waits gelöst). systemd startet neu. |
| Nicht-finite Logits | wie heute (generate.cpp:12617-12631): BDONE nonfinite, Slot frei |
| Overlap-Read zu einem ungünstigen Zeitpunkt | gegenseitige Verweigerung wie heute (ee403737) |
| Server legt Nutzer 1–4 auf einen Nicht-Lane-Slot | Pump, Grund im Log einmal pro Phase |
| Watchdog | Die Lanes melden `progress_beat` pro Verdict (12639). Stall-Report über `g_live`. |

## 11. Test-Knöpfe

Neue Knöpfe:

| Knopf | Wirkung |
|---|---|
| `STRATA_SPEC_LANES=N` (2–4) | Lmax. Ohne `STRATA_SOLO_PIPELINE=0` ist M1a dann mit eingeschaltet (gemappt), `STRATA_SOLO_PIPELINE=1` allein bleibt nur M1a. |
| `STRATA_LC_SHARE=0` | Keine Arena-Teilung: jede Lane und `ver_b` mit eigener Arena wie heute. Das ist der A/B-Arm für die Region und der Notausgang. |
| `STRATA_LC_POISON=1` | Bei jedem Moduswechsel (nach Drain, vor dem ersten Fenster des neuen Modus) werden alle geteilten Arenen per `poison_scratch` (NaN-Muster) und die Snapshot-Fläche überschrieben. Der Exaktheitsbeweis für §4.1. |
| `STRATA_SPEC_LANES_KEEP=0` | M1b-Verhalten: Drain, sobald n < n_min |
| `STRATA_SPEC_LANES_HOLD=k` | Generalisiert: frische Lane-Slots warten, bis k Nutzer aufgenommen sind. Damit fahren alle Slots „Admission → nur Lane“. Der Exaktheitstreiber für 3 und 4 Lanes. |
| `STRATA_LC_S4_POLICY=fifo\|chain` | Reihenfolge auf S4, siehe §7 |
| `STRATA_SPEC_LANES_CATCHUP_MAX=n` | siehe §7 |

Bleiben wie heute:

- `STRATA_SPEC_LANES_ONE`, `_SERIAL3`, `_LOG`, `_NOSTART`, `_CATCHUP`, `_GIVEBACK`
- `STRATA_PIPELINE_SNAP_OVERLAP`, FORCE_MISS

Ohne GPU (jetzt):

- Unit-Tests: `spec_lanes_test` (Gates, `region_bytes`, `lane_slots` für 4) und `solo_pipeline_test`.
- Build mit den angegebenen Flags.
- `sasscmp_ws.py` gegen `Strata-ll3-lanes/build` muss identisch sein.

Testfenster (später, mit Freigabe, ~2,5 h):

1. **Boot-Matrix:** Lmax 2/3/4 × 32K/28K × SHARE 0/1.
   - Gemessen: frei pro Karte, Loans (müssen 0 sein), Experten resident 100 %, gewähltes Lmax.
   - Geprüft wird das Modell aus §4.3 (±30 MiB).
2. **Exaktheit:**
   - HOLD mit 2, 3 und 4 Nutzern gegen serielles Solo, greedy und T = 0,7 / seed 1234.
   - Ein Lauf mit POISON, in dem `GEN`, Lanes und Pump abwechseln.
   - `batch_test16` env-aus, NOSTART und POISON-Wechsel gegen Prod.
   - SLOT_PARK_VERIFY mit Lane-Slots.
3. **HTTP ABBA mit 1/2/3/4/8/16 Nutzern:**
   - Ziele pro Nutzer: 1 → ≥ 138, 2 → ≥ 110, 3 → ≥ 100, 4 → ≥ 85; 8 und 16 Nutzer ±0.
   - Antworten bei 8 und 16 Nutzern identisch.
4. **Lange Kontexte:** 40K- und 100K-Slots in Lanes mit gestreamter K/V, bei 28K gegen 32K.
5. **Soak:** 30 min mit wechselnder Last von 1 bis 5 Nutzern. Gezählt werden Lane-Starts, Drains, Fehler und VRAM-Drift.

## 12. Risiken, nach Rang

1. **Versteckter persistenter Arena-Inhalt.** Lane 0 teilt sich die Arena mit dem Haupt-Verifier, Lane 1 mit `ver_b`.
   - Folge wären stille Token-Abweichungen im Pump oder im Solo nach einer Lane-Phase.
   - Abwehr: Audit der `carve`-Felder, kein Nullen bei fremder Arena, `carve_signature`, Poison-Test, `STRATA_LC_SHARE=0` als Ausweg.
   - Fällt der Test durch, gibt es zwei Auswege: Platz 0 bekommt eine eigene Arena (+100 MiB pro Karte, Lmax 3 passt bei 32K dann nicht mehr), oder der persistente Teil wird vor bzw. nach jeder Lane-Phase gesichert (eine kleine Kopie).
2. **S4-Sättigung ab 3 Lanes.** Pro Nutzer könnte es unter dem Modell bleiben (Host-Polling für 4 Lanes auf einem Thread, Takt bei voller CUDA3-Last).
   - Bei 4 Lanes droht weniger Gewinn als erhofft. Die Untergrenze liegt beim Pump-Wert 71, weil Lmax per Knopf auf 3 sinken kann.
3. **VRAM-Marge.** Bei Lmax 4 und 32K liegt CUDA0 unter der Schwelle, deshalb automatisch Lmax 3. Bei 28K sind ~960 MiB frei.
   - Die lazy Batch-Layouts des Pumps sind die einzige Laufzeit-Allokation, die übrig bleibt.
   - Die 28K kosten Lesebandbreite bei 170K-Chats (zu messen).
4. **Die Graph-Kosten sind nur bei L = 2 gemessen** (31 MiB pro Lane und Karte). Ein Teil davon könnte einmaliges Modul-Laden
   sein, dann sind sie bei L = 4 kleiner. Sie könnten aber auch größer sein. Die Boot-Messung ist maßgeblich.
5. **Code-Breite:** Alle `[2]`-Felder auf `[4]` zu bringen, ein globales S4-Event, `lanes_keep` statt `n == L`. Indexfehler
   gefährden das, was mit 2 Lanes schon bewiesen ist. Abwehr: HOLD-Tests bei L = 2 zuerst, bit-gleich zu M1b.
6. **Catch-up-Kosten** bei 5–16 Nutzern mit 4 Lane-Slots im Pump (bis zu 4 Drafter-Prefills pro Runde auf dem ohnehin
   vollsten Karte). Messen, notfalls `CATCHUP_MAX`.
7. **Pfadabhängigkeit** für Gespräche, die über Moduswechsel laufen. Die alten Referenz-Hashes gelten dort nicht. Nötig sind neue Referenzen, eine Fehlfunktion ist das nicht.
8. **Boot-Captures von M1a statt lazy:** Ein Fehlschlag fällt jetzt beim Boot auf, nicht beim ersten Request. Das ist eher ein Gewinn, ändert aber die Boot-Zeit (+~1 s).
9. **Server-Randfälle** (`releasing`-Slot, BGEN-Ablehnung #1994). Folge ist nur der Pump-Fallback, im Log sichtbar.
10. **`max_t`-Bindung:** Ändert jemand `--spec` über 8 oder `--batch` unter 8, unterscheiden sich die Schnitte. `carve_signature` schaltet die Lanes dann ab, statt falsch zu teilen.

## 13. Umsetzungsschritte

| Schritt | Inhalt | Aufwand |
|---|---|---|
| M2a | `set_external_arena`, `arena()`, `poison_scratch`, `carve_signature` (verify); Region für M1a + L = 2 mit Teilung; M1a-Captures beim Boot; Reserve pro Karte. Ziel: BOTH bei 32K mit ~940 MiB frei auf CUDA0. Commit: `engine: low-concurrency region shares the verifier arenas across modes (opt-in STRATA_SPEC_LANES=2 with STRATA_SOLO_PIPELINE=1)` | 5–7 h |
| M2b | Lmax 3/4: Felder `[4]`, `gate_start`/`gate_keep`, S4-FIFO-Event, HOLD=k, `CATCHUP_MAX`, Boot-Abbau von Lmax | 6–8 h |
| M2c | Tests ohne GPU (Unit, Build, SASS), Review | 2–3 h |
| Fenster | §11, 1.–5. | ~2,5 GPU-h, mit Freigabe |

Zwischenziel, das sich schon lohnt: M2a allein (Lmax 2 bei 32K, kein KV-Schnitt) bringt 1 und 2 Nutzer auf das gemessene
BOTH-Niveau. CUDA0 hat dabei ~180 MiB mehr Luft als BOTH-28k und behält die vollen 32K residenten Zellen.

Betroffene Dateien in `/media/raid0/StrataQ38FN/Strata-ll4-lanes`:

- `src/core/verify.cpp` und `include/strata/core/verify.hpp`: externe Arena, Poison, Signatur
- `include/strata/program/spec_lanes.hpp`: Env, Gates, Größenrechnung
- `src/program/generate.cpp`: Reserve, Region, Boot-Captures, Lane-Felder `[4]`, S4-FIFO, Gate in der Hauptschleife
- `tests/program/spec_lanes_test.cpp`, `tests/program/solo_pipeline_test.cpp`

## 14. Umsetzung und Korrekturen aus dem Review (Stand 2026-10-17)

Umgesetzt in `Strata-ll4-lanes`, Branch `ll4-lanes`: Commits `25221fc9` (verify), `a069ac58` (spec_lanes + Tests),
`abdc40fb` (Engine). Keine GPU benutzt. Build sauber, Kernels SASS-identisch zu `Strata-ll3-lanes/build`.

Abweichungen vom Entwurf und eingearbeitete Review-Punkte:

- **Keine eigene `LcRegion`.** Die Plätze sind die Arenen vorhandener Verifier: Platz 0 = Haupt-Verifier der Stufe
  (Lane 0), Platz 1 = `ver_b` (Lane 1), Plätze 2/3 = eigene Arenen der Lanes 2/3. Die GDN-Snapshots werden nach dem
  Lane-Aufbau per Boot-Umzug in Lane 2s Arena gelegt (generate.cpp `lc_snaps_place`). Gleiche VRAM-Bilanz wie
  `k·A + max(0, S − (k−1)·A)` (Unit-Test prüft die Formel).
- **M1 (Review):** Der Pump teilt Platz 0 mit Lane 0 (`record_window` nutzt `arena_`; `arena_b_` hält nur
  `commitb_`/`tail_snap_b_`). Fence und Poison decken Pump → Lanes und Lanes → Pump ab.
- **H1:** `lc_fence` vor jedem Lane-Start synchronisiert Haupt-/`ver_b`-Streams, `PlSnap.side`, Solo- und
  Lane-Drafter sowie die Lane-Streams; `pl_drain` wartet zusätzlich auf `PlSnap.side`. Lanes → Pump/GEN war schon sicher.
- **H2:** `poison_arena` lässt `arg_scratch_` (selbst-rücksetzender Zähler) und `one_` aus, nur `cudaMemsetAsync`.
  `STRATA_LC_POISON=<Byte>` (z. B. 0x3c und 0x5a, zwei Läufe) füllt bei jedem Wechsel in die Lanes und aus ihnen.
  Weitere Felder mit Lesen-vor-Schreiben wurden nicht gefunden (Atomics in den Kerneln: nur `arg_scratch_`,
  `TopRow` liegt nicht in der Arena, `qcnt_` außerhalb); der Poison-Lauf im Fenster ist der Beweis.
- **M2:** Boot-Regel `frei − lazy ≥ 768` mit `lazy = STRATA_LC_LAZY_MIB` (141). Vorab-Schätzung `fit_lanes`, danach
  Prüfung mit echten Zahlen; reicht es nicht, fällt die oberste Lane weg (bei Lane 2 ziehen die Snapshots zurück in
  eigene Puffer). Erwartung Prod 32K: 3 Lanes (CUDA0 und CUDA3 begrenzen), 28K: 4 Lanes.
- **M3:** Lane-Graphen nur bis `S_mtp` (T = 1..4 + Commit).
- **M1a-Graphen beim Boot:** nur wenn `STRATA_SPEC_LANES` gesetzt ist (M1a allein bleibt lazy wie heute).
- **Gates:** `gate_start` (2..L, mit `_ONE` 1..L), `gate_keep` (Default: läuft weiter, solange ein Lane-Slot dekodiert;
  `STRATA_SPEC_LANES_KEEP=0` = M1b). `HOLD=k` hat Vorrang vor dem Start.
- **S4:** weiterhin höchstens ein Lane-Fenster auf der letzten Stufe; ein Fenster wartet auf die Chains aller anderen
  Lanes (SERIAL3 verallgemeinert), eine Chain auf das letzte S4-Fenster. `STRATA_LC_S4_POLICY` ist **nicht** umgesetzt.
- **Reserve vor den Caches:** `own·120 + (L − own)·40` MiB pro Karte (Share), höchstens die 400 MiB von M1b; Share aus:
  `L·200`.
- **Env:** `STRATA_SPEC_LANES=2..4` schaltet M1a mit ein, außer `STRATA_SOLO_PIPELINE` ist gesetzt.
  `STRATA_LC_SHARE=0`, `STRATA_SPEC_LANES_KEEP=0`, `STRATA_SPEC_LANES_CATCHUP_MAX=n`, `STRATA_LC_POISON`, `STRATA_LC_LAZY_MIB`.

Offen für das GPU-Fenster: M4 (HOLD 3/4 mit lookup-lastigen Prompts), M5 (`f_lane` aus „lanes ran … ms“ / Wandzeit),
M6 (`STRATA_COMMIT_TIMING`, `_SERIAL3=0` bei 3/4 Lanes), die Boot-Matrix aus §11.

### 14.1 Zweites Review, Korrekturen (Commits `ba107591`, `f60f4ea5`)

- **Produktion = M1b bleibt M1b.** `production.json` (T16-1441f) setzt `STRATA_SPEC_LANES=2` und
  `STRATA_SOLO_PIPELINE=1`. `STRATA_LC_SHARE` ist jetzt nur für 3 und 4 Lanes Default; bei 2 ist es aus, und daran
  hängt das ganze M1b-Profil: eigene Arena pro Lane, `KEEP=0`, Reserve 400 MiB, M1b-Bootregel (vorher
  `frei ≥ L·200 + 512 + 256`, keine Nachprüfung), keine Boot-Capture der M1a-Graphen, Lane-Graphen bis `--spec`.
  `STRATA_LC_SHARE=1` bzw. `STRATA_SPEC_LANES_KEEP=1` schalten das neue Profil auch bei 2 Lanes ein.
- **Exaktheit präziser:** Garantiert ist, dass die Lane-Fenster eines Lane-Slots seriellem Solo-Spec entsprechen. Die
  Tokens eines Nutzers insgesamt hängen vom Pfad ab: Jede neue BGEN draint die Lanes, und während eines überlappten
  Prompt-Reads (bis ~10 s bei 32K) laufen die übrigen Lane-Nutzer im Pump (Batch-Numerik, #59). Referenz-Hashes
  pro Pfad; nur HOLD-Läufe prüfen den reinen Lane-Pfad. (Folgeidee: Lanes neben dem überlappten Read weiterlaufen
  lassen, braucht eigenes Audit.)
- **Snapshots in Lane 2:** begrenzt durch `Verifier::persist_offset()` (Beginn von `arg_scratch_`/`one_`), nicht durch
  die ganze Arena. Unit-Test der Grenze.
- **Drafter der Lanes 2/3** laden erst nach den Caches (Experten bleiben resident; wenn kein Platz, fällt die Lane weg).
  `fit_lanes` rechnet sie ein und gibt die nicht gebrauchten zurück.
- Nachprüfung: Karte ohne lesbaren freien Speicher zählt als knapp. `lc_snaps_place(false)`-Fehlerpfad gibt auch die
  schon zurückgezogenen Puffer frei.
- `HOLD=1/2` ist wieder exakt M1bs Regel (ein einzelner frischer Lane-Slot wartet, auch mit Bild/Penalty).
- Das Einfalten von M1a durch `STRATA_SPEC_LANES` gilt nur, wenn die Konfiguration die Lanes zulässt (`lanes_may`).
- Carve in `strata/core/arena_carve.hpp` (Host), Test `arena_carve_test` (Signatur, Ausrichtung, Keep-Range).
- Offen: Catch-ups bei ≥ 5 Nutzern laufen für bis zu 4 Lane-Slots pro Pump-Runde (CUDA3). 16-Nutzer-A/B: aus vs. 2 vs.
  4 vs. 4 + `STRATA_SPEC_LANES_CATCHUP_MAX=2`. Mit KEEP (Profil 3/4) läuft der letzte Nutzer allein in seiner Lane
  (~126 tok/s, nicht 141 wie M1a); messen, echter Fix bräuchte den Server.
