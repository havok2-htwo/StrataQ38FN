# Multiturn follow-up TTFT: 1.16 s (15.10) -> 21.66 / 9.80 s (17.10): diagnosis

I worked this out from logs and code only. No GPU was used and no service was touched.

Inputs: `results/cmpa-strata-mt.txt` (today, 09:58:41, isolated :8095) and `results/cbs1441-mt.txt` (15.10, 06:45:48, production :8090).

Engine logs:
- today: `Strata/strata-cmpv-1441g.log`, second engine start at line 2135. The mt section is lines 5669-8040.
- 15.10: `Strata/strata-unsloth-ud-iq4_xs.log`, engine start at line 1126269 (`strata-g3-1441a`). The mt section is lines 1128161-1130455.

Server log: `results/cmpa-strata-server.out`, lines 330-958.

## Verdict (most likely cause)

**Nothing got slower at the follow-up itself. The schedule changed.** The cause is `STRATA_PIPE_OVERLAP=1` + `STRATA_PIPE_OVERLAP_KVS=1` (#81). It is in `CMPV-1441g-text.json` and in production; the 15.10 engine (1441a) did not have it. Its effect, step by step:

1. **Decode now runs during the 16 cold turn-1 reads.**
   - On 15.10 the slots were paused while the engine read a prompt (#80: "stream pause 7.6 s"). The 16 cold reads ran back to back for about 28 s. No mt turn 1 finished before the 16th read was admitted.
   - Today the slots kept decoding during the reads ("the slots decoded 86 / 187 / 264 / ... rows meanwhile"; 12,173 rows over 47 overlapped reads). Turn-1 replies stream at 40-44 tok/s. That speed comes from the batch-window decode that runs beside the overlapped reads, not from the spec lanes: every engine line except BSTOP/BYIELD drains the lanes first, and `lanes_may_begin` waits while an overlapped read is active (`linux/patches/v0.1.44.1-ll3/all/0001-pipe-overlap-x-spec-lanes-the-gate-keeps-a-HOLD-lane.patch`, lines 16 and 43-44). The mt section has only 10 `lanes ran` lines, each 25-40 ms with 1 window and 1-4 tokens per slot, about 0.3 s in total.
   - So **13 of the 16 turn-1 replies finish before the last cold read is admitted.** In the server log the first `done: 237 tokens in 9 s` is at line 357; the last cold read (`11,342 tokens, 34 s so far`) is at line 511.
2. **Those users send turn 2 about 1 s later, and it waits behind the remaining cold turn-1 reads.**
   - Admission is one prompt read at a time ("control lines"), in strict arrival order.
   - The engine log shows 16 cold reads (lines 5669 ... 6842), then all 32 follow-ups (6877 ... 8040). Not one follow-up was taken earlier.
   - The first follow-up read shows `11,234 of 15,336 tokens, 26 s so far` (server log line 514): it had waited about 25 s for its turn. The turn-2 p95 of 27.07 s is that wait.
3. **`STRATA_ADMIT_SHORT_FIRST=8192` cannot help, because it ranks by the TOTAL prompt length, not by the new tokens.** It was active in both runs: the systemd drop-in on 15.10, the `window-agentic-cmp.sh:35` env today.
   - `Strata-143-prod/serve/server.py:1759` calls `_take_control(cancel, len(prompt), new=True)`.
   - `_admit_next` (`server.py:1572-1582`) compares `e[0]` (= `len(prompt)`) with 8192.
   - A follow-up has 15-20K tokens in total but only about 4K new. A cold turn 1 has 11K. Every prompt is "long", so the order is pure FIFO.
   - The #656 give-way (`_shorter_waiting`, `server.py:1610-1613`, also by total length) never fires either. There are 0 `BYIELD` lines in the mt section.
4. **Overlapped reads are slower (#81: "join TTFT x1.38"). This pushes the single read pipe close to saturation, so the backlog never drains.**

   | | cold 11K read | follow-up ~4.1K read | sum of all mt reads | wall | read-pipe busy |
   |---|---|---|---|---|---|
   | 15.10 (no overlap) | 1.67-1.75 s | 0.86-0.96 s | 56.2 s | 90 s | **62 %** |
   | 17.10 (overlap) | 1.8-2.5 s (mostly 2.2) | 1.01-1.35 s | 76.0 s | 86 s | **88 %** |

   At 88 % utilisation, queueing delay rises steeply. The ~11 follow-ups that piled up during the cold phase are served at about 1.3 s each while new ones keep arriving. Turn 3 (9.80 s) is still queueing, not reading: each read is about 1.3 s.

**Why throughput and decode p50 went up while follow-up TTFT got worse:** the same time moved from inside the decode (where `dec` measures it) to before the first token (where TTFT measures it).
- On 15.10, turn-1 `dec p50 9.6 tok/s` contains the about 28 s that each reply sat paused behind the other reads.
- Today the reply decodes at about 41 tok/s, but its next turn waits in the admission queue instead.
- Wall time is 86 s vs 90 s and output is 159.8 vs 145.5 tok/s: the work as a whole got slightly faster. So for this bench the TTFT number is a real queueing regression for follow-ups, but it is not a regression in the engine's read speed.

### Check with a model (no GPU)

`/tmp/claude-1000/-media-raid0-StrataQ38FN/6d620a76-28f9-4e43-abfe-4ef98460c18b/scratchpad/sim/sim3.py` (temporary) is a small event simulation. It models one read pipe, uses each run's real reply lengths from `multiturn.jsonl` (both runs' replies; the ranges below cover both) and 1 s think time. The switches are whether decode runs during reads, the read times, and the admission order.

Decode rates are set per run:
- **No overlap (15.10):** 400-450 tok/s aggregate, 45-125 tok/s per stream at most. The real aggregate was at least 386 tok/s: 13,128 output tokens in at most 34 s outside the reads (90 s wall minus 56.2 s of reads). An earlier version of this check used 230 tok/s here; that model ran 115.6-116.6 s against 90 s measured, so it was not calibrated.
- **Overlap (today):** 230-300 tok/s aggregate, 25-45 tok/s per stream at most.

| model | turn 1 / 2 / 3 TTFT p50 (s) | wall (s) | measured |
|---|---|---|---|
| A: no overlap, reads 1.7 / 0.9 s (15.10) | 14.5 / 1.5-2.4 / 0.9-1.3 | 86-94 | 15.25 / 1.16 / 1.17, 90 s |
| B: overlap, reads 2.2 / 1.27 s (today) | 18.8 / 15.8-21.4 / 7.3-13.2 | 82-89 | 18.56 / 21.66 / 9.80, 86 s |
| C: overlap, but 15.10 read speeds | 14.5 / 8.9-14.4 / 0.9-6.7 | 63-80 | - |
| D: no overlap, but today's (overlapped) read speeds | 18.8 / 2.2-3.5 / 1.3-2.1 | 106-114 | - |
| E: overlap + follow-ups strictly first | 22.6-34.0 / 2.3-3.0 / 1.5-2.7 | 91-106 | - |
| F: overlap + follow-ups first, 30 s limit (as `_admit_next` would do it today) | 22.6-34.0 / 3.9-7.7 / 1.6-8.4 | 81-95 | - |
| G: as F, limit 45-60 s | 22.6-34.0 / 2.3-3.2 / 1.5-2.8 | 83-106 | - |

A and B reproduce both measured runs, including wall time. Across all the settings I tried, the overlap ordering is needed for the regression: slower reads alone (D) add only about 1-2 s. The overlap with 15.10's read speeds (C) already gives most or all of the turn-2 wait. **How turn 2 and turn 3 split between ordering and slower reads depends on the parameters** (C's turn 3 is 0.9-6.7 s), so I do not attribute either turn to one of them. The two effects also compound: slower reads keep the saturated pipe busy for longer.

## Candidates checked

| candidate | finding |
|---|---|
| **STRATA_PIPE_OVERLAP(_KVS)** (#81) | **Main cause** (see above). 15.10's engine logs only "adaptive overlap: off" and has no `overlapped read` lines; today there are 105 overlapped reads in the second engine run (47 of them in the mt section); the 15.10 engine section has 0. |
| STRATA_ADMIT_SHORT_FIRST=8192 | It was set in both runs, so it did not change. It ranks by total `len(prompt)` (`server.py:1497, 1535-1544, 1759`). With these prompt sizes it is plain FIFO. It is the **lever for a fix**, not the cause. |
| spec lanes 4 / solo pipeline | **Not a cause; they barely run in this workload.** The mt section has 10 `lanes ran` lines (25-40 ms each, 1 window, 1-4 tokens per slot, about 0.3 s in total): lanes are drained by every engine line except BSTOP/BYIELD and do not start while an overlapped read is active (ll3 `all/0001`, lines 16, 43-44). The turn-1 streams at about 44 tok/s (`done: ... 43.8 tok/s`) are batch-window decode beside the overlapped reads. What the lane setting does change is slot placement: cold reads land on slots 0/4/8/12 and reuse them; the follow-up is then restored from the conversation cache (`restored 11232 tokens (live) in 33.5 ms`), which is cheap. Lanes do not hold admissions: there are no `lane slot is held` lines. The solo-to-batch hand-off of the first request costs 49 ms (line 5676). |
| --kv-resident 28672 vs 32768 | **No measurable cost.** The KV-streaming hit rate is the same (~75-80 % in both logs). The one non-overlapped cold 11,017-token read today (the first mt request, line 5669) took 1,702 ms; on 15.10 the same read took 1,749 ms (line 1128161; the 16 cold reads there were 1,668-1,749 ms). It is a single sample today, but it leaves no room for a 28K penalty. No A/B arm is needed. |
| MAX_QUEUE / ordering in the overlap path | The order is strict arrival order at the server (`_admit_next`). The engine takes one admission at a time in both builds. No re-ordering was found in the log. |
| a cold read monopolising | Cold reads are 1.8-2.5 s each, with no single outlier. The monopoly is the **16 cold reads together** (about 35 s) ahead of follow-ups that arrived later. |
| 15.10 disturbed by production clients | **Ruled out.** Between the first mt read (1128161) and the last (1130455) there is exactly one foreign request: 1,261 tokens, `slot 7 takes 1261` at line 1129307, about 0.4 s. Everything else is the bench. The disturbance can only have made 15.10 slower. |
| "parallel 16 -> it will use 8" warning | Cosmetic, and present in both runs: `server.py:2893` prints it, but still passes `--batch 16`, and the engine reports 16 slot sessions. |
| T=0.7 noise | Reply lengths per turn are similar in the two runs: turn-1 median about 290 vs 260 tokens. The simulation gives the same result with either run's lengths, so noise does not explain a factor of 18. |

## What I could not rule out from logs alone

- **How much comes from read speed vs scheduling.** The engine build differs (1441a vs 1441f-lanes4). The simulation says that the overlap ordering is necessary and that slower reads alone are not enough. The split between the two is sensitive to the decode parameters (see the model table), so it remains open. The A/B arms A and B below cannot settle it either: B removes both the ordering effect and the overlapped-read slowdown. Only arm E (overlap kept, order fixed) measures the split, and it needs a server patch first.
- **Per-request arrival times.** The logs have no per-line timestamps and `bench_multiturn.py` writes none. Arrival was inferred from the server's `s so far` counters (shared across requests, so approximate) and the done/read order. One small, safe bench change would help: add `t_send` / `t_first` / `t_end` (relative to t0) to each row.

## A/B plan to confirm (GPU window, production down, about 15-25 min)

What it settles:
- A vs B (in ABBA order) confirms that the overlap causes the regression.
- It does **not** split read speed from ordering, because B removes both. That split needs arm E. E needs the "new tokens first" server patch (fix idea 1), which has not been written or reviewed yet. Write and review it before asking for the window, or run A/B only and accept that the split stays open.
- Arms for spec lanes and `--kv-resident` are dropped. Lanes barely run in this workload (10 `lanes ran` lines, about 0.3 s). The non-overlapped read time is already the same at 28K (1,702 ms) as on 15.10 (1,749 ms). A 32K arm would also have to change the lane count (32K fits only 3 lanes, #84), so it would change two settings at once.

Setup:
- Same harness as `window-agentic-cmp.sh`: an isolated `strata-bench` scope on :8095 with the drop-in env (`STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1`).
- Run only `bench_multiturn.py --users 16 --turns 3`; add `--temperature 0` so reply hashes are comparable.
- Each arm takes about 100 s to boot plus about 90 s of bench.
- Order ABBA (2 reps each of A and B); then E twice, if the patch is ready.

| arm | config (based on `configs/CMPV-1441g-text.json`) | expectation if the diagnosis holds |
|---|---|---|
| A | unchanged | follow-up TTFT p50 about 12 s (turn 2 about 20, turn 3 about 10) |
| B | remove `STRATA_PIPE_OVERLAP` and `STRATA_PIPE_OVERLAP_KVS` from `env` | follow-up about 1-2 s, turn-1 dec about 10 tok/s, wall about 90 s; engine log has no `overlapped read` lines |
| E (needs the reviewed patch) | A + server patch "short first by new tokens", `STRATA_ADMIT_MAX_WAIT_S` left at 30 | turn 2 about 4-8 s, turn 3 about 2-8 s (simulation F); turn-1 TTFT p50 rises to about 23-34 s |
| E' (optional) | E with `STRATA_ADMIT_MAX_WAIT_S=60` | turn 2 about 2-3 s, turn 3 about 1.5-3 s (simulation G); turn-1 TTFT p50 as in E |

Write the configs as `configs/MTR-A.json`, `MTR-B.json` (E and E' are A plus the patched server and env), and label the runs `mtr-<arm>-<rep>`.

Pass criteria:
- the median difference A vs B is larger than the spread between reps;
- the read times in the engine log (`overlapped read of ~4.1K in ~1.25 s` vs `read in ~0.9 s`) match the table above;
- for E: E keeps the overlapped (slower) reads but fixes the order, so E's follow-up TTFT minus B's is what remains with the order fixed. That is an upper bound on the share that comes from slower overlapped reads (the 30 s limit of fix idea 1 adds to it; E' shows how much).

Total: A/B ABBA is 4 runs x about 3.5 min, about 15 min; E (+ E') adds about 7-14 min.

## Fix idea (cause is clear enough to design, not to deploy without E)

1. **Rank admission by the tokens that must be read, not by the total** (server only, exact: only the order changes).
   - In `generate_batched` (`server.py:1759`), pass an estimate of the new tokens as `plen` instead of `len(prompt)`: `len(prompt)` minus the longest prefix shared with a `slot_held` entry (the data `pick_slot` already uses, `server.py:2092`), or with the server's recent finished requests (what `prefix_diag` already tracks, with ids + raw_ids).
   - Use the same estimate in `_shorter_waiting` and in `give_way`, so a cold 11K read can yield (#656) to a 4K follow-up at a chunk boundary.
   - With 8192 as the threshold, follow-ups (about 4K new) go first.
   - **The 30 s wait limit undercuts this.** `_admit_next` (`server.py:1580-1581`, default `STRATA_ADMIT_MAX_WAIT_S=30` at `server.py:297`) makes a long prompt that has waited 30 s count as short again, and then it goes by arrival time. In this bench the remaining cold turn-1 reads pass 30 s (turn-1 p50 is about 23-34 s under the fix), so follow-ups queue behind them again. Simulated with the limit: turn 2 3.9-7.7 s, turn 3 1.6-8.4 s (row F). Without a limit: 2.3-3.0 / 1.5-2.7 s (row E). With 45-60 s: 2.3-3.2 / 1.5-2.8 s (row G).
   - So the limit has to be decided together with the fix. Options: (a) keep 30 s and accept follow-ups of about 4-8 s, already 3-5x better than today; (b) raise it, or tie it to the cold read time (for example a multiple of the expected read of the waiting prompt), so that cold reads wait longer but still not forever; (c) no limit for ranking by new tokens, which risks starving a cold read under a steady stream of follow-ups. Evaluate this trade-off explicitly, with E vs E' in the window.
   - Cost in every variant: a cold turn 1 that arrives during a burst of follow-ups waits longer (simulation: turn-1 p50 18.8 -> about 23-34 s). This is the usual agent trade-off: a running conversation goes before a new one. A cap on how many follow-ups may pass one cold read (for example 4) is another way to bound that cost.
2. Optionally, reduce the overlapped-read penalty (x1.3-1.4) for short reads only when many reads are queued, in other words fall back to a non-overlapped read when `waiting >= N`. That is an engine change with an exactness re-check, so it is lower priority than (1).
3. Bench hygiene: record per-request timestamps in `bench_multiturn.py`, so the next diagnosis needs no log archaeology.
