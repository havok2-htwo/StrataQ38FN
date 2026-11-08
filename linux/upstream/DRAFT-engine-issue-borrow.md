# [Engine] conversation cache: on a layer split, a sub-agent that matches only a shared head takes the whole parked conversation and drops its checkpoints

**Area:** [Engine]

## What do you want to do?

### Problem

On a multi-GPU layer split, the conversation cache never lends out a parked conversation. A new request can match only a shallow checkpoint of a parked conversation. A typical case is a sub-agent that shares the main agent's system prompt and tools. That request `take()`s the whole entry, rewrites every cell past the match and drops the checkpoints past it. The parked conversation then leaves the cache, and its own next turn has to read almost the whole prompt again.

Upstream's borrow (c9e571e0, #1164) solves this on one GPU. It is gated on `stages.empty()` in `src/program/generate.cpp` (v0.2.12, 146df3c4, line 16205):

```cpp
// (a layer split restores whole stage images only, so it never borrows; ...)
bool borrow = stages.empty() && parked.tokens > std::max(resume, slot_tokens) &&
              conversations.borrows(parked);
if (parked.tokens > std::max(resume, slot_tokens) && !borrow)
    incoming.emplace(conversations.take(parked.index));
```

As far as I can tell, the gate exists because the one-GPU borrow calls `conversation_snapshot_restore_prefix()` on the donor (generate.cpp:16446), and the split path restores each stage image whole with `conversation_snapshot_restore()` (generate.cpp:16472-16482). The comment in c9e571e0 says the same: "a layer split restores whole stage images only, so it never borrows". The commit message and the `docs/DETAILS.md` change give no other reason. A draft-less `restore_prefix` overload for later-stage images does exist (`conversation_snapshot.hpp:183`). The split path does not use it, and it has no per-stage checkpoint handling for a borrow.

After the take, the request's resume point is the shallow match. The prune at generate.cpp:16536-16544 (`off_path`: checkpoints with `ids.size() > resume`) then drops every deeper checkpoint of the donor.

The workload is an agent client: a main agent plus sub-agents. They share a head of 42 tools, about 13K tokens, and differ from tool #43 on. With `--prompt-cache-every 8192`, the sub-agent's deepest match is the 8192 grid checkpoint. Sometimes the main chat is only in the RAM cache, because an unrelated request took its slot. The sub-agent then takes the 137K-token main chat, and the main agent's next turn gets 8,192 tokens back.

### Environment

- Strata v0.2.12 (146df3c4). The stock arm uses a binary built from that tree with CUDA 13.0 and `-DSTRATA_MMQ_TAB=ON`, an upstream option. The "ours" arms use v0.2.12 plus our patch series.
- Linux, CUDA 13.0, 4x RTX 3090 24 GB, `--layer-split 12,25,37`.
- Qwen3.8-Flash-Next UD-IQ4_XS, `parallel` 16, `--kv int8`, `--max-context 262144`, MTP `--spec 4`.
- `--conversation-cache-mib 126976 --conversation-cache-slots 448 --conversation-cache-min-tokens 1024 --prompt-cache-every 8192 --prompt-cache 12`.
- presence_penalty 0, temperature 0.

### Reproduce

The requests are sent one after another to an otherwise idle server with a layer split. With pipelined batch groups, `pick_slot()` does not prefer the slot that holds a request's prefix, so each request lands in the same free slot. When OTHER is admitted, CHECK's conversation is parked, and after that the main chat exists only in the RAM cache.

1. **4 fillers:** unrelated conversations of 6.1K to 7.6K tokens, so the cache has entries.
2. **A:** a 42-tool head (about 13K tokens), a system prompt of about 6K tokens and a multi-turn history. 137,090 tokens in all.
3. **CHECK:** A, its reply and a short check question.
4. **OTHER:** an unrelated request of 888 tokens, with no tools and its own system prompt. It takes CHECK's slot, so A/CHECK is parked.
5. **SUB:** a sub-agent of 29,890 tokens. It has the same first 42 tools, then two tools of its own, its own system prompt and its own history. Its deepest match in the parked chat is the 8192 checkpoint.
6. **A2:** A, its reply and a new user message, without the check question.
7. A second cycle follows: CHECK2, OTHER2, SUB2, A3.

Commands (test server on 127.0.0.1:8091):

```sh
python serve/server.py --engine strata --config run.json --port 8091 --host 127.0.0.1
python repro_park_ram.py --url http://127.0.0.1:8091 --scenarios pingpong --sizes 136000 \
    --cycles 2 --fillers 4 --pp-other 900 --pressure none --engine-log engine.log --out repro.jsonl
```

`repro_park_ram.py` is our script. It needs only the Python standard library and an OpenAI-compatible endpoint. I can attach it.

**Expected:** A2 gets close to its full prefix (about 137K tokens) back from the parked conversation.

**Actual (stock v0.2.12):** A2 gets 8,192 tokens back, reads 129,437 tokens again and takes 23.6 s to the first token.

Engine log, stock, in log order. The lines come from SUB, from the park after CHECK2, from SUB2 and from A3:

```
strata serve: conversation cache: restored 8192 tokens (checkpoint) in 165.2 ms; parked=4 bytes=3469594444
strata serve: conversation cache: dropped 1 superseded copy of this conversation; parked=5
strata serve: conversation cache: restored 24576 tokens (checkpoint) in 64.5 ms; parked=5 bytes=5740305496
strata serve: conversation cache: restored 137622 tokens (checkpoint) in 192.8 ms; parked=5 bytes=4768222404
```

**Reproduction status:** reproduced locally on 2026-11-08, one run per arm, with a fresh server for each arm.

### Measured A/B

Arms:

- **Cb:** stock v0.2.12 binary and stock `serve/server.py`, with the same engine arguments as below. Our own opt-in `STRATA_*` switches are left out, because the stock binary does not have them.
- **Nb:** our tree, engine and server, with our production switches and `STRATA_PARK_BORROW=0`.
- **Bb:** the same as Nb with `STRATA_PARK_BORROW=1`.

Nb and Bb use the same binary and the same config. The only difference between them is this switch (apart from a per-arm log and disk-tier folder).

| Request | Prompt tokens | Cb stock: reused / TTFT | Nb borrow off: reused / TTFT | Bb borrow on: reused / TTFT |
|---|---:|---:|---:|---:|
| A (cold) | 137,090 | 0 / 24.9 s | 0 / 24.4 s | 0 / 24.5 s |
| CHECK | 137,358 | 137,290 / 0.38 s | 137,289 / 0.43 s | 137,289 / 0.43 s |
| SUB (the borrower) | 29,890 | 8,192 / 4.24 s | 8,192 / 4.25 s | 8,192 / 4.20 s |
| **A2 (next main turn)** | **137,629** | **8,192 (5.9 %) / 23.6 s** | **8,192 (5.9 %) / 23.0 s** | **137,292 (99.8 %) / 0.67 s** |
| A3 (second cycle) | 138,069 (Nb: 138,070) | 137,622 / 0.98 s | 137,677 / 0.67 s | 137,676 / 0.67 s |

What the table shows:

- The borrow on the split is what changes A2. With the switch off, our tree behaves like stock.
- A3 is fine in all three arms, because SUB2 then matches SUB's own parked copy.
- SUB2 reused 24,576 tokens on stock and 29,715 on ours, with or without the borrow. So that gap comes from other parts of our tree, not from this change.

Engine log, Bb, for SUB and A2:

```
strata serve: conversation cache: borrowed 8192 of a parked 137389-token conversation (checkpoint) in 209.5 ms; parked=5 bytes=5225704800 checkpoints=1 borrowed=1
strata serve: conversation cache: restored 137292 of 137389 tokens (checkpoint) in 211.1 ms; parked=5 bytes=4828323092
```

Costs and limits:

- On a split, the borrow restores the whole image. SUB's restore took 209.5 ms in Bb, against 165.9 ms for the take in Nb.
- The lowest free VRAM per card was within 3 MiB between the two arms: 955/950/1854/898 MiB in Bb, 958/950/1854/898 MiB in Nb.
- This is one run per arm, not a throughput benchmark.

Reply hashes in the same run, Nb against Bb (greedy):

- SUB, the borrower, has the same reply hash in both arms. So do A, CHECK, OTHER, OTHER2 and SUB2.
- A2 differs. Its prompt is the same in both arms, but Bb restores it and Nb reads it again. CHECK2 and A3 differ as well; their prompts contain A2's reply.
- The four filler replies also differ between Nb and Bb, although they run before any borrow on the same binary. So this harness does not establish token identity, and I am not claiming identity from it.

### Production data (not an A/B)

This comes from a live agent client on the same hardware: a main agent plus sub-agents that share a 42-tool head. The borrow ran on our earlier tree, which is based on 0.1.44.1, not on v0.2.12.

| | Before (07:44-10:27 UTC) | After the borrow (10:29-15:12 UTC) |
|---|---:|---:|
| Prompts of 100K tokens or more | 18 | 107 |
| Share of those prompts' tokens reused | 73.3 % | 91.7 % |
| Borrows | - | 118, 0 failed |

These are two different time windows with different volume. In an earlier stretch of the same production log, 13 of 19 misses of this kind had this cause. In one example, a sub-agent of about 29.5K tokens that shared only the 8192 root took the 132K main chat. The main chat's next turn then got 8,192 of 134,737 tokens back and read 126,545 tokens again (22.5 s).

### Proposed change

The change is opt-in behind `STRATA_PARK_BORROW=1`. With it unset, behaviour and log lines are v0.2.12's. It extends upstream's borrow rather than replacing it.

**1. A stricter rule.** `ConversationCache::borrows(m, shallow)` keeps upstream's conditions:

- the match is not the live end;
- the match is below the deepest checkpoint;
- no pinned checkpoint lies at or below the match;
- the serial check passes.

With `shallow`, it also requires the match to be less than half of the entry (`2 * match < live length`). A conversation that re-renders a few dozen tokens before its end then still takes its own image, instead of leaving a stale copy behind on each turn.

**2. On a layer split, the request borrows instead of taking:**

1. The entry is still `take()`n, so the outgoing park counts it as held and can neither evict nor supersede it. It is validated and restored whole, as today. There is no new per-stage prefix restore.
2. Before branching, the borrower copies those of the donor's checkpoints that are at or below its resume point and on its path. Each copy keeps its stage parts at the same index of every stage chain (`conversation_borrow_checkpoints`, which joins them with `conversation_checkpoints_merge`). The donor keeps all of its own checkpoints.
3. If the stage chains do not line up, or a copy hits `bad_alloc`, the request falls back to a plain take, as with the switch off. The log then says `(borrow failed (copy), taken)`.
4. On success, the borrower skips `retain()`: those K/V pages belong to the donor and go back with it. `ConversationCache::put_back(image, serial)` then returns the image to its LRU place, identified by the match's serial. It copies no K/V.
5. Anything that would have happened to the donor while parked happens at `put_back`:
   - if an entry parked after it supersedes it (the same conversation went on), the donor is dropped;
   - slot count and budget are enforced oldest-first, and that may evict the donor itself;
   - a donor left beside pinned entries only is dropped.
6. The split borrow counts in `borrowed()` and logs `borrowed N of a parked M-token conversation (checkpoint) ... checkpoints=C borrowed=K`.

With the switch on, a take also logs the donor's length (`restored N of M tokens`).

The cost of this approach is a whole-image restore for the borrower: about 210 ms at 137K tokens here, against about 166 ms for a take. A per-stage prefix restore would avoid that cost. It could build on the existing draft-less `conversation_snapshot_restore_prefix` overload, but the split path, the guard and the per-stage checkpoint parts would need more changes than this patch makes.

**3. Server.** `ConvCacheLog` in `serve/server.py` also counts the borrow lines. In v0.2.12, its `EVENT` pattern (`(parked|skipped|restored) (\d+) tokens`) does not match upstream's own one-GPU line `borrowed N tokens from a parked conversation`, so `/metrics` does not count one-GPU borrows. The patch counts that line, the split borrow line and the `restored N of M tokens` form.

### Tests

- `src/core/conversation_cache_test.cpp`: 4960 checks passed, run locally on the build in our tree. It covers:
  - the `shallow` rule next to upstream's rule, including the production case of an 8192 root against a 132,362-token entry with a message-boundary checkpoint at 132,245;
  - `put_back`: LRU place and serial identity, byte accounting, a full slot count (the oldest other entry goes, or the donor when it is the oldest), the budget, supersede while out, a donor beside pinned entries only, and a disabled cache;
  - `conversation_borrow_checkpoints`: copies with and without a split, stage parts index-aligned, misaligned or swapped chains rejected, and the donor keeping its own checkpoints.
- `serve/test_conv_cache_borrow.py`: 2 tests passed (`python -m unittest serve.test_conv_cache_borrow`). They check that the borrow and restore lines are counted, and that upstream's "cannot borrow" and "no room beside" lines are not.
- **Not run:**
  - a GPU test of the split borrow beyond the A/B above;
  - a token-identity check of the borrower against a cold read of the same prompt. I can add an identity case (borrower against `STRATA_PARK_BORROW=0`, greedy) if you say how you want it shaped;
  - HIP and SYCL builds.

## What do you do today instead?

On v0.2.12 there is no workaround on a layer split. In our tree, production runs this patch with `STRATA_PARK_BORROW=1`.

## Open questions

1. Is a whole-image restore followed by `put_back` acceptable as the split borrow? Or would you rather have a per-stage prefix restore, using the draft-less `conversation_snapshot_restore_prefix` on each stage image, so the borrower does not pay for the donor's full length?
2. Should the `shallow` rule (`2 * match < live length`) apply only on a split, or on one GPU as well? Our patch applies it on both when the switch is on, which makes one-GPU borrows stricter under the switch.
3. Does the take-then-`put_back` sequence match how you want the outgoing park and eviction to see the donor while it is out? The donor counts as held (`take()`), and `put_back` applies supersede and eviction to it afterwards.
4. Should the server change (counting the borrow lines in `ConvCacheLog`) be a separate small fix? It also fixes the counting of the one-GPU borrow line, which `/metrics` misses in v0.2.12.
5. Our patch also excludes images read back from our disk tier (`from_tier`) and slots pinned by our read-yield patch (`yield_pin`). Upstream has neither, so I would drop both conditions for upstream. Is there an equivalent case upstream where a borrow must not happen?

<!--
Before posting (submitter only, does not render):
- Search open/closed issues for #1164 follow-ups and "layer split borrow" first (CONTRIBUTING.md step 1).
- repro_park_ram.py has private paths (ROOT/Strata-143-prod, /media/raid0/...) and production references in its docstring; scrub it before attaching.
- The production numbers are from TESTLOG #94 on the 0.1.44.1-based tree (TESTLOG names base 28de225d; the borrow commit there is 09aedda6 + fixup 324b1d80); the v0.2.12 port is 3daa2f24 + cc84bdfb. Commit messages need the "port <sha>" / "ours" / TESTLOG wording removed before any PR.
- The "13 of 19" figure in TESTLOG #94 is qualified "before the RAM problem"; check the exact window before posting.
- The conversation_cache_test binary (Strata-212p/build, 2026-11-08 04:41) is newer than the last change to its sources (3daa2f24, 2026-11-07 15:52); re-run on HEAD before posting anyway.
- Engine for Nb/Bb: strata-212q-9743c131 with server Strata-212p@e55fa457; stock: strata-212-stocktab-f83fd816 with server Strata-212@146df3c4 (server env STRATA_ADMIT_SHORT_FIRST=8192 only).
- Raw data: /media/raid0/StrataQ38FN/linux/results/w212h/ (repro.jsonl, engine-{Bb,Nb,Cb}.log, cfg/run-*.json).
-->
