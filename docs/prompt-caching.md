# Prompt Caching

How `bedrock-converse` caches prompts across an agent loop, the Bedrock cache rules that shape the design, and the experiments that pinned down the one rule the docs don't state.

## TL;DR

- The SDK places **one rolling cache point on the last message** before every invoke, at a single `cache_ttl`. That's it.
- It is **purely additive and non-destructive**: it never removes or moves cache points you placed yourself, and it skips the tail if you already put a point there.
- Any **static / tiered** caching (system, tools, per-user prefixes) is **the consumer's job** — place those points where you build the agent.

```python
agent.with_prompt_caching(ttl="1h")   # or "5m" (default)
```

## Background: how Bedrock prompt caching works

Caching is **prefix-based**. A cache point at position *K* caches the whole token prefix `[0..K]`. On the next request Bedrock finds the longest live cached prefix, reads it cheaply, and writes only the new tokens beyond it.

| Mechanic | Value |
|---|---|
| Cache **read** / hit | 0.1× base input |
| Cache **write**, 5m TTL | 1.25× base input |
| Cache **write**, 1h TTL | 2× base input |
| TTL refresh | **free on every hit** — a cache that's used at least every TTL never expires |
| Max breakpoints | **4** per request |
| Min tokens / checkpoint | 4,096 (Claude 4.x), 1,024 (3.7) |
| 1h TTL support | Opus 4.5, Sonnet 4.5, Haiku 4.5 — **not** Sonnet 4.6 / Opus 4.6 (5m-only) |
| Telemetry | response `usage` → `cacheReadInputTokens`, `cacheWriteInputTokens`, `cacheDetails: [{ttl, inputTokens}]` |

Billing when mixing TTLs in one request: `read(A) + 1h_write(B−A) + 5m_write(C−B)`, where A = highest live hit, B = highest 1h breakpoint after A, C = last breakpoint. **Ordering rule: longer-TTL breakpoints must appear before shorter-TTL ones.**

## The rule the docs don't state — proven empirically

> **A cache read only reuses a prefix whose every segment has TTL ≥ the checkpoint TTL you request.** Equivalently: you cannot stack a 1h checkpoint on top of a 5m-cached region. Doing so makes Bedrock fall back to the last segment that is already ≥1h and **re-write the 5m region at the 1h rate.**

This is *why* the ordering rule exists, and it's the crux of the design. Anthropic's docs deliberately leave it unspecified, so we measured it on `au.anthropic.claude-sonnet-4-5-20250929-v1:0` (ap-southeast-2).

### Experiment 1 — clean two-arm test (no gap)

Identical setup `[base]@1h, [base,mid]@5m`; the only difference is the TTL of the third checkpoint placed above it. ~27k unique tokens per block.

| New checkpoint | `cacheReadInputTokens` | `cacheWriteInputTokens` | Interpretation |
|---|---|---|---|
| **5m** over the 5m region | **54,566** (base + mid) | 27,281 @5m (new tail only) | **reused** the 5m segment |
| **1h** over the 5m region | **27,285** (base only) | 54,562 @1h | **refused** to reuse — re-wrote `mid` at 1h |

Same prefix, same bytes; flipping the checkpoint from 5m→1h collapses the read from 54k to 27k and forces a 1h re-write of the middle. That is the invariant, demonstrated.

### Experiment 2 — gap test (does an expired 5m segment orphan a later 1h segment?)

Built `[A]@1h → [A,X]@5m → [A,X,B]@1h` across requests, idled 400s (5m expires, 1h survives), re-sent.

- Setup confirmed 1h survives the gap (control: a pure-1h chain read back fully at 45,986 after 400s).
- Probe read back the **full** prefix (68,977). **No orphaning, no mid-prefix "hole."** Because advancing a 1h checkpoint over the 5m region had already **re-written that region at 1h** (the invariant) — so by gap time the whole prefix was uniformly 1h.

**Conclusion:** there is never a `[1h][5m][1h]` sandwich. The invariant prevents it by re-writing — which is exactly the cost trap below.

## Why the old rolling scheme was wrong

The previous implementation pinned a **1h "anchor"** at each turn's opening message and rolled the **5m** tail behind it. But the anchor advances past the *previous* turn's 5m tail every turn — and by the invariant, advancing a 1h checkpoint over a 5m region **re-writes that tail at 1h**. So every tool-result tail was written **twice**: 1.25× when produced (5m), then 2× to promote it next turn.

Per-turn cost (tail ≈ user-msg in size):

| Scheme | Approx per-turn write cost |
|---|---|
| Old hybrid (1h anchor + 5m tail) | 2×user + **3.25×tail** |
| Single-TTL rolling (current) | 2×user + 2×tail, no double-write |

The hybrid was strictly worse than a single-TTL scheme for any multi-turn conversation.

## The current design

One rolling point, one TTL, non-destructive:

```python
def build_payload(self, messages):
    payload = super().build_payload(messages)
    if self.prompt_caching and self.caching_supported:
        self.cache_rolling_messages(payload.get('messages') or [])
    return payload

def cache_rolling_messages(self, messages):
    if messages and not any('cachePoint' in content for content in messages[-1]['content']):
        messages[-1]['content'].append(self.cache_block(self.cache_ttl))
```

- **One TTL.** `with_prompt_caching(ttl="5m" | "1h")`. No model gating (the consumer chooses; an unsupported 1h silently behaves as 5m at the API). Caching is skipped entirely on models without cache support.
- **Rolling tail covers the whole loop.** Tool-loop iterations and new user turns are the same event — both append to the message list — so a single point at the last message, refreshed each `build_payload` (run before every invoke), caches the growing conversation. Reads everything prior at 0.1×, writes only the new delta.
- **Payload-only, never mutates your `Message` objects.** The point is appended to the serialized payload dict each build, so it's fresh every invoke (no accumulation) and your persisted history stays clean.
- **Non-destructive to consumer points.** It only appends to the last message, and the `'cachePoint' in content` guard skips the add if you already placed a point on the tail. Points you place on system / tools / earlier messages are never touched. (`remove_invalid_caching` only strips points on *unsupported* models, where nothing is cacheable.)

## Consumer responsibility: static & tiered caching

The SDK deliberately does **not** cache the system prompt or tools. Place those points yourself when building the agent — you know which parts are stable and how widely they're shared. A good multi-tenant layering, most-shared first:

```
[ global system + tools ]   ← shared across all users      (add_system_cache_point / tools cache point)
[ + per-user prefix ]       ← shared across one user's convos
... conversation messages ...
[ rolling tail ]            ← the SDK's single moving point
```

Rules to respect:
- **≤ 4 breakpoints total** (your static points + the SDK's 1 rolling). Bedrock rejects more.
- **Longer TTL before shorter** along the prefix (e.g. 1h static before a 5m tail), per the invariant.
- **Don't place a manual point on the very last message** — that's the rolling point's slot (the guard will defer to yours if you do, but you lose the moving behaviour).

## Reproducing the experiments

The probes are plain `boto3` Converse calls reading `usage.cacheDetails`. Build `[base]@1h, [mid]@5m`, then probe with a 5m vs a 1h checkpoint and compare `cacheReadInputTokens`; for the gap test, sleep > 5 min between setup and probe. Unique high-entropy filler per block (> 4,096 tokens) keeps prefixes distinct and cacheable.
