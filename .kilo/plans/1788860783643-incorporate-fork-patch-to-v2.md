# Implementation Plan: Incorporate Fork Patch into GPT-Load v2.0

## Context & Overview
Our fork (`wannalink/gpt-load`) on top of v1.x contained two major extensions:
1. **Deployment configs**: `fly.toml` (Fly.io persistent SQLite configuration) and `docker-compose.yml` (local Dockerfile build).
2. **Gemini Proactive & Reactive Throttling**: A per-channel rate-limiting guard that prevents hitting Google's 429 quota exhaustion (`20 requests / 5m5s` sliding window, reactive 429 `Resource has been exhausted` latching, delay/queueing until window reset, and network failure counter rollback).

Upstream `tbphp/gpt-load` 2.x is a ground-up rewrite using Bifrost for multi-protocol conversion, a unified execution contract (`execution.Executor`), and a modular channel architecture. This plan defines how to rebase the fork onto 2.0 and port the patches cleanly to the new 2.0 architecture.

---

## Architectural Decisions

- **Throttling Scope**: Per-Credential (API Key) across **all routes targeting Google Gemini** (both Native Gemini protocol and Converted OpenAI/Anthropic/Responses protocols), as both consume the same upstream Google API quota.
- **Integration Layer**: Implemented as an `Executor` decorator / middleware layer in `internal/execution/gemini_throttle.go` (or integrated at the `ExecutionForwarder` / `bifrost.Runtime` boundary). This intercepts execution attempts cleanly without polluting protocol conversion or UI modules.
- **Data Persistence**: `fly.toml` will mount `/app/data` to persist both the SQLite database (`/app/data/gpt-load.db`) and `encryption.key` required by v2.x.

---

## Step-by-Step Implementation Tasks

### Step 1: Upstream Sync & Rebase to v2.0
1. Set up remote `upstream` pointing to `https://github.com/tbphp/gpt-load`.
2. Sync/rebase branch onto upstream `main` (v2.x release).
3. Clean up legacy v1-specific files (`internal/channel/gemini_throttle.go`, `internal/channel/gemini_channel.go` from v1).

### Step 2: Port Deployment Files
1. **`fly.toml`**:
   - Re-introduce `fly.toml` with:
     - `app = "gpt-load-fly"`
     - `primary_region = "lhr"`
     - `DB_DSN = "/app/data/gpt-load.db"`
     - Volume mount: `source = "gpt_storage"`, `destination = "/app/data"`
     - Port `3001` and memory settings.
2. **`docker-compose.yml`**:
   - Update `docker-compose.yml` to build from source (`build: context: . dockerfile: Dockerfile`) rather than pulling the public image.

### Step 3: Implement Gemini Rate Limiter & Throttler for v2.0
1. Create `internal/execution/gemini_throttle.go`:
   - Define sliding window duration (`5*time.Minute + 5*time.Second`) and proactive request limit (`20` requests).
   - Implement `geminiRateLimitState`:
     - Mutex-protected tracking of `windowStart`, `requestCount`, `throttled` flag, and `sendMu` for serialized dispatch during throttled state.
     - `beforeSend(ctx context.Context)`: Checks window expiration, calculates sleep duration if throttled/limit reached, pauses with `ctx.Done()` awareness, increments count, and returns `release(statusCode int, hitExhaustion429 bool)`.
     - `release`: Reverts count if `statusCode == 0` (network error / API not reached). If `hitExhaustion429` (HTTP 429 with `resource has been exhausted`), activates `throttled = true`.
   - Implement `geminiThrottleRegistry`:
     - Thread-safe map tracking `geminiRateLimitState` keyed by credential ID / API key hash.
2. Integrate with `ExecutionForwarder` or `bifrost.Runtime`:
   - Before executing an `AttemptSpec` where target channel is `gemini` (or `ProviderKind == spec.ProviderGemini`):
     - Exclude non-throttled paths (e.g. if path contains `/v1beta/openai`).
     - Acquire slot via `geminiRateLimitState.beforeSend(ctx)`.
     - Execute the attempt (`Execute` or `ExecuteStream`).
     - In result inspection, check status and response body/error summary for Gemini 429 quota exhaustion and call `release`.

### Step 4: Unit & Integration Tests
1. Create `internal/execution/gemini_throttle_test.go`:
   - Test sliding window expiration and reset after 5m5s.
   - Test proactive rate limiting: first 20 requests dispatch immediately; 21st request blocks until window reset.
   - Test reactive throttling: 429 with "Resource has been exhausted" engages throttle latch.
   - Test network failure (status 0) count rollback.
   - Test context cancellation while waiting in throttle queue.
   - Test per-credential isolation (Key A throttling does not block Key B).
2. Test end-to-end execution with converted routes (OpenAI -> Gemini) and native routes (Gemini -> Gemini).

### Step 5: Verification & Standards Check
1. Run `go test ./...` across all packages to verify no regressions in v2 test suite.
2. Verify docker build completes cleanly: `docker build -t gpt-load:test .`.

---

## Risks & Mitigations
- **Context Timeouts**: When proactive throttling sleeps until the 5-minute window resets, downstream client requests might timeout if their HTTP client timeout is shorter than the sleep.
  - *Mitigation*: The throttler listens to `ctx.Done()` so client cancellations abort immediately without leaking goroutines or incrementing request counts.
- **Multi-Instance Deployments**: Throttling state is kept in-memory per instance.
  - *Mitigation*: On Fly.io or single-instance setups, in-memory state is exact. In multi-instance setups, each instance manages its local dispatch, and reactive 429 latching acts as a secondary safety net.
