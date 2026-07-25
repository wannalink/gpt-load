package channel

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

// geminiNativeModelsPathPattern matches the Gemini native generative API
// endpoints (https://generativelanguage.googleapis.com/v1beta/models/*)
// that are subject to Gemini's per-project rate limiting. The OpenAI
// compatibility surface (/v1beta/openai/...) has its own separate quota and
// must never be throttled by this guard.
var geminiNativeModelsPathPattern = regexp.MustCompile(`/v1beta/models/`)

// geminiThrottleWindow is the sliding window used to group requests that
// count toward a single Gemini rate-limit episode.
const geminiThrottleWindow = 5*time.Minute + 5*time.Second

// isGeminiThrottledPath reports whether the request path belongs to the
// native Gemini generateContent surface and should be subject to the
// resource-exhausted throttling guard.
func isGeminiThrottledPath(path string) bool {
	if strings.Contains(path, "/v1beta/openai") {
		return false
	}
	return geminiNativeModelsPathPattern.MatchString(path)
}

// isResourceExhaustedResponse detects Gemini's exact resource exhaustion
// error: HTTP 429 with the specific body containing
// "Resource has been exhausted (e.g. check quota)."
func isResourceExhaustedResponse(statusCode int, body []byte) bool {
	if statusCode != http.StatusTooManyRequests {
		return false
	}
	return strings.Contains(strings.ToLower(string(body)), "resource has been exhausted (e.g. check quota).")
}

// geminiRateLimitState tracks the sliding throttling window for a single
// Gemini channel instance (one per group). When the upstream responds with
// a 429 "resource exhausted" error, all subsequent requests within the same
// 5-minute window (measured from the first request that opened the window)
// are paused and serialized until the window elapses. Once a request lands
// after the window has fully elapsed, it becomes the new starting point and
// the loop resets.
type geminiRateLimitState struct {
	mu          sync.Mutex
	windowStart time.Time
	throttled   bool

	// sendMu serializes upstream sends. This guarantees that whenever we are
	// recovering from (or freshly entering) a throttled state, requests are
	// dispatched and their responses verified one at a time rather than
	// concurrently, per the requirement that queued requests are sent
	// one-by-one with response verification in between.
	sendMu sync.Mutex
}

// beforeSend must be called immediately before dispatching a request that
// matches isGeminiThrottledPath. It blocks the caller while the current
// window is throttled, resets the window once it has fully elapsed, and
// returns a release function that must be invoked with the outcome of the
// request once it completes. If the request context is canceled while
// waiting, beforeSend returns the context error immediately.
func (s *geminiRateLimitState) beforeSend(ctx context.Context) (release func(hitLimit bool), err error) {
	s.mu.Lock()
	now := time.Now()
	if s.windowStart.IsZero() || now.Sub(s.windowStart) >= geminiThrottleWindow {
		// First request ever, or the previous 5m5s window has fully elapsed:
		// record this request's timestamp as the anchor for the current quota window.
		s.windowStart = now
		s.throttled = false
	}
	throttled := s.throttled
	windowStart := s.windowStart
	s.mu.Unlock()

	if throttled {
		if d := time.Until(windowStart.Add(geminiThrottleWindow)); d > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(d):
			}
		}

		s.mu.Lock()
		nowAfterSleep := time.Now()
		if nowAfterSleep.Sub(s.windowStart) >= geminiThrottleWindow {
			s.throttled = false
			s.windowStart = nowAfterSleep
		}
		s.mu.Unlock()
	}

	// Only one Gemini native request is ever in flight at a time for this
	// channel during recovery, ensuring queued requests are sent and verified sequentially.
	s.sendMu.Lock()
	if err := ctx.Err(); err != nil {
		s.sendMu.Unlock()
		return nil, err
	}

	var released bool
	release = func(hitLimit bool) {
		if released {
			return
		}
		released = true
		if hitLimit {
			s.mu.Lock()
			s.throttled = true
			s.mu.Unlock()
		}
		s.sendMu.Unlock()
	}
	return release, nil
}

// geminiThrottleTransport wraps an http.RoundTripper and applies the Gemini
// native-endpoint throttling guard around each matching request.
type geminiThrottleTransport struct {
	base  http.RoundTripper
	state *geminiRateLimitState
}

func (t *geminiThrottleTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !isGeminiThrottledPath(req.URL.Path) {
		return t.base.RoundTrip(req)
	}

	release, err := t.state.beforeSend(req.Context())
	if err != nil {
		return nil, err
	}

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		// Network-level or context failures are not quota signals.
		release(false)
		return resp, err
	}

	hitLimit := false
	if resp.StatusCode == http.StatusTooManyRequests && resp.Body != nil {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr == nil {
			hitLimit = isResourceExhaustedResponse(resp.StatusCode, bodyBytes)
			resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		} else {
			resp.Body = io.NopCloser(bytes.NewReader(nil))
		}
	}

	release(hitLimit)
	return resp, nil
}

// wrapWithGeminiThrottle returns a shallow copy of base with its Transport
// decorated by the throttling guard. The original *http.Client (which may be
// shared/cached by httpclient.Manager across groups) is left untouched;
// only this copy, held by a single GeminiChannel instance, is affected.
func wrapWithGeminiThrottle(base *http.Client, state *geminiRateLimitState) *http.Client {
	if base == nil {
		return base
	}
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	wrapped := *base
	wrapped.Transport = &geminiThrottleTransport{base: transport, state: state}
	return &wrapped
}
