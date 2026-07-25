package channel

import (
	"bytes"
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
const geminiThrottleWindow = 5 * time.Minute

// isGeminiThrottledPath reports whether the request path belongs to the
// native Gemini generateContent surface and should be subject to the
// resource-exhausted throttling guard.
func isGeminiThrottledPath(path string) bool {
	if strings.Contains(path, "/v1beta/openai") {
		return false
	}
	return geminiNativeModelsPathPattern.MatchString(path)
}

// isResourceExhaustedResponse detects Gemini's quota/rate-limit error shape:
// HTTP 429 with a body such as "Resource has been exhausted (e.g. check
// quota)."
func isResourceExhaustedResponse(statusCode int, body []byte) bool {
	if statusCode != http.StatusTooManyRequests {
		return false
	}
	lower := strings.ToLower(string(body))
	return strings.Contains(lower, "resource has been exhausted") ||
		strings.Contains(lower, "resource_exhausted") ||
		strings.Contains(lower, "check quota")
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
// request once it completes.
func (s *geminiRateLimitState) beforeSend() (release func(hitLimit bool)) {
	s.mu.Lock()
	now := time.Now()
	if s.windowStart.IsZero() || now.Sub(s.windowStart) >= geminiThrottleWindow {
		// First request ever, or the previous window has fully elapsed:
		// this request becomes the new starting point for the window.
		s.windowStart = now
		s.throttled = false
	}
	throttled := s.throttled
	windowStart := s.windowStart
	s.mu.Unlock()

	if throttled {
		if d := time.Until(windowStart.Add(geminiThrottleWindow)); d > 0 {
			time.Sleep(d)
		}

		// The wait has pushed us past the original window boundary; this
		// request starts a brand-new window.
		s.mu.Lock()
		s.windowStart = time.Now()
		s.throttled = false
		s.mu.Unlock()
	}

	// Only one Gemini native request is ever in flight at a time for this
	// channel, ensuring queued requests are sent and verified sequentially.
	s.sendMu.Lock()

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
	return release
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

	release := t.state.beforeSend()

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		// Network-level failures are not quota signals.
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
