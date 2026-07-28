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
// count toward a single Gemini rate-limit episode across native API endpoints.
const geminiThrottleWindow = 5*time.Minute + 5*time.Second

// geminiProactiveLimit is the maximum number of requests (with status code < 500)
// allowed within a single 5-minute sliding window before proactive throttling occurs.
const geminiProactiveLimit = 20

// isGeminiThrottledPath reports whether the request path belongs to the
// native Gemini generateContent surface and should be subject to the
// channel-wide throttling guard.
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

// geminiRateLimitState tracks the sliding rate-limiting window and request count for a single Gemini channel instance.
type geminiRateLimitState struct {
	mu           sync.Mutex
	windowStart  time.Time
	requestCount int
	throttled    bool

	// sendMu serializes upstream sends during recovery/throttled states.
	sendMu sync.Mutex
}

// beforeSend must be called immediately before dispatching a request on a native Gemini endpoint.
// It checks proactive and reactive limits, pauses if necessary until the window elapses,
// and returns a release function that updates the rate limit state based on the HTTP status code.
func (s *geminiRateLimitState) beforeSend(ctx context.Context) (release func(statusCode int, hitExhaustion429 bool), err error) {
	s.mu.Lock()
	now := time.Now()
	if s.windowStart.IsZero() || now.Sub(s.windowStart) >= geminiThrottleWindow {
		// First request ever or previous 5m5s window elapsed: start fresh window
		s.windowStart = now
		s.requestCount = 0
		s.throttled = false
	}
	isThrottled := s.throttled || s.requestCount >= geminiProactiveLimit
	windowStart := s.windowStart
	s.mu.Unlock()

	if isThrottled {
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
			s.requestCount = 0
			s.windowStart = nowAfterSleep
		}
		s.mu.Unlock()
	}

	s.sendMu.Lock()
	if err := ctx.Err(); err != nil {
		s.sendMu.Unlock()
		return nil, err
	}

	var released bool
	release = func(statusCode int, hitExhaustion429 bool) {
		if released {
			return
		}
		released = true
		defer s.sendMu.Unlock()

		s.mu.Lock()
		defer s.mu.Unlock()

		if hitExhaustion429 {
			// Reactive safety net triggered
			s.throttled = true
			return
		}

		// Proactive counting for response codes < 500
		if statusCode > 0 && statusCode < 500 {
			s.requestCount++
			if s.requestCount >= geminiProactiveLimit {
				s.throttled = true
			}
		}
	}
	return release, nil
}

// geminiThrottleTransport wraps an http.RoundTripper and applies the channel-wide throttling guard.
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
		// Network-level or context failures do not count toward quota limits.
		release(0, false)
		return resp, err
	}

	hitExhaustion429 := false
	if resp.StatusCode == http.StatusTooManyRequests && resp.Body != nil {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr == nil {
			hitExhaustion429 = isResourceExhaustedResponse(resp.StatusCode, bodyBytes)
			resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		} else {
			resp.Body = io.NopCloser(bytes.NewReader(nil))
		}
	}

	release(resp.StatusCode, hitExhaustion429)
	return resp, nil
}

// wrapWithGeminiThrottle returns a shallow copy of base with its Transport decorated by the throttling guard.
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
