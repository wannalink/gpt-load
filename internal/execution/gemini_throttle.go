package execution

import (
	"context"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// GeminiThrottleWindow is the sliding window duration used to group requests that
// count toward a single Gemini rate-limit episode.
const GeminiThrottleWindow = 5*time.Minute + 5*time.Second

// GeminiProactiveLimit is the maximum number of requests (reaching Google API)
// allowed within a single 5-minute sliding window before proactive throttling occurs.
const GeminiProactiveLimit = 20

// isGeminiThrottledTarget reports whether the attempt targets Google Gemini and
// is subject to proactive rate limiting and 429 throttling.
func isGeminiThrottledTarget(spec AttemptSpec) bool {
	if strings.Contains(spec.Path, "/v1beta/openai") {
		return false
	}
	channelID := strings.ToLower(strings.TrimSpace(spec.ChannelID))
	return channelID == "gemini"
}

// isResourceExhausted detects whether an attempt failed due to Gemini rate limiting or quota exhaustion:
// HTTP 429 Too Many Requests, FailureHintRateLimited, or error body/summary containing exhaustion markers.
func isResourceExhausted(statusCode int, body []byte, evidence *ErrorEvidence) bool {
	if statusCode == http.StatusTooManyRequests {
		return true
	}
	if evidence != nil {
		if evidence.StatusCode == http.StatusTooManyRequests || evidence.Hint == FailureHintRateLimited {
			return true
		}
		summary := strings.ToLower(evidence.Summary)
		code := strings.ToLower(evidence.Code)
		typeVal := strings.ToLower(evidence.Type)
		for _, marker := range []string{
			"resource_exhausted",
			"resource has been exhausted",
			"quota_exceeded",
			"quota exceeded",
			"rate_limit",
			"rate limit",
			"too_many_requests",
			"429",
		} {
			if strings.Contains(summary, marker) || strings.Contains(code, marker) || strings.Contains(typeVal, marker) {
				return true
			}
		}
	}
	if len(body) > 0 {
		lowerBody := strings.ToLower(string(body))
		for _, marker := range []string{
			"resource_exhausted",
			"resource has been exhausted",
			"quota_exceeded",
			"quota exceeded",
			"rate_limit",
			"rate limit",
		} {
			if strings.Contains(lowerBody, marker) {
				return true
			}
		}
	}
	return false
}

// geminiRateLimitState tracks the sliding rate-limiting window and request count for a single Gemini channel.
type geminiRateLimitState struct {
	mu           sync.Mutex
	windowStart  time.Time
	requestCount int
	throttled    bool

	// sendMu serializes upstream sends during recovery/throttled states.
	sendMu sync.Mutex

	// overridable for deterministic unit testing
	nowFunc   func() time.Time
	sleepFunc func(ctx context.Context, d time.Duration) error
}

func newGeminiRateLimitState() *geminiRateLimitState {
	return &geminiRateLimitState{
		nowFunc: time.Now,
		sleepFunc: func(ctx context.Context, d time.Duration) error {
			timer := time.NewTimer(d)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
				return nil
			}
		},
	}
}

func (s *geminiRateLimitState) now() time.Time {
	if s.nowFunc != nil {
		return s.nowFunc()
	}
	return time.Now()
}

func (s *geminiRateLimitState) sleep(ctx context.Context, d time.Duration) error {
	if s.sleepFunc != nil {
		return s.sleepFunc(ctx, d)
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// beforeSend must be called immediately before dispatching a request on Gemini.
// It checks proactive and reactive limits, pauses if necessary until the window elapses,
// increments the proactive dispatch count, and returns a release function to update state on response completion.
func (s *geminiRateLimitState) beforeSend(ctx context.Context) (release func(statusCode int, hitExhaustion429 bool), err error) {
	s.mu.Lock()
	now := s.now()
	if s.windowStart.IsZero() || now.Sub(s.windowStart) >= GeminiThrottleWindow {
		// First request ever or previous 5m5s window fully elapsed: start fresh window
		s.windowStart = now
		s.requestCount = 0
		s.throttled = false
	}
	isThrottled := s.throttled || s.requestCount >= GeminiProactiveLimit
	windowStart := s.windowStart
	s.mu.Unlock()

	if isThrottled {
		if d := windowStart.Add(GeminiThrottleWindow).Sub(now); d > 0 {
			log.Printf("[gemini-throttle] throttle active (count=%d, throttled=%v): pausing request for %v (window started at %v)",
				s.requestCount, s.throttled, d, windowStart.Format(time.RFC3339))
			if err := s.sleep(ctx, d); err != nil {
				log.Printf("[gemini-throttle] request canceled while waiting in throttle queue: %v", err)
				return nil, err
			}
			log.Printf("[gemini-throttle] throttle window elapsed, resuming dispatch")
		}

		s.mu.Lock()
		nowAfterSleep := s.now()
		if nowAfterSleep.Sub(s.windowStart) >= GeminiThrottleWindow {
			s.throttled = false
			s.requestCount = 0
		}
		s.mu.Unlock()
	}

	// Serialize upstream sends during recovery/throttling state
	s.sendMu.Lock()
	if err := ctx.Err(); err != nil {
		s.sendMu.Unlock()
		return nil, err
	}

	sendTime := s.now()

	s.mu.Lock()
	s.requestCount++
	if s.requestCount >= GeminiProactiveLimit {
		log.Printf("[gemini-throttle] proactive limit reached (%d/%d requests in window): engaging proactive throttle for subsequent requests",
			s.requestCount, GeminiProactiveLimit)
		s.throttled = true
	}
	s.mu.Unlock()

	var released bool
	release = func(statusCode int, hitExhaustion429 bool) {
		if released {
			return
		}
		released = true
		defer s.sendMu.Unlock()

		s.mu.Lock()
		defer s.mu.Unlock()

		if statusCode == 0 && !hitExhaustion429 {
			// Network error (Google API was NOT reached at all):
			// revert the requestCount increment.
			if s.requestCount > 0 {
				s.requestCount--
				if s.requestCount < GeminiProactiveLimit {
					s.throttled = false
				}
			}
			return
		}

		// Google API WAS reached (statusCode > 0, e.g. 200, 400, 429, 500, etc.).
		// Any request reaching Google API starts the 5m window if the timer is not currently running.
		if sendTime.Sub(s.windowStart) >= GeminiThrottleWindow {
			s.windowStart = sendTime
			s.requestCount = 1
			s.throttled = false
		}

		if hitExhaustion429 {
			// Reactive safety net triggered
			log.Printf("[gemini-throttle] 429 quota exhaustion detected (status=%d): engaging reactive throttle latch for window", statusCode)
			s.throttled = true
		}
	}
	return release, nil
}

// GeminiThrottleRegistry manages rate-limiting states per channel.
type GeminiThrottleRegistry struct {
	mu     sync.Mutex
	states map[string]*geminiRateLimitState
}

// NewGeminiThrottleRegistry creates a new registry for tracking Gemini throttle states.
func NewGeminiThrottleRegistry() *GeminiThrottleRegistry {
	return &GeminiThrottleRegistry{
		states: make(map[string]*geminiRateLimitState),
	}
}

func (r *GeminiThrottleRegistry) getState(spec AttemptSpec) *geminiRateLimitState {
	key := channelKey(spec)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, exists := r.states[key]
	if !exists {
		state = newGeminiRateLimitState()
		r.states[key] = state
	}
	return state
}

func channelKey(spec AttemptSpec) string {
	return strings.ToLower(strings.TrimSpace(spec.ChannelID))
}

// GeminiThrottledExecutor wraps an execution.Executor and enforces proactive rate limiting
// and reactive 429 throttling for Gemini channel attempts.
type GeminiThrottledExecutor struct {
	inner    Executor
	registry *GeminiThrottleRegistry
}

// NewGeminiThrottledExecutor creates a new GeminiThrottledExecutor wrapping inner.
func NewGeminiThrottledExecutor(inner Executor) *GeminiThrottledExecutor {
	return &GeminiThrottledExecutor{
		inner:    inner,
		registry: NewGeminiThrottleRegistry(),
	}
}

// Execute enforces the Gemini throttling guard before delegating to inner.Execute.
func (e *GeminiThrottledExecutor) Execute(ctx context.Context, spec AttemptSpec) AttemptResult {
	if e == nil || e.inner == nil {
		return AttemptResult{
			DispatchState: DispatchNotSent,
			Error: &ErrorEvidence{
				Kind:       ErrorKindInternal,
				OriginHint: ErrorOriginInternal,
				ScopeHint:  ErrorScopeRequest,
				Summary:    "executor is unavailable",
			},
		}
	}

	if !isGeminiThrottledTarget(spec) {
		return e.inner.Execute(ctx, spec)
	}

	state := e.registry.getState(spec)
	release, err := state.beforeSend(ctx)
	if err != nil {
		return AttemptResult{
			DispatchState: DispatchNotSent,
			Error: &ErrorEvidence{
				Kind:         ErrorKindCanceled,
				OriginHint:   ErrorOriginClient,
				ScopeHint:    ErrorScopeRequest,
				Summary:      "request canceled before dispatch: " + err.Error(),
				ReplaySafety: ReplaySafetyRejectedBeforeProcessing,
			},
		}
	}

	result := e.inner.Execute(ctx, spec)

	hitExhaustion := isResourceExhausted(result.StatusCode, result.Body, result.Error)
	statusCode := result.StatusCode
	if statusCode == 0 && result.Error != nil && result.Error.StatusCode != 0 {
		statusCode = result.Error.StatusCode
	}
	if hitExhaustion && statusCode == 0 {
		statusCode = http.StatusTooManyRequests
	} else if statusCode == 0 && result.DispatchState == DispatchNotSent {
		release(0, false)
		return result
	}

	release(statusCode, hitExhaustion)
	return result
}

// ExecuteStream enforces the Gemini throttling guard before delegating to inner.ExecuteStream.
func (e *GeminiThrottledExecutor) ExecuteStream(ctx context.Context, spec AttemptSpec, sink StreamSink) StreamResult {
	if e == nil || e.inner == nil {
		return StreamResult{
			DispatchState: DispatchNotSent,
			Error: &ErrorEvidence{
				Kind:       ErrorKindInternal,
				OriginHint: ErrorOriginInternal,
				ScopeHint:  ErrorScopeRequest,
				Summary:    "executor is unavailable",
			},
		}
	}

	if !isGeminiThrottledTarget(spec) {
		return e.inner.ExecuteStream(ctx, spec, sink)
	}

	state := e.registry.getState(spec)
	release, err := state.beforeSend(ctx)
	if err != nil {
		return StreamResult{
			DispatchState: DispatchNotSent,
			Error: &ErrorEvidence{
				Kind:         ErrorKindCanceled,
				OriginHint:   ErrorOriginClient,
				ScopeHint:    ErrorScopeRequest,
				Summary:      "request canceled before dispatch: " + err.Error(),
				ReplaySafety: ReplaySafetyRejectedBeforeProcessing,
			},
		}
	}

	result := e.inner.ExecuteStream(ctx, spec, sink)

	hitExhaustion := isResourceExhausted(result.StatusCode, nil, result.Error)
	statusCode := result.StatusCode
	if statusCode == 0 && result.Error != nil && result.Error.StatusCode != 0 {
		statusCode = result.Error.StatusCode
	}
	if hitExhaustion && statusCode == 0 {
		statusCode = http.StatusTooManyRequests
	} else if statusCode == 0 && result.DispatchState == DispatchNotSent {
		release(0, false)
		return result
	}

	release(statusCode, hitExhaustion)
	return result
}

var _ Executor = (*GeminiThrottledExecutor)(nil)
