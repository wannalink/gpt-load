package execution

import (
	"context"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// GeminiThrottleWindow is the sliding window duration used to group requests that
// count toward a single Gemini rate-limit episode.
const GeminiThrottleWindow = 5*time.Minute + 5*time.Second

// GeminiProactiveLimit is the maximum number of requests (reaching Google API)
// allowed within a single 5-minute sliding window before proactive throttling occurs.
const GeminiProactiveLimit = 20

// GeminiHighDemandMaxBackoff is the maximum pause duration for Gemini 503 high demand errors.
// It is capped at 4 minutes to ensure requests and pause queues do not exceed the 5-minute safeguard.
const GeminiHighDemandMaxBackoff = 4 * time.Minute

// isGeminiThrottledTarget reports whether the attempt targets Google Gemini and
// is subject to proactive rate limiting and 429 throttling.
func isGeminiThrottledTarget(spec AttemptSpec) bool {
	if strings.Contains(spec.Path, "/v1beta/openai") {
		return false
	}
	channelID := strings.ToLower(strings.TrimSpace(spec.ChannelID))
	return channelID == "gemini"
}

// isResourceExhausted detects Gemini's exact resource exhaustion response: HTTP 429 with
// the specific message "Resource has been exhausted (e.g. check quota)." in body or error summary.
func isResourceExhausted(statusCode int, body []byte, evidence *ErrorEvidence) bool {
	is429 := statusCode == http.StatusTooManyRequests ||
		(evidence != nil && evidence.StatusCode == http.StatusTooManyRequests)
	if !is429 {
		return false
	}

	const marker = "resource has been exhausted (e.g. check quota)."

	if len(body) > 0 && strings.Contains(strings.ToLower(string(body)), marker) {
		return true
	}
	if evidence != nil && strings.Contains(strings.ToLower(evidence.Summary), marker) {
		return true
	}
	return false
}

// isHighDemand503 detects Gemini's 503 high demand response ("This model is currently experiencing high demand...").
func isHighDemand503(statusCode int, body []byte, evidence *ErrorEvidence) bool {
	is503 := statusCode == http.StatusServiceUnavailable ||
		(evidence != nil && evidence.StatusCode == http.StatusServiceUnavailable)
	if !is503 {
		return false
	}
	const marker = "this model is currently experiencing high demand"
	if len(body) > 0 && strings.Contains(strings.ToLower(string(body)), marker) {
		return true
	}
	if evidence != nil {
		summary := strings.ToLower(evidence.Summary)
		code := strings.ToLower(evidence.Code)
		typeVal := strings.ToLower(evidence.Type)
		if strings.Contains(summary, marker) || strings.Contains(code, marker) || strings.Contains(typeVal, marker) {
			return true
		}
	}
	return false
}

// geminiTargetModel extracts the model name for Gemini rate limiting and endpoint pausing.
func geminiTargetModel(spec AttemptSpec) string {
	if spec.UpstreamModel != "" {
		return spec.UpstreamModel
	}
	if spec.ClientModel != "" {
		return spec.ClientModel
	}
	if idx := strings.Index(spec.Path, "/models/"); idx != -1 {
		rest := spec.Path[idx+len("/models/"):]
		if colon := strings.Index(rest, ":"); colon != -1 {
			return rest[:colon]
		}
		if slash := strings.Index(rest, "/"); slash != -1 {
			return rest[:slash]
		}
		return rest
	}
	return ""
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

// GeminiThrottleRegistry manages rate-limiting states per channel and model pauses.
type GeminiThrottleRegistry struct {
	mu          sync.Mutex
	states      map[string]*geminiRateLimitState
	modelPauses map[string]*geminiModelPauseState
}

type geminiModelPauseState struct {
	mu           sync.Mutex
	currentDelay time.Duration
	lastFailure  time.Time
	pausedUntil  time.Time
}

var activeGeminiThrottleRegistry atomic.Pointer[GeminiThrottleRegistry]

// ResetGeminiModelPause resets any high demand pause for the specified model in the active registry.
func ResetGeminiModelPause(model string) {
	if r := activeGeminiThrottleRegistry.Load(); r != nil {
		r.resetModelHighDemand(model)
	}
}

// NewGeminiThrottleRegistry creates a new registry for tracking Gemini throttle states.
func NewGeminiThrottleRegistry() *GeminiThrottleRegistry {
	r := &GeminiThrottleRegistry{
		states:      make(map[string]*geminiRateLimitState),
		modelPauses: make(map[string]*geminiModelPauseState),
	}
	activeGeminiThrottleRegistry.Store(r)
	return r
}

func (r *GeminiThrottleRegistry) getModelPauseState(model string) *geminiModelPauseState {
	model = strings.ToLower(strings.TrimSpace(model))
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.modelPauses == nil {
		r.modelPauses = make(map[string]*geminiModelPauseState)
	}
	state, exists := r.modelPauses[model]
	if !exists {
		state = &geminiModelPauseState{}
		r.modelPauses[model] = state
	}
	return state
}

func (r *GeminiThrottleRegistry) checkModelPause(
	ctx context.Context,
	model string,
	sleepFunc func(context.Context, time.Duration) error,
	nowFunc func() time.Time,
) error {
	if model == "" {
		return nil
	}
	pauseState := r.getModelPauseState(model)
	pauseState.mu.Lock()
	now := nowFunc()
	if pauseState.pausedUntil.After(now) {
		d := pauseState.pausedUntil.Sub(now)
		pausedUntil := pauseState.pausedUntil
		pauseState.mu.Unlock()
		log.Printf("[gemini-throttle] model %q high demand pause active: pausing request for %v (until %v)",
			model, d, pausedUntil.Format(time.RFC3339))
		if err := sleepFunc(ctx, d); err != nil {
			log.Printf("[gemini-throttle] request canceled while waiting in model %q pause queue: %v", model, err)
			return err
		}
		log.Printf("[gemini-throttle] model %q pause elapsed, resuming dispatch", model)
		return nil
	}
	pauseState.mu.Unlock()
	return nil
}

func (r *GeminiThrottleRegistry) recordModelHighDemand(model string, now time.Time) time.Duration {
	if model == "" {
		return 1 * time.Minute
	}
	pauseState := r.getModelPauseState(model)
	pauseState.mu.Lock()
	defer pauseState.mu.Unlock()

	if pauseState.currentDelay == 0 || now.Sub(pauseState.lastFailure) > pauseState.currentDelay*2+GeminiHighDemandMaxBackoff {
		pauseState.currentDelay = 1 * time.Minute
	} else {
		pauseState.currentDelay *= 2
		if pauseState.currentDelay > GeminiHighDemandMaxBackoff {
			pauseState.currentDelay = GeminiHighDemandMaxBackoff
		}
	}
	pauseState.lastFailure = now
	pauseState.pausedUntil = now.Add(pauseState.currentDelay)
	return pauseState.currentDelay
}

func (r *GeminiThrottleRegistry) resetModelHighDemand(model string) {
	if model == "" {
		return
	}
	pauseState := r.getModelPauseState(model)
	pauseState.mu.Lock()
	defer pauseState.mu.Unlock()

	pauseState.currentDelay = 0
	pauseState.lastFailure = time.Time{}
	pauseState.pausedUntil = time.Time{}
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
	model := geminiTargetModel(spec)

	if !spec.Synthetic {
		if err := e.registry.checkModelPause(ctx, model, state.sleep, state.now); err != nil {
			return AttemptResult{
				DispatchState: DispatchNotSent,
				Error: &ErrorEvidence{
					Kind:         ErrorKindCanceled,
					OriginHint:   ErrorOriginClient,
					ScopeHint:    ErrorScopeRequest,
					Summary:      "request canceled while waiting for model pause: " + err.Error(),
					ReplaySafety: ReplaySafetyRejectedBeforeProcessing,
				},
			}
		}
	}

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

	if isHighDemand503(result.StatusCode, result.Body, result.Error) && ctx.Err() == nil {
		delay := e.registry.recordModelHighDemand(model, state.now())
		if !spec.Synthetic {
			log.Printf("[gemini-throttle] model %q received 503 high demand: pausing request for %v before retry", model, delay)
			if sleepErr := state.sleep(ctx, delay); sleepErr == nil {
				log.Printf("[gemini-throttle] model %q pause elapsed, retrying request", model)
				result = e.inner.Execute(ctx, spec)
			}
		} else {
			log.Printf("[gemini-throttle] model %q received 503 high demand during synthetic route: failing over immediately without sleep", model)
		}
	}

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

	if statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices {
		e.registry.resetModelHighDemand(model)
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
	model := geminiTargetModel(spec)

	if !spec.Synthetic {
		if err := e.registry.checkModelPause(ctx, model, state.sleep, state.now); err != nil {
			return StreamResult{
				DispatchState: DispatchNotSent,
				Error: &ErrorEvidence{
					Kind:         ErrorKindCanceled,
					OriginHint:   ErrorOriginClient,
					ScopeHint:    ErrorScopeRequest,
					Summary:      "request canceled while waiting for model pause: " + err.Error(),
					ReplaySafety: ReplaySafetyRejectedBeforeProcessing,
				},
			}
		}
	}

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

	if isHighDemand503(result.StatusCode, nil, result.Error) && ctx.Err() == nil {
		delay := e.registry.recordModelHighDemand(model, state.now())
		if !spec.Synthetic {
			log.Printf("[gemini-throttle] model %q received 503 high demand: pausing stream for %v before retry", model, delay)
			if sleepErr := state.sleep(ctx, delay); sleepErr == nil {
				log.Printf("[gemini-throttle] model %q pause elapsed, retrying stream", model)
				result = e.inner.ExecuteStream(ctx, spec, sink)
			}
		} else {
			log.Printf("[gemini-throttle] model %q received 503 high demand during synthetic route: failing over immediately without sleep", model)
		}
	}

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

	if statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices {
		e.registry.resetModelHighDemand(model)
	}

	release(statusCode, hitExhaustion)
	return result
}

var _ Executor = (*GeminiThrottledExecutor)(nil)
