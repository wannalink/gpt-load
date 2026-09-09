package execution

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

func TestIsGeminiThrottledTarget(t *testing.T) {
	tests := []struct {
		name     string
		spec     AttemptSpec
		expected bool
	}{
		{
			name:     "gemini channel native route",
			spec:     AttemptSpec{ChannelID: "gemini", Path: "/v1beta/models/gemini-pro:generateContent"},
			expected: true,
		},
		{
			name:     "gemini channel converted route",
			spec:     AttemptSpec{ChannelID: "gemini", Path: "/v1/chat/completions"},
			expected: true,
		},
		{
			name:     "gemini channel case insensitive",
			spec:     AttemptSpec{ChannelID: "Gemini", Path: "/v1beta/models/gemini-pro:generateContent"},
			expected: true,
		},
		{
			name:     "gemini openai compatible surface excluded",
			spec:     AttemptSpec{ChannelID: "gemini", Path: "/v1beta/openai/chat/completions"},
			expected: false,
		},
		{
			name:     "openai channel excluded",
			spec:     AttemptSpec{ChannelID: "openai", Path: "/v1/chat/completions"},
			expected: false,
		},
		{
			name:     "claude channel excluded",
			spec:     AttemptSpec{ChannelID: "claude", Path: "/v1/messages"},
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := isGeminiThrottledTarget(tc.spec)
			if actual != tc.expected {
				t.Fatalf("expected isGeminiThrottledTarget = %v, got %v", tc.expected, actual)
			}
		})
	}
}

func TestIsResourceExhausted(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       []byte
		evidence   *ErrorEvidence
		expected   bool
	}{
		{
			name:       "exact 429 with matching body",
			statusCode: http.StatusTooManyRequests,
			body:       []byte(`{"error":{"message":"Resource has been exhausted (e.g. check quota)."}}`),
			expected:   true,
		},
		{
			name:       "exact 429 with matching error summary",
			statusCode: 0,
			evidence: &ErrorEvidence{
				StatusCode: http.StatusTooManyRequests,
				Summary:    "upstream error: Resource has been exhausted (e.g. check quota).",
			},
			expected: true,
		},
		{
			name:       "429 other rate limit message is ignored",
			statusCode: http.StatusTooManyRequests,
			body:       []byte(`{"error":{"message":"rate limit exceeded"}}`),
			expected:   false,
		},
		{
			name:       "429 without matching summary is ignored",
			statusCode: 0,
			evidence: &ErrorEvidence{
				StatusCode: http.StatusTooManyRequests,
				Summary:    "upstream rate limit exceeded",
			},
			expected: false,
		},
		{
			name:       "200 ok with body is ignored",
			statusCode: http.StatusOK,
			body:       []byte(`Resource has been exhausted (e.g. check quota).`),
			expected:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := isResourceExhausted(tc.statusCode, tc.body, tc.evidence)
			if actual != tc.expected {
				t.Fatalf("expected isResourceExhausted = %v, got %v", tc.expected, actual)
			}
		})
	}
}

func TestGeminiProactiveRateLimiting(t *testing.T) {
	currentTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	var sleepCalled bool
	var sleepDuration time.Duration

	state := newGeminiRateLimitState()
	state.nowFunc = func() time.Time { return currentTime }
	state.sleepFunc = func(ctx context.Context, d time.Duration) error {
		sleepCalled = true
		sleepDuration = d
		currentTime = currentTime.Add(d)
		return nil
	}

	// First 20 requests must not sleep
	for i := 1; i <= GeminiProactiveLimit; i++ {
		release, err := state.beforeSend(context.Background())
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		if sleepCalled {
			t.Fatalf("request %d slept unexpectedly", i)
		}
		release(http.StatusOK, false)
	}

	// 21st request must trigger sleep
	sleepCalled = false
	release, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("21st request failed: %v", err)
	}
	if !sleepCalled {
		t.Fatalf("21st request did not sleep")
	}
	if sleepDuration < 5*time.Minute {
		t.Fatalf("sleep duration %v was less than 5 minutes", sleepDuration)
	}
	release(http.StatusOK, false)

	// After sleep, window was reset and next request should proceed without sleep
	sleepCalled = false
	release2, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("request after reset failed: %v", err)
	}
	if sleepCalled {
		t.Fatalf("request after reset slept unexpectedly")
	}
	release2(http.StatusOK, false)
}

func TestGeminiReactiveThrottling(t *testing.T) {
	currentTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	var sleepCalled bool
	var sleepDuration time.Duration

	state := newGeminiRateLimitState()
	state.nowFunc = func() time.Time { return currentTime }
	state.sleepFunc = func(ctx context.Context, d time.Duration) error {
		sleepCalled = true
		sleepDuration = d
		currentTime = currentTime.Add(d)
		return nil
	}

	// Send 1st request which hits 429 Resource Exhausted
	release, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	release(http.StatusTooManyRequests, true)

	// 2nd request must sleep because throttled = true
	sleepCalled = false
	release2, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	if !sleepCalled {
		t.Fatalf("second request did not sleep after 429 exhaustion")
	}
	if sleepDuration < 5*time.Minute {
		t.Fatalf("sleep duration %v was less than 5 minutes", sleepDuration)
	}
	release2(http.StatusOK, false)
}

func TestGeminiNetworkErrorRollback(t *testing.T) {
	currentTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	state := newGeminiRateLimitState()
	state.nowFunc = func() time.Time { return currentTime }

	// Send 19 requests successfully
	for i := 1; i <= GeminiProactiveLimit-1; i++ {
		release, err := state.beforeSend(context.Background())
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		release(http.StatusOK, false)
	}

	// 20th request fails with network error (statusCode = 0)
	release, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("20th request failed: %v", err)
	}
	release(0, false)

	// Count should have rolled back to 19, so another request should NOT trigger throttling
	state.sleepFunc = func(ctx context.Context, d time.Duration) error {
		t.Fatalf("should not sleep when count is 19")
		return nil
	}
	releaseNext, err := state.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("next request failed: %v", err)
	}
	releaseNext(http.StatusOK, false)
}

func TestGeminiContextCancellation(t *testing.T) {
	currentTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	state := newGeminiRateLimitState()
	state.nowFunc = func() time.Time { return currentTime }
	state.sleepFunc = func(ctx context.Context, d time.Duration) error {
		return ctx.Err()
	}

	// Fill quota
	for i := 1; i <= GeminiProactiveLimit; i++ {
		release, err := state.beforeSend(context.Background())
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		release(http.StatusOK, false)
	}

	// Next request with canceled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := state.beforeSend(ctx)
	if err == nil {
		t.Fatalf("expected context error, got nil")
	}
}

func TestGeminiChannelWideSharingAcrossKeys(t *testing.T) {
	currentTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	registry := NewGeminiThrottleRegistry()

	specKey1 := AttemptSpec{
		ChannelID:  "gemini",
		Credential: NewCredentialSnapshot(1, 1, 1, []byte("key-1")),
	}
	specKey2 := AttemptSpec{
		ChannelID:  "gemini",
		Credential: NewCredentialSnapshot(2, 1, 1, []byte("key-2")),
	}

	state1 := registry.getState(specKey1)
	state2 := registry.getState(specKey2)

	if state1 != state2 {
		t.Fatalf("all credentials in the gemini channel must share the same rate limit state")
	}

	var sleepCalled bool
	state1.nowFunc = func() time.Time { return currentTime }
	state1.sleepFunc = func(ctx context.Context, d time.Duration) error {
		sleepCalled = true
		currentTime = currentTime.Add(d)
		return nil
	}

	// Key 1 sends 10 requests
	for i := 1; i <= 10; i++ {
		release, err := state1.beforeSend(context.Background())
		if err != nil {
			t.Fatalf("key1 request %d failed: %v", i, err)
		}
		release(http.StatusOK, false)
	}

	// Key 2 sends 10 requests (reaching 20 total for the channel)
	for i := 1; i <= 10; i++ {
		release, err := state2.beforeSend(context.Background())
		if err != nil {
			t.Fatalf("key2 request %d failed: %v", i, err)
		}
		release(http.StatusOK, false)
	}

	// 21st request from Key 1 or Key 2 must sleep
	sleepCalled = false
	releaseNext, err := state1.beforeSend(context.Background())
	if err != nil {
		t.Fatalf("21st request failed: %v", err)
	}
	if !sleepCalled {
		t.Fatalf("21st request across keys should have paused")
	}
	releaseNext(http.StatusOK, false)
}

type fakeExecutor struct {
	unaryCallCount  atomic.Int32
	streamCallCount atomic.Int32
	unaryFunc       func(context.Context, AttemptSpec) AttemptResult
	streamFunc      func(context.Context, AttemptSpec, StreamSink) StreamResult
}

func (f *fakeExecutor) Execute(ctx context.Context, spec AttemptSpec) AttemptResult {
	f.unaryCallCount.Add(1)
	if f.unaryFunc != nil {
		return f.unaryFunc(ctx, spec)
	}
	return AttemptResult{
		DispatchState:   DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
		Body:            []byte(`{"success":true}`),
	}
}

func (f *fakeExecutor) ExecuteStream(ctx context.Context, spec AttemptSpec, sink StreamSink) StreamResult {
	f.streamCallCount.Add(1)
	if f.streamFunc != nil {
		return f.streamFunc(ctx, spec, sink)
	}
	if sink != nil {
		_ = sink(StreamEvent{Kind: StreamEventReady, StatusCode: http.StatusOK})
		_ = sink(StreamEvent{Kind: StreamEventData, Data: []byte("data: chunk\n\n")})
	}
	return StreamResult{
		DispatchState:   DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
	}
}

func TestGeminiThrottledExecutor(t *testing.T) {
	fake := &fakeExecutor{}
	executor := NewGeminiThrottledExecutor(fake)

	// Non-Gemini attempt
	nonGeminiSpec := AttemptSpec{
		ChannelID:  "openai",
		Credential: NewCredentialSnapshot(1, 1, 1, []byte("key-1")),
	}
	result := executor.Execute(context.Background(), nonGeminiSpec)
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", result.StatusCode)
	}
	if fake.unaryCallCount.Load() != 1 {
		t.Fatalf("expected 1 unary call, got %d", fake.unaryCallCount.Load())
	}

	// Gemini attempt
	geminiSpec := AttemptSpec{
		ChannelID:  "gemini",
		Credential: NewCredentialSnapshot(1, 1, 1, []byte("gemini-key")),
	}
	result = executor.Execute(context.Background(), geminiSpec)
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", result.StatusCode)
	}
	if fake.unaryCallCount.Load() != 2 {
		t.Fatalf("expected 2 unary calls, got %d", fake.unaryCallCount.Load())
	}

	// Gemini streaming attempt
	streamResult := executor.ExecuteStream(context.Background(), geminiSpec, func(event StreamEvent) error {
		return nil
	})
	if streamResult.StatusCode != http.StatusOK {
		t.Fatalf("expected stream status 200, got %d", streamResult.StatusCode)
	}
	if fake.streamCallCount.Load() != 1 {
		t.Fatalf("expected 1 stream call, got %d", fake.streamCallCount.Load())
	}
}
