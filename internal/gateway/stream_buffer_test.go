package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/execution"
)

func TestExecutionForwarderBufferStreamPreventsIncompleteDelivery(t *testing.T) {
	t.Parallel()

	// 1. Incomplete stream with BufferStream=true: must NOT commit partial body
	executorIncomplete := fakeExecutionExecutor{stream: func(
		_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink,
	) execution.StreamResult {
		_ = sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: make(http.Header)})
		_ = sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")})
		// Abrupt end without terminal event
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	inputBuffered := executionForwardInput()
	inputBuffered.BufferStream = true
	recorderBuffered := httptest.NewRecorder()
	resultBuffered := NewExecutionForwarder(executorIncomplete).ForwardStream(context.Background(), inputBuffered, recorderBuffered)

	if resultBuffered.Committed {
		t.Fatalf("expected Committed=false for incomplete buffered stream, got true")
	}
	if recorderBuffered.Body.Len() != 0 {
		t.Fatalf("expected empty body for incomplete buffered stream, got %q", recorderBuffered.Body.String())
	}

	// 2. Complete stream with BufferStream=true: flushes full body
	executorComplete := fakeExecutionExecutor{stream: func(
		_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink,
	) execution.StreamResult {
		_ = sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: make(http.Header)})
		_ = sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("data: {\"choices\":[{\"delta\":{\"content\":\"full\"}}]}\n\n")})
		_ = sink(execution.StreamEvent{Sequence: 3, Kind: execution.StreamEventData, Data: []byte("data: [DONE]\n\n")})
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	recorderComplete := httptest.NewRecorder()
	resultComplete := NewExecutionForwarder(executorComplete).ForwardStream(context.Background(), inputBuffered, recorderComplete)

	if !resultComplete.Committed {
		t.Fatalf("expected Committed=true for complete buffered stream, got false")
	}
	if !strings.Contains(recorderComplete.Body.String(), "full") || !strings.Contains(recorderComplete.Body.String(), "[DONE]") {
		t.Fatalf("expected complete flushed body, got %q", recorderComplete.Body.String())
	}
}
