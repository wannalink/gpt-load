package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/state"
)

func TestSyntheticProxyQuotaExhaustionFallback(t *testing.T) {
	// Attempt 1 on Group 1 (claude-3-7-sonnet): 429 Rate limited / Resource exhausted
	// Attempt 2 on Group 2 (gemini-2.5-flash): 200 OK
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		withProviderErrorBeforeCommit(UpstreamResult{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"Resource exhausted","type":"rate_limit_error","code":"resource_exhausted"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:         execution.ErrorKindHTTP,
				StatusCode:   http.StatusTooManyRequests,
				Type:         "rate_limit_error",
				Code:         "resource_exhausted",
				Summary:      "Resource exhausted",
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
		}),
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-fallback","model":"gemini-2.5-flash","choices":[{"message":{"content":"served by fallback"}}]}`),
		},
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "anthropic-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "claude-3-7-sonnet"}}, Enabled: true,
			},
			{
				ConnectionType: "api_key", ID: 2, Name: "gemini-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gemini-2.5-flash"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
			{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2"},
		},
		SyntheticModels: []state.SyntheticModelConfig{
			{
				ID:           1,
				Name:         "auto-pro",
				TargetModels: []string{"claude-3-7-sonnet", "gemini-2.5-flash"},
				Enabled:      true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-anthropic"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"sk-gemini"}`)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2", EncryptedValue: enc2},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"auto-pro","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200, body = %s", recorder.Code, recorder.Body.String())
	}
	if !bytes.Contains(recorder.Body.Bytes(), []byte("served by fallback")) {
		t.Fatalf("body = %s, want choice served by fallback", recorder.Body.String())
	}
	if len(forwarder.inputs) != 2 {
		t.Fatalf("forward attempts = %d, want 2 (fallback occurred)", len(forwarder.inputs))
	}
}

func TestSyntheticProxyHighDemandFallback(t *testing.T) {
	// Attempt 1 on Group 1: 503 High demand / server overloaded
	// Attempt 2 on Group 2: 200 OK
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		withProviderErrorBeforeCommit(UpstreamResult{
			StatusCode: http.StatusServiceUnavailable,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"The model is overloaded. Please try again later.","type":"server_error","code":"service_unavailable"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:         execution.ErrorKindHTTP,
				StatusCode:   http.StatusServiceUnavailable,
				Type:         "server_error",
				Code:         "service_unavailable",
				Summary:      "The model is overloaded. Please try again later.",
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
		}),
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-fallback","model":"gemini-2.5-flash","choices":[{"message":{"content":"served after 503 high demand"}}]}`),
		},
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "primary-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "model-high-demand"}}, Enabled: true,
			},
			{
				ConnectionType: "api_key", ID: 2, Name: "fallback-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gemini-2.5-flash"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
			{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2"},
		},
		SyntheticModels: []state.SyntheticModelConfig{
			{
				ID:           1,
				Name:         "auto-pro",
				TargetModels: []string{"model-high-demand", "gemini-2.5-flash"},
				Enabled:      true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-primary"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"sk-fallback"}`)
	_ = registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2", EncryptedValue: enc2},
	})

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"auto-pro","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200, body = %s", recorder.Code, recorder.Body.String())
	}
	if !bytes.Contains(recorder.Body.Bytes(), []byte("served after 503 high demand")) {
		t.Fatalf("body = %s, want choice served after 503 high demand", recorder.Body.String())
	}
	if len(forwarder.inputs) != 2 {
		t.Fatalf("forward attempts = %d, want 2 (fallback occurred)", len(forwarder.inputs))
	}
}

func TestSyntheticProxyTokenLimitFailFast(t *testing.T) {
	// Group 1 returns 400 with context_length_exceeded
	// Group 2 should NOT be called; 400 error should be returned directly to user.
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusBadRequest,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"Context length exceeded","type":"invalid_request_error","code":"context_length_exceeded"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:       execution.ErrorKindInvalidRequest,
				StatusCode: http.StatusBadRequest,
				Type:       "invalid_request_error",
				Code:       "context_length_exceeded",
				Summary:    "Context length exceeded",
			},
		},
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "anthropic-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "claude-3-7-sonnet"}}, Enabled: true,
			},
			{
				ConnectionType: "api_key", ID: 2, Name: "gemini-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gemini-2.5-flash"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
			{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2"},
		},
		SyntheticModels: []state.SyntheticModelConfig{
			{
				ID:           1,
				Name:         "auto-pro",
				TargetModels: []string{"claude-3-7-sonnet", "gemini-2.5-flash"},
				Enabled:      true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-anthropic"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"sk-gemini"}`)
	_ = registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2", EncryptedValue: enc2},
	})

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"auto-pro","messages":[{"role":"user","content":"very long prompt"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400, body = %s", recorder.Code, recorder.Body.String())
	}
	if len(forwarder.inputs) != 1 {
		t.Fatalf("forward attempts = %d, want exactly 1 (no fallback on token limit rejection)", len(forwarder.inputs))
	}
}

func TestSyntheticProxyTotalExhaustionReturns503(t *testing.T) {
	// Both Group 1 and Group 2 return 429
	// Standard 503 no_available_candidate should be returned to user.
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		withProviderErrorBeforeCommit(UpstreamResult{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"Rate limit reached","type":"rate_limit_error","code":"rate_limit_exceeded"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:         execution.ErrorKindHTTP,
				StatusCode:   http.StatusTooManyRequests,
				Type:         "rate_limit_error",
				Code:         "rate_limit_exceeded",
				Summary:      "Rate limit reached",
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
		}),
		withProviderErrorBeforeCommit(UpstreamResult{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"Rate limit reached","type":"rate_limit_error","code":"rate_limit_exceeded"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:         execution.ErrorKindHTTP,
				StatusCode:   http.StatusTooManyRequests,
				Type:         "rate_limit_error",
				Code:         "rate_limit_exceeded",
				Summary:      "Rate limit reached",
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
		}),
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "anthropic-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "claude-3-7-sonnet"}}, Enabled: true,
			},
			{
				ConnectionType: "api_key", ID: 2, Name: "gemini-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gemini-2.5-flash"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
			{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2"},
		},
		SyntheticModels: []state.SyntheticModelConfig{
			{
				ID:           1,
				Name:         "auto-pro",
				TargetModels: []string{"claude-3-7-sonnet", "gemini-2.5-flash"},
				Enabled:      true,
			},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-anthropic"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"sk-gemini"}`)
	_ = registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2", EncryptedValue: enc2},
	})

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"auto-pro","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503, body = %s", recorder.Code, recorder.Body.String())
	}
	var errBody struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &errBody); err != nil {
		t.Fatalf("unmarshal error body: %v", err)
	}
	if errBody.Code != "no_available_candidate" {
		t.Fatalf("error.code = %q, want no_available_candidate", errBody.Code)
	}
}
