package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

func TestCandidateCooldownAutoRetriesAndSucceeds(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-ok","model":"gpt-4o","choices":[{"message":{"content":"cooldown cleared"}}]}`),
		},
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "main-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc, _ := handler.encryption.Encrypt(`{"api_key":"sk-test"}`)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	// Put credential on a short 100ms cooldown
	registry.SetCooldown(1, time.Now().Add(100*time.Millisecond))

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()

	start := time.Now()
	engine.ServeHTTP(response, request)
	elapsed := time.Since(start)

	if response.Code != http.StatusOK {
		t.Fatalf("response code = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "cooldown cleared") {
		t.Fatalf("response body = %s, want 'cooldown cleared'", response.Body.String())
	}
	if elapsed < 80*time.Millisecond {
		t.Fatalf("request completed too fast (%v), did not wait for cooldown", elapsed)
	}
	if len(forwarder.inputs) != 1 {
		t.Fatalf("forwarder inputs count = %d, want 1", len(forwarder.inputs))
	}
}

func TestCandidateCooldownExceedsBudgetFailsWith503(t *testing.T) {
	t.Setenv("GPT_LOAD_CANDIDATE_RETRY_TIMEOUT", "60ms")

	forwarder := &scriptedForwarder{results: []UpstreamResult{}}
	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "main-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc, _ := handler.encryption.Encrypt(`{"api_key":"sk-test"}`)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	// Put credential on a 1-hour cooldown
	registry.SetCooldown(1, time.Now().Add(time.Hour))

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()

	start := time.Now()
	engine.ServeHTTP(response, request)
	elapsed := time.Since(start)

	if response.Code != http.StatusServiceUnavailable && response.Code != http.StatusTooManyRequests {
		t.Fatalf("response code = %d, want 503 or 429; body = %s", response.Code, response.Body.String())
	}
	if elapsed > 5*time.Second {
		t.Fatalf("request hung for %v, exceeded timeout budget", elapsed)
	}
	if len(forwarder.inputs) != 0 {
		t.Fatalf("forwarder inputs count = %d, want 0", len(forwarder.inputs))
	}
}

func TestCandidateCooldownCanceledByContext(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{}}
	handler, manager, registry := newHandlerForTest(t, forwarder)
	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "main-group", ChannelID: channel.OpenAI,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1"},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}

	enc, _ := handler.encryption.Encrypt(`{"api_key":"sk-test"}`)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}

	// Put credential on 10s cooldown
	registry.SetCooldown(1, time.Now().Add(10*time.Second))

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`),
	).WithContext(ctx)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()

	start := time.Now()
	engine.ServeHTTP(response, request)
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Fatalf("cancellation did not abort in time: elapsed = %v", elapsed)
	}
	if len(forwarder.inputs) != 0 {
		t.Fatalf("forwarder inputs count = %d, want 0", len(forwarder.inputs))
	}
}

func TestAutoModelCandidateCooldownAutoRetriesAndSucceeds(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-ok","model":"gpt-4o","choices":[{"message":{"content":"auto model succeeded after wait"}}]}`),
		},
	}}

	handler, manager, registry := newHandlerForTest(t, forwarder)
	configureAutoModelTest(t, handler, manager, state.FilterSet{})

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"answer-key-1"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"answer-key-2"}`)
	enc3, _ := handler.encryption.Encrypt(`{"api_key":"decision-key"}`)
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 2, Fingerprint: "credential-2", EncryptedValue: enc2},
		{ID: 3, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 3, Fingerprint: "credential-3", EncryptedValue: enc3},
	}); err != nil {
		t.Fatal(err)
	}

	handler.decisionClient = autoDecisionClient(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{},
			Body:       io.NopCloser(strings.NewReader(`{"answers":{"preset":{"choice":"balanced","confidence":0.9}}}`)),
		}, nil
	})

	// Put both answer credentials on a 100ms cooldown
	registry.SetCooldown(1, time.Now().Add(100*time.Millisecond))
	registry.SetCooldown(2, time.Now().Add(100*time.Millisecond))

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"auto-probe","messages":[{"role":"user","content":"hello"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()

	start := time.Now()
	engine.ServeHTTP(response, request)
	elapsed := time.Since(start)

	if response.Code != http.StatusOK {
		t.Fatalf("response code = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "auto model succeeded after wait") {
		t.Fatalf("response body = %s, want 'auto model succeeded after wait'", response.Body.String())
	}
	if elapsed < 80*time.Millisecond {
		t.Fatalf("request completed too fast (%v), did not wait for cooldown", elapsed)
	}
	if len(forwarder.inputs) != 1 {
		t.Fatalf("forwarder inputs count = %d, want 1", len(forwarder.inputs))
	}
}
