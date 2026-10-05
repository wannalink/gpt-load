package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

func Test503CooldownQueuesRequestWhenArrivingDuringCooldown(t *testing.T) {
	// A request arrives while the candidate is already in 503 cooldown for 100ms from a prior request.
	// It should wait for the 100ms 503 cooldown to clear, then forward once and succeed 200 OK.
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-ok","model":"gpt-4o","choices":[{"message":{"content":"served after existing 503 cooldown cleared"}}]}`),
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

	// Put candidate on a short 100ms 503 model cooldown
	ref := state.CredentialRef{ID: 1, GroupID: 1, IdentityGeneration: 1}
	registry.SetModelCooldown(ref, "gpt-4o", time.Now().Add(100*time.Millisecond), time.Now())

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
	if !strings.Contains(response.Body.String(), "served after existing 503 cooldown cleared") {
		t.Fatalf("response body = %s, want 'served after existing 503 cooldown cleared'", response.Body.String())
	}
	if elapsed < 80*time.Millisecond {
		t.Fatalf("request completed too fast (%v), did not wait for 503 cooldown", elapsed)
	}
	if len(forwarder.inputs) != 1 {
		t.Fatalf("forwarder inputs count = %d, want 1", len(forwarder.inputs))
	}
}

func TestSynthetic503CooldownQueuesRequestUntilTargetRecovers(t *testing.T) {
	// Synthetic model "auto-pro" maps to ["claude-3-7-sonnet", "gemini-2.5-flash"].
	// Both targets start in 503 cooldown (100ms).
	// The request should queue and wait until the 100ms cooldown expires, then forward and succeed with 200 OK.
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-synthetic-recovered","choices":[{"message":{"content":"synthetic model recovered after wait"}}]}`),
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

	// Put both targets on a short 100ms model cooldown
	ref1 := state.CredentialRef{ID: 1, GroupID: 1, IdentityGeneration: 1}
	ref2 := state.CredentialRef{ID: 2, GroupID: 2, IdentityGeneration: 1}
	registry.SetModelCooldown(ref1, "claude-3-7-sonnet", time.Now().Add(100*time.Millisecond), time.Now())
	registry.SetModelCooldown(ref2, "gemini-2.5-flash", time.Now().Add(100*time.Millisecond), time.Now())

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"auto-pro","messages":[{"role":"user","content":"hello"}]}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()

	start := time.Now()
	engine.ServeHTTP(response, request)
	elapsed := time.Since(start)

	if response.Code != http.StatusOK {
		t.Fatalf("response code = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "synthetic model recovered after wait") {
		t.Fatalf("response body = %s, want 'synthetic model recovered after wait'", response.Body.String())
	}
	if elapsed < 80*time.Millisecond {
		t.Fatalf("request completed too fast (%v), did not wait for synthetic model 503 cooldown", elapsed)
	}
}
