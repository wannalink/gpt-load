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
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

type capturingTelemetrySink struct {
	events []telemetry.RequestEvent
}

func (c *capturingTelemetrySink) Emit(event telemetry.RequestEvent) {
	c.events = append(c.events, event)
}

type staticPriceTableProvider struct {
	table *pricing.Table
}

func (p staticPriceTableProvider) Load() *pricing.Table {
	return p.table
}

func TestUpstreamModelPricingRuleForSyntheticAndAliasedModels(t *testing.T) {
	// 1. Setup price table with distinct pricing for claude-3-7-sonnet vs gemini-2.5-flash
	priceTable, err := pricing.NewTable([]pricing.Rule{
		{
			Identity: pricing.Identity{ChannelID: "openai", ModelID: "claude-3-7-sonnet"},
			Prices: pricing.Prices{
				Input:  pricing.Price{NanoUSDPerMillion: 3_000_000_000, Set: true},  // $3/M
				Output: pricing.Price{NanoUSDPerMillion: 15_000_000_000, Set: true}, // $15/M
			},
		},
		{
			Identity: pricing.Identity{ChannelID: "openai", ModelID: "gemini-2.5-flash"},
			Prices: pricing.Prices{
				Input:  pricing.Price{NanoUSDPerMillion: 100_000_000, Set: true}, // $0.10/M
				Output: pricing.Price{NanoUSDPerMillion: 400_000_000, Set: true}, // $0.40/M
			},
		},
	})
	if err != nil {
		t.Fatalf("pricing.NewTable failed: %v", err)
	}

	// 2. Setup forwarder: 429 on claude-3-7-sonnet, success on gemini-2.5-flash with 1000 input, 500 output tokens
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		withProviderErrorBeforeCommit(UpstreamResult{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"error":{"message":"Rate limited","code":"rate_limit_exceeded"}}`),
			ExecutionError: &execution.ErrorEvidence{
				Kind:         execution.ErrorKindHTTP,
				StatusCode:   http.StatusTooManyRequests,
				Code:         "rate_limit_exceeded",
				ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
			},
		}),
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       []byte(`{"id":"chatcmpl-1","model":"gemini-2.5-flash","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1000,"completion_tokens":500,"total_tokens":1500}}`),
			Usage: usage.Result{
				State:  usage.StateComplete,
				Tokens: usage.Tokens{UncachedInput: 1000, Output: 500},
			},
		},
	}}

	telemetryCapture := &capturingTelemetrySink{}
	handler, manager, registry := newHandlerForTest(t, forwarder)
	handler.priceTables = staticPriceTableProvider{table: priceTable}
	handler.requestLogSink = telemetryCapture

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
				Name:         "auto-smart",
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

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-1"}`)
	enc2, _ := handler.encryption.Encrypt(`{"api_key":"sk-2"}`)
	_ = registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
		{ID: 2, GroupID: 2, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-2", EncryptedValue: enc2},
	})

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"auto-smart","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200, body = %s", recorder.Code, recorder.Body.String())
	}

	if len(telemetryCapture.events) != 1 {
		t.Fatalf("events count = %d, want 1", len(telemetryCapture.events))
	}

	ev := telemetryCapture.events[0]
	if ev.ClientModel != "auto-smart" {
		t.Fatalf("ClientModel = %q, want auto-smart", ev.ClientModel)
	}
	if ev.UpstreamModel != "gemini-2.5-flash" {
		t.Fatalf("UpstreamModel = %q, want gemini-2.5-flash", ev.UpstreamModel)
	}

	// 1000 input tokens @ $0.10/M = 100,000 nanoUSD
	// 500 output tokens @ $0.40/M = 200,000 nanoUSD
	// Total estimated cost = 300,000 nanoUSD
	if ev.Usage.Pricing.EstimatedCostNanoUSD != 300_000 {
		t.Fatalf("EstimatedCostNanoUSD = %d, want 300000 (calculated from gemini-2.5-flash rates)", ev.Usage.Pricing.EstimatedCostNanoUSD)
	}
}

func TestUpstreamReportedModelAttributionAndPricing(t *testing.T) {
	// Setup price table: $2/M for gemini-3.8-flash, $10/M for gemini-flash-latest
	priceTable, err := pricing.NewTable([]pricing.Rule{
		{
			Identity: pricing.Identity{ChannelID: "gemini", ModelID: "gemini-3.8-flash"},
			Prices: pricing.Prices{
				Input:  pricing.Price{NanoUSDPerMillion: 2_000_000_000, Set: true},  // $2/M
				Output: pricing.Price{NanoUSDPerMillion: 8_000_000_000, Set: true},  // $8/M
			},
		},
		{
			Identity: pricing.Identity{ChannelID: "gemini", ModelID: "gemini-flash-latest"},
			Prices: pricing.Prices{
				Input:  pricing.Price{NanoUSDPerMillion: 10_000_000_000, Set: true}, // $10/M
				Output: pricing.Price{NanoUSDPerMillion: 20_000_000_000, Set: true}, // $20/M
			},
		},
	})
	if err != nil {
		t.Fatalf("pricing.NewTable failed: %v", err)
	}

	// Provider returns response with UpstreamReportedModel = "gemini-3.8-flash"
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		{
			StatusCode:            http.StatusOK,
			Header:                http.Header{"Content-Type": {"application/json"}},
			Body:                  []byte(`{"id":"chatcmpl-1","model":"gemini-3.8-flash","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1000,"completion_tokens":500,"total_tokens":1500}}`),
			UpstreamReportedModel: "gemini-3.8-flash",
			ResponseModelObserved: true,
			ResponseModelMismatch: true,
			Usage: usage.Result{
				State:  usage.StateComplete,
				Tokens: usage.Tokens{UncachedInput: 1000, Output: 500},
			},
		},
	}}

	telemetryCapture := &capturingTelemetrySink{}
	handler, manager, registry := newHandlerForTest(t, forwarder)
	handler.priceTables = staticPriceTableProvider{table: priceTable}
	handler.requestLogSink = telemetryCapture

	if _, err := manager.Publish(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ConnectionType: "api_key", ID: 1, Name: "gemini-group", ChannelID: channel.Gemini,
				Params: json.RawMessage(`{}`),
				Models: []state.ModelConfig{{ID: "gemini-flash-latest"}}, Enabled: true,
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

	enc1, _ := handler.encryption.Encrypt(`{"api_key":"sk-1"}`)
	_ = registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 1, GroupID: 1, Status: state.CredentialStatusActive, Version: 1, IdentityGeneration: 1, Fingerprint: "cred-1", EncryptedValue: enc1},
	})

	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"gemini-flash-latest","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer gl-client")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	engine.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200, body = %s", recorder.Code, recorder.Body.String())
	}

	if len(telemetryCapture.events) != 1 {
		t.Fatalf("events count = %d, want 1", len(telemetryCapture.events))
	}

	ev := telemetryCapture.events[0]
	if ev.ClientModel != "gemini-flash-latest" {
		t.Fatalf("ClientModel = %q, want gemini-flash-latest", ev.ClientModel)
	}
	if ev.UpstreamReportedModel != "gemini-3.8-flash" {
		t.Fatalf("UpstreamReportedModel = %q, want gemini-3.8-flash", ev.UpstreamReportedModel)
	}
	if ev.Usage.Pricing.UpstreamModel != "gemini-3.8-flash" {
		t.Fatalf("Usage.Pricing.UpstreamModel = %q, want gemini-3.8-flash", ev.Usage.Pricing.UpstreamModel)
	}

	// 1000 input tokens @ $2/M = 2,000,000 nanoUSD
	// 500 output tokens @ $8/M = 4,000,000 nanoUSD
	// Total estimated cost = 6,000,000 nanoUSD (from gemini-3.8-flash rates, NOT gemini-flash-latest rates of 20,000,000)
	if ev.Usage.Pricing.EstimatedCostNanoUSD != 6_000_000 {
		t.Fatalf("EstimatedCostNanoUSD = %d, want 6000000 (calculated from gemini-3.8-flash rates)", ev.Usage.Pricing.EstimatedCostNanoUSD)
	}
}
