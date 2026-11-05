package gateway

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

type mockRuntimeRegistry struct {
	views map[uint][]state.CredentialRuntimeView
}

func (m *mockRuntimeRegistry) SchedulingState() *state.SchedulingState {
	return nil
}

func (m *mockRuntimeRegistry) CollectCredentialCandidates(groupIDs []uint, excluded func(uint) bool, now time.Time) []state.CredentialMeta {
	return nil
}

func (m *mockRuntimeRegistry) GroupRuntimeViews(groupID uint) []state.CredentialRuntimeView {
	return m.views[groupID]
}

func (m *mockRuntimeRegistry) CaptureActiveCredentialRefs(groupIDs []uint) []state.CredentialRef {
	return nil
}

func (m *mockRuntimeRegistry) CredentialRef(credentialID uint) (state.CredentialRef, bool) {
	return state.CredentialRef{}, false
}

func (m *mockRuntimeRegistry) ActiveEncryptedCredentialDataIfMatch(ref state.CredentialRef) (string, bool) {
	return "", false
}

func (m *mockRuntimeRegistry) SetCooldownWithChange(credentialID uint, until time.Time) (bool, bool) {
	return false, false
}

func (m *mockRuntimeRegistry) SetCooldownWithChangeIfVersion(credentialID uint, expectedVersion uint64, until time.Time) (bool, bool) {
	return false, false
}

func (m *mockRuntimeRegistry) SetModelCooldown(ref state.CredentialRef, model string, until, now time.Time) (bool, bool) {
	return true, true
}

func (m *mockRuntimeRegistry) SetQuotaCooldown(ref state.CredentialRef, model string, until, now time.Time) (bool, bool) {
	return true, true
}

func (m *mockRuntimeRegistry) SetGroupModelCooldown(groupID uint, model string, until, now time.Time) {}

func (m *mockRuntimeRegistry) IncrFailure(credentialID uint) (int, bool) {
	return 0, false
}

func (m *mockRuntimeRegistry) SetBlacklistedWithChange(credentialID uint) (bool, bool) {
	return false, false
}

func (m *mockRuntimeRegistry) ClearFailure(credentialID uint) bool {
	return false
}

func TestRoutingTableLoggerSingleLine(t *testing.T) {
	var buf bytes.Buffer
	logger := logrus.New()
	logger.SetOutput(&buf)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors: true,
	})

	now := time.Now()
	snapshot := &state.ConfigSnapshot{
		SyntheticModels: map[string][]string{
			"synthetic-group": {"model-a", "model-b"},
		},
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"model-a": {{GroupID: 1, UpstreamModelID: "model-a"}},
					"model-b": {{GroupID: 2, UpstreamModelID: "model-b"}},
				},
			},
		},
	}

	registry := &mockRuntimeRegistry{
		views: map[uint][]state.CredentialRuntimeView{
			1: {
				{ID: 101, GroupID: 1},
				{ID: 102, GroupID: 1, CooldownUntil: now.Add(2 * time.Minute)},
			},
			2: {
				{ID: 201, GroupID: 2, ModelCooldowns: map[string]time.Time{"model-b": now.Add(5 * time.Minute)}},
			},
		},
	}

	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
	}

	logRoutingTable(logger, snapshot, registry, "synthetic-group", query, now)

	logOutput := buf.String()
	if strings.Contains(logOutput, "\n================================") {
		t.Errorf("log output contains raw multiline table border; expected clean single line: %s", logOutput)
	}

	if !strings.Contains(logOutput, "Routing candidates for synthetic-group (synthetic): model-a(1/2, AVAILABLE, 0s) | model-b(0/1, COOLDOWN, 5m0s)") {
		t.Errorf("unexpected log message format: %s", logOutput)
	}
}

func TestRoutingTableLoggerNestedSynthetic(t *testing.T) {
	var buf bytes.Buffer
	logger := logrus.New()
	logger.SetOutput(&buf)
	logger.SetFormatter(&logrus.TextFormatter{DisableColors: true})

	now := time.Now()
	snapshot := &state.ConfigSnapshot{
		SyntheticModels: map[string][]string{
			"top-level":    {"nested-alias"},
			"nested-alias": {"real-model"},
		},
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"real-model": {{GroupID: 10, UpstreamModelID: "real-model"}},
				},
			},
		},
	}

	registry := &mockRuntimeRegistry{
		views: map[uint][]state.CredentialRuntimeView{
			10: {
				{ID: 1, GroupID: 10},
			},
		},
	}

	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
	}

	logRoutingTable(logger, snapshot, registry, "top-level", query, now)

	logOutput := buf.String()
	if !strings.Contains(logOutput, "nested-alias(1/1, AVAILABLE, 0s)") {
		t.Errorf("expected nested synthetic model target to resolve real model stats, got: %s", logOutput)
	}
}
