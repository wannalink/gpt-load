package scheduler_test

import (
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

type testCredentialSource struct {
	progress    *state.SchedulingState
	credentials map[uint][]state.CredentialMeta
}

func (s *testCredentialSource) SchedulingState() *state.SchedulingState {
	for _, creds := range s.credentials {
		for _, meta := range creds {
			s.progress.SyncCredential(state.CredentialRuntimeView{
				ID:                 meta.ID,
				GroupID:            meta.GroupID,
				IdentityGeneration: meta.IdentityGeneration,
				Status:             state.CredentialStatusActive,
			})
		}
	}
	return s.progress
}

func (s *testCredentialSource) CollectCredentialCandidates(groupIDs []uint, excluded func(uint) bool, now time.Time) []state.CredentialMeta {
	var res []state.CredentialMeta
	for _, gID := range groupIDs {
		for _, cred := range s.credentials[gID] {
			if !excluded(cred.ID) {
				res = append(res, cred)
			}
		}
	}
	return res
}

func TestSyntheticModelPriorityResolution(t *testing.T) {
	// Setup 2 groups: Group 1 has "model-primary", Group 2 has "model-fallback"
	targetPrimary := channel.ResolvedTarget{
		ChannelID: "openai",
	}
	targetFallback := channel.ResolvedTarget{
		ChannelID: "openai",
	}

	snapshot := &state.ConfigSnapshot{
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"model-primary": {
						{GroupID: 1, UpstreamModelID: "model-primary", Mode: channel.RouteNative, ResolvedTarget: targetPrimary},
					},
					"model-fallback": {
						{GroupID: 2, UpstreamModelID: "model-fallback", Mode: channel.RouteNative, ResolvedTarget: targetFallback},
					},
				},
			},
		},
		Groups: map[uint]state.GroupView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct"},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct"},
		},
		GroupCatalog: map[uint]state.GroupCatalogView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
		},
		SyntheticModels: map[string][]string{
			"auto-synthetic": {"model-primary", "model-fallback"},
		},
	}

	modelStr := "auto-synthetic"
	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
		ExternalModel:  &modelStr,
	}

	// 1. Check CandidateGroupIDsForQuery returns both groups 1 and 2
	groupIDs := scheduler.CandidateGroupIDsForQuery(snapshot, query)
	if len(groupIDs) != 2 || groupIDs[0] != 1 || groupIDs[1] != 2 {
		t.Fatalf("CandidateGroupIDsForQuery = %v, want [1, 2]", groupIDs)
	}

	// 2. Setup credentials: cred 10 for group 1, cred 20 for group 2
	weight := 1
	credSource := &testCredentialSource{
		progress: state.NewSchedulingState(),
		credentials: map[uint][]state.CredentialMeta{
			1: {{ID: 10, GroupID: 1, WeightManual: &weight}},
			2: {{ID: 20, GroupID: 2, WeightManual: &weight}},
		},
	}

	iter := scheduler.New(snapshot, credSource, query)

	// First Next() must yield Priority 1 (Group 1, Cred 10, model-primary)
	sel1, err := iter.Next()
	if err != nil {
		t.Fatalf("first Next() failed: %v", err)
	}
	if sel1.GroupID != 1 || sel1.CredentialID != 10 || sel1.UpstreamModelID == nil || *sel1.UpstreamModelID != "model-primary" {
		t.Fatalf("sel1 = %+v, want group 1 cred 10 model-primary", sel1)
	}

	// Second Next() must yield Priority 2 (Group 2, Cred 20, model-fallback)
	sel2, err := iter.Next()
	if err != nil {
		t.Fatalf("second Next() failed: %v", err)
	}
	if sel2.GroupID != 2 || sel2.CredentialID != 20 || sel2.UpstreamModelID == nil || *sel2.UpstreamModelID != "model-fallback" {
		t.Fatalf("sel2 = %+v, want group 2 cred 20 model-fallback", sel2)
	}

	// Third Next() must be ErrExhausted
	_, err = iter.Next()
	if err != scheduler.ErrExhausted {
		t.Fatalf("third Next() err = %v, want ErrExhausted", err)
	}
}

func TestSyntheticModelCooldownSkip(t *testing.T) {
	targetPrimary := channel.ResolvedTarget{ChannelID: "openai"}
	targetFallback := channel.ResolvedTarget{ChannelID: "openai"}

	snapshot := &state.ConfigSnapshot{
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"model-primary": {
						{GroupID: 1, UpstreamModelID: "model-primary", Mode: channel.RouteNative, ResolvedTarget: targetPrimary},
					},
					"model-fallback": {
						{GroupID: 2, UpstreamModelID: "model-fallback", Mode: channel.RouteNative, ResolvedTarget: targetFallback},
					},
				},
			},
		},
		Groups: map[uint]state.GroupView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct"},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct"},
		},
		GroupCatalog: map[uint]state.GroupCatalogView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
		},
		SyntheticModels: map[string][]string{
			"auto-synthetic": {"model-primary", "model-fallback"},
		},
	}

	modelStr := "auto-synthetic"
	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
		ExternalModel:  &modelStr,
	}

	weight := 1
	now := time.Now()
	// Cred 10 has model-primary in cooldown until 1 hour later
	credSource := &testCredentialSource{
		progress: state.NewSchedulingState(),
		credentials: map[uint][]state.CredentialMeta{
			1: {{
				ID: 10, GroupID: 1, WeightManual: &weight,
				ModelCooldowns: map[string]time.Time{
					"model-primary": now.Add(time.Hour),
				},
			}},
			2: {{ID: 20, GroupID: 2, WeightManual: &weight}},
		},
	}

	iter := scheduler.New(snapshot, credSource, query)

	// Since Priority 1 is in cooldown, Next() immediately yields Priority 2 (Group 2, Cred 20, model-fallback)
	sel, err := iter.Next()
	if err != nil {
		t.Fatalf("Next() failed: %v", err)
	}
	if sel.GroupID != 2 || sel.CredentialID != 20 || sel.UpstreamModelID == nil || *sel.UpstreamModelID != "model-fallback" {
		t.Fatalf("sel = %+v, want group 2 cred 20 model-fallback directly", sel)
	}
}

func TestSyntheticModelSameCredentialFallback(t *testing.T) {
	// A single group with 1 credential provides both "model-primary" and "model-fallback"
	targetPrimary := channel.ResolvedTarget{ChannelID: "gemini"}
	targetFallback := channel.ResolvedTarget{ChannelID: "gemini"}

	snapshot := &state.ConfigSnapshot{
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"model-primary": {
						{GroupID: 1, UpstreamModelID: "model-primary", Mode: channel.RouteNative, ResolvedTarget: targetPrimary},
					},
					"model-fallback": {
						{GroupID: 1, UpstreamModelID: "model-fallback", Mode: channel.RouteNative, ResolvedTarget: targetFallback},
					},
				},
			},
		},
		Groups: map[uint]state.GroupView{
			1: {ID: 1, Name: "GeminiGroup", ChannelID: "gemini", ConnectionType: "direct"},
		},
		GroupCatalog: map[uint]state.GroupCatalogView{
			1: {ID: 1, Name: "GeminiGroup", ChannelID: "gemini", ConnectionType: "direct", Enabled: true},
		},
		SyntheticModels: map[string][]string{
			"auto-synthetic": {"model-primary", "model-fallback"},
		},
	}

	modelStr := "auto-synthetic"
	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
		ExternalModel:  &modelStr,
	}

	weight := 1
	credSource := &testCredentialSource{
		progress: state.NewSchedulingState(),
		credentials: map[uint][]state.CredentialMeta{
			1: {{ID: 100, GroupID: 1, WeightManual: &weight}},
		},
	}

	iter := scheduler.New(snapshot, credSource, query)

	// 1. First attempt yields Cred 100 for model-primary (Tier 0)
	sel1, err := iter.Next()
	if err != nil {
		t.Fatalf("first Next() failed: %v", err)
	}
	if sel1.CredentialID != 100 || sel1.UpstreamModelID == nil || *sel1.UpstreamModelID != "model-primary" {
		t.Fatalf("sel1 = %+v, want cred 100 model-primary", sel1)
	}

	// 2. Next attempt MUST yield the SAME Cred 100 for model-fallback (Tier 1)
	sel2, err := iter.Next()
	if err != nil {
		t.Fatalf("second Next() failed: %v", err)
	}
	if sel2.CredentialID != 100 || sel2.UpstreamModelID == nil || *sel2.UpstreamModelID != "model-fallback" {
		t.Fatalf("sel2 = %+v, want cred 100 model-fallback on fallback tier", sel2)
	}

	// 3. Third attempt must be exhausted
	_, err = iter.Next()
	if err != scheduler.ErrExhausted {
		t.Fatalf("third Next() err = %v, want ErrExhausted", err)
	}
}

func TestSyntheticModelReturnsToPrimaryAfterCooldownExpires(t *testing.T) {
	targetPrimary := channel.ResolvedTarget{ChannelID: "openai"}
	targetFallback := channel.ResolvedTarget{ChannelID: "openai"}

	snapshot := &state.ConfigSnapshot{
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"model-primary": {
						{GroupID: 1, UpstreamModelID: "model-primary", Mode: channel.RouteNative, ResolvedTarget: targetPrimary},
					},
					"model-fallback": {
						{GroupID: 2, UpstreamModelID: "model-fallback", Mode: channel.RouteNative, ResolvedTarget: targetFallback},
					},
				},
			},
		},
		Groups: map[uint]state.GroupView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct"},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct"},
		},
		GroupCatalog: map[uint]state.GroupCatalogView{
			1: {ID: 1, Name: "PrimaryGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
			2: {ID: 2, Name: "FallbackGroup", ChannelID: "openai", ConnectionType: "direct", Enabled: true},
		},
		SyntheticModels: map[string][]string{
			"auto-synthetic": {"model-primary", "model-fallback"},
		},
	}

	modelStr := "auto-synthetic"
	query := scheduler.Query{
		ClientProtocol: protocol.OpenAICompletions,
		Operation:      execution.OperationChatCompletion,
		ExternalModel:  &modelStr,
	}

	weight := 1
	baseTime := time.Now()
	cooldownExpiry := baseTime.Add(10 * time.Minute)

	credSource := &testCredentialSource{
		progress: state.NewSchedulingState(),
		credentials: map[uint][]state.CredentialMeta{
			1: {{
				ID: 10, GroupID: 1, WeightManual: &weight,
				ModelCooldowns: map[string]time.Time{
					"model-primary": cooldownExpiry,
				},
			}},
			2: {{ID: 20, GroupID: 2, WeightManual: &weight}},
		},
	}

	// 1. When request occurs at baseTime (during cooldown of model-primary):
	// It must skip model-primary and pick model-fallback (Priority 2)
	iter1 := scheduler.New(snapshot, credSource, query)
	sel1, err := iter1.Next()
	if err != nil {
		t.Fatalf("first query Next() failed: %v", err)
	}
	if sel1.GroupID != 2 || *sel1.UpstreamModelID != "model-fallback" {
		t.Fatalf("expected fallback model during cooldown, got %v", *sel1.UpstreamModelID)
	}

	// 2. Clear/expire the cooldown (simulate time advancing past cooldownExpiry)
	credSource.credentials[1][0].ModelCooldowns["model-primary"] = baseTime.Add(-time.Minute)

	// A new request arriving now MUST return to Priority 1 (model-primary)
	iter2 := scheduler.New(snapshot, credSource, query)
	sel2, err := iter2.Next()
	if err != nil {
		t.Fatalf("second query Next() failed: %v", err)
	}
	if sel2.GroupID != 1 || *sel2.UpstreamModelID != "model-primary" {
		t.Fatalf("expected primary model after cooldown expired, got %v", *sel2.UpstreamModelID)
	}
}
