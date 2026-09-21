package state_test

import (
	"testing"

	"gpt-load/internal/state"
)

func TestCompileSyntheticModels(t *testing.T) {
	t.Run("valid synthetic models", func(t *testing.T) {
		input := state.CompileInput{
			SyntheticModels: []state.SyntheticModelConfig{
				{
					ID:           1,
					Name:         "auto-smart",
					Description:  "Smart fallback model",
					TargetModels: []string{"claude-3-7-sonnet", "gpt-4o", "gemini-2.5-pro"},
					Enabled:      true,
				},
				{
					ID:           2,
					Name:         "auto-fast",
					Description:  "Fast fallback model",
					TargetModels: []string{"gemini-2.5-flash", "gpt-4o-mini"},
					Enabled:      true,
				},
				{
					ID:           3,
					Name:         "disabled-model",
					TargetModels: []string{"gpt-4o"},
					Enabled:      false,
				},
			},
		}

		snapshot, err := state.Compile(input)
		if err != nil {
			t.Fatalf("state.Compile failed: %v", err)
		}

		if len(snapshot.SyntheticModels) != 2 {
			t.Fatalf("got %d synthetic models, want 2", len(snapshot.SyntheticModels))
		}

		smartTargets, ok := snapshot.SyntheticModels["auto-smart"]
		if !ok {
			t.Fatal("auto-smart missing from snapshot")
		}
		if len(smartTargets) != 3 || smartTargets[0] != "claude-3-7-sonnet" || smartTargets[1] != "gpt-4o" || smartTargets[2] != "gemini-2.5-pro" {
			t.Fatalf("auto-smart targets = %v, want [claude-3-7-sonnet gpt-4o gemini-2.5-pro]", smartTargets)
		}

		if _, exists := snapshot.SyntheticModels["disabled-model"]; exists {
			t.Fatal("disabled-model should not be compiled in snapshot")
		}
	})

	t.Run("duplicate name rejection", func(t *testing.T) {
		input := state.CompileInput{
			SyntheticModels: []state.SyntheticModelConfig{
				{Name: "dup", TargetModels: []string{"m1"}, Enabled: true},
				{Name: "dup", TargetModels: []string{"m2"}, Enabled: true},
			},
		}
		if _, err := state.Compile(input); err == nil {
			t.Fatal("expected duplicate synthetic model error")
		}
	})

	t.Run("empty target models rejection", func(t *testing.T) {
		input := state.CompileInput{
			SyntheticModels: []state.SyntheticModelConfig{
				{Name: "empty", TargetModels: []string{}, Enabled: true},
			},
		}
		if _, err := state.Compile(input); err == nil {
			t.Fatal("expected empty target models error")
		}
	})

	t.Run("self target rejection", func(t *testing.T) {
		input := state.CompileInput{
			SyntheticModels: []state.SyntheticModelConfig{
				{Name: "self-loop", TargetModels: []string{"self-loop"}, Enabled: true},
			},
		}
		if _, err := state.Compile(input); err == nil {
			t.Fatal("expected self target rejection error")
		}
	})
}
