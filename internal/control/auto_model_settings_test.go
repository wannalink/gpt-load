package control

import (
	"encoding/json"
	"strings"
	"testing"

	"gpt-load/internal/automodel"
	"gpt-load/internal/storage/models"
)

func TestAutoModelSettingsPersistOnlyChannelBackedConfig(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	const raw = `{"enabled":false,"model":"jev-router","timeout_seconds":4,"models":[]}`
	response, err := fixture.service.UpdateSettings(t.Context(), SettingsUpdateRequest{
		Settings: map[string]json.RawMessage{"auto_model": json.RawMessage(raw)},
	})
	if err != nil {
		t.Fatalf("save automatic model settings: %v", err)
	}
	encoded, _ := json.Marshal(response)
	for _, removed := range []string{"api_key", "api_key_configured", "provider", "input_price", "output_price"} {
		if strings.Contains(string(encoded), `"`+removed+`"`) {
			t.Fatalf("removed field %q returned: %s", removed, encoded)
		}
	}

	var row models.SystemSetting
	if err := fixture.db.Where("key = ?", automodel.SettingKey).Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	plaintext, err := fixture.encryption.Decrypt(row.Value)
	if err != nil || plaintext != raw {
		t.Fatalf("stored config = %q, error = %v", plaintext, err)
	}
}

func TestAutoModelSettingsAcceptSyntheticModelTarget(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	// 1. Create a synthetic model
	_, err := fixture.service.CreateSyntheticModel(t.Context(), SyntheticModelCreateRequest{
		Name:         "gemini-flash-auto-best",
		TargetModels: []string{"gemini-flash-latest", "gemini-3.8-flash"},
	})
	if err != nil {
		t.Fatalf("create synthetic model: %v", err)
	}

	// 2. Create a JEV channel group for the decision model
	_, err = fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name:           stringPointer("Jev Router"),
		ChannelID:      "jev",
		ConnectionType: models.ConnectionTypeAPIKey,
		Params:         json.RawMessage(`{"base_url":"https://api.typesafe.ai/v1"}`),
		Credentials:    "test-key",
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "von-latest", Alias: "von-latest", AliasEnabled: true},
			},
		},
	})
	if err != nil {
		t.Fatalf("create group: %v", err)
	}

	// 3. Save auto_model settings targeting the synthetic model
	autoModelJSON := `{
		"enabled": true,
		"model": "von-latest",
		"timeout_seconds": 2,
		"models": [
			{
				"id": "entry-1",
				"name": "auto-smart",
				"fallback": "preset-1",
				"presets": [
					{
						"id": "preset-1",
						"name": "Low",
						"model": "gemini-flash-auto-best",
						"description": "Routine tasks",
						"parameter_overrides": []
					}
				]
			}
		]
	}`

	_, err = fixture.service.UpdateSettings(t.Context(), SettingsUpdateRequest{
		Settings: map[string]json.RawMessage{"auto_model": json.RawMessage(autoModelJSON)},
	})
	if err != nil {
		t.Fatalf("UpdateSettings with synthetic model target failed: %v", err)
	}
}
