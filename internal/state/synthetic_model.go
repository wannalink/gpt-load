package state

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"gpt-load/internal/storage/models"
)

type SyntheticModelConfig struct {
	ID           uint
	Name         string
	Description  string
	TargetModels []string
	Enabled      bool
}

func compileSyntheticModels(snapshot *ConfigSnapshot, configs []SyntheticModelConfig) error {
	snapshot.SyntheticModels = make(map[string][]string, len(configs))
	for _, cfg := range configs {
		if !cfg.Enabled {
			continue
		}
		name := strings.TrimSpace(cfg.Name)
		if name == "" {
			return fmt.Errorf("synthetic model name is required")
		}
		if _, exists := snapshot.SyntheticModels[name]; exists {
			return fmt.Errorf("duplicate synthetic model %q", name)
		}
		if len(cfg.TargetModels) == 0 {
			return fmt.Errorf("synthetic model %q must have at least one target model", name)
		}
		cleanedTargets := make([]string, 0, len(cfg.TargetModels))
		for _, target := range cfg.TargetModels {
			t := strings.TrimSpace(target)
			if t == "" {
				return fmt.Errorf("synthetic model %q contains empty target model", name)
			}
			if t == name {
				return fmt.Errorf("synthetic model %q cannot target itself", name)
			}
			cleanedTargets = append(cleanedTargets, t)
		}
		snapshot.SyntheticModels[name] = cleanedTargets
	}
	return nil
}

// MapSyntheticModels transforms synthetic model DB rows and SYNTHETIC_MODELS environment overrides into configs.
func MapSyntheticModels(rows []models.SyntheticModel) ([]SyntheticModelConfig, error) {
	configByName := make(map[string]SyntheticModelConfig, len(rows))
	order := make([]string, 0, len(rows))
	for _, row := range rows {
		var targetModels []string
		if len(row.TargetModels) > 0 {
			if err := json.Unmarshal(row.TargetModels, &targetModels); err != nil {
				return nil, fmt.Errorf("unmarshal synthetic model %d target models: %w", row.ID, err)
			}
		}
		configByName[row.Name] = SyntheticModelConfig{
			ID:           row.ID,
			Name:         row.Name,
			Description:  row.Description,
			TargetModels: targetModels,
			Enabled:      row.Enabled,
		}
		order = append(order, row.Name)
	}
	if envJSON := strings.TrimSpace(os.Getenv("SYNTHETIC_MODELS")); envJSON != "" {
		type envSyntheticModel struct {
			Name         string   `json:"name"`
			Description  string   `json:"description"`
			TargetModels []string `json:"target_models"`
			Enabled      *bool    `json:"enabled"`
		}
		var envModels []envSyntheticModel
		if err := json.Unmarshal([]byte(envJSON), &envModels); err == nil {
			for _, em := range envModels {
				if strings.TrimSpace(em.Name) == "" {
					continue
				}
				enabled := true
				if em.Enabled != nil {
					enabled = *em.Enabled
				}
				cfg, exists := configByName[em.Name]
				if exists {
					cfg.TargetModels = em.TargetModels
					if em.Description != "" {
						cfg.Description = em.Description
					}
					cfg.Enabled = enabled
					configByName[em.Name] = cfg
				} else {
					configByName[em.Name] = SyntheticModelConfig{
						Name:         em.Name,
						Description:  em.Description,
						TargetModels: em.TargetModels,
						Enabled:      enabled,
					}
					order = append(order, em.Name)
				}
			}
		}
	}
	configs := make([]SyntheticModelConfig, 0, len(order))
	for _, name := range order {
		configs = append(configs, configByName[name])
	}
	return configs, nil
}
