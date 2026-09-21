package state

import (
	"fmt"
	"strings"
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
