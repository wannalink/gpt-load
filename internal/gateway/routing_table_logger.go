package gateway

import (
	"fmt"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/platform/utils"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

type modelCandidateStats struct {
	modelName        string
	totalKeys        int
	availableKeys    int
	quotaExhausted   int
	isModelBusy      bool
	modelBusyTime    time.Duration
	isQuotaExhausted bool
	quotaExhaustTime time.Duration
}

func logRoutingTable(
	logger *logrus.Logger,
	snapshot *state.ConfigSnapshot,
	registry runtimeCredentialRegistry,
	requestedModel string,
	query scheduler.Query,
	now time.Time,
) {
	if snapshot == nil || registry == nil || strings.TrimSpace(requestedModel) == "" {
		return
	}

	modelType := "real"
	targetModels := []string{requestedModel}

	// 1. Check if model is a Synthetic Model
	if targets, exists := snapshot.SyntheticModels[requestedModel]; exists && len(targets) > 0 {
		modelType = "synthetic"
		targetModels = targets
	} else if snapshot.AutoModels != nil { // 2. Check if model is a Smart/AutoModel
		if autoEntry, ok := snapshot.AutoModels.Lookup(requestedModel); ok {
			modelType = "smart"
			models := make([]string, 0, len(autoEntry.Presets))
			for _, preset := range autoEntry.Presets {
				if preset.Model != "" {
					models = append(models, preset.Model)
				}
			}
			if len(models) > 0 {
				targetModels = models
			}
		}
	}

	// 2. Gather key & cooldown stats for each target model
	statsList := make([]modelCandidateStats, 0, len(targetModels))
	for _, m := range targetModels {
		statsList = append(statsList, computeModelStats(snapshot, registry, m, requestedModel, query, now))
	}

	// 3. Format and log the table
	formatAndLogRoutingTable(logger, requestedModel, modelType, statsList)
}

func computeModelStats(
	snapshot *state.ConfigSnapshot,
	registry runtimeCredentialRegistry,
	model string,
	requestedModel string,
	query scheduler.Query,
	now time.Time,
) modelCandidateStats {
	stats := modelCandidateStats{
		modelName: model,
	}

	// Resolve actual real model targets if 'model' is itself a synthetic or auto model alias
	realModels := resolveRealModels(snapshot, model)

	// Find all group IDs and associated upstream model IDs supporting these real models
	matchedGroups := make(map[uint]string) // groupID -> upstreamModelID

	for _, realModel := range realModels {
		for clientProtocol, ops := range snapshot.ExecutionCandidates {
			if query.ClientProtocol != "" && clientProtocol != query.ClientProtocol {
				continue
			}
			for op, models := range ops {
				if query.Operation != "" && op != query.Operation {
					continue
				}
				for key, routes := range models {
					isMatch := key == realModel || key == state.NoModelRouteKey
					for _, route := range routes {
						if isMatch || route.UpstreamModelID == realModel {
							matchedGroups[route.GroupID] = route.UpstreamModelID
						}
					}
				}
			}
		}
	}

	var minQuotaCooldownUntil time.Time
	var hasMinQuotaCooldown bool
	var minModelCooldownUntil time.Time
	var hasMinModelCooldown bool

	for groupID, upstreamModelID := range matchedGroups {
		views := registry.GroupRuntimeViews(groupID)
		for _, view := range views {
			stats.totalKeys++

			// Check key-level quota cooldown
			if view.CooldownUntil.After(now) {
				stats.quotaExhausted++
				if !hasMinQuotaCooldown || view.CooldownUntil.Before(minQuotaCooldownUntil) {
					minQuotaCooldownUntil = view.CooldownUntil
					hasMinQuotaCooldown = true
				}
			} else {
				stats.availableKeys++
			}

			// Check model-level high demand / busy endpoint cooldown
			if view.ModelCooldowns != nil {
				for _, k := range []string{model, realModels[0], upstreamModelID, requestedModel} {
					if k != "" {
						if mUntil, ok := view.ModelCooldowns[k]; ok && mUntil.After(now) {
							if !hasMinModelCooldown || mUntil.Before(minModelCooldownUntil) {
								minModelCooldownUntil = mUntil
								hasMinModelCooldown = true
							}
						}
					}
				}
			}
		}
	}

	if hasMinModelCooldown && minModelCooldownUntil.After(now) {
		stats.isModelBusy = true
		stats.modelBusyTime = minModelCooldownUntil.Sub(now)
	}

	if stats.availableKeys == 0 && hasMinQuotaCooldown && minQuotaCooldownUntil.After(now) {
		stats.isQuotaExhausted = true
		stats.quotaExhaustTime = minQuotaCooldownUntil.Sub(now)
	}

	return stats
}

func resolveRealModels(snapshot *state.ConfigSnapshot, model string) []string {
	visited := make(map[string]bool)
	var resolve func(m string) []string
	resolve = func(m string) []string {
		if visited[m] {
			return nil
		}
		visited[m] = true

		if targets, ok := snapshot.SyntheticModels[m]; ok && len(targets) > 0 {
			var realTargets []string
			for _, t := range targets {
				realTargets = append(realTargets, resolve(t)...)
			}
			if len(realTargets) > 0 {
				return realTargets
			}
		}
		if snapshot.AutoModels != nil {
			if autoEntry, ok := snapshot.AutoModels.Lookup(m); ok {
				var realTargets []string
				for _, preset := range autoEntry.Presets {
					if preset.Model != "" {
						realTargets = append(realTargets, resolve(preset.Model)...)
					}
				}
				if len(realTargets) > 0 {
					return realTargets
				}
			}
		}
		return []string{m}
	}
	return resolve(model)
}

func formatAndLogRoutingTable(
	logger *logrus.Logger,
	requestedModel string,
	modelType string,
	statsList []modelCandidateStats,
) {
	summaryLines := make([]string, 0, len(statsList))
	for _, s := range statsList {
		statusStr := "AVAILABLE"
		timeStr := "0s"
		if s.isModelBusy {
			statusStr = "MODEL BUSY"
			timeStr = s.modelBusyTime.Round(time.Second).String()
		} else if s.isQuotaExhausted {
			statusStr = "QUOTA EXHAUSTED"
			timeStr = s.quotaExhaustTime.Round(time.Second).String()
		} else if s.totalKeys == 0 {
			statusStr = "NO KEYS"
			timeStr = "N/A"
		}

		summaryLines = append(
			summaryLines,
			fmt.Sprintf("%s(%d/%d, %s, %s)", s.modelName, s.availableKeys, s.totalKeys, statusStr, timeStr),
		)
	}

	targetsSummary := strings.Join(summaryLines, " | ")
	msg := fmt.Sprintf("Routing candidates for %s (%s): %s", requestedModel, modelType, targetsSummary)

	utils.LogPlaneBestEffort(
		logger,
		logrus.InfoLevel,
		utils.LogPlaneData,
		logrus.Fields{
			"event":           "routing_table",
			"requested_model": requestedModel,
			"model_type":      modelType,
			"targets_summary": targetsSummary,
		},
		msg,
	)
}
