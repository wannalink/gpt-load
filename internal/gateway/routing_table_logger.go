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
	modelName      string
	totalKeys      int
	availableKeys  int
	cooldownedKeys int
	timeUntilAvail time.Duration
	isOnCooldown   bool
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
		statsList = append(statsList, computeModelStats(snapshot, registry, m, now))
	}

	// 3. Format and log the table
	formatAndLogRoutingTable(logger, requestedModel, modelType, statsList)
}

func computeModelStats(
	snapshot *state.ConfigSnapshot,
	registry runtimeCredentialRegistry,
	model string,
	now time.Time,
) modelCandidateStats {
	stats := modelCandidateStats{
		modelName: model,
	}

	// Find all groups supporting this model
	groupIDs := make(map[uint]struct{})
	for _, ops := range snapshot.ExecutionCandidates {
		for _, models := range ops {
			if routes, ok := models[model]; ok {
				for _, route := range routes {
					groupIDs[route.GroupID] = struct{}{}
				}
			}
		}
	}

	var minCooldownUntil time.Time
	var hasMinCooldown bool

	for groupID := range groupIDs {
		views := registry.GroupRuntimeViews(groupID)
		for _, view := range views {
			stats.totalKeys++

			// Check credential-level or model-level cooldown
			var cooldownUntil time.Time
			if view.CooldownUntil.After(now) {
				cooldownUntil = view.CooldownUntil
			}
			if view.ModelCooldowns != nil {
				if mUntil, ok := view.ModelCooldowns[model]; ok && mUntil.After(cooldownUntil) {
					cooldownUntil = mUntil
				}
			}

			if cooldownUntil.After(now) {
				stats.cooldownedKeys++
				if !hasMinCooldown || cooldownUntil.Before(minCooldownUntil) {
					minCooldownUntil = cooldownUntil
					hasMinCooldown = true
				}
			} else {
				stats.availableKeys++
			}
		}
	}

	if stats.availableKeys == 0 && hasMinCooldown && minCooldownUntil.After(now) {
		stats.isOnCooldown = true
		stats.timeUntilAvail = minCooldownUntil.Sub(now)
	}

	return stats
}

func formatAndLogRoutingTable(
	logger *logrus.Logger,
	requestedModel string,
	modelType string,
	statsList []modelCandidateStats,
) {
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("\n================================ ROUTING CANDIDATES TABLE ================================\n"))
	builder.WriteString(fmt.Sprintf("Requested Model: %s | Type: %s\n", requestedModel, modelType))
	builder.WriteString(fmt.Sprintf("----------------------------------------------------------------------------------------\n"))
	builder.WriteString(fmt.Sprintf("%-25s | %-22s | %-12s | %-20s\n", "TARGET MODEL", "AVAILABLE / TOTAL KEYS", "STATUS", "TIME UNTIL AVAIL"))
	builder.WriteString(fmt.Sprintf("----------------------------------------------------------------------------------------\n"))

	summaryLines := make([]string, 0, len(statsList))
	for _, s := range statsList {
		statusStr := "AVAILABLE"
		timeStr := "0s"
		if s.isOnCooldown {
			statusStr = "COOLDOWN"
			timeStr = s.timeUntilAvail.Round(time.Second).String()
		} else if s.totalKeys == 0 {
			statusStr = "NO KEYS"
			timeStr = "N/A"
		}

		keysStr := fmt.Sprintf("%d / %d keys", s.availableKeys, s.totalKeys)
		builder.WriteString(fmt.Sprintf("%-25s | %-22s | %-12s | %-20s\n", s.modelName, keysStr, statusStr, timeStr))

		summaryLines = append(summaryLines, fmt.Sprintf("%s(%d/%d, %s, %s)", s.modelName, s.availableKeys, s.totalKeys, statusStr, timeStr))
	}
	builder.WriteString(fmt.Sprintf("================================================================------------------------"))

	utils.LogPlaneBestEffort(
		logger,
		logrus.InfoLevel,
		utils.LogPlaneData,
		logrus.Fields{
			"event":           "routing_table",
			"requested_model": requestedModel,
			"model_type":      modelType,
			"targets_summary": strings.Join(summaryLines, " | "),
		},
		builder.String(),
	)
}
