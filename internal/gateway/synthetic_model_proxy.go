package gateway

import (
	"strings"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
)

func isSyntheticModelRequest(snapshot *state.ConfigSnapshot, externalModel string) bool {
	if snapshot == nil || snapshot.SyntheticModels == nil || externalModel == "" {
		return false
	}
	_, ok := snapshot.SyntheticModels[externalModel]
	return ok
}

func isExplicitTerminalRejection(result UpstreamResult) bool {
	if result.ExecutionError != nil {
		if execution.ExplicitRequestRejection(
			result.ExecutionError.Type,
			result.ExecutionError.Code,
			result.ExecutionError.Summary,
		) {
			return true
		}
		summary := strings.ToLower(result.ExecutionError.Summary)
		if strings.Contains(summary, "generativelanguage.googleapis.com/generate_content_free_tier_requests") ||
			strings.Contains(summary, "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests") {
			return true
		}
	}
	if len(result.Body) > 0 {
		body := strings.ToLower(string(result.Body))
		if strings.Contains(body, "generativelanguage.googleapis.com/generate_content_free_tier_requests") ||
			strings.Contains(body, "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests") {
			return true
		}
	}
	return false
}

func forwardAttemptLimitForModel(snapshot *state.ConfigSnapshot, model string) int {
	limit := retryAttemptLimit(snapshot.Settings.RetryCount)
	if targets, ok := snapshot.SyntheticModels[model]; ok && len(targets) > 0 {
		limit = len(targets) * limit
		if limit < 50 {
			limit = 50
		}
	}
	return limit
}

func shouldRetrySyntheticProviderError(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	if decision.Retry == health.RetryNone || isExplicitTerminalRejection(result) {
		return false
	}
	return isSynthetic
}

func shouldRetrySyntheticResponse(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	if decision.Retry == health.RetryNone || isExplicitTerminalRejection(result) {
		return false
	}
	return isSynthetic && result.StatusCode >= 400
}

func shouldRetrySyntheticAttempt(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	if decision.Retry == health.RetryNone || isExplicitTerminalRejection(result) {
		return false
	}
	return isSynthetic
}
