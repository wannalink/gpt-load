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
		markers := strings.ToLower(strings.Join([]string{
			result.ExecutionError.Type,
			result.ExecutionError.Code,
			result.ExecutionError.Summary,
		}, " "))
		if (strings.Contains(markers, "input_token_count") ||
			strings.Contains(markers, "generate_content_free_tier_input_token_count") ||
			strings.Contains(markers, "token count exceeds")) &&
			!strings.Contains(markers, "per minute") && !strings.Contains(markers, "tpm") {
			return true
		}
	}
	if len(result.Body) > 0 {
		body := strings.ToLower(string(result.Body))
		if (strings.Contains(body, "input_token_count") ||
			strings.Contains(body, "generate_content_free_tier_input_token_count") ||
			strings.Contains(body, "token count exceeds")) &&
			!strings.Contains(body, "per minute") && !strings.Contains(body, "tpm") {
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
	if decision.Category == health.FailureCategoryClientError || isExplicitTerminalRejection(result) {
		return false
	}
	return decision.Retry != health.RetryNone || isSynthetic
}

func shouldRetrySyntheticResponse(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	if decision.Category == health.FailureCategoryClientError || isExplicitTerminalRejection(result) {
		return false
	}
	return decision.Retry != health.RetryNone || (isSynthetic && result.StatusCode >= 400)
}

func shouldRetrySyntheticAttempt(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	if decision.Category == health.FailureCategoryClientError || isExplicitTerminalRejection(result) {
		return false
	}
	return decision.Retry != health.RetryNone || isSynthetic
}
