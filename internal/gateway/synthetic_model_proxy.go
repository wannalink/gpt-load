package gateway

import (
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

func isExplicitTokenLimitRejection(result UpstreamResult) bool {
	if result.ExecutionError != nil {
		return execution.ExplicitRequestRejection(
			result.ExecutionError.Type,
			result.ExecutionError.Code,
			result.ExecutionError.Summary,
		)
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
	return decision.Retry != health.RetryNone || (isSynthetic && !isExplicitTokenLimitRejection(result))
}

func shouldRetrySyntheticResponse(decision health.Decision, isSynthetic bool, result UpstreamResult) bool {
	return decision.Retry != health.RetryNone || (isSynthetic && result.StatusCode >= 400 && !isExplicitTokenLimitRejection(result))
}

func shouldRetrySyntheticAttempt(decision health.Decision, isSynthetic bool) bool {
	return decision.Retry != health.RetryNone || isSynthetic
}
