package gateway

import (
	"gpt-load/internal/execution"
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
