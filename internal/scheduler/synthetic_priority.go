package scheduler

import (
	"gpt-load/internal/state"
)

type priorityTier struct {
	regular         candidatePool
	storeDowngraded candidatePool
	tried           map[uint]struct{}
}

func isSyntheticModel(snapshot *state.ConfigSnapshot, query Query) ([]string, bool) {
	if snapshot == nil || snapshot.SyntheticModels == nil || query.ExternalModel == nil {
		return nil, false
	}
	targetModels, ok := snapshot.SyntheticModels[*query.ExternalModel]
	return targetModels, ok && len(targetModels) > 0
}

func candidateGroupIDsForSingleModel(snapshot *state.ConfigSnapshot, query Query) []uint {
	decisions, _, err := evaluateTargets(
		snapshot,
		snapshot.ExecutionCandidates,
		normalizeQuery(query),
	)
	if err != nil {
		return []uint{}
	}
	groupIDs := make([]uint, 0, len(decisions))
	for _, decision := range decisions {
		if decision.included {
			groupIDs = append(groupIDs, decision.target.GroupID)
		}
	}
	return groupIDs
}
