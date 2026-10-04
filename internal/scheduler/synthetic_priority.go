package scheduler

import (
	"time"

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

// EarliestCandidateAvailable checks all candidate targets across priority tiers and returns
// the earliest time when a candidate's cooldown will expire, or now if one is already available.
// If no credentials or groups exist for the query, it returns false.
func (iterator *Iterator) EarliestCandidateAvailable(now time.Time) (time.Time, bool) {
	if iterator == nil || iterator.credentials == nil {
		return time.Time{}, false
	}
	source, ok := iterator.credentials.(interface {
		Snapshot() []state.CredentialRuntimeView
	})
	if !ok {
		return time.Time{}, false
	}
	views := source.Snapshot()
	viewsByGroup := make(map[uint][]state.CredentialRuntimeView)
	for _, v := range views {
		if v.Status == state.CredentialStatusActive && !v.Blacklisted {
			viewsByGroup[v.GroupID] = append(viewsByGroup[v.GroupID], v)
		}
	}

	var earliest time.Time
	hasCandidates := false

	checkPool := func(pool *candidatePool) {
		for groupID, targets := range pool.targetsByGroup {
			if _, skipped := iterator.skippedGroups[groupID]; skipped {
				continue
			}
			creds := viewsByGroup[groupID]
			if len(creds) == 0 {
				continue
			}
			for _, target := range targets {
				modelID := target.target.UpstreamModelID
				for _, cred := range creds {
					if iterator.allowedCredentialIDs != nil {
						if _, allowed := iterator.allowedCredentialIDs[cred.ID]; !allowed {
							continue
						}
					}
					if iterator.allowedCredentialRefs != nil {
						ref, allowed := iterator.allowedCredentialRefs[cred.ID]
						if !allowed {
							continue
						}
						if ref.GroupID != cred.GroupID {
							continue
						}
						if ref.IdentityGeneration != cred.IdentityGeneration {
							continue
						}
					}
					hasCandidates = true
					until := cred.CooldownUntil
					if modelID != "" && cred.ModelCooldowns != nil {
						if mUntil := cred.ModelCooldowns[modelID]; mUntil.After(until) {
							until = mUntil
						}
					}
					if !until.After(now) {
						earliest = now
						return
					}
					if earliest.IsZero() || until.Before(earliest) {
						earliest = until
					}
				}
			}
		}
	}

	for i := range iterator.priorityTiers {
		tier := &iterator.priorityTiers[i]
		checkPool(&tier.regular)
		checkPool(&tier.storeDowngraded)
	}

	if !hasCandidates {
		return time.Time{}, false
	}
	return earliest, !earliest.IsZero()
}
