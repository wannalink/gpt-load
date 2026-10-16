package scheduler

import (
	"slices"
	"strings"
	"time"

	"gpt-load/internal/channel"
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

func candidateGroupIDsForSynthetic(snapshot *state.ConfigSnapshot, query Query, targetModels []string) []uint {
	seen := make(map[uint]struct{})
	var groupIDs []uint
	for _, targetModel := range targetModels {
		q := query
		model := targetModel
		q.ExternalModel = &model
		for _, gID := range candidateGroupIDsForSingleModel(snapshot, q) {
			if _, exists := seen[gID]; !exists {
				seen[gID] = struct{}{}
				groupIDs = append(groupIDs, gID)
			}
		}
	}
	return groupIDs
}

func initSyntheticIterator(iterator *Iterator, snapshot *state.ConfigSnapshot, query Query, targetModels []string) {
	iterator.priorityTiers = make([]priorityTier, 0, len(targetModels))
	for _, targetModel := range targetModels {
		q := query
		m := targetModel
		q.ExternalModel = &m
		targets, reason := filterTargetsWithReason(snapshot, q)
		if iterator.staticReason == "" && reason != "" {
			iterator.staticReason = reason
		}
		tier := priorityTier{
			regular:         newCandidatePool(),
			storeDowngraded: newCandidatePool(),
			tried:           make(map[uint]struct{}),
		}
		for _, target := range targets {
			pool := &tier.regular
			if target.responsesStoreDowngraded {
				pool = &tier.storeDowngraded
			}
			mode := target.target.Mode
			groupID := target.target.GroupID
			pool.targetsByGroup[groupID] = append(pool.targetsByGroup[groupID], target)
			if !slices.Contains(pool.groupIDsByMode[mode], groupID) {
				pool.groupIDsByMode[mode] = append(pool.groupIDsByMode[mode], groupID)
			}
		}
		if len(targets) > 0 {
			iterator.staticReason = ""
		}
		iterator.priorityTiers = append(iterator.priorityTiers, tier)
	}
	for i := range iterator.priorityTiers {
		tier := &iterator.priorityTiers[i]
		for _, pool := range []*candidatePool{&tier.regular, &tier.storeDowngraded} {
			for _, targets := range pool.targetsByGroup {
				slices.SortFunc(targets, func(a, b candidateTarget) int {
					return strings.Compare(a.target.UpstreamModelID, b.target.UpstreamModelID)
				})
			}
		}
	}
	if len(iterator.priorityTiers) > 0 {
		iterator.regular = iterator.priorityTiers[0].regular
		iterator.storeDowngraded = iterator.priorityTiers[0].storeDowngraded
	}
}

func (iterator *Iterator) Snapshot() *state.ConfigSnapshot {
	if iterator == nil {
		return nil
	}
	return iterator.snapshot
}

func (iterator *Iterator) ResetTried() {
	if iterator == nil {
		return
	}
	iterator.tried = make(map[uint]struct{})
	for i := range iterator.priorityTiers {
		iterator.priorityTiers[i].tried = make(map[uint]struct{})
	}
	iterator.skippedGroups = make(map[uint]struct{})
}

func (iterator *Iterator) advanceSyntheticPriority(selection Selection) {
	if iterator == nil {
		return
	}
	priority := selection.Group.Priority
	for i := range iterator.priorityTiers {
		tier := &iterator.priorityTiers[i]
		tierPool := &tier.regular
		if selection.ResponsesStoreDowngraded {
			tierPool = &tier.storeDowngraded
		}
		tierPool.priorityLimit, tierPool.lowerPriority = &priority, true
	}
}

func (iterator *Iterator) nextSyntheticTier() (Selection, error) {
	for tierIdx := 0; tierIdx < len(iterator.priorityTiers); tierIdx++ {
		tier := &iterator.priorityTiers[tierIdx]
		for _, pool := range []*candidatePool{&tier.regular, &tier.storeDowngraded} {
			var selected state.CredentialMeta
			var target candidateTarget
			var found bool
			now := iterator.now()
			iterator.withWeightedPoolForTier(tier, pool, []channel.RouteMode{channel.RouteNative, channel.RouteConverted}, now, func(weighted []weightedCredential) {
				weighted = iterator.priorityCandidates(pool, weighted)
				for _, modes := range iterator.routeModeTiers {
					eligible := make([]weightedCredential, 0, len(weighted))
					for _, candidate := range weighted {
						for _, candidateTarget := range pool.targetsByGroup[candidate.meta.GroupID] {
							if iterator.targetAvailable(candidateTarget, candidate.meta, modes, now) {
								eligible = append(eligible, candidate)
								break
							}
						}
					}
					selected, found = iterator.selectCredential(eligible, iterator.preferredCredentialID)
					if !found {
						continue
					}
					target = iterator.selectTarget(pool, modes, selected, now)
					return
				}
			})
			if !found {
				continue
			}
			tier.tried[selected.ID] = struct{}{}
			iterator.tried[selected.ID] = struct{}{}
			if pool.priorityLimit != nil {
				priority := target.group.Priority
				pool.priorityLimit, pool.lowerPriority = &priority, false
			}
			return newSelection(selected, target), nil
		}
	}
	return Selection{}, ErrExhausted
}

func (iterator *Iterator) withWeightedPoolForTier(tier *priorityTier, candidates *candidatePool, modes []channel.RouteMode, now time.Time, fn func([]weightedCredential)) {
	var groupIDs []uint
	seenGroups := make(map[uint]struct{})
	for _, mode := range modes {
		for _, groupID := range candidates.groupIDsByMode[mode] {
			if _, exists := seenGroups[groupID]; !exists {
				seenGroups[groupID] = struct{}{}
				groupIDs = append(groupIDs, groupID)
			}
		}
	}
	excluded := func(id uint) bool {
		if tier != nil {
			if _, tried := tier.tried[id]; tried {
				return true
			}
		} else {
			if _, tried := iterator.tried[id]; tried {
				return true
			}
		}
		return false
	}
	consume := func(pool []state.CredentialMeta) {
		weighted := make([]weightedCredential, 0, len(pool))
		for _, credential := range pool {
			if iterator.allowedCredentialIDs != nil {
				if _, ok := iterator.allowedCredentialIDs[credential.ID]; !ok {
					continue
				}
			}
			if iterator.allowedCredentialRefs != nil {
				ref, ok := iterator.allowedCredentialRefs[credential.ID]
				if !ok || ref.GroupID != credential.GroupID || ref.IdentityGeneration != credential.IdentityGeneration {
					continue
				}
			}
			if _, skipped := iterator.skippedGroups[credential.GroupID]; skipped {
				continue
			}
			for _, target := range candidates.targetsByGroup[credential.GroupID] {
				if !iterator.targetAvailable(target, credential, modes, now) {
					continue
				}
				weight := effectiveWeight(target.group.WeightManual, credential.WeightManual)
				if weight > 0 {
					weighted = append(weighted, weightedCredential{meta: credential, weight: weight})
				}
				break
			}
		}
		fn(weighted)
	}
	if source, ok := iterator.credentials.(interface {
		WithCredentialCandidates([]uint, func(uint) bool, time.Time, func([]state.CredentialMeta))
	}); ok {
		source.WithCredentialCandidates(groupIDs, excluded, now, consume)
	} else {
		consume(iterator.credentials.CollectCredentialCandidates(groupIDs, excluded, now))
	}
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
			creds := viewsByGroup[groupID]
			if len(creds) == 0 {
				continue
			}
			for _, target := range targets {
				modelID := target.target.UpstreamModelID
				for _, cred := range creds {
					if _, tried := iterator.tried[cred.ID]; tried {
						continue
					}
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
						if _, skipped := iterator.skippedGroups[groupID]; !skipped {
							earliest = now
							return
						}
						continue
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
