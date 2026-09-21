package health

import (
	"strings"
	"sync"
	"time"
	_ "time/tzdata"

	"gpt-load/internal/execution"
)

var pacificLoc *time.Location

func init() {
	var err error
	pacificLoc, err = time.LoadLocation("America/Los_Angeles")
	if err != nil {
		pacificLoc = time.FixedZone("Pacific Time", -8*60*60)
	}
}

// GeminiFreeTierQuotaMetric is the specific metric string returned in the Gemini 429 error.
const GeminiFreeTierQuotaMetric = "generativelanguage.googleapis.com/generate_content_free_tier_requests"

// GeminiFreeTierQuotaErrorSubstring is the error prefix indicating free tier quota exhaustion.
const GeminiFreeTierQuotaErrorSubstring = "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:"

// GeminiHighDemandErrorSubstring is the error message indicating high model demand.
const GeminiHighDemandErrorSubstring = "this model is currently experiencing high demand. spikes in demand are usually temporary. please try again later."

// GeminiHighDemandMaxBackoff is the maximum backoff duration for Gemini 503 high demand errors.
// It is capped at 4 minutes to stay below the 5-minute request timeout / cancellation safeguard.
const GeminiHighDemandMaxBackoff = 4 * time.Minute

type geminiBackoffKey struct {
	groupID uint
	model   string
}

type geminiBackoffState struct {
	currentDelay time.Duration
	lastFailure  time.Time
}

var (
	geminiBackoffMu sync.Mutex
	geminiBackoffs  = make(map[geminiBackoffKey]*geminiBackoffState)
)

func getNextGeminiBackoff(groupID uint, model string, now time.Time) time.Duration {
	if groupID == 0 || model == "" {
		return 1 * time.Minute
	}
	key := geminiBackoffKey{groupID: groupID, model: model}

	geminiBackoffMu.Lock()
	defer geminiBackoffMu.Unlock()

	state, ok := geminiBackoffs[key]
	if !ok {
		state = &geminiBackoffState{
			currentDelay: 1 * time.Minute,
			lastFailure:  now,
		}
		geminiBackoffs[key] = state
		return state.currentDelay
	}

	// Simple pruning of very old records when map grows large
	if len(geminiBackoffs) > 1000 {
		for k, v := range geminiBackoffs {
			if now.Sub(v.lastFailure) > 24*time.Hour {
				delete(geminiBackoffs, k)
			}
		}
	}

	// If a long time has passed since the last failure, reset backoff
	if now.Sub(state.lastFailure) > state.currentDelay*2+GeminiHighDemandMaxBackoff {
		state.currentDelay = 1 * time.Minute
	} else {
		// Double the delay
		state.currentDelay *= 2
		if state.currentDelay > GeminiHighDemandMaxBackoff {
			state.currentDelay = GeminiHighDemandMaxBackoff
		}
	}
	state.lastFailure = now
	return state.currentDelay
}

// geminiFreeTierQuotaDecision inspects the execution attempt for Gemini's free tier quota error
// ("Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:")
// and returns a Decision forcing a model-specific cooldown (EffectCooldownModel) for the specific key-model pair
// until the next 12:00 AM midnight Pacific Time.
func geminiFreeTierQuotaDecision(attempt ExecutionAttempt) (Decision, bool) {
	if attempt.Evidence == nil {
		return Decision{}, false
	}

	markers := strings.ToLower(strings.Join([]string{
		attempt.Evidence.Type,
		attempt.Evidence.Code,
		attempt.Evidence.Summary,
	}, " "))

	if !strings.Contains(markers, GeminiFreeTierQuotaMetric) &&
		!strings.Contains(markers, strings.ToLower(GeminiFreeTierQuotaErrorSubstring)) {
		return Decision{}, false
	}

	retry := retryUnlessExplicitlyUnknown(attempt.Evidence)
	if retry == RetryNone {
		retry = RetryNextCandidate
	}

	res := decision(
		FailureCategoryRateLimited,
		originForEvidence(attempt.Evidence),
		execution.ErrorScopeModel,
		retry,
		EffectCooldownModel,
		RuleID("gemini.free_tier_quota_model_cooldown"),
	)

	nowPT := attempt.Now.In(pacificLoc)
	nextMidnightPT := time.Date(nowPT.Year(), nowPT.Month(), nowPT.Day()+1, 0, 0, 0, 0, pacificLoc)
	res.CooldownUntil = nextMidnightPT

	return res, true
}

// geminiHighDemandDecision inspects the execution attempt for Gemini's high demand 503 error
// ("This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.")
// and returns a Decision forcing a model-specific exponential backoff cooldown (EffectCooldownModel).
func geminiHighDemandDecision(attempt ExecutionAttempt, context DecisionContext) (Decision, bool) {
	if attempt.Evidence == nil {
		return Decision{}, false
	}

	markers := strings.ToLower(strings.Join([]string{
		attempt.Evidence.Type,
		attempt.Evidence.Code,
		attempt.Evidence.Summary,
	}, " "))

	if !strings.Contains(markers, GeminiHighDemandErrorSubstring) {
		return Decision{}, false
	}

	retry := retryUnlessExplicitlyUnknown(attempt.Evidence)
	if retry == RetryNone {
		retry = RetryNextCandidate
	}

	res := decision(
		FailureCategoryRateLimited,
		originForEvidence(attempt.Evidence),
		execution.ErrorScopeModel,
		retry,
		EffectCooldownModel,
		RuleID("gemini.high_demand_model_cooldown"),
	)

	delay := getNextGeminiBackoff(context.GroupID, context.Model, attempt.Now)
	res.CooldownUntil = attempt.Now.Add(delay)

	return res, true
}

// ResetGeminiBackoff instantly resets the high demand exponential backoff timer for a group-model pair.
func ResetGeminiBackoff(groupID uint, model string) {
	if model != "" {
		execution.ResetGeminiModelPause(model)
	}
	if groupID == 0 || model == "" {
		return
	}
	key := geminiBackoffKey{groupID: groupID, model: model}

	geminiBackoffMu.Lock()
	defer geminiBackoffMu.Unlock()

	delete(geminiBackoffs, key)
}
