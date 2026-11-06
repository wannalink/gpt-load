package health

import (
	"strings"
	"sync"
	"time"

	"gpt-load/internal/execution"
)

// GeminiFreeTierQuotaMetric is the specific metric string returned in the Gemini 429 error.
const GeminiFreeTierQuotaMetric = "generativelanguage.googleapis.com/generate_content_free_tier_requests"

// GeminiFreeTierQuotaErrorSubstring is the error prefix indicating free tier quota exhaustion.
const GeminiFreeTierQuotaErrorSubstring = "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:"

// GeminiHighDemandErrorSubstring is the error message indicating high model demand.
const GeminiHighDemandErrorSubstring = "this model is currently experiencing high demand. spikes in demand are usually temporary. please try again later."

var geminiBackoffSteps = []time.Duration{
	30 * time.Second,
	60 * time.Second,
	120 * time.Second,
	290 * time.Second,
}

type geminiBackoffKey struct {
	groupID uint
	model   string
}

type geminiBackoffState struct {
	step        int
	lastFailure time.Time
}

var (
	geminiBackoffMu sync.Mutex
	geminiBackoffs  = make(map[geminiBackoffKey]*geminiBackoffState)
)

func getNextGeminiBackoff(groupID uint, model string, now time.Time) time.Duration {
	if groupID == 0 || model == "" {
		return geminiBackoffSteps[0]
	}
	key := geminiBackoffKey{groupID: groupID, model: model}

	geminiBackoffMu.Lock()
	defer geminiBackoffMu.Unlock()

	state, ok := geminiBackoffs[key]
	if !ok {
		state = &geminiBackoffState{
			step:        0,
			lastFailure: now,
		}
		geminiBackoffs[key] = state
		return geminiBackoffSteps[0]
	}

	// Simple pruning of very old records when map grows large
	if len(geminiBackoffs) > 1000 {
		for k, v := range geminiBackoffs {
			if now.Sub(v.lastFailure) > 24*time.Hour {
				delete(geminiBackoffs, k)
			}
		}
	}

	// If a long time has passed since the last failure, reset backoff step
	if now.Sub(state.lastFailure) > 10*time.Minute {
		state.step = 0
	} else if state.step < len(geminiBackoffSteps)-1 {
		state.step++
	}
	state.lastFailure = now
	return geminiBackoffSteps[state.step]
}

// parseGeminiRetryDuration parses duration from Gemini's rate limit error message
// (e.g. "Please retry in 3h7m1.473191079s.").
func parseGeminiRetryDuration(text string) (time.Duration, bool) {
	lower := strings.ToLower(text)
	idx := strings.Index(lower, "please retry in ")
	if idx == -1 {
		return 0, false
	}
	remaining := strings.TrimSpace(text[idx+len("please retry in "):])
	fields := strings.Fields(remaining)
	if len(fields) == 0 {
		return 0, false
	}
	rawDuration := strings.TrimRight(fields[0], ".,;:!?")
	d, err := time.ParseDuration(rawDuration)
	if err != nil || d <= 0 {
		return 0, false
	}
	return d, true
}

// geminiFreeTierQuotaDecision inspects the execution attempt for Gemini's free tier quota error
// ("Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:")
// and returns a Decision forcing a model-specific cooldown (EffectCooldownModel) for the specific key-model pair
// until the next 12:00 AM midnight UTC or the duration parsed from the error message.
func geminiFreeTierQuotaDecision(attempt ExecutionAttempt) (Decision, bool) {
	if attempt.Evidence == nil {
		return Decision{}, false
	}

	markers := strings.ToLower(strings.Join([]string{
		attempt.Evidence.Type,
		attempt.Evidence.Code,
		attempt.Evidence.Summary,
	}, " "))

	// generate_content_free_tier_input_token_count is an explicit terminal rejection for the request
	// (input token quota exhausted/exceeded). Google's error description explicitly includes
	// the phrase "input tokens per minute limit exceeded on Free Tier", so it must not be
	// bypassed by the "per minute" or "tpm" checks intended for generic TPM rate limits.
	if strings.Contains(markers, "generate_content_free_tier_input_token_count") ||
		((strings.Contains(markers, "input_token_count") || strings.Contains(markers, "token count exceeds")) &&
			!strings.Contains(markers, "per minute") && !strings.Contains(markers, "tpm")) {
		return decision(
			FailureCategoryClientError,
			originForEvidence(attempt.Evidence),
			execution.ErrorScopeRequest,
			RetryNone,
			EffectNone,
			RuleID("gemini.input_token_limit_exceeded"),
		), true
	}

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

	if d, ok := parseGeminiRetryDuration(attempt.Evidence.Summary); ok {
		res.CooldownUntil = attempt.Now.Add(d)
	} else if d, ok := parseGeminiRetryDuration(markers); ok {
		res.CooldownUntil = attempt.Now.Add(d)
	} else {
		nowUTC := attempt.Now.UTC()
		nextMidnightUTC := time.Date(nowUTC.Year(), nowUTC.Month(), nowUTC.Day()+1, 0, 0, 0, 0, time.UTC)
		res.CooldownUntil = nextMidnightUTC
	}

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

// ReduceGeminiBackoffOnSuccess reduces the high demand backoff for a group-model pair by 2 steps on successful response.
func ReduceGeminiBackoffOnSuccess(groupID uint, model string) {
	if model != "" {
		execution.ReduceGeminiModelPauseOnSuccess(model)
	}
	if groupID == 0 || model == "" {
		return
	}
	key := geminiBackoffKey{groupID: groupID, model: model}

	geminiBackoffMu.Lock()
	defer geminiBackoffMu.Unlock()

	state, ok := geminiBackoffs[key]
	if !ok {
		return
	}
	state.step -= 2
	if state.step < 0 {
		delete(geminiBackoffs, key)
	}
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
