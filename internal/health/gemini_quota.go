package health

import (
	"strings"
	"time"

	"gpt-load/internal/execution"
)

// GeminiFreeTierQuotaCooldown is the forced cooldown duration for a key that exceeds the free tier request quota.
const GeminiFreeTierQuotaCooldown = 5 * time.Minute

// GeminiFreeTierQuotaMetric is the specific metric string returned in the Gemini 429 error.
const GeminiFreeTierQuotaMetric = "generativelanguage.googleapis.com/generate_content_free_tier_requests"

// GeminiFreeTierQuotaErrorSubstring is the error prefix indicating free tier quota exhaustion.
const GeminiFreeTierQuotaErrorSubstring = "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:"

// geminiFreeTierQuotaDecision inspects the execution attempt for Gemini's free tier quota error
// ("Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit:")
// and returns a Decision forcing a 5-minute credential cooldown using the app's built-in cooldown mechanism.
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
		execution.ErrorScopeCredential,
		retry,
		EffectCooldownCredential,
		RuleID("gemini.free_tier_quota_credential_cooldown"),
	)
	res.CooldownUntil = attempt.Now.Add(GeminiFreeTierQuotaCooldown)
	return res, true
}
