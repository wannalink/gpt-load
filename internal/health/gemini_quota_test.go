package health

import (
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

func TestGeminiFreeTierQuotaDecision(t *testing.T) {
	now := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name       string
		attempt    ExecutionAttempt
		wantMatch  bool
		wantScope  execution.ErrorScope
		wantEffect Effect
		wantRetry  RetryDirective
		wantRuleID RuleID
	}{
		{
			name: "exact gemini free tier quota error message",
			attempt: ExecutionAttempt{
				StatusCode: http.StatusTooManyRequests,
				Now:        now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusTooManyRequests,
					Summary:    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15",
				},
			},
			wantMatch:  true,
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectCooldownCredential,
			wantRetry:  RetryNextCandidate,
			wantRuleID: RuleID("gemini.free_tier_quota_credential_cooldown"),
		},
		{
			name: "case insensitive metric match in code or summary",
			attempt: ExecutionAttempt{
				StatusCode: http.StatusTooManyRequests,
				Now:        now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusTooManyRequests,
					Summary:    "Resource has been exhausted: generativelanguage.googleapis.com/generate_content_free_tier_requests",
				},
			},
			wantMatch:  true,
			wantScope:  execution.ErrorScopeCredential,
			wantEffect: EffectCooldownCredential,
			wantRetry:  RetryNextCandidate,
			wantRuleID: RuleID("gemini.free_tier_quota_credential_cooldown"),
		},
		{
			name: "other rate limit error does not trigger gemini free tier cooldown",
			attempt: ExecutionAttempt{
				StatusCode: http.StatusTooManyRequests,
				Now:        now,
				Evidence: &execution.ErrorEvidence{
					Kind:       execution.ErrorKindHTTP,
					StatusCode: http.StatusTooManyRequests,
					Summary:    "rate limit exceeded",
				},
			},
			wantMatch: false,
		},
		{
			name: "nil evidence does not match",
			attempt: ExecutionAttempt{
				StatusCode: http.StatusTooManyRequests,
				Now:        now,
			},
			wantMatch: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decision, matched := geminiFreeTierQuotaDecision(tc.attempt)
			if matched != tc.wantMatch {
				t.Fatalf("matched = %v, want %v", matched, tc.wantMatch)
			}
			if !matched {
				return
			}
			if decision.Category != FailureCategoryRateLimited {
				t.Errorf("Category = %v, want %v", decision.Category, FailureCategoryRateLimited)
			}
			if decision.Scope != tc.wantScope {
				t.Errorf("Scope = %v, want %v", decision.Scope, tc.wantScope)
			}
			if decision.Effect != tc.wantEffect {
				t.Errorf("Effect = %v, want %v", decision.Effect, tc.wantEffect)
			}
			if decision.Retry != tc.wantRetry {
				t.Errorf("Retry = %v, want %v", decision.Retry, tc.wantRetry)
			}
			if decision.RuleID != tc.wantRuleID {
				t.Errorf("RuleID = %v, want %v", decision.RuleID, tc.wantRuleID)
			}
			expectedCooldown := tc.attempt.Now.Add(GeminiFreeTierQuotaCooldown)
			if !decision.CooldownUntil.Equal(expectedCooldown) {
				t.Errorf("CooldownUntil = %v, want %v", decision.CooldownUntil, expectedCooldown)
			}
		})
	}
}

func TestJudgeExecutionGeminiFreeTierQuota(t *testing.T) {
	now := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)
	attempt := ExecutionAttempt{
		StatusCode: http.StatusTooManyRequests,
		Now:        now,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			StatusCode: http.StatusTooManyRequests,
			Summary:    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15 per minute",
		},
	}

	context := DecisionContext{
		Operation:                execution.OperationChatCompletion,
		DefaultRateLimitCooldown: time.Minute,
	}

	decision := JudgeExecution(attempt, context)

	if decision.Category != FailureCategoryRateLimited {
		t.Errorf("Category = %v, want %v", decision.Category, FailureCategoryRateLimited)
	}
	if decision.Effect != EffectCooldownCredential {
		t.Errorf("Effect = %v, want %v (built-in credential cooldown)", decision.Effect, EffectCooldownCredential)
	}
	if decision.Scope != execution.ErrorScopeCredential {
		t.Errorf("Scope = %v, want %v", decision.Scope, execution.ErrorScopeCredential)
	}
	if decision.Retry != RetryNextCandidate {
		t.Errorf("Retry = %v, want %v", decision.Retry, RetryNextCandidate)
	}
	expectedCooldown := now.Add(10 * time.Minute)
	if !decision.CooldownUntil.Equal(expectedCooldown) {
		t.Errorf("CooldownUntil = %v, want %v (10 minutes cooldown)", decision.CooldownUntil, expectedCooldown)
	}
}
