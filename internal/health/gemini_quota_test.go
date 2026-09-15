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
			wantScope:  execution.ErrorScopeModel,
			wantEffect: EffectCooldownModel,
			wantRetry:  RetryNextCandidate,
			wantRuleID: RuleID("gemini.free_tier_quota_model_cooldown"),
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
			wantScope:  execution.ErrorScopeModel,
			wantEffect: EffectCooldownModel,
			wantRetry:  RetryNextCandidate,
			wantRuleID: RuleID("gemini.free_tier_quota_model_cooldown"),
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

			// Calculate expected cooldown using America/Los_Angeles (Pacific Time)
			loc, err := time.LoadLocation("America/Los_Angeles")
			if err != nil {
				loc = time.FixedZone("Pacific Time", -8*60*60)
			}
			nowPT := tc.attempt.Now.In(loc)
			expectedCooldown := time.Date(nowPT.Year(), nowPT.Month(), nowPT.Day()+1, 0, 0, 0, 0, loc)
			if !decision.CooldownUntil.Equal(expectedCooldown) {
				t.Errorf("CooldownUntil = %v, want %v", decision.CooldownUntil, expectedCooldown)
			}
		})
	}
}

func TestJudgeExecutionGeminiFreeTierQuota(t *testing.T) {
	now := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)
	attempt := ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusTooManyRequests,
		Now:           now,
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
	if decision.Effect != EffectCooldownModel {
		t.Errorf("Effect = %v, want %v (built-in model cooldown)", decision.Effect, EffectCooldownModel)
	}
	if decision.Scope != execution.ErrorScopeModel {
		t.Errorf("Scope = %v, want %v", decision.Scope, execution.ErrorScopeModel)
	}
	if decision.Retry != RetryNextCandidate {
		t.Errorf("Retry = %v, want %v", decision.Retry, RetryNextCandidate)
	}

	loc, err := time.LoadLocation("America/Los_Angeles")
	if err != nil {
		loc = time.FixedZone("Pacific Time", -8*60*60)
	}
	nowPT := now.In(loc)
	expectedCooldown := time.Date(nowPT.Year(), nowPT.Month(), nowPT.Day()+1, 0, 0, 0, 0, loc)
	if !decision.CooldownUntil.Equal(expectedCooldown) {
		t.Errorf("CooldownUntil = %v, want %v (midnight PT)", decision.CooldownUntil, expectedCooldown)
	}
}

func TestGeminiHighDemandDecision(t *testing.T) {
	now := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)
	summary := "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later."
	attempt := ExecutionAttempt{
		StatusCode: http.StatusServiceUnavailable,
		Now:        now,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			StatusCode: http.StatusServiceUnavailable,
			Summary:    summary,
		},
	}

	context := DecisionContext{
		CredentialID: 42,
		GroupID:      1,
		Model:        "gemini-1.5-flash",
	}

	// 1st hit -> 1m delay
	d1, matched := geminiHighDemandDecision(attempt, context)
	if !matched {
		t.Fatalf("first attempt should match")
	}
	if d1.Effect != EffectCooldownModel || d1.Scope != execution.ErrorScopeModel {
		t.Errorf("Effect/Scope mismatch: effect=%v, scope=%v", d1.Effect, d1.Scope)
	}
	if d1.RuleID != "gemini.high_demand_model_cooldown" {
		t.Errorf("RuleID mismatch: %v", d1.RuleID)
	}
	expected1 := now.Add(1 * time.Minute)
	if !d1.CooldownUntil.Equal(expected1) {
		t.Errorf("CooldownUntil = %v, want %v", d1.CooldownUntil, expected1)
	}

	// 2nd hit (soon after) -> 2m delay
	attempt2 := attempt
	attempt2.Now = now.Add(time.Second)
	d2, _ := geminiHighDemandDecision(attempt2, context)
	expected2 := attempt2.Now.Add(2 * time.Minute)
	if !d2.CooldownUntil.Equal(expected2) {
		t.Errorf("CooldownUntil = %v, want %v (doubled)", d2.CooldownUntil, expected2)
	}

	// 3rd hit (soon after) -> 4m delay
	attempt3 := attempt
	attempt3.Now = attempt2.Now.Add(time.Second)
	d3, _ := geminiHighDemandDecision(attempt3, context)
	expected3 := attempt3.Now.Add(4 * time.Minute)
	if !d3.CooldownUntil.Equal(expected3) {
		t.Errorf("CooldownUntil = %v, want %v (doubled)", d3.CooldownUntil, expected3)
	}

	// Reset decay test -> 4th hit after a long delay (e.g. 1 hour) -> should reset back to 1m
	attempt4 := attempt
	attempt4.Now = attempt3.Now.Add(time.Hour)
	d4, _ := geminiHighDemandDecision(attempt4, context)
	expected4 := attempt4.Now.Add(1 * time.Minute)
	if !d4.CooldownUntil.Equal(expected4) {
		t.Errorf("CooldownUntil = %v, want %v (reset to 1m)", d4.CooldownUntil, expected4)
	}

	// 5th hit soon after -> should double delay to 2m
	attempt5 := attempt
	attempt5.Now = attempt4.Now.Add(time.Second)
	d5, _ := geminiHighDemandDecision(attempt5, context)
	expected5 := attempt5.Now.Add(2 * time.Minute)
	if !d5.CooldownUntil.Equal(expected5) {
		t.Errorf("CooldownUntil = %v, want %v (doubled)", d5.CooldownUntil, expected5)
	}

	// Successful request triggers ResetGeminiBackoff
	ResetGeminiBackoff(context.GroupID, context.Model)

	// 6th hit soon after -> should reset back to 1m!
	attempt6 := attempt
	attempt6.Now = attempt5.Now.Add(time.Second)
	d6, _ := geminiHighDemandDecision(attempt6, context)
	expected6 := attempt6.Now.Add(1 * time.Minute)
	if !d6.CooldownUntil.Equal(expected6) {
		t.Errorf("CooldownUntil = %v, want %v (reset by ResetGeminiBackoff)", d6.CooldownUntil, expected6)
	}
}

func TestJudgeExecutionGeminiHighDemand(t *testing.T) {
	now := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)
	attempt := ExecutionAttempt{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusServiceUnavailable,
		Now:           now,
		Evidence: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			StatusCode: http.StatusServiceUnavailable,
			Summary:    "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.",
		},
	}

	context := DecisionContext{
		Operation:                execution.OperationChatCompletion,
		DefaultRateLimitCooldown: time.Minute,
		CredentialID:             99,
		GroupID:                  1,
		Model:                    "gemini-2.5-pro",
	}

	decision := JudgeExecution(attempt, context)

	if decision.Category != FailureCategoryRateLimited {
		t.Errorf("Category = %v, want %v", decision.Category, FailureCategoryRateLimited)
	}
	if decision.Effect != EffectCooldownModel {
		t.Errorf("Effect = %v, want %v", decision.Effect, EffectCooldownModel)
	}
	if decision.Scope != execution.ErrorScopeModel {
		t.Errorf("Scope = %v, want %v", decision.Scope, execution.ErrorScopeModel)
	}
	if decision.Retry != RetryNextCandidate {
		t.Errorf("Retry = %v, want %v", decision.Retry, RetryNextCandidate)
	}
	if decision.RuleID != "gemini.high_demand_model_cooldown" {
		t.Errorf("RuleID = %v, want gemini.high_demand_model_cooldown", decision.RuleID)
	}
	expected := now.Add(1 * time.Minute)
	if !decision.CooldownUntil.Equal(expected) {
		t.Errorf("CooldownUntil = %v, want %v", decision.CooldownUntil, expected)
	}
}
