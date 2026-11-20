package gateway

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/state"
)

func syntheticModelName(externalModel string, originalMetadata dialect.RequestMetadata) string {
	if originalMetadata.Model != nil && *originalMetadata.Model != "" {
		return *originalMetadata.Model
	}
	return externalModel
}

func (handler *Handler) applyForkGroupDecisionEffect(
	group state.GroupView,
	decision health.Decision,
	statusCode int,
	attemptNow time.Time,
	model string,
) bool {
	if decision.Effect == health.EffectCooldownModel && decision.RuleID == "gemini.high_demand_model_cooldown" {
		handler.registry.SetGroupModelCooldown(group.ID, model, decision.CooldownUntil, attemptNow)
		utils.LogPlaneBestEffort(handler.logger, logrus.WarnLevel, utils.LogPlaneData,
			logrus.Fields{"event": "model_cooldown", "group_id": group.ID, "model": model,
				"cooldown_until": decision.CooldownUntil, "status_code": statusCode}, "Upstream model entered cooldown (high demand)")
		return true
	}
	return false
}

func applyForkTimeoutDecision(decision *health.Decision, result UpstreamResult, now time.Time, isSynthetic bool) {
	if isSynthetic && decision != nil && result.DispatchState != execution.DispatchNotSent && isTimeoutResult(result) {
		decision.Effect = health.EffectCooldownModel
		decision.RuleID = "gemini.high_demand_model_cooldown"
		decision.CooldownUntil = now.Add(20 * time.Second)
		decision.Retry = health.RetryNextCandidate
	}
}

func isSyntheticModelRequest(snapshot *state.ConfigSnapshot, externalModel string) bool {
	if snapshot == nil || snapshot.SyntheticModels == nil || externalModel == "" {
		return false
	}
	_, ok := snapshot.SyntheticModels[externalModel]
	return ok
}

func isExplicitTerminalRejection(result UpstreamResult) bool {
	if result.ExecutionError != nil {
		if execution.ExplicitRequestRejection(
			result.ExecutionError.Type,
			result.ExecutionError.Code,
			result.ExecutionError.Summary,
		) {
			return true
		}
		markers := strings.ToLower(strings.Join([]string{
			result.ExecutionError.Type,
			result.ExecutionError.Code,
			result.ExecutionError.Summary,
		}, " "))
		if health.IsGeminiInputTokenLimitMarkers(markers) {
			return true
		}
	}
	if len(result.Body) > 0 && health.IsGeminiInputTokenLimitMarkers(strings.ToLower(string(result.Body))) {
		return true
	}
	return false
}

func forwardAttemptLimitForModel(snapshot *state.ConfigSnapshot, model string) int {
	if snapshot == nil {
		return 1
	}
	limit := retryAttemptLimit(snapshot.Settings.RetryCount)
	if targets, ok := snapshot.SyntheticModels[model]; ok && len(targets) > 0 {
		limit = len(targets) * limit
		if limit < 50 {
			limit = 50
		}
		return limit
	}
	if snapshot.AutoModels != nil {
		if autoEntry, ok := snapshot.AutoModels.Lookup(model); ok && len(autoEntry.Presets) > 0 {
			limit = len(autoEntry.Presets) * limit
			if limit < 50 {
				limit = 50
			}
			return limit
		}
	}
	return limit
}

func isTimeoutResult(result UpstreamResult) bool {
	if result.StatusCode == http.StatusGatewayTimeout || result.StatusCode == http.StatusRequestTimeout {
		return true
	}
	if result.ExecutionError != nil {
		if result.ExecutionError.Kind == execution.ErrorKindTimeout {
			return true
		}
		switch result.ExecutionError.Code {
		case "upstream_timeout", "upstream_stream_idle_timeout":
			return true
		}
	}
	return false
}

func isRecoverableUncommittedError(result UpstreamResult, isSynthetic, bufferedStream bool) bool {
	if result.Committed {
		return false
	}
	if isSynthetic && isTimeoutResult(result) {
		return true
	}
	if result.ExecutionError == nil {
		switch result.Stream.EndReason {
		case StreamEndProviderIncomplete, StreamEndUpstreamTerminated, StreamEndUpstreamProtocolError, StreamEndIdleTimeout, StreamEndSSEError:
			return true
		}
		return errors.Is(result.Err, ErrUpstreamProtocol)
	}
	switch result.ExecutionError.Kind {
	case execution.ErrorKindTransport:
		return !result.RequestWritten || result.ExecutionError.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing || isSynthetic || (bufferedStream && result.ResponseStarted)
	case execution.ErrorKindTimeout:
		return !result.RequestWritten || result.ExecutionError.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing || isSynthetic
	case execution.ErrorKindProvider:
		if result.ExecutionError.Code == "" && result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices {
			return true
		}
		switch result.ExecutionError.Code {
		case "upstream_protocol_error", "upstream_response_incomplete", "upstream_stream_terminated", "upstream_stream_idle_timeout":
			return true
		}
	case execution.ErrorKindInternal:
		if result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices {
			return true
		}
		if bufferedStream && (result.ExecutionError.Summary == "execution runtime returned an empty stream" ||
			result.ExecutionError.Summary == "invalid upstream SSE stream") {
			return true
		}
	}
	return false
}

func is2xxCompletedResponse(result UpstreamResult) bool {
	return result.StatusCode >= 200 && result.StatusCode < 300 &&
		!result.ProviderErrorBeforeCommit &&
		result.ExecutionError == nil &&
		result.Err == nil &&
		(result.Stream.EndReason == StreamEndNone || result.Stream.EndReason == StreamEndCleanEOF)
}

func shouldRetrySynthetic(decision health.Decision, isSynthetic bool, bufferedStream bool, result UpstreamResult) bool {
	if is2xxCompletedResponse(result) {
		return false
	}
	if isSynthetic && isTimeoutResult(result) {
		return true
	}
	if decision.Category == health.FailureCategoryClientError || isExplicitTerminalRejection(result) {
		return false
	}
	if (isSynthetic || bufferedStream) && isRecoverableUncommittedError(result, isSynthetic, bufferedStream) {
		return true
	}
	return decision.Retry != health.RetryNone || (isSynthetic && result.StatusCode >= 400)
}
