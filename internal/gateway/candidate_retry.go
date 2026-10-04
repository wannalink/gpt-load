package gateway

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/platform/utils"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

func candidateRetryTimeoutBudget(snapshot *state.ConfigSnapshot) time.Duration {
	if envVal := strings.TrimSpace(os.Getenv("GPT_LOAD_CANDIDATE_RETRY_TIMEOUT")); envVal != "" {
		if d, err := time.ParseDuration(envVal); err == nil && d > 0 {
			return d
		}
		if sec, err := strconv.Atoi(envVal); err == nil && sec > 0 {
			return time.Duration(sec) * time.Second
		}
	}

	timeout := 120 * time.Second
	if snapshot != nil {
		if snapshot.Settings.FirstByteTimeout > 0 {
			timeout = snapshot.Settings.FirstByteTimeout
		} else if snapshot.Settings.RequestTimeout > 0 {
			timeout = snapshot.Settings.RequestTimeout
		}
	}
	return timeout
}

func (handler *Handler) waitCandidateCooldown(
	ctx context.Context,
	snapshot *state.ConfigSnapshot,
	iterator *scheduler.Iterator,
	requestStartedAt time.Time,
	forwardAttempts int,
) bool {
	if ctx.Err() != nil {
		return false
	}
	now := handler.now()
	earliest, hasCandidates := iterator.EarliestCandidateAvailable(now)
	if !hasCandidates {
		return false
	}

	timeoutBudget := candidateRetryTimeoutBudget(snapshot)
	effectiveDeadline := requestStartedAt.Add(timeoutBudget - 2*time.Second)

	if ctxDeadline, ok := ctx.Deadline(); ok && ctxDeadline.Before(effectiveDeadline) {
		effectiveDeadline = ctxDeadline.Add(-500 * time.Millisecond)
	}

	remaining := effectiveDeadline.Sub(now)
	if remaining <= 0 {
		return false
	}

	var waitDuration time.Duration
	if earliest.After(now) {
		waitDuration = earliest.Sub(now)
	} else {
		waitDuration = 50 * time.Millisecond
	}

	if waitDuration > remaining {
		waitDuration = remaining
	}
	if waitDuration > 2*time.Second {
		waitDuration = 2 * time.Second
	}
	if waitDuration < 20*time.Millisecond {
		waitDuration = 20 * time.Millisecond
	}

	utils.LogPlaneBestEffort(
		handler.logger,
		logrus.InfoLevel,
		utils.LogPlaneData,
		logrus.Fields{
			"event":          "candidate_retry_wait",
			"wait_duration":  waitDuration.String(),
			"cooldown_until": earliest.Format(time.RFC3339),
			"attempts":       forwardAttempts,
		},
		"No candidates available immediately; waiting for candidate cooldown",
	)

	timer := time.NewTimer(waitDuration)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
	}

	iterator.ResetTried()
	return true
}
