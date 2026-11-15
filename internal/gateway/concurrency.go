package gateway

import (
	"net/http"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

var reasonConcurrencyLimit = reason{Status: http.StatusTooManyRequests, Code: "concurrency_limit_exceeded", Message: "Concurrency limit exceeded."}

// 每次准入都在发布锁内读取完整的新配置；活动计数不跟随快照重建。
func (h *Handler) acquireRequestConcurrency(keyID uint) (release func(), failure *reason) {
	failure = &reasonConfigurationChanged
	h.manager.WithCurrentSnapshotRead(func(snapshot *state.ConfigSnapshot) bool {
		if snapshot == nil {
			return false
		}
		key, exists := snapshot.AccessKeysByID[keyID]
		if !exists {
			return false
		}
		limit := snapshot.Settings.DefaultAccessKeyConcurrencyLimit
		if key.ConcurrencyLimit != nil {
			limit = *key.ConcurrencyLimit
		}
		innerRelease, allowed := h.manager.Concurrency().TryAcquireRequest(keyID, snapshot.Settings.GlobalConcurrencyLimit, limit)
		if !allowed {
			failure = &reasonConcurrencyLimit
			snap := h.manager.Concurrency().Snapshot()
			logrus.WithFields(logrus.Fields{
				"event":        "concurrency_limit_exceeded",
				"key_id":       keyID,
				"key_name":     key.Name,
				"key_limit":    limit,
				"global":       snap.Global,
				"global_limit": snapshot.Settings.GlobalConcurrencyLimit,
			}).Warn("[CONCURRENCY] Request concurrency limit exceeded")
			return true
		}
		failure = nil
		start := time.Now()
		snap := h.manager.Concurrency().Snapshot()
		logrus.WithFields(logrus.Fields{
			"event":     "concurrency_request_acquired",
			"key_id":    keyID,
			"key_name":  key.Name,
			"global":    snap.Global,
			"key_count": snap.AccessKeys[keyID],
		}).Info("[CONCURRENCY] Request concurrency acquired")

		release = sync.OnceFunc(func() {
			innerRelease()
			duration := time.Since(start)
			afterSnap := h.manager.Concurrency().Snapshot()
			logrus.WithFields(logrus.Fields{
				"event":       "concurrency_request_released",
				"key_id":      keyID,
				"duration_ms": duration.Milliseconds(),
				"global":      afterSnap.Global,
				"key_count":   afterSnap.AccessKeys[keyID],
			}).Info("[CONCURRENCY] Request concurrency released")
		})
		return true
	})
	return
}

func (h *Handler) acquireGroupConcurrency(groupID uint) (release func(), failure *reason) {
	failure = &reasonConfigurationChanged
	h.manager.WithCurrentSnapshotRead(func(snapshot *state.ConfigSnapshot) bool {
		if snapshot == nil {
			return false
		}
		group, exists := snapshot.Groups[groupID]
		if !exists {
			return false
		}
		innerRelease, allowed := h.manager.Concurrency().TryAcquireGroup(groupID, group.ConcurrencyLimit)
		if !allowed {
			failure = &reasonConcurrencyLimit
			snap := h.manager.Concurrency().Snapshot()
			logrus.WithFields(logrus.Fields{
				"event":       "group_concurrency_limit_exceeded",
				"group_id":    groupID,
				"group_name":  group.Name,
				"group_limit": group.ConcurrencyLimit,
				"group_count": snap.Groups[groupID],
			}).Warn("[CONCURRENCY] Group concurrency limit exceeded")
			return true
		}
		failure = nil
		start := time.Now()
		snap := h.manager.Concurrency().Snapshot()
		logrus.WithFields(logrus.Fields{
			"event":       "concurrency_group_acquired",
			"group_id":    groupID,
			"group_name":  group.Name,
			"group_count": snap.Groups[groupID],
		}).Info("[CONCURRENCY] Group concurrency acquired")

		release = sync.OnceFunc(func() {
			innerRelease()
			duration := time.Since(start)
			afterSnap := h.manager.Concurrency().Snapshot()
			logrus.WithFields(logrus.Fields{
				"event":       "concurrency_group_released",
				"group_id":    groupID,
				"group_name":  group.Name,
				"duration_ms": duration.Milliseconds(),
				"group_count": afterSnap.Groups[groupID],
			}).Info("[CONCURRENCY] Group concurrency released")
		})
		return true
	})
	return
}

func concurrencyControlRequest(request *http.Request, route route) bool {
	if route.Protocol != protocol.OpenAIResponses {
		return false
	}
	metadata, err := dialect.NewOpenAIResponses().InspectRequest(&dialect.ParsedRequest{Method: request.Method, Path: request.URL.Path, RawQuery: request.URL.RawQuery})
	return err == nil && metadata.Operation == execution.OperationResponsesCancel
}
