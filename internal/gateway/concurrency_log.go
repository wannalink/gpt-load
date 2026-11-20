package gateway

import (
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/state"
)

func (h *Handler) logRequestConcurrencyLimit(keyID uint, keyName string, limit int64, globalLimit int64) {
	snap := h.manager.Concurrency().Snapshot()
	logrus.WithFields(logrus.Fields{
		"event":        "concurrency_limit_exceeded",
		"key_id":       keyID,
		"key_name":     keyName,
		"key_limit":    limit,
		"global":       snap.Global,
		"global_limit": globalLimit,
	}).Warn("[CONCURRENCY] Request concurrency limit exceeded")
}

func (h *Handler) wrapRequestConcurrencyRelease(keyID uint, keyName string, innerRelease func()) func() {
	start := time.Now()
	snap := h.manager.Concurrency().Snapshot()
	logrus.WithFields(logrus.Fields{
		"event":     "concurrency_request_acquired",
		"key_id":    keyID,
		"key_name":  keyName,
		"global":    snap.Global,
		"key_count": snap.AccessKeys[keyID],
	}).Info("[CONCURRENCY] Request concurrency acquired")

	return sync.OnceFunc(func() {
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
}

func (h *Handler) logGroupConcurrencyLimit(groupID uint, group state.GroupView) {
	snap := h.manager.Concurrency().Snapshot()
	logrus.WithFields(logrus.Fields{
		"event":       "group_concurrency_limit_exceeded",
		"group_id":    groupID,
		"group_name":  group.Name,
		"group_limit": group.ConcurrencyLimit,
		"group_count": snap.Groups[groupID],
	}).Warn("[CONCURRENCY] Group concurrency limit exceeded")
}

func (h *Handler) wrapGroupConcurrencyRelease(groupID uint, group state.GroupView, innerRelease func()) func() {
	start := time.Now()
	snap := h.manager.Concurrency().Snapshot()
	logrus.WithFields(logrus.Fields{
		"event":       "concurrency_group_acquired",
		"group_id":    groupID,
		"group_name":  group.Name,
		"group_count": snap.Groups[groupID],
	}).Info("[CONCURRENCY] Group concurrency acquired")

	return sync.OnceFunc(func() {
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
}
