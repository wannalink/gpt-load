package gateway

import (
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"
)

func (handler *Handler) writeForkReasonResponse(context *gin.Context, value reason) error {
	headers := http.Header{"Content-Type": {"application/json; charset=utf-8"}}
	if value.Status == http.StatusServiceUnavailable || value.Status == http.StatusTooManyRequests {
		headers.Set("Retry-After", "30")
	}

	body, err := json.Marshal(struct {
		Error struct {
			Message string  `json:"message"`
			Type    string  `json:"type"`
			Param   *string `json:"param"`
			Code    string  `json:"code"`
		} `json:"error"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}{
		Error: struct {
			Message string  `json:"message"`
			Type    string  `json:"type"`
			Param   *string `json:"param"`
			Code    string  `json:"code"`
		}{
			Message: value.Message,
			Type:    "server_error",
			Param:   nil,
			Code:    value.Code,
		},
		Code:    value.Code,
		Message: value.Message,
	})
	if err != nil {
		return err
	}
	return handler.writeBufferedResponse(
		context,
		value.Status,
		headers,
		body,
	)
}
