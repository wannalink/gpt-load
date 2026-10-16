package gateway

import (
	"fmt"
	"io"
	"net/http"

	"gpt-load/internal/execution"
)

type streamBuffer struct {
	enabled bool
	chunks  [][]byte
}

func newStreamBuffer(enabled bool) *streamBuffer {
	return &streamBuffer{enabled: enabled}
}

func (b *streamBuffer) push(data []byte, terminalInChunk bool, onTerminal func() bool) bool {
	if b == nil || !b.enabled {
		return false
	}
	b.chunks = append(b.chunks, append([]byte(nil), data...))
	if terminalInChunk && onTerminal != nil {
		_ = onTerminal()
	}
	return true
}

func (b *streamBuffer) shouldReplay(downstreamErr error, terminal execution.StreamResult, ready *execution.StreamEvent, firstWasError bool) bool {
	if b == nil || !b.enabled {
		return false
	}
	return downstreamErr == nil && terminal.Error == nil && ready != nil &&
		ready.StatusCode >= http.StatusOK && ready.StatusCode < http.StatusMultipleChoices &&
		!firstWasError
}

func (b *streamBuffer) replay(controller *streamWriteController, ready *execution.StreamEvent, onStreamReady func()) error {
	defer func() {
		if b != nil {
			b.chunks = nil
		}
	}()
	return replayBufferedStream(controller, ready, b.chunks, onStreamReady)
}

func replayBufferedStream(
	controller *streamWriteController,
	ready *execution.StreamEvent,
	bufferedData [][]byte,
	onStreamReady func(),
) error {
	var initialChunk []byte
	if len(bufferedData) > 0 {
		initialChunk = bufferedData[0]
	}
	if err := commitStream(controller, ready.StatusCode, ready.Header, initialChunk); err != nil {
		return err
	}
	if onStreamReady != nil {
		onStreamReady()
	}
	if len(bufferedData) > 1 {
		for _, chunk := range bufferedData[1:] {
			written, err := controller.write(chunk)
			if err != nil {
				return &streamFailure{
					kind: streamFailureDownstreamWrite,
					err:  fmt.Errorf("write execution stream: %w", err),
				}
			}
			if written != len(chunk) {
				return &streamFailure{
					kind: streamFailureDownstreamWrite,
					err:  fmt.Errorf("write execution stream: %w", io.ErrShortWrite),
				}
			}
		}
	}
	if err := controller.flush(); err != nil {
		return &streamFailure{
			kind: streamFailureDownstreamWrite,
			err:  fmt.Errorf("flush execution stream: %w", err),
		}
	}
	return nil
}
