package gateway

import (
	"fmt"
	"io"

	"gpt-load/internal/execution"
)

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
