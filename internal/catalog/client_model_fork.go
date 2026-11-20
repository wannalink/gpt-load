package catalog

import "slices"

func preferForkReasoningLevel(levels []string) (string, bool) {
	if slices.Contains(levels, "high") {
		return "high", true
	}
	return "", false
}

func normalizeForkCodexDefaultReasoningLevel(level string) string {
	if level == "low" || level == "" {
		return "high"
	}
	return level
}
