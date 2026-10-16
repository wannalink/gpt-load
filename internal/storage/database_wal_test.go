package storage

import (
	"path/filepath"
	"testing"

	"gpt-load/internal/platform/config"
)

func TestOpenConfiguredAppliesFileBackedSQLitePool(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "wal_pool_test.db")

	cfg := &config.Config{
		DatabaseDSN: dbPath,
		DatabaseMetadata: config.DatabaseMetadata{
			Source: config.DatabaseSourceExternal,
			Driver: config.DatabaseDriverSQLite,
		},
		DatabasePool: config.DatabasePoolConfig{
			MaxOpenConnections: 15,
			MaxIdleConnections: 5,
		},
	}

	db, err := OpenConfigured(cfg)
	if err != nil {
		t.Fatalf("OpenConfigured() error = %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	t.Cleanup(func() {
		_ = sqlDB.Close()
	})

	stats := sqlDB.Stats()
	if stats.MaxOpenConnections != 15 {
		t.Fatalf("MaxOpenConnections = %d, want 15", stats.MaxOpenConnections)
	}
}

func TestApplyFileBackedSQLitePoolDefaults(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "wal_default_pool_test.db")

	cfg := &config.Config{
		DatabaseDSN: dbPath,
		DatabaseMetadata: config.DatabaseMetadata{
			Source: config.DatabaseSourceExternal,
			Driver: config.DatabaseDriverSQLite,
		},
		DatabasePool: config.DatabasePoolConfig{
			MaxOpenConnections: 0,
			MaxIdleConnections: 0,
		},
	}

	db, err := OpenConfigured(cfg)
	if err != nil {
		t.Fatalf("OpenConfigured() error = %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	t.Cleanup(func() {
		_ = sqlDB.Close()
	})

	stats := sqlDB.Stats()
	if stats.MaxOpenConnections != 10 {
		t.Fatalf("default MaxOpenConnections = %d, want 10", stats.MaxOpenConnections)
	}
}
