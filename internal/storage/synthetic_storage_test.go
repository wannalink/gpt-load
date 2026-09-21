package storage_test

import (
	"testing"

	"gpt-load/internal/storage"
	"gpt-load/internal/storage/models"
)

func TestSyntheticModelAutoMigrationAndLedgerReconciliation(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatalf("Open(:memory:) error = %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	t.Cleanup(func() {
		_ = sqlDB.Close()
	})

	// 1. First AutoMigrate should succeed and create synthetic_models table
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("storage.AutoMigrate failed: %v", err)
	}

	if !db.Migrator().HasTable(&models.SyntheticModel{}) {
		t.Fatal("synthetic_models table was not created")
	}

	// 2. Simulate a legacy database that has old fork migration IDs in schema_migrations
	legacyID := "0018_synthetic_models"
	if err := db.Exec("INSERT INTO schema_migrations (id) VALUES (?)", legacyID).Error; err != nil {
		t.Fatalf("insert legacy migration ID: %v", err)
	}

	// 3. Subsequent AutoMigrate should self-heal, clean up legacy ID, and succeed
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("AutoMigrate with legacy ledger failed: %v", err)
	}

	var count int64
	if err := db.Table("schema_migrations").Where("id LIKE '%synthetic%'").Count(&count).Error; err != nil {
		t.Fatalf("count legacy migrations: %v", err)
	}
	if count != 0 {
		t.Fatalf("legacy migration was not cleaned up, count = %d", count)
	}
}
