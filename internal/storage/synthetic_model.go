package storage

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

func cleanLegacyMigrationLedger(db *gorm.DB) {
	if db != nil && db.Migrator().HasTable(migrationLedgerTable) {
		_ = db.Exec("DELETE FROM " + migrationLedgerTable + " WHERE id LIKE '%synthetic%' OR id LIKE '%recalculate%'").Error
	}
}

func autoMigrateSyntheticModels(db *gorm.DB) error {
	if db == nil {
		return nil
	}

	if !db.Migrator().HasTable(&models.SyntheticModel{}) {
		if err := db.AutoMigrate(&models.SyntheticModel{}); err != nil {
			return fmt.Errorf("auto migrate synthetic models: %w", err)
		}
	}

	if db.Migrator().HasTable("system_settings") {
		_ = db.Where(&models.SystemSetting{Key: "fork_upstream_usage_recalc_v1"}).Delete(&models.SystemSetting{}).Error
	}

	if db.Migrator().HasTable("request_log_attempts") {
		_ = db.Exec("CREATE INDEX IF NOT EXISTS idx_request_log_attempts_group_request ON request_log_attempts(group_id, request_id)").Error
	}

	return nil
}
