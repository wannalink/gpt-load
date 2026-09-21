package storage

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

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
		_ = db.Exec("DELETE FROM system_settings WHERE key = 'fork_upstream_usage_recalc_v1'").Error
	}

	return nil
}
