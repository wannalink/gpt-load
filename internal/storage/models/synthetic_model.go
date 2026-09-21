package models

// SyntheticModel stores a virtual model definition mapping to an ordered list of target models.
type SyntheticModel struct {
	ID           uint   `gorm:"primaryKey;autoIncrement"`
	Name         string `gorm:"type:varchar(255);not null;uniqueIndex"`
	Description  string `gorm:"type:varchar(255);not null;default:''"`
	TargetModels JSON   `gorm:"type:json;not null"`
	Enabled      bool   `gorm:"not null;default:true"`
	CreatedAtMS  int64  `gorm:"column:created_at_ms;not null;autoCreateTime:milli;check:chk_synthetic_model_created_at,created_at_ms >= 0"`
	UpdatedAtMS  int64  `gorm:"column:updated_at_ms;not null;autoUpdateTime:milli;check:chk_synthetic_model_updated_at,updated_at_ms >= 0"`
}

func (SyntheticModel) TableName() string {
	return "synthetic_models"
}
