package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/storage/models"
)

type SyntheticModelDTO struct {
	ID           uint     `json:"id"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	TargetModels []string `json:"target_models"`
	Enabled      bool     `json:"enabled"`
	CreatedAtMS  int64    `json:"created_at_ms"`
	UpdatedAtMS  int64    `json:"updated_at_ms"`
}

type SyntheticModelCreateRequest struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	TargetModels []string `json:"target_models"`
	Enabled      *bool    `json:"enabled"`
}

type SyntheticModelUpdateRequest struct {
	Name         *string  `json:"name"`
	Description  *string  `json:"description"`
	TargetModels []string `json:"target_models"`
	Enabled      *bool    `json:"enabled"`
}

func (s *Service) ListSyntheticModels(ctx context.Context) ([]SyntheticModelDTO, error) {
	var rows []models.SyntheticModel
	if err := s.db.WithContext(ctx).Order("id ASC").Find(&rows).Error; err != nil {
		return nil, err
	}
	result := make([]SyntheticModelDTO, 0, len(rows))
	for _, row := range rows {
		dto, err := mapSyntheticModelDTO(row)
		if err != nil {
			return nil, err
		}
		result = append(result, dto)
	}
	return result, nil
}

func (s *Service) GetSyntheticModel(ctx context.Context, id uint) (SyntheticModelDTO, error) {
	if id == 0 {
		return SyntheticModelDTO{}, app_errors.ErrValidation
	}
	var row models.SyntheticModel
	if err := s.db.WithContext(ctx).Where("id = ?", id).First(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return SyntheticModelDTO{}, app_errors.ErrResourceNotFound
		}
		return SyntheticModelDTO{}, err
	}
	return mapSyntheticModelDTO(row)
}

func (s *Service) CreateSyntheticModel(ctx context.Context, req SyntheticModelCreateRequest) (SyntheticModelDTO, error) {
	name := strings.TrimSpace(req.Name)
	if name == "" {
		return SyntheticModelDTO{}, fmt.Errorf("name is required: %w", app_errors.ErrValidation)
	}
	if len(req.TargetModels) == 0 {
		return SyntheticModelDTO{}, fmt.Errorf("at least one target model is required: %w", app_errors.ErrValidation)
	}
	cleanedTargets := make([]string, 0, len(req.TargetModels))
	for _, t := range req.TargetModels {
		target := strings.TrimSpace(t)
		if target == "" {
			return SyntheticModelDTO{}, fmt.Errorf("target model cannot be empty: %w", app_errors.ErrValidation)
		}
		if target == name {
			return SyntheticModelDTO{}, fmt.Errorf("synthetic model cannot target itself: %w", app_errors.ErrValidation)
		}
		cleanedTargets = append(cleanedTargets, target)
	}

	targetsJSON, err := json.Marshal(cleanedTargets)
	if err != nil {
		return SyntheticModelDTO{}, fmt.Errorf("marshal targets: %w", err)
	}

	enabled := true
	if req.Enabled != nil {
		enabled = *req.Enabled
	}

	row := models.SyntheticModel{
		Name:         name,
		Description:  strings.TrimSpace(req.Description),
		TargetModels: models.JSON(targetsJSON),
		Enabled:      enabled,
	}

	if _, err := s.writeConfig(ctx, func(tx *gorm.DB) error {
		return tx.Create(&row).Error
	}, nil); err != nil {
		return SyntheticModelDTO{}, err
	}

	return mapSyntheticModelDTO(row)
}

func (s *Service) UpdateSyntheticModel(ctx context.Context, id uint, req SyntheticModelUpdateRequest) (SyntheticModelDTO, error) {
	if id == 0 {
		return SyntheticModelDTO{}, app_errors.ErrValidation
	}

	var row models.SyntheticModel
	if err := s.db.WithContext(ctx).Where("id = ?", id).First(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return SyntheticModelDTO{}, app_errors.ErrResourceNotFound
		}
		return SyntheticModelDTO{}, err
	}

	if req.Name != nil {
		name := strings.TrimSpace(*req.Name)
		if name == "" {
			return SyntheticModelDTO{}, fmt.Errorf("name cannot be empty: %w", app_errors.ErrValidation)
		}
		row.Name = name
	}

	if req.Description != nil {
		row.Description = strings.TrimSpace(*req.Description)
	}

	if req.TargetModels != nil {
		if len(req.TargetModels) == 0 {
			return SyntheticModelDTO{}, fmt.Errorf("at least one target model is required: %w", app_errors.ErrValidation)
		}
		cleanedTargets := make([]string, 0, len(req.TargetModels))
		for _, t := range req.TargetModels {
			target := strings.TrimSpace(t)
			if target == "" {
				return SyntheticModelDTO{}, fmt.Errorf("target model cannot be empty: %w", app_errors.ErrValidation)
			}
			if target == row.Name {
				return SyntheticModelDTO{}, fmt.Errorf("synthetic model cannot target itself: %w", app_errors.ErrValidation)
			}
			cleanedTargets = append(cleanedTargets, target)
		}
		targetsJSON, err := json.Marshal(cleanedTargets)
		if err != nil {
			return SyntheticModelDTO{}, fmt.Errorf("marshal targets: %w", err)
		}
		row.TargetModels = models.JSON(targetsJSON)
	}

	if req.Enabled != nil {
		row.Enabled = *req.Enabled
	}

	if _, err := s.writeConfig(ctx, func(tx *gorm.DB) error {
		return tx.Save(&row).Error
	}, nil); err != nil {
		return SyntheticModelDTO{}, err
	}

	return mapSyntheticModelDTO(row)
}

func (s *Service) DeleteSyntheticModel(ctx context.Context, id uint) error {
	if id == 0 {
		return app_errors.ErrValidation
	}
	_, err := s.writeConfig(ctx, func(tx *gorm.DB) error {
		res := tx.Where("id = ?", id).Delete(&models.SyntheticModel{})
		if res.Error != nil {
			return res.Error
		}
		if res.RowsAffected == 0 {
			return app_errors.ErrResourceNotFound
		}
		return nil
	}, nil)
	return err
}

func (s *Server) handleListSyntheticModels(c *gin.Context) {
	result, err := s.service.ListSyntheticModels(c.Request.Context())
	if err != nil {
		writeServiceError(c, "list_synthetic_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetSyntheticModel(c *gin.Context) {
	id, err := parseSyntheticModelID(c)
	if err != nil {
		writeServiceError(c, "get_synthetic_model", app_errors.ErrValidation)
		return
	}
	result, err := s.service.GetSyntheticModel(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_synthetic_model", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleCreateSyntheticModel(c *gin.Context) {
	var req SyntheticModelCreateRequest
	if err := bindStrictJSON(c, &req); err != nil {
		writeServiceError(c, "create_synthetic_model", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateSyntheticModel(c.Request.Context(), req)
	if err != nil {
		writeServiceError(c, "create_synthetic_model", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateSyntheticModel(c *gin.Context) {
	id, err := parseSyntheticModelID(c)
	if err != nil {
		writeServiceError(c, "update_synthetic_model", app_errors.ErrValidation)
		return
	}
	var req SyntheticModelUpdateRequest
	if err := bindStrictJSON(c, &req); err != nil {
		writeServiceError(c, "update_synthetic_model", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateSyntheticModel(c.Request.Context(), id, req)
	if err != nil {
		writeServiceError(c, "update_synthetic_model", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDeleteSyntheticModel(c *gin.Context) {
	id, err := parseSyntheticModelID(c)
	if err != nil {
		writeServiceError(c, "delete_synthetic_model", app_errors.ErrValidation)
		return
	}
	if err := s.service.DeleteSyntheticModel(c.Request.Context(), id); err != nil {
		writeServiceError(c, "delete_synthetic_model", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func parseSyntheticModelID(c *gin.Context) (uint, error) {
	raw := strings.TrimSpace(c.Param("id"))
	val, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || val == 0 {
		return 0, fmt.Errorf("invalid synthetic model id")
	}
	return uint(val), nil
}

func mapSyntheticModelDTO(row models.SyntheticModel) (SyntheticModelDTO, error) {
	var targets []string
	if len(row.TargetModels) > 0 {
		if err := json.Unmarshal(row.TargetModels, &targets); err != nil {
			return SyntheticModelDTO{}, fmt.Errorf("unmarshal targets: %w", err)
		}
	}
	return SyntheticModelDTO{
		ID:           row.ID,
		Name:         row.Name,
		Description:  row.Description,
		TargetModels: targets,
		Enabled:      row.Enabled,
		CreatedAtMS:  row.CreatedAtMS,
		UpdatedAtMS:  row.UpdatedAtMS,
	}, nil
}
