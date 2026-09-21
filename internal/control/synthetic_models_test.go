package control

import (
	"testing"
)

func TestSyntheticModelsCRUD(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)

	// 1. Create a synthetic model
	created, err := fixture.service.CreateSyntheticModel(t.Context(), SyntheticModelCreateRequest{
		Name:         "auto-smart",
		Description:  "Smart fallback model",
		TargetModels: []string{"claude-3-7-sonnet", "gemini-2.5-pro", "gemini-2.5-flash"},
	})
	if err != nil {
		t.Fatalf("CreateSyntheticModel failed: %v", err)
	}
	if created.ID == 0 || created.Name != "auto-smart" || len(created.TargetModels) != 3 {
		t.Fatalf("created = %+v, want valid DTO with 3 targets", created)
	}

	// Verify snapshot updated
	currentSnapshot := fixture.manager.Current()
	if currentSnapshot == nil || len(currentSnapshot.SyntheticModels["auto-smart"]) != 3 {
		t.Fatalf("currentSnapshot.SyntheticModels[auto-smart] = %v, want 3 targets", currentSnapshot.SyntheticModels["auto-smart"])
	}

	// 2. List synthetic models
	list, err := fixture.service.ListSyntheticModels(t.Context())
	if err != nil {
		t.Fatalf("ListSyntheticModels failed: %v", err)
	}
	if len(list) != 1 || list[0].ID != created.ID {
		t.Fatalf("list = %+v, want 1 model with ID %d", list, created.ID)
	}

	// 3. Get synthetic model
	detail, err := fixture.service.GetSyntheticModel(t.Context(), created.ID)
	if err != nil {
		t.Fatalf("GetSyntheticModel failed: %v", err)
	}
	if detail.ID != created.ID || detail.Name != "auto-smart" {
		t.Fatalf("detail = %+v, want ID %d", detail, created.ID)
	}

	// 4. Update synthetic model
	newDesc := "Updated smart description"
	newTargets := []string{"gemini-2.5-pro", "gemini-2.5-flash"}
	updated, err := fixture.service.UpdateSyntheticModel(t.Context(), created.ID, SyntheticModelUpdateRequest{
		Description:  &newDesc,
		TargetModels: newTargets,
	})
	if err != nil {
		t.Fatalf("UpdateSyntheticModel failed: %v", err)
	}
	if updated.Description != newDesc || len(updated.TargetModels) != 2 {
		t.Fatalf("updated = %+v, want 2 targets and updated description", updated)
	}

	// Verify snapshot updated after edit
	currentSnapshot = fixture.manager.Current()
	if len(currentSnapshot.SyntheticModels["auto-smart"]) != 2 {
		t.Fatalf("currentSnapshot.SyntheticModels[auto-smart] after update = %v, want 2 targets", currentSnapshot.SyntheticModels["auto-smart"])
	}

	// 5. Delete synthetic model
	if err := fixture.service.DeleteSyntheticModel(t.Context(), created.ID); err != nil {
		t.Fatalf("DeleteSyntheticModel failed: %v", err)
	}

	// Verify list is empty and snapshot cleared
	list, err = fixture.service.ListSyntheticModels(t.Context())
	if err != nil {
		t.Fatalf("ListSyntheticModels after delete failed: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("len(list) = %d after delete, want 0", len(list))
	}
	currentSnapshot = fixture.manager.Current()
	if _, exists := currentSnapshot.SyntheticModels["auto-smart"]; exists {
		t.Fatal("auto-smart should not exist in snapshot after delete")
	}
}

func TestSyntheticModelsValidation(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)

	t.Run("empty name", func(t *testing.T) {
		_, err := fixture.service.CreateSyntheticModel(t.Context(), SyntheticModelCreateRequest{
			Name:         "",
			TargetModels: []string{"gpt-4o"},
		})
		if err == nil {
			t.Fatal("expected error on empty name")
		}
	})

	t.Run("empty targets", func(t *testing.T) {
		_, err := fixture.service.CreateSyntheticModel(t.Context(), SyntheticModelCreateRequest{
			Name:         "auto-test",
			TargetModels: []string{},
		})
		if err == nil {
			t.Fatal("expected error on empty targets")
		}
	})

	t.Run("self target", func(t *testing.T) {
		_, err := fixture.service.CreateSyntheticModel(t.Context(), SyntheticModelCreateRequest{
			Name:         "auto-self",
			TargetModels: []string{"auto-self"},
		})
		if err == nil {
			t.Fatal("expected error on self target")
		}
	})
}
