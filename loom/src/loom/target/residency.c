// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/residency.h"

#include <string.h>

#include "iree/base/internal/math.h"

uint64_t loom_target_residency_round_resource_units(uint64_t units,
                                                    uint32_t granularity) {
  IREE_ASSERT_NE(granularity, 0);
  const uint64_t round_mask = (uint64_t)granularity - 1u;
  if ((granularity & (granularity - 1u)) == 0) {
    return units > UINT64_MAX - round_mask ? UINT64_MAX
                                           : (units + round_mask) & ~round_mask;
  }
  const uint64_t remainder = units % granularity;
  if (remainder == 0) return units;
  const uint64_t delta = granularity - remainder;
  return units > UINT64_MAX - delta ? UINT64_MAX : units + delta;
}

void loom_target_residency_evaluate_cliffs(
    const loom_target_residency_cliff_t* cliffs, iree_host_size_t cliff_count,
    uint32_t initial_tier, uint64_t units,
    loom_target_residency_cliff_evaluation_t* out_evaluation) {
  *out_evaluation = (loom_target_residency_cliff_evaluation_t){
      .tier = initial_tier,
  };
  const loom_target_residency_cliff_t* last_crossed_cliff = NULL;
  for (iree_host_size_t i = 0; i < cliff_count; ++i) {
    const loom_target_residency_cliff_t* cliff = &cliffs[i];
    if (units < cliff->cliff_units) {
      out_evaluation->worse_tier = cliff->tier_after;
      out_evaluation->worse_cliff_units = cliff->cliff_units;
      out_evaluation->additional_units_to_worse_tier =
          cliff->cliff_units - units;
      out_evaluation->flags |=
          LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER;
      break;
    }
    out_evaluation->tier = cliff->tier_after;
    last_crossed_cliff = cliff;
  }
  if (last_crossed_cliff != NULL) {
    out_evaluation->better_tier = last_crossed_cliff->tier_before;
    out_evaluation->reduction_units_to_better_tier =
        units - last_crossed_cliff->cliff_units + 1u;
    out_evaluation->flags |=
        LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_BETTER_TIER;
  }
}

static uint64_t loom_target_residency_direct_resource_units_with_override(
    const uint32_t* direct_resource_units, uint16_t resource_id,
    uint16_t override_resource_id, uint32_t override_units) {
  return resource_id == override_resource_id
             ? override_units
             : direct_resource_units[resource_id];
}

uint32_t loom_target_residency_evaluate_tier_with_direct_resource_override(
    const loom_target_residency_model_t* model,
    const uint32_t* direct_resource_units, uint16_t direct_resource_id,
    uint32_t override_units) {
  IREE_ASSERT_ARGUMENT(model);
  IREE_ASSERT_ARGUMENT(direct_resource_units);
  IREE_ASSERT_LT(direct_resource_id, model->direct_resources.resource_count);

  uint32_t tier = model->best_tier;
  for (uint16_t resource_id = 0;
       resource_id < model->direct_resources.resource_count; ++resource_id) {
    const loom_target_residency_cliff_range_t range =
        model->direct_resources.cliff_ranges[resource_id];
    const loom_target_residency_cliff_t* cliffs =
        range.count == 0 ? NULL : &model->direct_resources.cliffs[range.start];
    const uint64_t units =
        loom_target_residency_direct_resource_units_with_override(
            direct_resource_units, resource_id, direct_resource_id,
            override_units);
    loom_target_residency_cliff_evaluation_t evaluation;
    loom_target_residency_evaluate_cliffs(cliffs, range.count, model->best_tier,
                                          units, &evaluation);
    tier = iree_min(tier, evaluation.tier);
  }

  for (uint16_t resource_id = 0;
       resource_id < model->derived_resources.resource_count; ++resource_id) {
    const loom_target_residency_derived_resource_t* resource =
        &model->derived_resources.resources[resource_id];
    uint64_t resource_units = 0;
    for (uint16_t i = 0; i < resource->member_count; ++i) {
      const loom_target_residency_derived_member_t* member =
          &model->derived_resources.members[resource->member_start + i];
      const uint64_t direct_units =
          loom_target_residency_direct_resource_units_with_override(
              direct_resource_units, member->direct_resource_id,
              direct_resource_id, override_units);
      const uint64_t contribution = loom_target_residency_round_resource_units(
          direct_units, member->contribution_granularity);
      resource_units =
          iree_math_saturating_add_u64(resource_units, contribution);
    }
    const loom_target_residency_cliff_t* cliffs =
        resource->cliff_count == 0
            ? NULL
            : &model->derived_resources.cliffs[resource->cliff_start];
    loom_target_residency_cliff_evaluation_t evaluation;
    loom_target_residency_evaluate_cliffs(cliffs, resource->cliff_count,
                                          model->best_tier, resource_units,
                                          &evaluation);
    tier = iree_min(tier, evaluation.tier);
  }
  return tier;
}

static iree_status_t loom_target_residency_validate_cliff_chain(
    const loom_target_residency_cliff_t* cliffs, iree_host_size_t cliff_count,
    uint16_t expected_resource_id, uint32_t initial_tier,
    iree_string_view_t resource_name) {
  if (cliff_count == 0) return iree_ok_status();
  if (cliffs == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency resource '%.*s' has a null non-empty cliff chain",
        (int)resource_name.size, resource_name.data);
  }
  uint32_t previous_cliff_units = 0;
  uint32_t expected_tier = initial_tier;
  for (iree_host_size_t i = 0; i < cliff_count; ++i) {
    const loom_target_residency_cliff_t* cliff = &cliffs[i];
    if (cliff->resource_id != expected_resource_id) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency resource '%.*s' cliff %zu names resource %u instead of "
          "%u",
          (int)resource_name.size, resource_name.data, i,
          (unsigned)cliff->resource_id, (unsigned)expected_resource_id);
    }
    if (cliff->cliff_units == 0 ||
        (i != 0 && cliff->cliff_units <= previous_cliff_units)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency resource '%.*s' cliff %zu threshold %u is not strictly "
          "increasing and nonzero",
          (int)resource_name.size, resource_name.data, i,
          (unsigned)cliff->cliff_units);
    }
    if (cliff->tier_before != expected_tier ||
        cliff->tier_after >= cliff->tier_before) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency resource '%.*s' cliff %zu has noncontiguous or "
          "nondescending tier transition %u to %u",
          (int)resource_name.size, resource_name.data, i,
          (unsigned)cliff->tier_before, (unsigned)cliff->tier_after);
    }
    previous_cliff_units = cliff->cliff_units;
    expected_tier = cliff->tier_after;
  }
  return iree_ok_status();
}

static iree_status_t loom_target_residency_validate_direct_resources(
    const loom_target_residency_model_t* model) {
  const loom_target_residency_direct_resource_table_t* table =
      &model->direct_resources;
  if (table->cliff_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency direct cliff table has %zu rows but ranges use 32-bit "
        "offsets",
        table->cliff_count);
  }
  if (table->resource_count == 0) {
    if (table->cliff_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency model has direct cliffs without direct resources");
    }
    return iree_ok_status();
  }
  if (table->names == NULL || table->cliff_ranges == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency model has %u direct resources without names and ranges",
        (unsigned)table->resource_count);
  }
  if (table->cliff_count != 0 && table->cliffs == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency model has a null non-empty direct cliff table");
  }
  iree_host_size_t expected_cliff_start = 0;
  for (uint16_t resource_id = 0; resource_id < table->resource_count;
       ++resource_id) {
    const iree_string_view_t resource_name = table->names[resource_id];
    if (iree_string_view_is_empty(resource_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency direct resource %u has an empty stable name",
          (unsigned)resource_id);
    }
    const loom_target_residency_cliff_range_t range =
        table->cliff_ranges[resource_id];
    if (range.start != expected_cliff_start ||
        range.count > table->cliff_count - expected_cliff_start) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency direct resource '%.*s' has malformed cliff range "
          "[%u, %u)",
          (int)resource_name.size, resource_name.data, (unsigned)range.start,
          (unsigned)(range.start + range.count));
    }
    const loom_target_residency_cliff_t* cliffs =
        range.count == 0 ? NULL : &table->cliffs[range.start];
    IREE_RETURN_IF_ERROR(loom_target_residency_validate_cliff_chain(
        cliffs, range.count, resource_id, model->best_tier, resource_name));
    expected_cliff_start += range.count;
  }
  if (expected_cliff_start != table->cliff_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency direct-resource ranges cover %zu of %zu cliff rows",
        expected_cliff_start, table->cliff_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_residency_validate_derived_resources(
    const loom_target_residency_model_t* model) {
  const loom_target_residency_direct_resource_table_t* direct_table =
      &model->direct_resources;
  const loom_target_residency_derived_resource_table_t* table =
      &model->derived_resources;
  if (table->resource_count == 0) {
    if (table->member_count != 0 || table->cliff_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency model has derived rows without derived resources");
    }
    return iree_ok_status();
  }
  if (table->resources == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency model has %u derived resources without resource rows",
        (unsigned)table->resource_count);
  }
  if (table->member_count != 0 && table->members == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency model has a null non-empty derived member table");
  }
  if (table->cliff_count != 0 && table->cliffs == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency model has a null non-empty derived cliff table");
  }
  iree_host_size_t expected_member_start = 0;
  iree_host_size_t expected_cliff_start = 0;
  for (uint16_t resource_id = 0; resource_id < table->resource_count;
       ++resource_id) {
    const loom_target_residency_derived_resource_t* resource =
        &table->resources[resource_id];
    if (iree_string_view_is_empty(resource->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency derived resource %u has an empty stable name",
          (unsigned)resource_id);
    }
    if (resource->pool_units == 0 || resource->allocation_granularity == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency derived resource '%.*s' has zero pool or allocation "
          "granularity",
          (int)resource->name.size, resource->name.data);
    }
    if (resource->member_count == 0 ||
        resource->member_start != expected_member_start ||
        resource->member_count > table->member_count - expected_member_start) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency derived resource '%.*s' has a malformed member range",
          (int)resource->name.size, resource->name.data);
    }
    if (resource->cliff_start != expected_cliff_start ||
        resource->cliff_count > table->cliff_count - expected_cliff_start) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency derived resource '%.*s' has a malformed cliff range",
          (int)resource->name.size, resource->name.data);
    }
    for (uint16_t i = 0; i < resource->member_count; ++i) {
      const loom_target_residency_derived_member_t* member =
          &table->members[resource->member_start + i];
      if (member->resource_id != resource_id ||
          member->direct_resource_id >= direct_table->resource_count ||
          member->contribution_granularity == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "residency derived resource '%.*s' member %u is malformed",
            (int)resource->name.size, resource->name.data, (unsigned)i);
      }
      for (uint16_t j = 0; j < i; ++j) {
        const loom_target_residency_derived_member_t* previous_member =
            &table->members[resource->member_start + j];
        if (previous_member->direct_resource_id == member->direct_resource_id) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "residency derived resource '%.*s' repeats direct resource %u",
              (int)resource->name.size, resource->name.data,
              (unsigned)member->direct_resource_id);
        }
      }
    }
    const loom_target_residency_cliff_t* cliffs =
        resource->cliff_count == 0 ? NULL
                                   : &table->cliffs[resource->cliff_start];
    IREE_RETURN_IF_ERROR(loom_target_residency_validate_cliff_chain(
        cliffs, resource->cliff_count, resource_id, model->best_tier,
        resource->name));
    expected_member_start += resource->member_count;
    expected_cliff_start += resource->cliff_count;
  }
  if (expected_member_start != table->member_count ||
      expected_cliff_start != table->cliff_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency derived-resource ranges do not cover their tables");
  }

  const bool has_member_indices =
      table->member_indices_by_direct_resource != NULL;
  const bool has_member_ranges =
      table->member_ranges_by_direct_resource != NULL;
  if (has_member_indices != has_member_ranges) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency derived-resource reverse index is only partially present");
  }
  if (!has_member_indices) return iree_ok_status();

  iree_host_size_t expected_index_start = 0;
  for (uint16_t direct_resource_id = 0;
       direct_resource_id < direct_table->resource_count;
       ++direct_resource_id) {
    const loom_target_residency_derived_member_range_t range =
        table->member_ranges_by_direct_resource[direct_resource_id];
    if (range.start != expected_index_start ||
        range.count > table->member_count - expected_index_start) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency direct resource %u has a malformed member reverse range",
          (unsigned)direct_resource_id);
    }
    for (uint16_t i = 0; i < range.count; ++i) {
      const uint16_t member_index =
          table->member_indices_by_direct_resource[range.start + i];
      if (member_index >= table->member_count ||
          table->members[member_index].direct_resource_id !=
              direct_resource_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "residency direct resource %u has a malformed member reverse "
            "index",
            (unsigned)direct_resource_id);
      }
      for (iree_host_size_t j = 0; j < expected_index_start + i; ++j) {
        if (table->member_indices_by_direct_resource[j] == member_index) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "residency member reverse index repeats member %u",
              (unsigned)member_index);
        }
      }
    }
    expected_index_start += range.count;
  }
  if (expected_index_start != table->member_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency member reverse index covers %zu of %u members",
        expected_index_start, (unsigned)table->member_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_residency_validate_model(
    const loom_target_residency_model_t* model) {
  if (model->best_tier == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "residency model best tier must be nonzero");
  }
  IREE_RETURN_IF_ERROR(loom_target_residency_validate_direct_resources(model));
  return loom_target_residency_validate_derived_resources(model);
}

static void loom_target_residency_resource_cliffs(
    const loom_target_residency_model_t* model,
    const loom_target_residency_resource_evaluation_t* resource_evaluation,
    const loom_target_residency_cliff_t** out_cliffs,
    iree_host_size_t* out_cliff_count) {
  *out_cliffs = NULL;
  *out_cliff_count = 0;
  if (resource_evaluation->kind == LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DIRECT) {
    const loom_target_residency_direct_resource_table_t* table =
        &model->direct_resources;
    const loom_target_residency_cliff_range_t range =
        table->cliff_ranges[resource_evaluation->resource_id];
    if (range.count != 0) {
      *out_cliffs = &table->cliffs[range.start];
      *out_cliff_count = range.count;
    }
    return;
  }
  const loom_target_residency_derived_resource_table_t* table =
      &model->derived_resources;
  const loom_target_residency_derived_resource_t* resource =
      &table->resources[resource_evaluation->resource_id];
  if (resource->cliff_count != 0) {
    *out_cliffs = &table->cliffs[resource->cliff_start];
    *out_cliff_count = resource->cliff_count;
  }
}

static void loom_target_residency_fill_resource_evaluation(
    iree_string_view_t name, loom_target_residency_resource_kind_t kind,
    uint16_t resource_id, uint64_t units,
    const loom_target_residency_cliff_t* cliffs, iree_host_size_t cliff_count,
    uint32_t best_tier,
    loom_target_residency_resource_evaluation_t* out_evaluation) {
  loom_target_residency_cliff_evaluation_t cliff_evaluation;
  loom_target_residency_evaluate_cliffs(cliffs, cliff_count, best_tier, units,
                                        &cliff_evaluation);
  *out_evaluation = (loom_target_residency_resource_evaluation_t){
      .name = name,
      .kind = kind,
      .resource_id = resource_id,
      .units = units,
      .tier = cliff_evaluation.tier,
  };
}

iree_status_t loom_target_residency_query(
    const loom_target_residency_model_t* model,
    const uint64_t* direct_resource_units,
    iree_host_size_t direct_resource_unit_count, iree_arena_allocator_t* arena,
    loom_target_residency_query_t* out_query) {
  if (out_query == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "residency query output must be non-null");
  }
  *out_query = (loom_target_residency_query_t){0};
  if (model == NULL) {
    if (direct_resource_unit_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "an unavailable residency model cannot consume direct resources");
    }
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_residency_validate_model(model));
  if (direct_resource_unit_count != model->direct_resources.resource_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency query has %zu direct resource values; model requires %u",
        direct_resource_unit_count,
        (unsigned)model->direct_resources.resource_count);
  }
  if (direct_resource_unit_count != 0 && direct_resource_units == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency query has a null non-empty direct resource vector");
  }

  const iree_host_size_t resource_count =
      direct_resource_unit_count + model->derived_resources.resource_count;
  loom_target_residency_resource_evaluation_t* resource_evaluations = NULL;
  if (resource_count != 0) {
    if (arena == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "residency query with resources requires an allocation arena");
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resource_evaluations),
        (void**)&resource_evaluations));
    memset(resource_evaluations, 0,
           resource_count * sizeof(*resource_evaluations));
  }

  uint32_t tier = model->best_tier;
  for (uint16_t resource_id = 0;
       resource_id < model->direct_resources.resource_count; ++resource_id) {
    const loom_target_residency_cliff_range_t range =
        model->direct_resources.cliff_ranges[resource_id];
    const loom_target_residency_cliff_t* cliffs =
        range.count == 0 ? NULL : &model->direct_resources.cliffs[range.start];
    loom_target_residency_resource_evaluation_t* resource_evaluation =
        &resource_evaluations[resource_id];
    loom_target_residency_fill_resource_evaluation(
        model->direct_resources.names[resource_id],
        LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DIRECT, resource_id,
        direct_resource_units[resource_id], cliffs, range.count,
        model->best_tier, resource_evaluation);
    tier = iree_min(tier, resource_evaluation->tier);
  }

  const iree_host_size_t derived_resource_offset =
      model->direct_resources.resource_count;
  for (uint16_t resource_id = 0;
       resource_id < model->derived_resources.resource_count; ++resource_id) {
    const loom_target_residency_derived_resource_t* resource =
        &model->derived_resources.resources[resource_id];
    uint64_t resource_units = 0;
    for (uint16_t i = 0; i < resource->member_count; ++i) {
      const loom_target_residency_derived_member_t* member =
          &model->derived_resources.members[resource->member_start + i];
      const uint64_t contribution = loom_target_residency_round_resource_units(
          direct_resource_units[member->direct_resource_id],
          member->contribution_granularity);
      resource_units =
          iree_math_saturating_add_u64(resource_units, contribution);
    }
    const loom_target_residency_cliff_t* cliffs =
        resource->cliff_count == 0
            ? NULL
            : &model->derived_resources.cliffs[resource->cliff_start];
    loom_target_residency_resource_evaluation_t* resource_evaluation =
        &resource_evaluations[derived_resource_offset + resource_id];
    loom_target_residency_fill_resource_evaluation(
        resource->name, LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DERIVED,
        resource_id, resource_units, cliffs, resource->cliff_count,
        model->best_tier, resource_evaluation);
    tier = iree_min(tier, resource_evaluation->tier);
  }

  iree_host_size_t limiting_resource_count = 0;
  uint32_t next_better_tier = model->best_tier;
  if (tier < model->best_tier) {
    for (iree_host_size_t i = 0; i < resource_count; ++i) {
      loom_target_residency_resource_evaluation_t* resource_evaluation =
          &resource_evaluations[i];
      if (resource_evaluation->tier != tier) {
        next_better_tier =
            iree_min(next_better_tier, resource_evaluation->tier);
        continue;
      }
      resource_evaluation->flags |=
          LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_LIMITING;
      ++limiting_resource_count;
      const loom_target_residency_cliff_t* cliffs = NULL;
      iree_host_size_t cliff_count = 0;
      loom_target_residency_resource_cliffs(model, resource_evaluation, &cliffs,
                                            &cliff_count);
      loom_target_residency_cliff_evaluation_t cliff_evaluation;
      loom_target_residency_evaluate_cliffs(
          cliffs, cliff_count, model->best_tier, resource_evaluation->units,
          &cliff_evaluation);
      IREE_ASSERT_TRUE(iree_any_bit_set(
          cliff_evaluation.flags,
          LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_BETTER_TIER));
      next_better_tier =
          iree_min(next_better_tier, cliff_evaluation.better_tier);
    }
  }

  const bool has_next_better_tier = next_better_tier > tier;
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    loom_target_residency_resource_evaluation_t* resource_evaluation =
        &resource_evaluations[i];
    const loom_target_residency_cliff_t* cliffs = NULL;
    iree_host_size_t cliff_count = 0;
    loom_target_residency_resource_cliffs(model, resource_evaluation, &cliffs,
                                          &cliff_count);
    if (has_next_better_tier && resource_evaluation->tier < next_better_tier) {
      loom_target_residency_cliff_evaluation_t cliff_evaluation;
      loom_target_residency_evaluate_cliffs(
          cliffs, cliff_count, model->best_tier, resource_evaluation->units,
          &cliff_evaluation);
      IREE_ASSERT_GE(cliff_evaluation.better_tier, next_better_tier);
      resource_evaluation->reduction_units_to_next_better_tier =
          cliff_evaluation.reduction_units_to_better_tier;
    }
    for (iree_host_size_t j = 0; j < cliff_count; ++j) {
      const loom_target_residency_cliff_t* cliff = &cliffs[j];
      if (resource_evaluation->units >= cliff->cliff_units ||
          cliff->tier_after >= tier) {
        continue;
      }
      resource_evaluation->next_worse_tier = cliff->tier_after;
      resource_evaluation->next_worse_cliff_units = cliff->cliff_units;
      resource_evaluation->additional_units_to_next_worse_tier =
          cliff->cliff_units - resource_evaluation->units;
      resource_evaluation->flags |=
          LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_HAS_NEXT_WORSE_TIER;
      break;
    }
  }

  *out_query = (loom_target_residency_query_t){
      .model_available = true,
      .best_tier = model->best_tier,
      .tier = tier,
      .has_next_better_tier = has_next_better_tier,
      .next_better_tier =
          has_next_better_tier ? next_better_tier : model->best_tier,
      .resources = resource_evaluations,
      .resource_count = resource_count,
      .direct_resource_count = direct_resource_unit_count,
      .limiting_resource_count = limiting_resource_count,
  };
  return iree_ok_status();
}
