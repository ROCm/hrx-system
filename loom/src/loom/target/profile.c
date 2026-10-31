// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/target/facts_builder.h"

static bool loom_target_profile_bundle_is_complete(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL;
}

void loom_target_profile_selection_release(
    loom_target_profile_selection_t* selection, iree_allocator_t allocator) {
  if (selection->profile != NULL && selection->destroy != NULL) {
    selection->destroy(selection->profile, allocator);
  }
  *selection = (loom_target_profile_selection_t){0};
}

static bool loom_target_profile_snapshot_equal(
    const loom_target_snapshot_t* lhs, const loom_target_snapshot_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  return iree_string_view_equal(lhs->name, rhs->name) &&
         lhs->codegen_format == rhs->codegen_format &&
         lhs->artifact_format == rhs->artifact_format &&
         lhs->subgroup_partition == rhs->subgroup_partition &&
         lhs->default_pointer_bitwidth == rhs->default_pointer_bitwidth &&
         lhs->index_bitwidth == rhs->index_bitwidth &&
         lhs->offset_bitwidth == rhs->offset_bitwidth &&
         lhs->max_workgroup_size.x == rhs->max_workgroup_size.x &&
         lhs->max_workgroup_size.y == rhs->max_workgroup_size.y &&
         lhs->max_workgroup_size.z == rhs->max_workgroup_size.z &&
         lhs->max_flat_workgroup_size == rhs->max_flat_workgroup_size &&
         lhs->max_workgroup_storage_bytes == rhs->max_workgroup_storage_bytes &&
         lhs->subgroup_size == rhs->subgroup_size &&
         lhs->max_grid_size.x == rhs->max_grid_size.x &&
         lhs->max_grid_size.y == rhs->max_grid_size.y &&
         lhs->max_grid_size.z == rhs->max_grid_size.z &&
         lhs->max_flat_grid_size == rhs->max_flat_grid_size &&
         lhs->max_workgroup_count.x == rhs->max_workgroup_count.x &&
         lhs->max_workgroup_count.y == rhs->max_workgroup_count.y &&
         lhs->max_workgroup_count.z == rhs->max_workgroup_count.z &&
         lhs->memory_spaces.generic == rhs->memory_spaces.generic &&
         lhs->memory_spaces.global == rhs->memory_spaces.global &&
         lhs->memory_spaces.workgroup == rhs->memory_spaces.workgroup &&
         lhs->memory_spaces.constant == rhs->memory_spaces.constant &&
         lhs->memory_spaces.private_memory ==
             rhs->memory_spaces.private_memory &&
         lhs->memory_spaces.host == rhs->memory_spaces.host &&
         lhs->memory_spaces.descriptor == rhs->memory_spaces.descriptor;
}

static bool loom_target_profile_export_equal(
    const loom_target_export_plan_t* lhs,
    const loom_target_export_plan_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  return iree_string_view_equal(lhs->name, rhs->name) &&
         iree_string_view_equal(lhs->export_symbol, rhs->export_symbol) &&
         iree_string_view_equal(lhs->calling_convention,
                                rhs->calling_convention) &&
         lhs->abi_kind == rhs->abi_kind && lhs->linkage == rhs->linkage &&
         lhs->hal_kernel.required_workgroup_size.x ==
             rhs->hal_kernel.required_workgroup_size.x &&
         lhs->hal_kernel.required_workgroup_size.y ==
             rhs->hal_kernel.required_workgroup_size.y &&
         lhs->hal_kernel.required_workgroup_size.z ==
             rhs->hal_kernel.required_workgroup_size.z &&
         lhs->hal_kernel.flat_workgroup_size_min ==
             rhs->hal_kernel.flat_workgroup_size_min &&
         lhs->hal_kernel.flat_workgroup_size_max ==
             rhs->hal_kernel.flat_workgroup_size_max;
}

static bool loom_target_profile_config_equal(const loom_target_config_t* lhs,
                                             const loom_target_config_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  return iree_string_view_equal(lhs->name, rhs->name) &&
         lhs->schedule_model_stable_id == rhs->schedule_model_stable_id &&
         lhs->contract_set.stable_id == rhs->contract_set.stable_id &&
         iree_string_view_equal(lhs->contract_set.spelling,
                                rhs->contract_set.spelling) &&
         lhs->contract_feature_bits == rhs->contract_feature_bits;
}

bool loom_target_profile_equal_common(const loom_target_profile_t* lhs,
                                      const loom_target_profile_t* rhs) {
  if (lhs->explicit_fields != rhs->explicit_fields) {
    return false;
  }
  const loom_target_bundle_t* lhs_bundle = lhs->target_bundle;
  const loom_target_bundle_t* rhs_bundle = rhs->target_bundle;
  if (lhs_bundle == rhs_bundle) {
    return true;
  }
  if (!lhs_bundle || !rhs_bundle) {
    return false;
  }
  return iree_string_view_equal(lhs_bundle->name, rhs_bundle->name) &&
         loom_target_profile_snapshot_equal(lhs_bundle->snapshot,
                                            rhs_bundle->snapshot) &&
         loom_target_profile_export_equal(lhs_bundle->export_plan,
                                          rhs_bundle->export_plan) &&
         loom_target_profile_config_equal(lhs_bundle->config,
                                          rhs_bundle->config);
}

bool loom_target_profile_equal(const loom_target_profile_t* lhs,
                               const loom_target_profile_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  return lhs && rhs && lhs->type == rhs->type && lhs->type->equal(lhs, rhs);
}

iree_status_t loom_target_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t** out_facts) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;
  if (profile == NULL || profile->type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target profile has no family type");
  }
  const loom_target_profile_type_t* profile_type = profile->type;
  if (!loom_target_profile_bundle_is_complete(profile->target_bundle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile family '%.*s' has no complete target bundle",
        (int)profile_type->name.size, profile_type->name.data);
  }
  if (profile_type->fact_type == NULL ||
      profile_type->fact_type->storage_size < sizeof(loom_target_facts_t) ||
      profile_type->project_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target profile family '%.*s' has no typed fact projector",
        (int)profile_type->name.size, profile_type->name.data);
  }

  loom_target_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, profile_type->fact_type->storage_size, (void**)&facts));
  memset(facts, 0, profile_type->fact_type->storage_size);
  loom_target_facts_builder_initialize(profile_type->fact_type,
                                       profile->target_bundle, facts);
  facts->explicit_fields = profile->explicit_fields;
  IREE_RETURN_IF_ERROR(profile_type->project_facts(profile, arena, facts));
  if (facts->fact_type->rebind != NULL) {
    facts->fact_type->rebind(facts);
  }
  *out_facts = facts;
  return iree_ok_status();
}
