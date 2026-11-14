// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/facts.h"

bool loom_target_facts_satisfy_identity_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  if (effective == requirement) {
    return true;
  }
  if (effective->fact_type != requirement->fact_type ||
      effective->fact_type->satisfies_identity_requirement == NULL) {
    return false;
  }
  return effective->fact_type->satisfies_identity_requirement(effective,
                                                              requirement);
}

bool loom_target_facts_selector_satisfies_identity_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  return effective->fact_type == requirement->fact_type &&
         effective->selector == requirement->selector;
}

bool loom_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  if (effective == requirement) {
    return true;
  }
  if (effective->fact_type != requirement->fact_type ||
      effective->fact_type->satisfies_specialization_requirement == NULL) {
    return false;
  }
  return effective->fact_type->satisfies_specialization_requirement(
      effective, requirement);
}

bool loom_target_facts_are_equivalent(const loom_target_facts_t* lhs,
                                      const loom_target_facts_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (lhs == rhs) {
    return true;
  }
  if (lhs->fact_type != rhs->fact_type ||
      lhs->explicit_fields != rhs->explicit_fields ||
      lhs->storage.config.schedule_model_stable_id !=
          rhs->storage.config.schedule_model_stable_id ||
      !loom_target_facts_satisfy_specialization_requirement(lhs, rhs) ||
      !loom_target_facts_satisfy_specialization_requirement(rhs, lhs)) {
    return false;
  }
  return lhs->fact_type->family_state_equivalent == NULL ||
         lhs->fact_type->family_state_equivalent(lhs, rhs);
}

bool loom_target_facts_structural_satisfy_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  if (effective->fact_type != requirement->fact_type ||
      effective->selector != requirement->selector) {
    return false;
  }
  return loom_target_snapshot_satisfies_specialization_requirement(
             &effective->storage.snapshot, &requirement->storage.snapshot) &&
         effective->storage.config.contract_set.stable_id ==
             requirement->storage.config.contract_set.stable_id &&
         iree_all_bits_set(effective->storage.config.contract_feature_bits,
                           requirement->storage.config.contract_feature_bits);
}

static bool loom_target_limit_satisfies(uint64_t effective_limit,
                                        uint64_t required_limit) {
  return required_limit == 0 || effective_limit >= required_limit;
}

bool loom_target_snapshot_has_same_data_layout(
    const loom_target_snapshot_t* lhs, const loom_target_snapshot_t* rhs) {
  return lhs->default_pointer_bitwidth == rhs->default_pointer_bitwidth &&
         lhs->index_bitwidth == rhs->index_bitwidth &&
         loom_target_snapshot_maximum_index_bitwidth(lhs) ==
             loom_target_snapshot_maximum_index_bitwidth(rhs) &&
         lhs->offset_bitwidth == rhs->offset_bitwidth &&
         lhs->memory_spaces.generic == rhs->memory_spaces.generic &&
         lhs->memory_spaces.global == rhs->memory_spaces.global &&
         lhs->memory_spaces.workgroup == rhs->memory_spaces.workgroup &&
         lhs->memory_spaces.constant == rhs->memory_spaces.constant &&
         lhs->memory_spaces.private_memory ==
             rhs->memory_spaces.private_memory &&
         lhs->memory_spaces.host == rhs->memory_spaces.host &&
         lhs->memory_spaces.descriptor == rhs->memory_spaces.descriptor;
}

bool loom_target_snapshot_satisfies_specialization_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement) {
  IREE_ASSERT_ARGUMENT(effective_snapshot);
  IREE_ASSERT_ARGUMENT(target_requirement);
  return loom_target_snapshot_has_same_data_layout(effective_snapshot,
                                                   target_requirement) &&
         effective_snapshot->codegen_format ==
             target_requirement->codegen_format &&
         effective_snapshot->artifact_format ==
             target_requirement->artifact_format &&
         effective_snapshot->subgroup_partition >=
             target_requirement->subgroup_partition &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.x,
             target_requirement->max_workgroup_size.x) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.y,
             target_requirement->max_workgroup_size.y) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.z,
             target_requirement->max_workgroup_size.z) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_flat_workgroup_size,
             target_requirement->max_flat_workgroup_size) &&
         (target_requirement->subgroup_size == 0 ||
          effective_snapshot->subgroup_size ==
              target_requirement->subgroup_size) &&
         (target_requirement->execution_unit_count == 0 ||
          effective_snapshot->execution_unit_count ==
              target_requirement->execution_unit_count) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.x,
                                     target_requirement->max_grid_size.x) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.y,
                                     target_requirement->max_grid_size.y) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.z,
                                     target_requirement->max_grid_size.z) &&
         loom_target_limit_satisfies(effective_snapshot->max_flat_grid_size,
                                     target_requirement->max_flat_grid_size) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.x,
             target_requirement->max_workgroup_count.x) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.y,
             target_requirement->max_workgroup_count.y) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.z,
             target_requirement->max_workgroup_count.z) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_storage_bytes,
             target_requirement->max_workgroup_storage_bytes);
}
