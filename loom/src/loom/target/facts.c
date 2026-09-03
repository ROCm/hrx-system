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
  return lhs->fact_type == rhs->fact_type &&
         lhs->explicit_fields == rhs->explicit_fields &&
         loom_target_facts_satisfy_specialization_requirement(lhs, rhs) &&
         loom_target_facts_satisfy_specialization_requirement(rhs, lhs);
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
  return loom_target_facts_product_contract_satisfies_specialization_requirement(
             effective, requirement) &&
         loom_target_snapshot_satisfies_specialization_requirement(
             &effective->storage.snapshot, &requirement->storage.snapshot) &&
         iree_string_view_equal(effective->storage.config.contract_set_key,
                                requirement->storage.config.contract_set_key) &&
         iree_all_bits_set(effective->storage.config.contract_feature_bits,
                           requirement->storage.config.contract_feature_bits);
}

static bool loom_target_facts_explicit_field_matches(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement, loom_target_fact_field_t field,
    bool values_match) {
  return !loom_target_facts_field_is_explicit(requirement, field) ||
         !loom_target_facts_field_is_explicit(effective, field) || values_match;
}

bool loom_target_facts_product_contract_satisfies_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  return loom_target_facts_explicit_field_matches(
             effective, requirement, LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT,
             effective->storage.snapshot.codegen_format ==
                 requirement->storage.snapshot.codegen_format) &&
         loom_target_facts_explicit_field_matches(
             effective, requirement, LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT,
             effective->storage.snapshot.artifact_format ==
                 requirement->storage.snapshot.artifact_format) &&
         loom_target_facts_explicit_field_matches(
             effective, requirement, LOOM_TARGET_FACT_FIELD_ABI,
             effective->storage.export_plan.abi_kind ==
                 requirement->storage.export_plan.abi_kind) &&
         loom_target_facts_explicit_field_matches(
             effective, requirement, LOOM_TARGET_FACT_FIELD_LINKAGE,
             effective->storage.export_plan.linkage ==
                 requirement->storage.export_plan.linkage);
}

static bool loom_target_limit_satisfies(uint64_t effective_limit,
                                        uint64_t required_limit) {
  return required_limit == 0 || effective_limit >= required_limit;
}

bool loom_target_snapshot_satisfies_specialization_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement) {
  IREE_ASSERT_ARGUMENT(effective_snapshot);
  IREE_ASSERT_ARGUMENT(target_requirement);
  return effective_snapshot->default_pointer_bitwidth ==
             target_requirement->default_pointer_bitwidth &&
         effective_snapshot->index_bitwidth ==
             target_requirement->index_bitwidth &&
         effective_snapshot->offset_bitwidth ==
             target_requirement->offset_bitwidth &&
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
             target_requirement->max_workgroup_storage_bytes) &&
         effective_snapshot->memory_spaces.generic ==
             target_requirement->memory_spaces.generic &&
         effective_snapshot->memory_spaces.global ==
             target_requirement->memory_spaces.global &&
         effective_snapshot->memory_spaces.workgroup ==
             target_requirement->memory_spaces.workgroup &&
         effective_snapshot->memory_spaces.constant ==
             target_requirement->memory_spaces.constant &&
         effective_snapshot->memory_spaces.private_memory ==
             target_requirement->memory_spaces.private_memory &&
         effective_snapshot->memory_spaces.host ==
             target_requirement->memory_spaces.host &&
         effective_snapshot->memory_spaces.descriptor ==
             target_requirement->memory_spaces.descriptor;
}
