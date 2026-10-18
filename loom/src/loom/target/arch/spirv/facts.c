// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/facts.h"

#include "loom/target/facts_builder.h"

static uint32_t loom_spirv_target_maximum_workgroup_size(
    const loom_spirv_target_facts_t* facts, uint32_t subgroup_size) {
  uint32_t maximum = facts->base.storage.snapshot.max_flat_workgroup_size;
  if (subgroup_size != facts->subgroup.default_size &&
      facts->subgroup.maximum_workgroup_subgroup_count != 0) {
    const uint64_t subgroup_limit =
        (uint64_t)subgroup_size *
        facts->subgroup.maximum_workgroup_subgroup_count;
    if (maximum == 0 || subgroup_limit < maximum) {
      maximum = (uint32_t)iree_min(subgroup_limit, UINT32_MAX);
    }
  }
  return maximum;
}

static bool loom_spirv_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  if (effective->selector != requirement->selector) {
    return false;
  }
  const loom_spirv_target_facts_t* facts =
      (const loom_spirv_target_facts_t*)effective;
  loom_target_snapshot_t snapshot = effective->storage.snapshot;
  const uint32_t required_size = requirement->storage.snapshot.subgroup_size;
  if (required_size != 0 && snapshot.subgroup_size == 0) {
    if (required_size != facts->subgroup.default_size &&
        (facts->subgroup.minimum_size == 0 ||
         required_size < facts->subgroup.minimum_size ||
         required_size > facts->subgroup.maximum_size ||
         !iree_device_size_is_power_of_two(required_size))) {
      return false;
    }
    snapshot.subgroup_size = required_size;
    snapshot.max_flat_workgroup_size =
        loom_spirv_target_maximum_workgroup_size(facts, required_size);
  }
  return loom_target_snapshot_satisfies_specialization_requirement(
             &snapshot, &requirement->storage.snapshot) &&
         iree_string_view_equal(effective->storage.config.contract_set_key,
                                requirement->storage.config.contract_set_key) &&
         iree_all_bits_set(effective->storage.config.contract_feature_bits,
                           requirement->storage.config.contract_feature_bits);
}

static void loom_spirv_target_facts_apply_specialization_requirement(
    const loom_target_facts_t* requirement, loom_target_facts_t* effective) {
  if (!loom_target_facts_field_is_explicit(
          requirement, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE)) {
    return;
  }
  const uint32_t subgroup_size = effective->storage.snapshot.subgroup_size;
  if (subgroup_size == 0) {
    return;
  }
  loom_spirv_target_facts_t* facts = (loom_spirv_target_facts_t*)effective;
  const uint32_t maximum_workgroup_size =
      loom_spirv_target_maximum_workgroup_size(facts, subgroup_size);
  if (maximum_workgroup_size ==
      effective->storage.snapshot.max_flat_workgroup_size) {
    return;
  }
  effective->storage.snapshot.max_flat_workgroup_size = maximum_workgroup_size;
  loom_target_fact_field_set_insert(
      &effective->explicit_fields,
      LOOM_TARGET_FACT_FIELD_MAX_FLAT_WORKGROUP_SIZE);
}

static iree_status_t loom_spirv_target_facts_select_execution(
    const loom_target_facts_t* source, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = source;
  const loom_spirv_target_facts_t* facts =
      (const loom_spirv_target_facts_t*)source;
  const uint32_t subgroup_size = source->storage.snapshot.subgroup_size != 0
                                     ? source->storage.snapshot.subgroup_size
                                     : facts->subgroup.default_size;
  if (subgroup_size == 0) {
    return iree_ok_status();
  }
  const uint32_t maximum_workgroup_size =
      loom_spirv_target_maximum_workgroup_size(facts, subgroup_size);
  if (subgroup_size == source->storage.snapshot.subgroup_size &&
      maximum_workgroup_size ==
          source->storage.snapshot.max_flat_workgroup_size) {
    return iree_ok_status();
  }
  loom_target_facts_t* selected = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_facts_builder_clone(source, arena, &selected));
  selected->storage.snapshot.subgroup_size = subgroup_size;
  selected->storage.snapshot.max_flat_workgroup_size = maximum_workgroup_size;
  loom_target_fact_field_set_insert(&selected->explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  if (maximum_workgroup_size !=
      source->storage.snapshot.max_flat_workgroup_size) {
    loom_target_fact_field_set_insert(
        &selected->explicit_fields,
        LOOM_TARGET_FACT_FIELD_MAX_FLAT_WORKGROUP_SIZE);
  }
  *out_facts = selected;
  return iree_ok_status();
}

const loom_target_fact_type_t loom_spirv_target_fact_type = {
    .name = IREE_SVL("spirv"),
    .storage_size = sizeof(loom_spirv_target_facts_t),
    .satisfies_identity_requirement =
        loom_target_facts_selector_satisfies_identity_requirement,
    .satisfies_specialization_requirement =
        loom_spirv_target_facts_satisfy_specialization_requirement,
    .apply_specialization_requirement =
        loom_spirv_target_facts_apply_specialization_requirement,
    .select_execution = loom_spirv_target_facts_select_execution,
};
