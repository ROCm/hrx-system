// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/facts_builder.h"

#include <string.h>

static void loom_target_facts_builder_rebind(loom_target_facts_t* facts) {
  loom_target_bundle_storage_rebind(&facts->storage);
  if (facts->fact_type->rebind != NULL) {
    facts->fact_type->rebind(facts);
  }
}

void loom_target_facts_builder_initialize(
    const loom_target_fact_type_t* fact_type,
    const loom_target_bundle_t* bundle, loom_target_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(fact_type);
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(bundle->snapshot);
  IREE_ASSERT_ARGUMENT(bundle->export_plan);
  IREE_ASSERT_ARGUMENT(bundle->config);
  IREE_ASSERT_ARGUMENT(out_facts);
  memset(out_facts, 0, sizeof(*out_facts));
  out_facts->fact_type = fact_type;
  out_facts->storage = (loom_target_bundle_storage_t){
      .snapshot = *bundle->snapshot,
      .export_plan = *bundle->export_plan,
      .config = *bundle->config,
      .bundle = *bundle,
  };
  loom_target_bundle_storage_rebind(&out_facts->storage);
}

iree_status_t loom_target_facts_builder_clone(const loom_target_facts_t* source,
                                              iree_arena_allocator_t* arena,
                                              loom_target_facts_t** out_facts) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(source->fact_type);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;
  loom_target_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, source->fact_type->storage_size, (void**)&facts));
  memcpy(facts, source, source->fact_type->storage_size);
  loom_target_facts_builder_rebind(facts);
  *out_facts = facts;
  return iree_ok_status();
}

static void loom_target_facts_builder_apply_field(
    loom_target_fact_field_t field,
    const loom_target_bundle_storage_t* requirement,
    loom_target_bundle_storage_t* effective) {
  switch (field) {
    case LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT:
      effective->snapshot.codegen_format = requirement->snapshot.codegen_format;
      break;
    case LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT:
      effective->snapshot.artifact_format =
          requirement->snapshot.artifact_format;
      break;
    case LOOM_TARGET_FACT_FIELD_DEFAULT_POINTER_BITWIDTH:
      effective->snapshot.default_pointer_bitwidth =
          requirement->snapshot.default_pointer_bitwidth;
      break;
    case LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH:
      effective->snapshot.index_bitwidth = requirement->snapshot.index_bitwidth;
      break;
    case LOOM_TARGET_FACT_FIELD_OFFSET_BITWIDTH:
      effective->snapshot.offset_bitwidth =
          requirement->snapshot.offset_bitwidth;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_X:
      effective->snapshot.max_workgroup_size.x =
          requirement->snapshot.max_workgroup_size.x;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Y:
      effective->snapshot.max_workgroup_size.y =
          requirement->snapshot.max_workgroup_size.y;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Z:
      effective->snapshot.max_workgroup_size.z =
          requirement->snapshot.max_workgroup_size.z;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_FLAT_WORKGROUP_SIZE:
      effective->snapshot.max_flat_workgroup_size =
          requirement->snapshot.max_flat_workgroup_size;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_STORAGE_BYTES:
      effective->snapshot.max_workgroup_storage_bytes =
          requirement->snapshot.max_workgroup_storage_bytes;
      break;
    case LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE:
      effective->snapshot.subgroup_size = requirement->snapshot.subgroup_size;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_X:
      effective->snapshot.max_grid_size.x =
          requirement->snapshot.max_grid_size.x;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_Y:
      effective->snapshot.max_grid_size.y =
          requirement->snapshot.max_grid_size.y;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_Z:
      effective->snapshot.max_grid_size.z =
          requirement->snapshot.max_grid_size.z;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_FLAT_GRID_SIZE:
      effective->snapshot.max_flat_grid_size =
          requirement->snapshot.max_flat_grid_size;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_X:
      effective->snapshot.max_workgroup_count.x =
          requirement->snapshot.max_workgroup_count.x;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Y:
      effective->snapshot.max_workgroup_count.y =
          requirement->snapshot.max_workgroup_count.y;
      break;
    case LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Z:
      effective->snapshot.max_workgroup_count.z =
          requirement->snapshot.max_workgroup_count.z;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_GENERIC:
      effective->snapshot.memory_spaces.generic =
          requirement->snapshot.memory_spaces.generic;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_GLOBAL:
      effective->snapshot.memory_spaces.global =
          requirement->snapshot.memory_spaces.global;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_WORKGROUP:
      effective->snapshot.memory_spaces.workgroup =
          requirement->snapshot.memory_spaces.workgroup;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_CONSTANT:
      effective->snapshot.memory_spaces.constant =
          requirement->snapshot.memory_spaces.constant;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_PRIVATE:
      effective->snapshot.memory_spaces.private_memory =
          requirement->snapshot.memory_spaces.private_memory;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_HOST:
      effective->snapshot.memory_spaces.host =
          requirement->snapshot.memory_spaces.host;
      break;
    case LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_DESCRIPTOR:
      effective->snapshot.memory_spaces.descriptor =
          requirement->snapshot.memory_spaces.descriptor;
      break;
    case LOOM_TARGET_FACT_FIELD_ABI:
      effective->export_plan.abi_kind = requirement->export_plan.abi_kind;
      break;
    case LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL:
      effective->export_plan.export_symbol =
          requirement->export_plan.export_symbol;
      break;
    case LOOM_TARGET_FACT_FIELD_LINKAGE:
      effective->export_plan.linkage = requirement->export_plan.linkage;
      break;
    case LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY:
      effective->config.contract_set_key = requirement->config.contract_set_key;
      break;
    case LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS:
      effective->config.contract_feature_bits =
          requirement->config.contract_feature_bits;
      break;
    case LOOM_TARGET_FACT_FIELD_COUNT_:
      break;
  }
}

void loom_target_facts_builder_apply_requirement(
    const loom_target_facts_t* requirement, loom_target_facts_t* effective) {
  IREE_ASSERT_ARGUMENT(requirement);
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT(effective->fact_type == requirement->fact_type);
  IREE_ASSERT(loom_target_facts_satisfy_specialization_requirement(
      effective, requirement));
  for (loom_target_fact_field_t field = 0;
       field < LOOM_TARGET_FACT_FIELD_COUNT_; ++field) {
    if (!loom_target_facts_field_is_explicit(requirement, field)) {
      continue;
    }
    loom_target_facts_builder_apply_field(field, &requirement->storage,
                                          &effective->storage);
  }
  effective->explicit_fields |= requirement->explicit_fields;
  loom_target_facts_builder_rebind(effective);
}

void loom_target_facts_builder_replace_bundle(
    const loom_target_bundle_t* bundle, loom_target_facts_t* facts) {
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(bundle->snapshot);
  IREE_ASSERT_ARGUMENT(bundle->export_plan);
  IREE_ASSERT_ARGUMENT(bundle->config);
  IREE_ASSERT_ARGUMENT(facts);
  facts->storage = (loom_target_bundle_storage_t){
      .snapshot = *bundle->snapshot,
      .export_plan = *bundle->export_plan,
      .config = *bundle->config,
      .bundle = *bundle,
  };
  loom_target_facts_builder_rebind(facts);
}
