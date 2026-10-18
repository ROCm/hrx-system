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
#define LOOM_COPY_FIELD(name, group, member)             \
  case LOOM_TARGET_FACT_FIELD_##name:                    \
    effective->group.member = requirement->group.member; \
    break;
    LOOM_COPY_FIELD(CODEGEN_FORMAT, snapshot, codegen_format)
    LOOM_COPY_FIELD(ARTIFACT_FORMAT, snapshot, artifact_format)
    LOOM_COPY_FIELD(DEFAULT_POINTER_BITWIDTH, snapshot,
                    default_pointer_bitwidth)
    LOOM_COPY_FIELD(INDEX_BITWIDTH, snapshot, index_bitwidth)
    LOOM_COPY_FIELD(OFFSET_BITWIDTH, snapshot, offset_bitwidth)
    LOOM_COPY_FIELD(MAX_WORKGROUP_SIZE_X, snapshot, max_workgroup_size.x)
    LOOM_COPY_FIELD(MAX_WORKGROUP_SIZE_Y, snapshot, max_workgroup_size.y)
    LOOM_COPY_FIELD(MAX_WORKGROUP_SIZE_Z, snapshot, max_workgroup_size.z)
    LOOM_COPY_FIELD(MAX_FLAT_WORKGROUP_SIZE, snapshot, max_flat_workgroup_size)
    LOOM_COPY_FIELD(MAX_WORKGROUP_STORAGE_BYTES, snapshot,
                    max_workgroup_storage_bytes)
    LOOM_COPY_FIELD(SUBGROUP_SIZE, snapshot, subgroup_size)
    LOOM_COPY_FIELD(MAX_GRID_SIZE_X, snapshot, max_grid_size.x)
    LOOM_COPY_FIELD(MAX_GRID_SIZE_Y, snapshot, max_grid_size.y)
    LOOM_COPY_FIELD(MAX_GRID_SIZE_Z, snapshot, max_grid_size.z)
    LOOM_COPY_FIELD(MAX_FLAT_GRID_SIZE, snapshot, max_flat_grid_size)
    LOOM_COPY_FIELD(MAX_WORKGROUP_COUNT_X, snapshot, max_workgroup_count.x)
    LOOM_COPY_FIELD(MAX_WORKGROUP_COUNT_Y, snapshot, max_workgroup_count.y)
    LOOM_COPY_FIELD(MAX_WORKGROUP_COUNT_Z, snapshot, max_workgroup_count.z)
    LOOM_COPY_FIELD(MEMORY_SPACE_GENERIC, snapshot, memory_spaces.generic)
    LOOM_COPY_FIELD(MEMORY_SPACE_GLOBAL, snapshot, memory_spaces.global)
    LOOM_COPY_FIELD(MEMORY_SPACE_WORKGROUP, snapshot, memory_spaces.workgroup)
    LOOM_COPY_FIELD(MEMORY_SPACE_CONSTANT, snapshot, memory_spaces.constant)
    LOOM_COPY_FIELD(MEMORY_SPACE_PRIVATE, snapshot,
                    memory_spaces.private_memory)
    LOOM_COPY_FIELD(MEMORY_SPACE_HOST, snapshot, memory_spaces.host)
    LOOM_COPY_FIELD(MEMORY_SPACE_DESCRIPTOR, snapshot, memory_spaces.descriptor)
    LOOM_COPY_FIELD(ABI, export_plan, abi_kind)
    LOOM_COPY_FIELD(EXPORT_SYMBOL, export_plan, export_symbol)
    LOOM_COPY_FIELD(LINKAGE, export_plan, linkage)
    LOOM_COPY_FIELD(CONTRACT_SET_KEY, config, contract_set_key)
    LOOM_COPY_FIELD(CONTRACT_FEATURE_BITS, config, contract_feature_bits)
#undef LOOM_COPY_FIELD
    case LOOM_TARGET_FACT_FIELD_COUNT_:
      break;
  }
}

iree_status_t loom_target_facts_builder_select_execution(
    const loom_target_facts_t* source, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = source;
  return source->fact_type->select_execution != NULL
             ? source->fact_type->select_execution(source, arena, out_facts)
             : iree_ok_status();
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
  if (effective->fact_type->apply_specialization_requirement != NULL) {
    effective->fact_type->apply_specialization_requirement(requirement,
                                                           effective);
  }
  loom_target_facts_builder_rebind(effective);
}

iree_status_t loom_target_facts_builder_project_worker(
    const loom_target_facts_t* source, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = source;
  return source != NULL && source->fact_type->project_worker != NULL
             ? source->fact_type->project_worker(source, arena, out_facts)
             : iree_ok_status();
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

void loom_target_facts_builder_set_worker_contract(
    uint8_t selector, const loom_target_bundle_t* bundle,
    loom_target_facts_t* facts) {
  facts->selector = selector;
  facts->storage.snapshot.name = bundle->snapshot->name;
  facts->storage.snapshot.codegen_format = bundle->snapshot->codegen_format;
  facts->storage.snapshot.artifact_format = bundle->snapshot->artifact_format;
  facts->storage.export_plan = *bundle->export_plan;
  facts->storage.config = *bundle->config;
  facts->storage.bundle.name = bundle->name;
  const loom_target_fact_field_set_t artifact_fields =
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ABI) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_LINKAGE) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY) |
      (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS);
  facts->explicit_fields &= ~artifact_fields;
  loom_target_facts_builder_rebind(facts);
}
