// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"

static const loom_target_snapshot_t kGenericReferenceSnapshot = {
    .name = IREE_SVL("target-generic-reference"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
};

static const loom_target_export_plan_t kGenericReferenceExportPlan = {
    .name = IREE_SVL("target-generic-reference"),
    .abi_kind = LOOM_TARGET_ABI_VM_MODULE_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_config_t kGenericReferenceConfig = {
    .name = IREE_SVL("target.generic.reference"),
};

static const loom_target_bundle_t kGenericReferenceBundle = {
    .name = IREE_SVL("target-generic-reference"),
    .snapshot = &kGenericReferenceSnapshot,
    .export_plan = &kGenericReferenceExportPlan,
    .config = &kGenericReferenceConfig,
};

static const loom_target_bundle_t* const kGenericTargetBundleValues[] = {
    NULL,
    &kGenericReferenceBundle,
};

const loom_target_bundle_table_t loom_target_generic_target_bundles = {
    .values = kGenericTargetBundleValues,
    .count = IREE_ARRAYSIZE(kGenericTargetBundleValues),
};

static void loom_target_bundle_storage_initialize_from_bundle(
    loom_target_bundle_storage_t* storage, iree_string_view_t name,
    const loom_target_bundle_t* bundle) {
  storage->snapshot = *bundle->snapshot;
  storage->export_plan = *bundle->export_plan;
  storage->config = *bundle->config;
  storage->snapshot.name = name;
  storage->export_plan.name = name;
  storage->config.name = name;
  storage->bundle = (loom_target_bundle_t){
      .name = name,
      .snapshot = &storage->snapshot,
      .export_plan = &storage->export_plan,
      .config = &storage->config,
  };
}

static void loom_target_projection_apply(
    const loom_module_t* module, const loom_op_t* op,
    const loom_target_projection_t* projection,
    loom_target_bundle_storage_t* storage) {
  loom_attribute_t attr = loom_op_attrs(op)[projection->attr_index];
  if (loom_attr_is_absent(attr)) return;

  uint8_t* storage_base = (uint8_t*)storage;
  void* destination = storage_base + projection->storage_offset;
  switch (projection->value_kind) {
    case LOOM_TARGET_PROJECTION_VALUE_ENUM_U8:
      *(uint8_t*)destination = (uint8_t)loom_attr_as_enum(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32:
      *(uint32_t*)destination = (uint32_t)loom_attr_as_i64(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64:
      *(uint64_t*)destination = (uint64_t)loom_attr_as_i64(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW:
      *(iree_string_view_t*)destination =
          module->strings.entries[loom_attr_as_string_id(attr)];
      break;
  }
}

static iree_status_t loom_target_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;

  loom_target_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  facts->name = module->strings.entries[symbol->name_id];

  loom_target_like_t target =
      loom_target_like_cast(module, symbol->defining_op);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  facts->selector = loom_attr_as_enum(loom_target_like_selector(target));
  facts->row_bundle = descriptor->bundle_table->values[facts->selector];
  loom_target_bundle_storage_initialize_from_bundle(
      &facts->storage, facts->name, facts->row_bundle);
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    loom_target_projection_apply(module, target.op, &descriptor->projections[i],
                                 &facts->storage);
  }

  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_target_symbol_fact_domain = {
    .compute = loom_target_symbol_fact_compute,
};

const loom_target_symbol_facts_t* loom_target_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_target_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_target_symbol_facts_t*)facts;
}
