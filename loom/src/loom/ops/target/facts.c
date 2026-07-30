// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/materialization.h"

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

static iree_status_t loom_target_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;

  loom_target_like_t target =
      loom_target_like_cast(module, symbol->defining_op);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  const uint8_t selector = loom_attr_as_enum(loom_target_like_selector(target));
  const loom_target_bundle_t* row_bundle =
      loom_target_bundle_table_lookup(descriptor->bundle_table, selector);
  if (row_bundle == NULL) {
    return iree_ok_status();
  }

  loom_target_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->target = target;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  facts->name = module->strings.entries[symbol->name_id];

  facts->selector = selector;
  facts->row_bundle = row_bundle;
  const bool resolved = loom_target_record_projection_resolve(
      module, target, facts->name, &facts->storage);
  IREE_ASSERT(resolved);
  (void)resolved;

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
