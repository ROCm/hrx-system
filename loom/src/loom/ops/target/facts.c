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
  const loom_target_fact_type_t* fact_type = descriptor->fact_type;
  IREE_ASSERT(fact_type != NULL);
  IREE_ASSERT(fact_type->storage_size >= sizeof(loom_target_facts_t));

  loom_target_facts_t* projection = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, fact_type->storage_size, (void**)&projection));
  memset(projection, 0, fact_type->storage_size);
  projection->fact_type = fact_type;
  projection->selector = selector;
  const bool resolved = loom_target_record_projection_resolve(
      module, target, module->strings.entries[symbol->name_id],
      &projection->storage, &projection->authored_attrs);
  IREE_ASSERT(resolved);
  (void)resolved;
  if (fact_type->project != NULL) {
    fact_type->project(module, target.op, projection);
  }

  loom_target_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->projection = projection;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  facts->name = module->strings.entries[symbol->name_id];

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

bool loom_target_facts_satisfy_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  if (effective == requirement) {
    return true;
  }
  if (effective->fact_type != requirement->fact_type ||
      effective->fact_type->satisfies_requirement == NULL) {
    return false;
  }
  return effective->fact_type->satisfies_requirement(effective, requirement);
}

bool loom_target_facts_structural_satisfy_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement) {
  IREE_ASSERT_ARGUMENT(effective);
  IREE_ASSERT_ARGUMENT(requirement);
  if (effective->fact_type != requirement->fact_type ||
      effective->selector != requirement->selector) {
    return false;
  }
  return loom_target_snapshot_satisfies_requirement(
             &effective->storage.snapshot, &requirement->storage.snapshot) &&
         iree_string_view_equal(effective->storage.config.contract_set_key,
                                requirement->storage.config.contract_set_key) &&
         iree_all_bits_set(effective->storage.config.contract_feature_bits,
                           requirement->storage.config.contract_feature_bits);
}

static bool loom_target_limit_satisfies(uint64_t effective_limit,
                                        uint64_t required_limit) {
  return required_limit == 0 || effective_limit >= required_limit;
}

bool loom_target_snapshot_satisfies_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement) {
  IREE_ASSERT_ARGUMENT(effective_snapshot);
  IREE_ASSERT_ARGUMENT(target_requirement);
  return effective_snapshot->codegen_format ==
             target_requirement->codegen_format &&
         effective_snapshot->artifact_format ==
             target_requirement->artifact_format &&
         effective_snapshot->default_pointer_bitwidth ==
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
