// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/ops/target.h"

#include "loom/ir/module.h"
#include "loom/target/arch/spirv/facts.h"
#include "loom/target/arch/spirv/ops/ops.h"

iree_status_t loom_spirv_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_spirv_target_facts_t* facts =
      loom_spirv_target_facts_cast(resolved_target->facts);
  IREE_ASSERT(facts != NULL);
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "SPIR-V target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "SPIR-V target flags must follow target fact ordinals");
  static_assert(LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "SPIR-V target flags must follow target fact ordinals");

  const loom_spirv_target_build_flags_t build_flags =
      (loom_spirv_target_build_flags_t)facts->base.explicit_fields;
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.export_plan.export_symbol,
        &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->base.storage.config.contract_set_key,
        &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->base.storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &facts->base.storage.export_plan;
  const loom_target_config_t* config = &facts->base.storage.config;
  loom_op_t* target_op = NULL;
  return loom_spirv_target_build(
      builder, build_flags, (loom_spirv_target_kind_t)facts->base.selector,
      symbol, snapshot->codegen_format, snapshot->artifact_format,
      snapshot->default_pointer_bitwidth, snapshot->index_bitwidth,
      snapshot->offset_bitwidth, snapshot->max_workgroup_size.x,
      snapshot->max_workgroup_size.y, snapshot->max_workgroup_size.z,
      snapshot->max_flat_workgroup_size, snapshot->max_workgroup_storage_bytes,
      snapshot->subgroup_size, snapshot->max_grid_size.x,
      snapshot->max_grid_size.y, snapshot->max_grid_size.z,
      snapshot->max_flat_grid_size, snapshot->max_workgroup_count.x,
      snapshot->max_workgroup_count.y, snapshot->max_workgroup_count.z,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, export_plan->abi_kind, export_symbol,
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      location, &target_op);
}
