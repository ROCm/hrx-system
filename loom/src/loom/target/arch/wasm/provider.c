// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/provider.h"

#include "loom/ir/module.h"
#include "loom/target/arch/wasm/descriptors/low_registry.h"
#include "loom/target/arch/wasm/low_verify.h"
#include "loom/target/arch/wasm/math_policy.h"
#include "loom/target/arch/wasm/ops/ops.h"
#include "loom/target/arch/wasm/ops/registry.h"
#include "loom/target/arch/wasm/records/target_records.h"
#include "loom/target/emit/wasm/lower/lower.h"

static iree_status_t loom_wasm_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  out_facts->selector = LOOM_WASM_TARGET_KIND_SIMD128;
  return iree_ok_status();
}

static const loom_target_profile_type_t kProfileType = {
    .name = IREE_SVL("wasm"),
    .fact_type = &loom_wasm_target_fact_type,
    .project_facts = loom_wasm_profile_project_facts,
};

static const loom_target_profile_t kSimd128Profile = {
    .type = &kProfileType,
    .target_bundle = &loom_wasm_low_target_bundle_core_simd128,
};

static iree_status_t loom_wasm_select_profile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (!iree_string_view_equal(selector, IREE_SV("simd128"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown Wasm target profile '%.*s'",
                            (int)selector.size, selector.data);
  }
  *out_profile = &kSimd128Profile;
  return iree_ok_status();
}

static iree_status_t loom_wasm_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_target_facts_t* facts = resolved_target->facts;
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "wasm target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_WASM_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "wasm target flags must follow target fact ordinals");
  static_assert(LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "wasm target flags must follow target fact ordinals");

  const loom_wasm_target_build_flags_t build_flags =
      (loom_wasm_target_build_flags_t)facts->explicit_fields;
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_WASM_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.export_plan.export_symbol, &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.config.contract_set_key, &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->storage.snapshot;
  const loom_target_export_plan_t* export_plan = &facts->storage.export_plan;
  const loom_target_config_t* config = &facts->storage.config;
  loom_op_t* target_op = NULL;
  return loom_wasm_target_build(
      builder, build_flags, (loom_wasm_target_kind_t)facts->selector, symbol,
      snapshot->codegen_format, snapshot->artifact_format,
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

static const loom_low_verify_provider_t* const kLoomWasmLowVerifyProviders[] = {
    &loom_wasm_low_verify_provider,
};

const loom_target_provider_t loom_wasm_target_provider = {
    .profile_type = &kProfileType,
    .select_profile = loom_wasm_select_profile,
    .materialize_definition = loom_wasm_materialize_definition,
    .select_low_call_policy = loom_target_select_low_call_policy_direct,
    .register_context = loom_wasm_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_wasm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_wasm_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_wasm_math_policy_registry_initialize,
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomWasmLowVerifyProviders),
            .values = kLoomWasmLowVerifyProviders,
        },
    .target_fact_type = &loom_wasm_target_fact_type,
};

static const loom_target_provider_t* const kLoomWasmTargetProviders[] = {
    &loom_wasm_target_provider,
};

const loom_target_provider_set_t loom_wasm_target_provider_set = {
    .providers = kLoomWasmTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomWasmTargetProviders),
};
