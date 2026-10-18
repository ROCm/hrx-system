// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization.h"
#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/registry.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/target.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/pass.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"

static iree_status_t loom_aie2p_target_provider_select_profile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  const loom_aie2p_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_target_profile_select(selector, &profile));
  *out_profile = profile ? &profile->base : NULL;
  return iree_ok_status();
}

static const loom_low_verify_provider_t* const kAie2pLowVerifyProviders[] = {
    &loom_aie2p_low_verify_provider,
};

static const loom_target_legalizer_provider_t* const
    kAie2pLegalizerProviders[] = {
        &loom_aie2p_target_legalizer_provider_storage,
};

static iree_status_t loom_aie2p_provider_build_string_attr(
    loom_builder_t* builder, iree_string_view_t name, iree_string_view_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, name, &name_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, value, &value_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_string(value_id),
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_provider_build_kernel_realization_run(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, IREE_SV("aie2p-realize-kernel"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_aie2p_provider_build_pipeline_lower_run(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, IREE_SV("aie2p-lower-pipeline"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_aie2p_provider_build_kernel_target_where(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_named_attr_t contract_attr = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_provider_build_string_attr(
      builder, IREE_SV("contract"), IREE_SV("amd.xdna.aie2p.array"),
      &contract_attr));
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS, IREE_SV("target"),
      loom_make_named_attr_slice(&contract_attr, 1),
      loom_aie2p_provider_build_kernel_realization_run, NULL, &where_op);
}

static iree_status_t loom_aie2p_provider_build_pipeline_target_where(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_named_attr_t contract_attr = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_provider_build_string_attr(
      builder, IREE_SV("contract"), IREE_SV("amd.xdna.aie2p.array"),
      &contract_attr));
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS, IREE_SV("target"),
      loom_make_named_attr_slice(&contract_attr, 1),
      loom_aie2p_provider_build_pipeline_lower_run, NULL, &where_op);
}

static loom_target_call_policy_t loom_aie2p_select_call_policy(
    const loom_resolved_target_t* resolved_target, const loom_module_t* module,
    loom_call_like_kind_t kind, loom_call_like_t call,
    loom_func_like_t callee) {
  (void)resolved_target;
  (void)call;
  if ((kind != LOOM_CALL_LIKE_KIND_SEMANTIC &&
       kind != LOOM_CALL_LIKE_KIND_LOW_INTERNAL) ||
      !module || !loom_func_like_isa(callee) ||
      loom_func_like_inline_policy(callee) != LOOM_INLINE_POLICY_NOINLINE ||
      !loom_func_like_is_module_internal(callee) ||
      !loom_func_like_body(callee)) {
    return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
  }

  loom_aie2p_call_abi_requirements_t argument_requirements = {0};
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(callee, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, arguments[i]);
    const bool supported =
        kind == LOOM_CALL_LIKE_KIND_SEMANTIC
            ? loom_aie2p_call_abi_add_source_type(type, &argument_requirements)
            : loom_aie2p_call_abi_add_low_type(type, &argument_requirements);
    if (!supported) {
      return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
    }
  }
  if (!loom_aie2p_call_abi_requirements_fit(argument_requirements)) {
    return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
  }

  loom_aie2p_call_abi_requirements_t result_requirements = {0};
  const loom_value_id_t* results = loom_op_const_results(callee.op);
  for (uint16_t i = 0; i < callee.op->result_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, results[i]);
    const bool supported =
        kind == LOOM_CALL_LIKE_KIND_SEMANTIC
            ? loom_aie2p_call_abi_add_source_type(type, &result_requirements)
            : loom_aie2p_call_abi_add_low_type(type, &result_requirements);
    if (!supported) {
      return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
    }
  }
  return loom_aie2p_call_abi_requirements_fit(result_requirements)
             ? LOOM_TARGET_CALL_POLICY_DIRECT
             : LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
}

static iree_status_t loom_aie2p_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  loom_op_t* for_op = NULL;
  switch (contribution->phase) {
    case LOOM_TARGET_PIPELINE_PHASE_SOURCE_EXECUTION_REALIZATION:
      return loom_pass_ir_build_for(
          contribution->builder, LOOM_PASS_ANCHOR_FUNC,
          loom_aie2p_provider_build_kernel_target_where, NULL, &for_op);
    case LOOM_TARGET_PIPELINE_PHASE_SOURCE_EXECUTION_LOWERING:
      return loom_pass_ir_build_for(
          contribution->builder, LOOM_PASS_ANCHOR_FUNC,
          loom_aie2p_provider_build_pipeline_target_where, NULL, &for_op);
    default:
      return iree_ok_status();
  }
}

const loom_target_provider_t loom_aie2p_target_provider = {
    .profile_type = &loom_aie2p_target_profile_type,
    .select_call_policy = loom_aie2p_select_call_policy,
    .materialize_definition = loom_aie2p_target_materialize_definition,
    .register_context = loom_aie2p_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_aie2p_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_aie2p_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_aie2p_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLegalizerProviders),
            .values = kAie2pLegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLowVerifyProviders),
            .values = kAie2pLowVerifyProviders,
        },
    .pass_registry = &loom_aie2p_pipeline_pass_registry,
    .contribute_pipeline = loom_aie2p_provider_contribute_pipeline,
    .select_profile = loom_aie2p_target_provider_select_profile,
    .target_fact_type = &loom_aie2p_target_fact_type,
};

static const loom_target_provider_t* const kAie2pTargetProviders[] = {
    &loom_aie2p_target_provider,
};

const loom_target_provider_set_t loom_aie2p_target_provider_set = {
    .providers = kAie2pTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pTargetProviders),
};
