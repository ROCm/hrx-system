// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/provider.h"

#include "loom/ir/module.h"
#include "loom/ops/callable_effects.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/diagnostics/packet_diagnostics.h"
#include "loom/target/arch/amdgpu/legalization.h"
#include "loom/target/arch/amdgpu/low_asm_diagnostics.h"
#include "loom/target/arch/amdgpu/low_verify.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/math_policy.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/pass_registry.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/transforms/pipeline/workgroup.h"
#include "loom/transforms/scalar/wide_division_legalization.h"

static const loom_target_low_legality_provider_t*
    kLoomAmdgpuLowLegalityProviders[] = {
        &loom_amdgpu_low_legality_provider_storage,
};

static const loom_target_legalizer_provider_t* kLoomAmdgpuLegalizerProviders[] =
    {
        &loom_amdgpu_target_legalizer_provider_storage,
        &loom_scalar_wide_division_legalizer_provider_storage,
};

static const loom_target_low_packet_diagnostic_provider_t*
    kLoomAmdgpuLowPacketDiagnosticProviders[] = {
        &loom_amdgpu_low_packet_diagnostic_provider_storage,
};

static const loom_target_low_asm_diagnostic_provider_t*
    kLoomAmdgpuLowAsmDiagnosticProviders[] = {
        &loom_amdgpu_low_asm_diagnostic_provider,
};

static const loom_low_verify_provider_t* kLoomAmdgpuLowVerifyProviders[] = {
    &loom_amdgpu_low_verify_provider,
};

static iree_status_t loom_amdgpu_provider_build_string_attr(
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

static iree_status_t loom_amdgpu_provider_build_run_pass(
    loom_builder_t* builder, iree_string_view_t key) {
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, key, loom_named_attr_slice_empty(),
                                &run_op);
}

static iree_status_t loom_amdgpu_provider_build_hal_buffer_descriptors_pass(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_amdgpu_provider_build_run_pass(
      builder, IREE_SV("amdgpu-materialize-hal-buffer-descriptors"));
}

static iree_status_t loom_amdgpu_provider_build_hal_kernel_abi_pass(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_amdgpu_provider_build_run_pass(
      builder, IREE_SV("amdgpu-materialize-hal-kernel-abi"));
}

static iree_status_t loom_amdgpu_provider_build_cfg_converge_pass(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_amdgpu_provider_build_run_pass(builder, IREE_SV("cfg-converge"));
}

static iree_status_t loom_amdgpu_provider_build_source_finalization(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      builder, IREE_SV("family"), IREE_SV("amdgpu"), &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      builder, IREE_SV("codegen"), IREE_SV("low_native"), &attrs[1]));
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS, IREE_SV("target"),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      loom_amdgpu_provider_build_cfg_converge_pass, NULL, &where_op);
}

static iree_status_t loom_amdgpu_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase ==
      LOOM_TARGET_PIPELINE_PHASE_SOURCE_EXECUTION_LOWERING) {
    return loom_pipeline_workgroup_build_pipeline(contribution->builder,
                                                  IREE_SV("amdgpu"));
  }
  loom_pass_ir_body_build_fn_t build_body = NULL;
  if (contribution->phase == LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW) {
    // Shared source preparation has finished threading branch conditions.
    // Masked lowering needs explicit convergence for the shared alternatives
    // that those valid condition proofs can expose.
    loom_op_t* for_op = NULL;
    return loom_pass_ir_build_for(
        contribution->builder, LOOM_PASS_ANCHOR_FUNC,
        loom_amdgpu_provider_build_source_finalization, NULL, &for_op);
  } else if (contribution->phase ==
             LOOM_TARGET_PIPELINE_PHASE_SOURCE_LOW_ARTIFACT_PREPARATION) {
    build_body = loom_amdgpu_provider_build_hal_buffer_descriptors_pass;
  } else if (contribution->phase ==
             LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION) {
    build_body = loom_amdgpu_provider_build_hal_kernel_abi_pass;
  } else {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[3] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("family"), IREE_SV("amdgpu"), &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("codegen"), IREE_SV("low_native"),
      &attrs[1]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("abi"), IREE_SV("hal_kernel"), &attrs[2]));

  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      contribution->builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS,
      IREE_SV("target"),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), build_body,
      NULL, &where_op);
}

static iree_status_t loom_amdgpu_provider_select_profile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  const loom_amdgpu_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_select(selector, &profile));
  *out_profile = profile ? &profile->base : NULL;
  return iree_ok_status();
}

static loom_target_call_policy_t loom_amdgpu_select_call_policy(
    const loom_resolved_target_t* resolved_target, const loom_module_t* module,
    loom_call_like_kind_t kind, loom_call_like_t call,
    loom_func_like_t callee) {
  (void)resolved_target;
  if (kind != LOOM_CALL_LIKE_KIND_SEMANTIC &&
      kind != LOOM_CALL_LIKE_KIND_LOW_INTERNAL) {
    return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
  }
  if (kind == LOOM_CALL_LIKE_KIND_SEMANTIC &&
      loom_callable_effects_may_create_resources(callee)) {
    // Helpers without resource creation return borrowed references, including
    // helpers that write to caller-owned storage. Other reference returns stay
    // semantic until inlining exposes their allocation and lifetime effects.
    // Apply the same policy before and after view-to-buffer projection.
    const loom_value_id_t* results = loom_op_const_results(callee.op);
    for (uint16_t i = 0; i < callee.op->result_count; ++i) {
      const loom_type_t result_type =
          loom_module_value_type(module, results[i]);
      if (loom_type_is_buffer(result_type) || loom_type_is_view(result_type)) {
        return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
      }
    }
  }
  const bool explicit_noinline =
      loom_call_like_inline_policy(call) == LOOM_INLINE_POLICY_NOINLINE ||
      loom_func_like_inline_policy(callee) == LOOM_INLINE_POLICY_NOINLINE;
  return explicit_noinline ? LOOM_TARGET_CALL_POLICY_DIRECT
                           : LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
}

const loom_target_provider_t loom_amdgpu_target_provider = {
    .profile_type = &loom_amdgpu_target_profile_type,
    .materialize_definition = loom_amdgpu_target_materialize_definition,
    .select_call_policy = loom_amdgpu_select_call_policy,
    .view_boundary_carrier = LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET,
    .loop_predicate_carrier = LOOM_TARGET_LOOP_PREDICATE_CARRIER_I32,
    .loop_predicate_max_vector_element_count =
        LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES,
    .register_context = loom_amdgpu_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_amdgpu_math_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowLegalityProviders),
            .values = kLoomAmdgpuLowLegalityProviders,
        },
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLegalizerProviders),
            .values = kLoomAmdgpuLegalizerProviders,
        },
    .low_packet_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowPacketDiagnosticProviders),
            .values = kLoomAmdgpuLowPacketDiagnosticProviders,
        },
    .low_asm_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowAsmDiagnosticProviders),
            .values = kLoomAmdgpuLowAsmDiagnosticProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowVerifyProviders),
            .values = kLoomAmdgpuLowVerifyProviders,
        },
    .pass_registry = &loom_amdgpu_pass_registry,
    .contribute_pipeline = loom_amdgpu_provider_contribute_pipeline,
    .target_fact_type = &loom_amdgpu_target_fact_type,
    .select_profile = loom_amdgpu_provider_select_profile,
};

static const loom_target_provider_t* const kLoomAmdgpuTargetProviders[] = {
    &loom_amdgpu_target_provider,
};

const loom_target_provider_set_t loom_amdgpu_target_provider_set = {
    .providers = kLoomAmdgpuTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomAmdgpuTargetProviders),
};
