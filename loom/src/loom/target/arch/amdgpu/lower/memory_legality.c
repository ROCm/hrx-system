// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"

static iree_string_view_t loom_amdgpu_cache_policy_scope_param(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_all_bits_set(policy->build_flags,
                           LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE)
             ? loom_amdgpu_cache_scope_name(policy->cache_scope)
             : IREE_SV("<missing>");
}

static iree_string_view_t loom_amdgpu_cache_policy_temporal_param(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_all_bits_set(policy->build_flags,
                           LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL)
             ? loom_amdgpu_cache_temporal_name(policy->cache_temporal)
             : IREE_SV("<missing>");
}

static iree_status_t loom_amdgpu_emit_memory_cache_policy_rejection(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  loom_diagnostic_param_t
      params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 5];
  loom_amdgpu_low_legality_make_context_params(context, op, params);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT] =
      loom_param_string(loom_amdgpu_memory_cache_policy_rejection_key(
          descriptor_set, access, policy));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 1] = loom_param_string(
      loom_amdgpu_memory_space_name(access->source.memory_space));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 2] =
      loom_param_string(loom_amdgpu_cache_policy_scope_param(policy));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 3] =
      loom_param_string(loom_amdgpu_cache_policy_temporal_param(policy));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 4] =
      loom_param_string(loom_low_descriptor_set_string(
          descriptor_set, descriptor_set->key_string_offset));
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_024_REF, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_low_legality_verify_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (op->kind == LOOM_OP_VIEW_PREFETCH) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_record_view_prefetch_diagnostic(
        context, op, descriptor_set));
    return iree_ok_status();
  }
  if (op->kind != LOOM_OP_VECTOR_LOAD && op->kind != LOOM_OP_VECTOR_STORE &&
      op->kind != LOOM_OP_VIEW_LOAD && op->kind != LOOM_OP_VIEW_STORE) {
    *out_handled = false;
    return iree_ok_status();
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_amdgpu_memory_access_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  if (!loom_amdgpu_memory_access_plan_select(
          module, loom_target_low_legality_fact_table(context), descriptor_set,
          view_regions, loom_target_low_legality_function(context), op, &source,
          &plan, &source_diagnostic, &diagnostic)) {
    bool handled = false;
    if (diagnostic.rejection_bits != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
          context, op, &source, &diagnostic, &handled));
      if (handled) {
        return iree_ok_status();
      }
    }
    const iree_string_view_t constraint_key =
        source_diagnostic.rejection_bits != 0
            ? loom_low_source_memory_access_rejection_key(
                  source_diagnostic.rejection_bits)
            : loom_amdgpu_memory_access_rejection_key(
                  diagnostic.rejection_bits);
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  for (uint32_t i = 0; i < plan.packet_count; ++i) {
    const loom_amdgpu_memory_access_t* access = &plan.packets[i].access;
    if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, access)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_record_memory_cache_policy_rejection_diagnostic(
              context, op, descriptor_set, access));
      return loom_amdgpu_emit_memory_cache_policy_rejection(
          context, op, descriptor_set, access);
    }
  }
  for (uint32_t i = 0; i < plan.packet_count; ++i) {
    const loom_amdgpu_memory_access_t* access = &plan.packets[i].access;
    const loom_amdgpu_memory_operation_kind_t kind =
        loom_amdgpu_memory_operation_kind_from_source(&access->source);
    IREE_RETURN_IF_ERROR(loom_amdgpu_record_memory_cache_policy_diagnostic(
        context, op, descriptor_set, access, kind));
    IREE_RETURN_IF_ERROR(loom_amdgpu_record_memory_access_diagnostic(
        context, op, descriptor_set, access, kind));
  }
  return iree_ok_status();
}
