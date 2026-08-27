// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/legality.h"

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

static iree_string_view_t loom_amdgpu_nonempty(iree_string_view_t value,
                                               iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

void loom_amdgpu_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  params[0] =
      loom_param_string(loom_amdgpu_nonempty(bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(
      loom_amdgpu_nonempty(bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(
      loom_amdgpu_nonempty(bundle->config->name, IREE_SV("<empty>")));
  params[3] =
      loom_param_string(loom_target_low_legality_function_name(context));
  params[4] = loom_param_string(
      loom_op_name(loom_target_low_legality_module(context), op));
}

iree_status_t loom_amdgpu_low_legality_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key) {
  loom_diagnostic_param_t
      params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 1];
  loom_amdgpu_low_legality_make_context_params(context, op, params);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT] =
      loom_param_string(constraint_key);
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_023_REF, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirement(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t constraint_key) {
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_target_low_legality_descriptor_set(context), descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirements(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_low_legality_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count) {
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, requirements[i].descriptor_ref,
        requirements[i].constraint_key));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_subgroup_wavefront(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  *out_wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(*out_wavefront_size)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_direct_subgroup_width(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    uint32_t source_wavefront_size, uint32_t required_width,
    iree_string_view_t constraint_key) {
  if (!loom_amdgpu_target_supports_direct_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          source_wavefront_size, required_width)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_full_wave_direct_subgroup_width(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t wavefront_constraint_key,
    iree_string_view_t direct_width_constraint_key,
    uint32_t* out_wavefront_size) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, wavefront_constraint_key, out_wavefront_size));
  return loom_amdgpu_low_legality_verify_direct_subgroup_width(
      context, op, *out_wavefront_size, *out_wavefront_size,
      direct_width_constraint_key);
}

bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_starts_with(bundle->config->contract_set_key,
                                      IREE_SV("amdgpu."));
}

bool loom_amdgpu_low_legality_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_AMDGPU_TARGET_STABLE_ID;
}

bool loom_amdgpu_low_legality_context_is_amdgpu(
    loom_target_low_legality_context_t* context) {
  return loom_amdgpu_low_legality_bundle_is_amdgpu(
             loom_target_low_legality_bundle(context)) ||
         loom_amdgpu_low_legality_descriptor_set_is_amdgpu(
             loom_target_low_legality_descriptor_set(context));
}
