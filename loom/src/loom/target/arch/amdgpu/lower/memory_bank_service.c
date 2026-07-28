// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/control_uniformity.h"
#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"

typedef struct loom_amdgpu_memory_bank_service_state_t {
  // Reusable execution-uniformity analysis for the source function.
  loom_control_uniformity_info_t control_uniformity;
} loom_amdgpu_memory_bank_service_state_t;

static int loom_amdgpu_memory_bank_service_state_key;

static const loom_amdgpu_lds_bank_service_model_t*
loom_amdgpu_memory_bank_service_model(loom_low_lower_context_t* context,
                                      const loom_low_descriptor_t* descriptor) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (loom_amdgpu_target_gfx1250_revision_from_ref(
          module, loom_low_lower_context_target_ref(context)) ==
      LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED) {
    return NULL;
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  switch (
      loom_amdgpu_descriptor_ref_for_descriptor(descriptor_set, descriptor)) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128:
      return loom_amdgpu_lds_bank_service_gfx1250_b128_model(
          LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ);
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128:
      return loom_amdgpu_lds_bank_service_gfx1250_b128_model(
          LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE);
    default:
      return NULL;
  }
}

static void loom_amdgpu_memory_bank_service_initialize_report(
    const loom_amdgpu_lds_bank_service_model_t* model,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  *out_report = (loom_low_lower_memory_bank_service_report_t){
      .proof = IREE_SVL("unknown"),
      .model_key = model->key,
      .model_revision = model->revision,
      .model_evidence = loom_amdgpu_lds_bank_service_evidence_class_name(
          model->evidence_class),
      .request_policy = loom_amdgpu_lds_bank_service_request_policy_name(
          model->request_policy),
      .lane_address_proof = IREE_SVL("unproven"),
      .active_lane_proof = IREE_SVL("unproven"),
      .base_residue_proof = IREE_SVL("unproven"),
      .wave_size = model->wave_size,
      .bank_count = model->bank_count,
      .bank_word_byte_count = model->bank_word_byte_count,
      .packet_word_count = model->packet_word_count,
      .phase_count = model->phase_count,
  };
  for (uint8_t phase = 0; phase < model->phase_count; ++phase) {
    out_report->phase_lane_counts[phase] =
        (uint8_t)iree_math_count_ones_u64(model->phase_lane_masks[phase]);
  }
}

static iree_status_t loom_amdgpu_memory_bank_service_control_uniformity(
    loom_low_lower_context_t* context,
    loom_control_uniformity_info_t** out_control_uniformity) {
  loom_amdgpu_memory_bank_service_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_memory_bank_service_state_key, sizeof(*state),
      (void**)&state));
  if (state->control_uniformity.module == NULL) {
    loom_control_uniformity_info_initialize(
        loom_low_lower_context_module(context),
        loom_low_lower_context_fact_table(context),
        loom_low_lower_context_scratch_arena(context),
        &state->control_uniformity);
  }
  *out_control_uniformity = &state->control_uniformity;
  return iree_ok_status();
}

static void loom_amdgpu_memory_bank_service_mark_unknown(
    iree_string_view_t reason,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  out_report->unknown_reason = reason;
}

iree_status_t loom_amdgpu_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  *out_report = (loom_low_lower_memory_bank_service_report_t){0};
  const loom_amdgpu_lds_bank_service_model_t* model =
      loom_amdgpu_memory_bank_service_model(context, descriptor);
  if (model == NULL) return iree_ok_status();
  loom_amdgpu_memory_bank_service_initialize_report(model, out_report);

  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(source);
  if (term == NULL) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-dynamic-term-count"), out_report);
    return iree_ok_status();
  }
  if (term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      term->dimension != LOOM_KERNEL_DIMENSION_X) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-not-workitem-x"), out_report);
    return iree_ok_status();
  }
  if (term->stride_value_count != 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-dynamic-stride"), out_report);
    return iree_ok_status();
  }
  if (term->byte_stride <= 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-nonpositive-stride"), out_report);
    return iree_ok_status();
  }
  const uint64_t byte_stride = (uint64_t)term->byte_stride;
  if (source->minimum_alignment < model->bank_word_byte_count ||
      byte_stride % model->bank_word_byte_count != 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-bank-word-alignment-unproven"), out_report);
    return iree_ok_status();
  }
  out_report->lane_address_proof = IREE_SV("affine-workitem-x-byte-stride");
  out_report->base_residue_proof =
      IREE_SV("all-bank-word-residues-common-translation");
  out_report->base_residue_count = model->bank_count;

  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_amdgpu_required_workgroup_size_from_facts(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context), bundle,
          loom_low_lower_context_fact_table(context), &workgroup_size) ||
      workgroup_size.x == 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("active-lane-workgroup-size-unknown"), out_report);
    return iree_ok_status();
  }
  if (workgroup_size.x % model->wave_size != 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("active-lane-workitem-x-wrap"), out_report);
    return iree_ok_status();
  }

  loom_control_uniformity_info_t* control_uniformity = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_bank_service_control_uniformity(
      context, &control_uniformity));
  loom_control_uniformity_failure_t failure = {0};
  bool control_is_uniform = false;
  IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_execution(
      control_uniformity, source_op, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP,
      &failure, &control_is_uniform));
  if (!control_is_uniform) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("active-lane-control-not-uniform"), out_report);
    return iree_ok_status();
  }
  out_report->active_lane_proof = IREE_SV("subgroup-uniform-control-full-wave");

  // Default LDS packet selection already proved the complete address range
  // fits the 32-bit DS address domain. The target model supplies the verified
  // gfx1250 wave size, so these relative lane addresses cannot overflow.
  uint64_t lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE] =
      {0};
  for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
    lane_base_byte_offsets[lane] = (uint64_t)lane * byte_stride;
  }
  const uint64_t active_lane_mask =
      model->wave_size == 64 ? UINT64_MAX
                             : (UINT64_C(1) << model->wave_size) - UINT64_C(1);
  loom_amdgpu_lds_bank_service_result_t result = {0};
  loom_amdgpu_lds_bank_service_evaluate(model, active_lane_mask,
                                        lane_base_byte_offsets, &result);

  out_report->proof = IREE_SV("exact");
  out_report->classification = result.extra_rounds == 0
                                   ? IREE_SV("conflict-free")
                                   : IREE_SV("conflicted");
  out_report->unknown_reason = iree_string_view_empty();
  for (uint8_t phase = 0; phase < result.phase_count; ++phase) {
    out_report->phase_required_rounds[phase] =
        result.phase_required_rounds[phase];
  }
  out_report->required_rounds = result.required_rounds;
  out_report->uncontended_rounds = result.uncontended_rounds;
  out_report->extra_rounds = result.extra_rounds;
  out_report->maximum_request_multiplicity =
      result.maximum_request_multiplicity;
  return iree_ok_status();
}
