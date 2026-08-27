// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static const loom_amdgpu_lds_bank_service_model_t*
loom_amdgpu_memory_bank_service_model(loom_low_lower_context_t* context,
                                      const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context));
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU bank-service analysis requires AMDGPU target facts");
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  return loom_amdgpu_lds_bank_service_model_lookup(
      target_facts->properties.lds_bank_service_model_set_ordinal,
      loom_amdgpu_descriptor_ref_for_descriptor(descriptor_set, descriptor));
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

static void loom_amdgpu_memory_bank_service_mark_unknown(
    iree_string_view_t reason,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  out_report->unknown_reason = reason;
}

static void loom_amdgpu_memory_bank_service_evaluate_full_wave(
    const loom_amdgpu_lds_bank_service_model_t* model,
    const uint64_t
        lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE],
    loom_low_lower_memory_bank_service_report_t* out_report) {
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

  loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
      context, source_op, model->wave_size,
      LOOM_AMDGPU_MEMORY_LANE_SOURCE_WORKITEM_X, &active_lane_proof));
  if (!active_lane_proof.is_full_subgroup) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        active_lane_proof.unknown_reason, out_report);
    return iree_ok_status();
  }
  out_report->active_lane_proof = active_lane_proof.proof;

  // Default LDS packet selection already proved the complete address range
  // fits the 32-bit DS address domain. The target model supplies the verified
  // wave size, so these relative lane addresses cannot overflow.
  uint64_t lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE] =
      {0};
  for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
    lane_base_byte_offsets[lane] = (uint64_t)lane * byte_stride;
  }
  loom_amdgpu_memory_bank_service_evaluate_full_wave(
      model, lane_base_byte_offsets, out_report);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_fragment_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  *out_report = (loom_low_lower_memory_bank_service_report_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            packet->descriptor_ref);
  const loom_amdgpu_lds_bank_service_model_t* model =
      loom_amdgpu_memory_bank_service_model(context, descriptor);
  if (model == NULL) return iree_ok_status();
  loom_amdgpu_memory_bank_service_initialize_report(model, out_report);

  if (layout->wave_size != model->wave_size) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-layout-wave-size-mismatch"), out_report);
    return iree_ok_status();
  }
  if (!plan->dynamic_base_is_subgroup_uniform) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-dynamic-base-not-subgroup-uniform"), out_report);
    return iree_ok_status();
  }
  if (!loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
          plan, packet->register_index, element_index)) {
    loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
        context, source_op, model->wave_size,
        LOOM_AMDGPU_MEMORY_LANE_SOURCE_SUBGROUP_LANE, &active_lane_proof));
    if (active_lane_proof.is_full_subgroup) {
      out_report->active_lane_proof = active_lane_proof.proof;
    }
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-runtime-fragment-stride"), out_report);
    return iree_ok_status();
  }

  uint64_t packet_byte_offset =
      plan->address_layout.register_byte_offsets[packet->register_index];
  if (element_index != 0) {
    packet_byte_offset += (uint64_t)element_index *
                          plan->address_layout.packed_element_byte_stride;
  }
  if (plan->source.minimum_alignment < model->bank_word_byte_count) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-bank-word-alignment-unproven"), out_report);
    return iree_ok_status();
  }

  uint64_t lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE] =
      {0};
  for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
    lane_base_byte_offsets[lane] =
        packet_byte_offset +
        loom_amdgpu_fragment_memory_relative_lane_byte_offset(
            &plan->address_layout, lane);
    if (lane_base_byte_offsets[lane] % model->bank_word_byte_count != 0) {
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-bank-word-alignment-unproven"), out_report);
      return iree_ok_status();
    }
  }
  out_report->lane_address_proof =
      IREE_SV("compiled-fragment-lane-register-layout");
  out_report->base_residue_proof =
      IREE_SV("subgroup-uniform-common-translation-all-bank-word-residues");
  out_report->base_residue_count = model->bank_count;

  loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
      context, source_op, model->wave_size,
      LOOM_AMDGPU_MEMORY_LANE_SOURCE_SUBGROUP_LANE, &active_lane_proof));
  if (!active_lane_proof.is_full_subgroup) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        active_lane_proof.unknown_reason, out_report);
    return iree_ok_status();
  }
  out_report->active_lane_proof = active_lane_proof.proof;

  loom_amdgpu_memory_bank_service_evaluate_full_wave(
      model, lane_base_byte_offsets, out_report);
  return iree_ok_status();
}
