// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

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
      .packet_byte_count = model->packet_byte_count,
      .phase_count = model->phase_count,
  };
  for (uint8_t phase = 0; phase < model->phase_count; ++phase) {
    out_report->phase_lane_counts[phase] =
        (uint8_t)iree_math_count_ones_u64(model->phase_lane_masks[phase]);
  }
}

static const loom_amdgpu_lds_bank_service_model_t*
loom_amdgpu_memory_bank_service_prepare_report(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint8_t wave_size,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context));
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU bank-service analysis requires AMDGPU target facts");
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_lds_bank_service_model_t* model =
      loom_amdgpu_lds_bank_service_model_lookup(
          target_facts->properties.lds_bank_service_model_set_ordinal,
          loom_amdgpu_descriptor_ref_for_descriptor(descriptor_set, descriptor),
          wave_size);
  if (model == NULL) {
    out_report->proof = IREE_SV("unmodeled");
    out_report->wave_size = wave_size;
    out_report->unknown_reason =
        target_facts->properties.lds_bank_service_model_set_ordinal ==
                LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE
            ? IREE_SV("target-model-unavailable")
            : IREE_SV("packet-wave-model-unavailable");
  } else {
    loom_amdgpu_memory_bank_service_initialize_report(model, out_report);
  }
  return model;
}

static void loom_amdgpu_memory_bank_service_mark_unknown(
    iree_string_view_t reason,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  out_report->unknown_reason = reason;
}

// A common word-sized translation only rotates bank indices. Retain the
// subword residue using root alignment, exact offsets, and dynamic
// divisibility. The source access alignment also constrains the combined origin
// address.
static uint64_t loom_amdgpu_memory_bank_service_source_residues(
    const loom_low_source_memory_access_plan_t* source,
    loom_value_facts_t dynamic_offset, uint8_t bank_word_byte_count) {
  uint64_t translation = (uint64_t)source->static_byte_offset;
  uint64_t step = iree_math_gcd_i64(
      bank_word_byte_count, iree_max(source->root_minimum_alignment, 1u));
  if (dynamic_offset.range_lo == dynamic_offset.range_hi) {
    translation += (uint64_t)dynamic_offset.range_lo;
  } else {
    step = iree_math_gcd_i64((int64_t)step, dynamic_offset.known_divisor);
  }
  const uint64_t access_alignment = iree_math_gcd_i64(
      bank_word_byte_count, iree_max(source->minimum_alignment, 1u));
  uint64_t residues = 0;
  for (uint64_t residue = 0; residue < bank_word_byte_count; residue += step) {
    const uint64_t origin = (residue + translation) % bank_word_byte_count;
    if (origin % access_alignment == 0) {
      residues |= UINT64_C(1) << origin;
    }
  }
  return residues;
}

static uint64_t loom_amdgpu_memory_bank_service_add_offset_residues(
    uint64_t source_residues, loom_value_facts_t offset,
    uint8_t bank_word_byte_count) {
  const bool exact = offset.range_lo == offset.range_hi;
  const uint64_t first =
      exact ? (uint64_t)offset.range_lo % bank_word_byte_count : 0;
  const uint64_t step =
      exact ? bank_word_byte_count
            : iree_math_gcd_i64(bank_word_byte_count, offset.known_divisor);
  uint64_t residues = 0;
  for (uint8_t source = 0; source < bank_word_byte_count; ++source) {
    if ((source_residues & (UINT64_C(1) << source)) == 0) {
      continue;
    }
    for (uint64_t added = first; added < bank_word_byte_count; added += step) {
      residues |= UINT64_C(1) << ((source + added) % bank_word_byte_count);
    }
  }
  return residues;
}

static void loom_amdgpu_memory_bank_service_record_result(
    const loom_amdgpu_lds_bank_service_result_t* result,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  out_report->base_residue_count = result->base_residue_count;
  if (result->base_residue_count > out_report->bank_count) {
    out_report->base_residue_proof =
        IREE_SV("all-compatible-byte-residues-common-translation");
  }

  out_report->proof = IREE_SV("exact");
  out_report->classification = result->extra_rounds == 0
                                   ? IREE_SV("conflict-free")
                                   : IREE_SV("conflicted");
  out_report->unknown_reason = iree_string_view_empty();
  for (uint8_t phase = 0; phase < result->phase_count; ++phase) {
    out_report->phase_required_rounds[phase] =
        result->phase_required_rounds[phase];
  }
  out_report->required_rounds = result->required_rounds;
  out_report->uncontended_rounds = result->uncontended_rounds;
  out_report->extra_rounds = result->extra_rounds;
  out_report->maximum_request_multiplicity =
      result->maximum_request_multiplicity;
}

static void loom_amdgpu_memory_bank_service_evaluate_full_wave(
    const loom_amdgpu_lds_bank_service_model_t* model,
    const uint64_t
        lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE],
    uint64_t common_base_byte_residues,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  const uint64_t active_lane_mask =
      model->wave_size == 64 ? UINT64_MAX
                             : (UINT64_C(1) << model->wave_size) - UINT64_C(1);
  loom_amdgpu_lds_bank_service_result_t result = {0};
  if (!loom_amdgpu_lds_bank_service_evaluate(
          model, active_lane_mask, lane_base_byte_offsets,
          common_base_byte_residues, &result)) {
    out_report->base_residue_proof = IREE_SV("unproven");
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-base-residue-unproven"), out_report);
    return;
  }
  loom_amdgpu_memory_bank_service_record_result(&result, out_report);
}

void loom_amdgpu_memory_calculate_source_bank_service(
    const loom_amdgpu_lds_bank_service_model_t* model,
    const loom_low_source_memory_access_plan_t* source,
    const loom_symbolic_expr_context_t* expressions,
    const loom_target_workgroup_size_t* workgroup_size,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  loom_amdgpu_memory_bank_service_initialize_report(model, out_report);
  if (source->root_uniform_scope < LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-root-not-subgroup-uniform"), out_report);
    return;
  }

  uint64_t coordinate_byte_strides[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  struct {
    // Shared producer's exact numeric function of one workitem coordinate.
    const loom_symbolic_projection_t* projection;
    // Physical byte coefficient supplied by the canonical source plan.
    uint64_t byte_stride;
    // Native workitem coordinate selected by the projection's root facts.
    uint8_t dimension;
  } projected_terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  uint8_t projected_term_count = 0;
  loom_value_facts_t common_offset = loom_value_facts_exact_i64(0);
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    if (loom_value_facts_is_subgroup_uniform(term->byte_facts)) {
      loom_value_facts_addi(&common_offset, &term->byte_facts, &common_offset);
      continue;
    }
    loom_symbolic_expr_summary_t summary = {0};
    const loom_value_fact_topology_domain_t* domain = NULL;
    int64_t byte_stride = term->byte_stride;
    uint8_t dimension = (uint8_t)term->dimension;
    if (term->source !=
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID) {
      // A retained coordinate may materialize an affine wrapper around a digit
      // or workitem ID. Consume that shared summary without changing the
      // address plan's chosen SSA value or walking its producers.
      if (loom_symbolic_expr_context_try_lookup_summary(
              expressions, term->index, &summary) &&
          loom_symbolic_expr_is_linear(&summary.expression) &&
          summary.expression.term_count == 1 &&
          summary.expression.terms[0].coefficient > 0) {
        const loom_symbolic_term_t* coordinate = summary.expression.terms;
        int64_t constant_byte_offset = 0;
        if (!iree_checked_mul_i64(summary.expression.constant, byte_stride,
                                  &constant_byte_offset) ||
            !iree_checked_mul_i64(coordinate->coefficient, byte_stride,
                                  &byte_stride)) {
          loom_amdgpu_memory_bank_service_mark_unknown(
              IREE_SV("address-varying-term-unproven"), out_report);
          return;
        }
        const loom_value_facts_t translation =
            loom_value_facts_exact_i64(constant_byte_offset);
        loom_value_facts_addi(&common_offset, &translation, &common_offset);
        domain = loom_value_facts_topology_domain(loom_value_fact_table_lookup(
            expressions->fact_table, coordinate->value_id));
        if (!domain ||
            domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID) {
          if (loom_symbolic_expr_context_try_lookup_summary(
                  expressions, coordinate->value_id, &summary) &&
              summary.projection) {
            domain =
                loom_value_facts_topology_domain(loom_value_fact_table_lookup(
                    expressions->fact_table, summary.projection->value_id));
          }
        }
      }
      if (!domain ||
          domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID) {
        loom_amdgpu_memory_bank_service_mark_unknown(
            IREE_SV("address-varying-term-unproven"), out_report);
        return;
      }
      dimension = (uint8_t)domain->axis;
    }
    if (term->stride_value_count != 0) {
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-dynamic-stride"), out_report);
      return;
    }
    if (byte_stride < 0) {
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-negative-stride"), out_report);
      return;
    }
    if (!summary.projection) {
      coordinate_byte_strides[dimension] += (uint64_t)byte_stride;
      continue;
    }
    if ((uint64_t)byte_stride % model->packet_byte_count != 0) {
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-packet-alignment-unproven"), out_report);
      return;
    }
    projected_terms[projected_term_count].projection = summary.projection;
    projected_terms[projected_term_count].byte_stride = (uint64_t)byte_stride;
    projected_terms[projected_term_count].dimension = dimension;
    ++projected_term_count;
  }
  const uint32_t packet_alignment = model->packet_byte_count;
  if (source->minimum_alignment < packet_alignment ||
      coordinate_byte_strides[0] % packet_alignment != 0 ||
      coordinate_byte_strides[1] % packet_alignment != 0 ||
      coordinate_byte_strides[2] % packet_alignment != 0) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-packet-alignment-unproven"), out_report);
    return;
  }
  out_report->lane_address_proof =
      projected_term_count != 0
          ? IREE_SV("canonical-workitem-digits-all-waves")
          : IREE_SV("canonical-workitem-coordinates-all-waves");
  out_report->base_residue_proof =
      IREE_SV("subgroup-uniform-common-translation-all-bank-word-residues");

  const uint64_t common_base_byte_residues =
      loom_amdgpu_memory_bank_service_source_residues(
          source, common_offset, model->bank_word_byte_count);
  const uint64_t active_lane_mask =
      model->wave_size == 64 ? UINT64_MAX
                             : (UINT64_C(1) << model->wave_size) - UINT64_C(1);
  const uint32_t plane_size = workgroup_size->x * workgroup_size->y;
  const uint32_t flat_size = plane_size * workgroup_size->z;
  loom_amdgpu_lds_bank_service_result_t result = {0};
  for (uint32_t wave_begin = 0; wave_begin < flat_size;
       wave_begin += model->wave_size) {
    uint64_t lane_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE] = {0};
    for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
      const uint32_t linear_id = wave_begin + lane;
      const uint32_t x = linear_id % workgroup_size->x;
      const uint32_t y = (linear_id / workgroup_size->x) % workgroup_size->y;
      const uint32_t z = linear_id / plane_size;
      // The selected DS packet already proves a 32-bit address envelope.
      // These nonnegative canonical contributions cannot overflow it.
      lane_offsets[lane] = x * coordinate_byte_strides[0] +
                           y * coordinate_byte_strides[1] +
                           z * coordinate_byte_strides[2];
      const uint32_t coordinates[] = {x, y, z};
      for (uint8_t i = 0; i < projected_term_count; ++i) {
        const loom_symbolic_projection_t* projection =
            projected_terms[i].projection;
        uint64_t digit = ((uint64_t)coordinates[projected_terms[i].dimension] *
                              (uint64_t)projection->scale +
                          (uint64_t)projection->offset) /
                         (uint64_t)projection->divisor;
        if (projection->modulus != 0) {
          digit %= (uint64_t)projection->modulus;
        }
        lane_offsets[lane] += digit * projected_terms[i].byte_stride;
      }
    }
    loom_amdgpu_lds_bank_service_result_t candidate = {0};
    if (!loom_amdgpu_lds_bank_service_evaluate(
            model, active_lane_mask, lane_offsets, common_base_byte_residues,
            &candidate)) {
      out_report->base_residue_proof = IREE_SV("unproven");
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-base-residue-unproven"), out_report);
      return;
    }
    if (wave_begin == 0) {
      result = candidate;
    } else {
      for (uint8_t phase = 0; phase < model->phase_count; ++phase) {
        if (candidate.phase_required_rounds[phase] !=
            result.phase_required_rounds[phase]) {
          loom_amdgpu_memory_bank_service_mark_unknown(
              IREE_SV("address-wave-profiles-differ"), out_report);
          return;
        }
      }
    }
  }
  loom_amdgpu_memory_bank_service_record_result(&result, out_report);
}

iree_status_t loom_amdgpu_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  *out_report = (loom_low_lower_memory_bank_service_report_t){0};
  if (source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  const uint8_t wave_size =
      loom_low_lower_context_bundle(context)->snapshot->subgroup_size;
  const loom_amdgpu_lds_bank_service_model_t* model =
      loom_amdgpu_memory_bank_service_prepare_report(context, descriptor,
                                                     wave_size, out_report);
  if (model == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
      context, source_op, model->wave_size, &active_lane_proof));
  if (!active_lane_proof.is_full_subgroup) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        active_lane_proof.unknown_reason, out_report);
    return iree_ok_status();
  }
  loom_amdgpu_memory_calculate_source_bank_service(
      model, source, loom_low_lower_context_symbolic_expr_context(context),
      &active_lane_proof.workgroup_size, out_report);
  out_report->active_lane_proof = active_lane_proof.proof;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_fragment_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index,
    const loom_amdgpu_fragment_memory_packet_offset_t* runtime_offset,
    loom_low_lower_memory_bank_service_report_t* out_report) {
  *out_report = (loom_low_lower_memory_bank_service_report_t){0};
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            packet->descriptor_ref);
  const loom_amdgpu_lds_bank_service_model_t* model =
      loom_amdgpu_memory_bank_service_prepare_report(
          context, descriptor, layout->wave_size, out_report);
  if (model == NULL) {
    return iree_ok_status();
  }
  if (!plan->dynamic_base_is_subgroup_uniform) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-dynamic-base-not-subgroup-uniform"), out_report);
    return iree_ok_status();
  }
  if (!runtime_offset->is_subgroup_uniform) {
    loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
        context, source_op, model->wave_size, &active_lane_proof));
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
  const uint32_t packet_alignment = model->packet_byte_count;
  if (plan->source.minimum_alignment < packet_alignment ||
      !((runtime_offset->byte_facts.range_lo == 0 &&
         runtime_offset->byte_facts.range_hi == 0) ||
        loom_value_facts_divisible_by(runtime_offset->byte_facts,
                                      packet_alignment))) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        IREE_SV("address-packet-alignment-unproven"), out_report);
    return iree_ok_status();
  }

  uint64_t lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE] =
      {0};
  for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
    lane_base_byte_offsets[lane] =
        packet_byte_offset +
        loom_amdgpu_fragment_memory_relative_lane_byte_offset(
            &plan->address_layout, lane);
    if (lane_base_byte_offsets[lane] % packet_alignment != 0) {
      loom_amdgpu_memory_bank_service_mark_unknown(
          IREE_SV("address-packet-alignment-unproven"), out_report);
      return iree_ok_status();
    }
  }
  out_report->lane_address_proof =
      IREE_SV("compiled-fragment-lane-register-layout");
  out_report->base_residue_proof =
      IREE_SV("subgroup-uniform-common-translation-all-bank-word-residues");

  loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
      context, source_op, model->wave_size, &active_lane_proof));
  if (!active_lane_proof.is_full_subgroup) {
    loom_amdgpu_memory_bank_service_mark_unknown(
        active_lane_proof.unknown_reason, out_report);
    return iree_ok_status();
  }
  out_report->active_lane_proof = active_lane_proof.proof;

  const loom_value_facts_t source_offset =
      loom_low_source_memory_dynamic_offset_facts(&plan->source, 0);
  const uint64_t source_residues =
      loom_amdgpu_memory_bank_service_source_residues(
          &plan->source, source_offset, model->bank_word_byte_count);
  const uint64_t common_base_byte_residues =
      loom_amdgpu_memory_bank_service_add_offset_residues(
          source_residues, runtime_offset->byte_facts,
          model->bank_word_byte_count);
  loom_amdgpu_memory_bank_service_evaluate_full_wave(
      model, lane_base_byte_offsets, common_base_byte_residues, out_report);
  return iree_ok_status();
}
