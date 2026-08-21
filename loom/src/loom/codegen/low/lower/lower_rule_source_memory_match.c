// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"

void loom_low_lower_rule_source_memory_state_initialize(
    const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* access_plan_storage,
    loom_low_lower_rule_source_memory_state_t* out_state) {
  *out_state = (loom_low_lower_rule_source_memory_state_t){
      .source_op = source_op,
      .access_plan = access_plan_storage,
  };
}

static const loom_low_source_memory_access_plan_t*
loom_low_lower_rule_source_memory_state_resolve(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op,
    loom_low_lower_rule_source_memory_state_t* state) {
  IREE_ASSERT_EQ(state->source_op, source_op);
  if (!state->plan_attempted) {
    state->plan_attempted = true;
    if (match_context->view_regions != NULL) {
      IREE_ASSERT(state->access_plan != NULL);
      state->plan_available = loom_low_source_memory_access_plan_build(
          match_context->view_regions, source_op, state->access_plan,
          &state->diagnostic);
    }
  }
  return state->plan_available ? state->access_plan : NULL;
}

uint32_t loom_low_lower_rule_source_memory_minimum_alignment(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op) {
  loom_low_lower_rule_source_memory_state_t* state =
      match_context->source_memory_state;
  IREE_ASSERT(state != NULL);
  const loom_low_source_memory_access_plan_t* access =
      loom_low_lower_rule_source_memory_state_resolve(match_context, source_op,
                                                      state);
  return access != NULL ? access->minimum_alignment : 0;
}

bool loom_low_lower_rule_memory_space_matches(
    loom_low_lower_memory_space_mask_t memory_space_mask,
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_UNKNOWN);
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL);
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_WORKGROUP);
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_PRIVATE);
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_CONSTANT);
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_HOST);
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_DESCRIPTOR);
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_MEMORY_SPACE_GENERIC);
    default:
      return false;
  }
}

static bool loom_low_lower_rule_source_memory_dynamic_index_source_matches(
    loom_low_source_memory_dynamic_index_source_t required_source,
    loom_low_source_memory_dynamic_index_source_t actual_source) {
  if (required_source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE) {
    return actual_source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE ||
           actual_source ==
               LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
           actual_source ==
               LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
  }
  return actual_source == required_source;
}

static bool loom_low_lower_rule_source_memory_dynamic_terms_match(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  if (source_memory->dynamic_term_count ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_TERM_COUNT_ANY) {
    return access->dynamic_term_count >=
           source_memory->dynamic_term_count_minimum;
  }
  if (access->dynamic_term_count != source_memory->dynamic_term_count) {
    return false;
  }
  if (source_memory->dynamic_term_count == 0) {
    return true;
  }
  const bool any_byte_stride = iree_any_bit_set(
      source_memory->flags,
      LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_BYTE_STRIDE_ANY);
  const bool allow_dynamic_stride_values =
      iree_any_bit_set(source_memory->flags,
                       LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_STRIDE_VALUES);
  for (uint8_t i = 0; i < access->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->dynamic_terms[i];
    if (!allow_dynamic_stride_values && term->stride_value_count != 0) {
      return false;
    }
    if (!loom_low_lower_rule_source_memory_dynamic_index_source_matches(
            source_memory->dynamic_index_source, term->source) ||
        (!any_byte_stride &&
         term->byte_stride != source_memory->dynamic_byte_stride)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_source_memory_root_matches(
    const loom_module_t* module,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  switch (source_memory->root_kind) {
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY:
      return true;
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT:
      return loom_low_source_memory_value_is_block_argument(
          module, access->root_value_id);
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ALLOCA: {
      const loom_value_t* root =
          loom_module_value(module, access->root_value_id);
      return !loom_value_is_block_arg(root) &&
             loom_buffer_alloca_isa(loom_value_def_op(root));
    }
    default:
      return false;
  }
}

static bool loom_low_lower_rule_source_memory_address_layout_matches(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  switch (source_memory->address_layout) {
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_ANY:
      return true;
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR:
      return access->address_layout ==
             LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR;
  }
  return false;
}

static bool loom_low_lower_rule_source_memory_dynamic_offset_matches(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  const uint8_t bit_count = source_memory->dynamic_offset_unsigned_bit_count;
  if (bit_count == 0) {
    return true;
  }
  return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      access, access->static_byte_offset, bit_count);
}

static bool loom_low_lower_rule_source_memory_address_input_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_source_memory_t* source_memory,
    loom_value_id_t source_value_id) {
  const loom_type_t source_type =
      loom_module_value_type(match_context->module, source_value_id);
  if (source_memory->address_coordinate_type ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX) {
    if (!loom_type_equal(source_type,
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
      return false;
    }
    const loom_value_facts_t facts = loom_value_fact_table_lookup(
        match_context->fact_table, source_value_id);
    return !loom_value_facts_is_float(facts) &&
           facts.range_lo >= source_memory->address_coordinate_minimum &&
           facts.range_hi <= source_memory->address_coordinate_maximum;
  }
  IREE_ASSERT_EQ(source_memory->address_coordinate_type,
                 LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET);
  if (loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
    return true;
  }
  if (!loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return false;
  }
  return loom_value_facts_fit_unsigned_bit_count(
      loom_value_fact_table_lookup(match_context->fact_table, source_value_id),
      31);
}

static bool loom_low_lower_rule_source_memory_address_facts_fit_byte_range(
    loom_value_facts_t facts, int64_t minimum_byte_offset,
    int64_t maximum_byte_offset) {
  return !loom_value_facts_is_float(facts) &&
         facts.range_lo >= minimum_byte_offset &&
         facts.range_hi <= maximum_byte_offset;
}

static bool loom_low_lower_rule_emit_materializes_source_memory_address(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  for (uint16_t operand_ordinal = 0; operand_ordinal < emit->operand_ref_count;
       ++operand_ordinal) {
    const uint16_t value_ref_index =
        (uint16_t)(emit->operand_ref_start + operand_ordinal);
    if (rule_set->value_refs[value_ref_index].kind ==
        LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_rule_source_memory_address_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    uint16_t* out_diagnostic_index) {
  if (!loom_low_lower_rule_emit_materializes_source_memory_address(rule_set,
                                                                   emit)) {
    return true;
  }
  if (out_diagnostic_index != NULL) {
    *out_diagnostic_index = source_memory->address_diagnostic_index;
  }

  const int64_t coordinate_unit_byte_count =
      source_memory->address_coordinate_unit_byte_count;
  int64_t minimum_byte_offset = 0;
  int64_t maximum_byte_offset = 0;
  if (coordinate_unit_byte_count <= 0 ||
      !iree_checked_mul_i64(source_memory->address_coordinate_minimum,
                            coordinate_unit_byte_count, &minimum_byte_offset) ||
      !iree_checked_mul_i64(source_memory->address_coordinate_maximum,
                            coordinate_unit_byte_count, &maximum_byte_offset) ||
      access->static_byte_offset % coordinate_unit_byte_count != 0 ||
      access->static_byte_offset < minimum_byte_offset ||
      access->static_byte_offset > maximum_byte_offset) {
    return false;
  }

  const loom_value_facts_t complete_byte_facts =
      loom_low_source_memory_dynamic_offset_facts(access,
                                                  access->static_byte_offset);
  if (!loom_low_lower_rule_source_memory_address_facts_fit_byte_range(
          complete_byte_facts, minimum_byte_offset, maximum_byte_offset)) {
    return false;
  }

  if (source_memory->address_base_kind ==
          LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW &&
      loom_low_source_memory_access_base_view_value_id(access) ==
          LOOM_VALUE_ID_INVALID) {
    return false;
  }

  uint8_t first_canonical_term = 0;
  if (coordinate_unit_byte_count == 1 &&
      source_memory->address_coordinate_type ==
          LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET &&
      access->dynamic_view_base_term_count != 0 &&
      access->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID) {
    const loom_type_t view_base_type = loom_module_value_type(
        match_context->module, access->dynamic_view_base_value_id);
    if (!loom_type_equal(view_base_type,
                         loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
      return false;
    }
    first_canonical_term = access->dynamic_view_base_term_count;
  }
  for (uint8_t term_ordinal = first_canonical_term;
       term_ordinal < access->dynamic_term_count; ++term_ordinal) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->dynamic_terms[term_ordinal];
    if (term->byte_stride % coordinate_unit_byte_count != 0 ||
        !loom_low_lower_rule_source_memory_address_facts_fit_byte_range(
            term->byte_facts, minimum_byte_offset, maximum_byte_offset) ||
        !loom_low_lower_rule_source_memory_address_input_matches(
            match_context, source_memory, term->index)) {
      return false;
    }
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      if (!loom_low_lower_rule_source_memory_address_input_matches(
              match_context, source_memory,
              term->stride_values[stride_ordinal])) {
        return false;
      }
    }
  }
  return true;
}

bool loom_low_lower_rule_source_memory_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    uint16_t* out_diagnostic_index) {
  if (out_diagnostic_index != NULL) {
    *out_diagnostic_index = source_memory->diagnostic_index;
  }
  if (access == NULL) {
    return false;
  }
  if (access->operation_kind != source_memory->operation_kind ||
      access->root_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_low_lower_rule_source_memory_root_matches(match_context->module,
                                                      source_memory, access) ||
      !loom_low_lower_rule_memory_space_matches(
          source_memory->memory_space_mask, access->memory_space) ||
      access->element_byte_count != source_memory->element_byte_count ||
      access->vector_lane_count != source_memory->vector_lane_count ||
      (source_memory->vector_lane_count > 1 &&
       access->vector_lane_byte_stride !=
           source_memory->vector_lane_byte_stride) ||
      access->static_byte_offset < source_memory->static_byte_offset_minimum ||
      access->static_byte_offset > source_memory->static_byte_offset_maximum ||
      (source_memory->minimum_alignment != 0 &&
       access->minimum_alignment < source_memory->minimum_alignment) ||
      access->cache_policy.build_flags !=
          source_memory->cache_policy_build_flags ||
      (iree_any_bit_set(
           source_memory->flags,
           LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_PRESERVE_SOURCE_INDEX) &&
       access->source_index_static_offset_extracted) ||
      (source_memory->dynamic_view_base_term_count !=
           LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_VIEW_BASE_TERM_COUNT_ANY &&
       access->dynamic_view_base_term_count !=
           source_memory->dynamic_view_base_term_count) ||
      !loom_low_lower_rule_source_memory_dynamic_terms_match(source_memory,
                                                             access)) {
    return false;
  }
  if (!loom_low_lower_rule_source_memory_address_layout_matches(source_memory,
                                                                access)) {
    if (out_diagnostic_index != NULL) {
      *out_diagnostic_index = source_memory->address_layout_diagnostic_index;
    }
    return false;
  }
  if (!loom_low_lower_rule_source_memory_dynamic_offset_matches(source_memory,
                                                                access)) {
    if (out_diagnostic_index != NULL) {
      *out_diagnostic_index = source_memory->dynamic_offset_diagnostic_index;
    }
    return false;
  }
  return true;
}

loom_low_lower_rule_source_memory_match_t
loom_low_lower_rule_source_memory_emits_match(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  loom_low_lower_rule_source_memory_match_t match = {
      .has_source_memory = false,
      .constraints_compatible = false,
      .all_emits_match = true,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
  };
  loom_low_lower_rule_source_memory_state_t* state =
      match_context->source_memory_state;
  const loom_low_source_memory_access_plan_t* source_memory_access = NULL;
  bool all_constraints_compatible = true;
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    if (emit->source_memory_ordinal == 0) continue;
    if (!match.has_source_memory) {
      IREE_ASSERT(state != NULL);
      source_memory_access = loom_low_lower_rule_source_memory_state_resolve(
          match_context, source_op, state);
    }
    match.has_source_memory = true;
    const uint16_t source_memory_index =
        (uint16_t)(emit->source_memory_ordinal - 1);
    const loom_low_lower_source_memory_t* source_memory =
        &rule_set->source_memories[source_memory_index];
    uint16_t diagnostic_index = source_memory->diagnostic_index;
    if (!loom_low_lower_rule_source_memory_matches(match_context, source_memory,
                                                   source_memory_access,
                                                   &diagnostic_index)) {
      all_constraints_compatible = false;
      match.all_emits_match = false;
    } else if (!loom_low_lower_rule_source_memory_address_matches(
                   match_context, rule_set, emit, source_memory,
                   source_memory_access, &diagnostic_index)) {
      match.all_emits_match = false;
    }
    if (!match.all_emits_match &&
        match.diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE) {
      match.diagnostic_index = diagnostic_index;
    }
  }
  match.constraints_compatible =
      match.has_source_memory && all_constraints_compatible;
  return match;
}
