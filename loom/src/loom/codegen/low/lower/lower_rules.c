// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_rules.h"

#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/lower/lower_internal.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/registers.h"
#include "loom/util/math.h"

static const loom_low_lower_rule_span_t* loom_low_lower_rule_set_find_span(
    const loom_low_lower_rule_set_t* rule_set, loom_op_kind_t source_op_kind) {
  uint16_t low = 0;
  uint16_t high = rule_set->span_count;
  while (low < high) {
    uint16_t mid = low + (uint16_t)((high - low) / 2);
    const loom_low_lower_rule_span_t* span = &rule_set->spans[mid];
    if (span->source_op_kind == source_op_kind) {
      return span;
    }
    if (span->source_op_kind < source_op_kind) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_emit_no_mapping(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_string_view_t loom_low_lower_rule_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_rule_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_lower_rule_function_name(
    const loom_low_lower_rule_match_context_t* match_context) {
  if (!loom_func_like_isa(match_context->function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_rule_symbol_name(
      match_context->module, loom_func_like_callee(match_context->function));
}

static uint32_t loom_low_lower_rule_source_memory_minimum_alignment(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op) {
  if (match_context->fact_table == NULL) {
    return 0;
  }
  loom_low_source_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(
          match_context->module, match_context->fact_table, source_op, &access,
          &diagnostic)) {
    return 0;
  }
  return access.minimum_alignment;
}

static iree_string_view_t loom_low_lower_rule_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_rule_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->export_plan->name,
                                      IREE_SV("<empty>"));
}

typedef struct loom_low_lower_u32_divisor_magic_info_t {
  // Multiplication constant consumed by the unsigned quotient recipe.
  uint32_t multiplier;
  // Final logical right shift applied after the high-half multiply.
  uint8_t post_shift;
  // Whether the quotient recipe needs the unsigned add-adjustment step.
  bool is_add;
} loom_low_lower_u32_divisor_magic_info_t;

static loom_low_lower_u32_divisor_magic_info_t
loom_low_lower_u32_divisor_magic_info(uint32_t divisor) {
  IREE_ASSERT_GT(divisor, 1u);
  const uint64_t u32_mask = UINT32_MAX;
  const uint64_t two31 = ((uint64_t)1) << 31;
  const uint64_t two31_minus_one = two31 - 1;

  const uint64_t nc = u32_mask - ((u32_mask + 1u - divisor) % divisor);
  uint32_t p = 31;
  uint64_t q1 = two31 / nc;
  uint64_t r1 = two31 % nc;
  uint64_t q2 = two31_minus_one / divisor;
  uint64_t r2 = two31_minus_one % divisor;
  bool is_add = false;
  for (;;) {
    ++p;
    if (r1 >= nc - r1) {
      q1 = ((q1 << 1) + 1) & u32_mask;
      r1 = ((r1 << 1) - nc) & u32_mask;
    } else {
      q1 = (q1 << 1) & u32_mask;
      r1 = (r1 << 1) & u32_mask;
    }
    if (r2 + 1 >= divisor - r2) {
      if (q2 >= two31_minus_one) {
        is_add = true;
      }
      q2 = ((q2 << 1) + 1) & u32_mask;
      r2 = ((r2 << 1) + 1 - divisor) & u32_mask;
    } else {
      if (q2 >= two31) {
        is_add = true;
      }
      q2 = (q2 << 1) & u32_mask;
      r2 = ((r2 << 1) + 1) & u32_mask;
    }
    const uint64_t delta = (divisor - 1 - r2) & u32_mask;
    if (!(p < 64 && (q1 < delta || (q1 == delta && r1 == 0)))) {
      break;
    }
  }

  loom_low_lower_u32_divisor_magic_info_t info = {
      .multiplier = (uint32_t)((q2 + 1) & u32_mask),
      .post_shift = (uint8_t)(p - 32),
      .is_add = is_add,
  };
  if (info.is_add) {
    IREE_ASSERT_GT(info.post_shift, 0);
    --info.post_shift;
  }
  return info;
}

static iree_string_view_t loom_low_lower_rule_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

static const loom_low_lower_value_materializer_t*
loom_low_lower_rule_value_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_value_ref_t* value_ref) {
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  return &rule_set->materializers[materializer_index];
}

static loom_value_id_t loom_low_lower_rule_source_value(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      IREE_ASSERT_LT(value_ref->index, source_op->operand_count);
      return loom_op_const_operands(source_op)[value_ref->index];
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      return loom_op_const_results(source_op)[value_ref->index];
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_UNREACHABLE("temporary value ref has no source value");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
      IREE_ASSERT_UNREACHABLE(
          "source-memory dynamic term value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      IREE_ASSERT_UNREACHABLE(
          "source-memory byte offset value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_lower_rule_can_materialize_value(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  if (match_context->can_materialize.fn == NULL) {
    return false;
  }
  return match_context->can_materialize.fn(
      match_context->can_materialize.user_data, match_context, rule_set,
      source_op, value_ref_index,
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index));
}

iree_status_t loom_low_lower_rule_resolve_descriptor_ref(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_ref, rule_set->descriptor_ref_count);
  if (match_context->descriptor_ref.fn != NULL) {
    return match_context->descriptor_ref.fn(
        match_context->descriptor_ref.user_data, match_context, rule_set,
        descriptor_ref, out_descriptor);
  }
  IREE_ASSERT(match_context->descriptor_set != NULL);
  IREE_ASSERT(rule_set->descriptor_refs != NULL);
  const iree_string_view_t key = rule_set->descriptor_refs[descriptor_ref].key;
  const uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor(
      match_context->descriptor_set, key);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }
  *out_descriptor = loom_low_descriptor_set_descriptor_at(
      match_context->descriptor_set, descriptor_ordinal);
  IREE_ASSERT(*out_descriptor != NULL);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_descriptor_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, bool* out_available) {
  *out_available = false;
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
      match_context, rule_set, descriptor_ref, &descriptor));
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t word_index = descriptor->feature_mask_word_start + i;
    const uint64_t required_bits =
        match_context->descriptor_set->feature_mask_words[word_index];
    const uint64_t available_bits = i == 0 ? match_context->feature_bits : 0;
    if ((required_bits & ~available_bits) != 0) {
      return iree_ok_status();
    }
  }
  *out_available = true;
  return iree_ok_status();
}

typedef struct loom_low_lower_rule_emit_state_t {
  // Rule-local low SSA values captured by earlier emit rows.
  loom_value_id_t* temporaries;
  // Number of entries in temporaries.
  uint16_t temporary_count;
} loom_low_lower_rule_emit_state_t;

static iree_status_t loom_low_lower_rule_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_low_lower_rule_t* rule,
    loom_low_lower_rule_emit_state_t* out_state) {
  *out_state = (loom_low_lower_rule_emit_state_t){
      .temporaries = NULL,
      .temporary_count = rule->temporary_count,
  };
  if (rule->temporary_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->temporary_count, sizeof(*out_state->temporaries),
      (void**)&out_state->temporaries));
  for (uint16_t i = 0; i < rule->temporary_count; ++i) {
    out_state->temporaries[i] = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_resolve_materializer_descriptor(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    iree_string_view_t descriptor_purpose,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  IREE_ASSERT_NE(descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
  const loom_low_lower_rule_match_context_t match_context = {
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
  };
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
      &match_context, rule_set, descriptor_ref, &descriptor));
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "generated source-memory byte-offset materializer references missing "
        "'%.*s' descriptor",
        (int)descriptor_purpose.size, descriptor_purpose.data);
  }
  return loom_low_lower_resolve_descriptor_row(context, descriptor,
                                               out_descriptor);
}

static iree_status_t loom_low_lower_rule_emit_i64_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_t* source_memory, int64_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_materializer_descriptor(
      context, rule_set, source_memory->byte_offset_const_i64_descriptor_ref,
      IREE_SV("const.i64"), &descriptor));

  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context),
      source_memory->byte_offset_const_i64_immediate, &immediate_name_id));
  const loom_named_attr_t attr = {
      .name_id = immediate_name_id,
      .value = loom_attr_i64(value),
  };

  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&attr, 1), result_type,
      location, &const_op));
  *out_value_id = loom_low_const_result(const_op);
  return iree_ok_status();
}

static bool loom_low_lower_rule_descriptor_operand_is_result(
    loom_low_operand_role_t role) {
  return role == LOOM_LOW_OPERAND_ROLE_RESULT ||
         role == LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT;
}

static uint16_t loom_low_lower_rule_materializer_result_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  uint16_t result_index = 0;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_lower_rule_descriptor_operand_is_result(operand->role)) {
      continue;
    }
    if (i == descriptor_operand_index) return result_index;
    ++result_index;
  }
  IREE_ASSERT_UNREACHABLE("descriptor result operand index is invalid");
  return 0;
}

static uint16_t loom_low_lower_rule_materializer_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) continue;
    if (i == descriptor_operand_index) return packet_operand_index;
    ++packet_operand_index;
  }
  IREE_ASSERT_UNREACHABLE("descriptor packet operand index is invalid");
  return 0;
}

static iree_status_t loom_low_lower_rule_materializer_copy_operands(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint16_t copy_operand_mask, uint16_t operand_count,
    loom_value_id_t* operands) {
  if (copy_operand_mask == 0) return iree_ok_status();
  IREE_ASSERT_LE(operand_count, 16);
  for (uint16_t i = 0; i < operand_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(copy_operand_mask, operand_bit)) continue;
    const loom_type_t copy_type = loom_module_value_type(
        loom_low_lower_context_module(context), operands[i]);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_copy_build(loom_low_lower_context_builder(context),
                            operands[i], copy_type, location, &copy_op));
    operands[i] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_materializer_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    uint16_t operand_count, loom_value_id_t* operands,
    loom_location_id_t location, const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor_row = descriptor->descriptor;
  uint16_t copy_operand_mask = 0;
  uint16_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor_row->constraint_start +
                  (uint64_t)descriptor_row->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor_row->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[descriptor_row->constraint_start + (uint32_t)i];
    switch (constraint->kind) {
      case LOOM_LOW_CONSTRAINT_KIND_TIED:
        ++tied_result_count;
        break;
      case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE: {
        IREE_ASSERT_NE(constraint->rhs_operand_index, LOOM_LOW_ID_NONE);
        const uint16_t packet_operand_index =
            loom_low_lower_rule_materializer_packet_operand_index(
                descriptor_set, descriptor_row, constraint->rhs_operand_index);
        IREE_ASSERT_LT(packet_operand_index, operand_count);
        copy_operand_mask |= (uint16_t)((uint16_t)1u << packet_operand_index);
        break;
      }
      default:
        break;
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materializer_copy_operands(
      context, location, copy_operand_mask, operand_count, operands));
  if (tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  uint16_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor_row->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[descriptor_row->constraint_start + (uint32_t)i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) continue;
    IREE_ASSERT_NE(constraint->rhs_operand_index, LOOM_LOW_ID_NONE);
    const uint16_t result_index = loom_low_lower_rule_materializer_result_index(
        descriptor_set, descriptor_row, constraint->lhs_operand_index);
    const uint16_t packet_operand_index =
        loom_low_lower_rule_materializer_packet_operand_index(
            descriptor_set, descriptor_row, constraint->rhs_operand_index);
    IREE_ASSERT_LT(packet_operand_index, operand_count);
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = result_index,
        .operand_index = packet_operand_index,
        .has_type_change = false,
    };
  }
  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_emit_i64_binary_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    iree_string_view_t descriptor_purpose, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_materializer_descriptor(
      context, rule_set, descriptor_ref, descriptor_purpose, &descriptor));

  const loom_value_id_t operands[2] = {lhs, rhs};
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  loom_value_id_t materializer_operands[2] = {operands[0], operands[1]};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materializer_tied_results(
      context, &descriptor, IREE_ARRAYSIZE(materializer_operands),
      materializer_operands, location, &tied_results, &tied_result_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, materializer_operands,
      IREE_ARRAYSIZE(materializer_operands), loom_named_attr_slice_empty(),
      &result_type, 1, tied_results, tied_result_count, location, &low_op));
  const loom_value_slice_t results = loom_low_op_results(low_op);
  IREE_ASSERT_EQ(results.count, 1u);
  *out_value_id = results.values[0];
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &index));
  loom_value_id_t accumulator = index;
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &stride_value));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_i64_binary_op(
        context, rule_set, source_memory->byte_offset_mul_i64_descriptor_ref,
        IREE_SV("mul.i64"), accumulator, stride_value, source_op->location,
        &accumulator));
  }

  if (term->byte_stride == 1) {
    *out_value_id = accumulator;
    return iree_ok_status();
  }

  const loom_type_t index_type = loom_module_value_type(
      loom_low_lower_context_module(context), accumulator);
  if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
      source_memory->byte_offset_shl_i64_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_i64_const(
        context, rule_set, source_memory, term->byte_shift, index_type,
        source_op->location, &shift));
    return loom_low_lower_rule_emit_i64_binary_op(
        context, rule_set, source_memory->byte_offset_shl_i64_descriptor_ref,
        IREE_SV("shl.i64"), accumulator, shift, source_op->location,
        out_value_id);
  }

  loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_i64_const(
      context, rule_set, source_memory, term->byte_stride, index_type,
      source_op->location, &stride));
  return loom_low_lower_rule_emit_i64_binary_op(
      context, rule_set, source_memory->byte_offset_mul_i64_descriptor_ref,
      IREE_SV("mul.i64"), accumulator, stride, source_op->location,
      out_value_id);
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(source_memory_access != NULL);
  IREE_ASSERT_GT(source_memory_access->dynamic_term_count, 0);

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < source_memory_access->dynamic_term_count; ++i) {
    loom_value_id_t term_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_source_memory_term(
        context, rule_set, source_op, source_memory,
        &source_memory_access->dynamic_terms[i], &term_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = term_value;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_i64_binary_op(
        context, rule_set, source_memory->byte_offset_add_i64_descriptor_ref,
        IREE_SV("add.i64"), accumulator, term_value, source_op->location,
        &accumulator));
  }

  IREE_ASSERT_NE(accumulator, LOOM_VALUE_ID_INVALID);
  *out_value_id = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint16_t value_ref_index, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, value_ref_index);
      if (value_ref->materializer_index != 0) {
        const loom_low_lower_value_materializer_t* materializer =
            loom_low_lower_rule_value_materializer(rule_set, value_ref);
        return materializer->materialize(context, source_op, source_value_id,
                                         out_low_value_id);
      }
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      IREE_ASSERT(state->temporaries[value_ref->index] !=
                  LOOM_VALUE_ID_INVALID);
      *out_low_value_id = state->temporaries[value_ref->index];
      return iree_ok_status();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      IREE_ASSERT_LT(value_ref->index,
                     source_memory_access->dynamic_term_count);
      const loom_value_id_t source_value_id =
          source_memory_access->dynamic_terms[value_ref->index].index;
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_byte_offset(
          context, rule_set, source_op, source_memory, source_memory_access,
          out_low_value_id);
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_lower_rule_type_matches(
    const loom_low_lower_type_pattern_t* pattern, loom_type_t type) {
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND) &&
      loom_type_kind(type) != pattern->type_kind) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT) &&
      !iree_any_bit_set(
          pattern->element_type_mask,
          LOOM_LOW_LOWER_SCALAR_TYPE_BIT(loom_type_element_type(type)))) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK) &&
      loom_type_rank(type) != pattern->rank) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 0) != pattern->static_dim0) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
    if (static_dim0 < pattern->static_dim0_min ||
        static_dim0 > pattern->static_dim0_max) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1)) {
    if (loom_type_rank(type) < 2 || loom_type_dim_is_dynamic_at(type, 1)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 1) != pattern->static_dim1) {
      return false;
    }
  }
  if (iree_any_bit_set(
          pattern->flags,
          LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE)) {
    uint64_t static_element_count = 0;
    if (!loom_type_static_element_count(type, &static_element_count)) {
      return false;
    }
    if (static_element_count < pattern->static_element_count_min ||
        static_element_count > pattern->static_element_count_max) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_vector_extract_tail_type_matches(
    loom_type_t source_type, uint16_t consumed_rank, loom_type_t result_type) {
  const uint8_t source_rank = loom_type_rank(source_type);
  if (consumed_rank > source_rank) return false;
  if (loom_type_element_type(source_type) !=
      loom_type_element_type(result_type)) {
    return false;
  }
  if (loom_type_is_scalar(result_type)) {
    return consumed_rank == source_rank;
  }
  if (!loom_type_is_vector(result_type)) return false;
  const uint8_t result_rank = loom_type_rank(result_type);
  if (consumed_rank + result_rank != source_rank) return false;
  for (uint8_t i = 0; i < result_rank; ++i) {
    if (loom_type_dim(source_type, consumed_rank + i) !=
        loom_type_dim(result_type, i)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_vector_extract_shape_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  if (!loom_vector_extract_isa(source_op)) return false;
  if (guard->attr_index >= source_op->attribute_count) return false;
  loom_attribute_t static_indices =
      loom_op_const_attrs(source_op)[guard->attr_index];
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) return false;

  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t result_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->other_value_ref_index);
  const loom_type_t source_type =
      loom_module_value_type(match_context->module, source_value_id);
  const loom_type_t result_type =
      loom_module_value_type(match_context->module, result_value_id);
  if (!loom_type_is_vector(source_type)) return false;

  const loom_value_slice_t dynamic_indices =
      loom_vector_extract_indices(source_op);
  if (static_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN) {
    return dynamic_indices.count == 1 && loom_type_rank(source_type) == 1 &&
           loom_type_element_type(source_type) ==
               loom_type_element_type(result_type) &&
           loom_type_is_scalar(result_type);
  }

  if (dynamic_indices.count != 0 ||
      static_indices.count > loom_type_rank(source_type)) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    const int64_t index = static_indices.i64_array[i];
    if (index < 0 || loom_type_dim_is_dynamic_at(source_type, i) ||
        index >= loom_type_dim_static_size_at(source_type, i)) {
      return false;
    }
  }
  return loom_low_lower_rule_vector_extract_tail_type_matches(
      source_type, static_indices.count, result_type);
}

static loom_scalar_type_t loom_low_lower_rule_type_pattern_element(
    const loom_low_lower_type_pattern_t* pattern) {
  IREE_ASSERT(iree_any_bit_set(pattern->flags,
                               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT));
  const uint64_t element_type_mask = pattern->element_type_mask;
  IREE_ASSERT_NE(element_type_mask, 0u);
  IREE_ASSERT_EQ(element_type_mask & (element_type_mask - 1), 0u);
  uint32_t element_type = 0;
  uint64_t shifted_mask = element_type_mask;
  while ((shifted_mask & 1u) == 0u) {
    ++element_type;
    shifted_mask >>= 1;
  }
  return (loom_scalar_type_t)element_type;
}

static loom_type_t loom_low_lower_rule_type_pattern_exact_type(
    const loom_low_lower_type_pattern_t* pattern) {
  IREE_ASSERT(iree_all_bits_set(pattern->flags,
                                LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                                    LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT));
  const loom_scalar_type_t element_type =
      loom_low_lower_rule_type_pattern_element(pattern);
  if (pattern->type_kind == LOOM_TYPE_SCALAR) {
    return loom_type_scalar(element_type);
  }
  IREE_ASSERT(iree_all_bits_set(
      pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                          LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0));
  IREE_ASSERT_GE(pattern->static_dim0, 0);
  if (pattern->rank == 2) {
    IREE_ASSERT(iree_all_bits_set(
        pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1));
    IREE_ASSERT_GE(pattern->static_dim1, 0);
    return loom_type_shaped_2d(pattern->type_kind, element_type,
                               loom_dim_pack_static(pattern->static_dim0),
                               loom_dim_pack_static(pattern->static_dim1),
                               /*encoding_id=*/0);
  }
  IREE_ASSERT_EQ(pattern->rank, 1);
  return loom_type_shaped_1d(pattern->type_kind, element_type,
                             loom_dim_pack_static(pattern->static_dim0), 0);
}

static iree_status_t loom_low_lower_rule_mapped_value(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  IREE_ASSERT(match_context->map_value.fn != NULL);
  loom_value_id_t source_value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  return match_context->map_value.fn(match_context->map_value.user_data,
                                     match_context, source_op, source_value_id,
                                     out_mapped_value);
}

static iree_status_t loom_low_lower_rule_mapped_value_register_class_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_rule_mapped_value_t mapped_value,
    uint16_t descriptor_register_class_id, bool* out_matches) {
  *out_matches = false;
  if (!mapped_value.is_register) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_register_class_id,
                 match_context->descriptor_set->reg_class_count);
  *out_matches =
      mapped_value.descriptor_register_class_id == descriptor_register_class_id;
  return iree_ok_status();
}

static bool loom_low_lower_rule_integer_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

static bool loom_low_lower_rule_integer_element_range_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }

  const loom_type_t type = loom_module_value_type(module, value_id);
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_scalar(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    *out_facts = facts;
    return true;
  }
  if (!loom_type_is_vector(type) ||
      loom_scalar_type_is_float(loom_type_element_type(type))) {
    return false;
  }

  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    *out_facts = uniform.element;
    return true;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context, facts,
                                                &lanes)) {
    if (lanes.count == 0) {
      return false;
    }
    loom_value_facts_t aggregate = lanes.lanes[0];
    for (iree_host_size_t i = 1; i < lanes.count; ++i) {
      loom_value_facts_meet(&aggregate, &lanes.lanes[i], &aggregate);
    }
    *out_facts = aggregate;
    return true;
  }

  loom_value_fact_vector_iota_t iota = {0};
  uint64_t lane_count = 0;
  int64_t base = 0;
  int64_t step = 0;
  if (loom_value_facts_query_vector_iota(&fact_table->context, facts, &iota) &&
      loom_type_static_element_count(type, &lane_count) && lane_count > 0 &&
      loom_value_facts_as_exact_i64(iota.base, &base) &&
      loom_value_facts_as_exact_i64(iota.step, &step) &&
      lane_count <= (uint64_t)INT64_MAX) {
    int64_t final_delta = 0;
    int64_t final_value = 0;
    if (!loom_checked_mul_i64((int64_t)(lane_count - 1), step, &final_delta) ||
        !loom_checked_add_i64(base, final_delta, &final_value)) {
      return false;
    }
    *out_facts = loom_value_facts_make(loom_min_i64(base, final_value),
                                       loom_max_i64(base, final_value), 1);
    return true;
  }

  return false;
}

static bool loom_low_lower_rule_result_index_assume_facts(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  if (value_ref->kind != LOOM_LOW_LOWER_VALUE_REF_RESULT) {
    return false;
  }

  // A single-use identity assume is the only observable result contract.
  // Borrowing its facts here keeps target guards strict for unassumed or
  // multiply-used producers while allowing authored postconditions to prove a
  // dynamic address calculation.
  const loom_value_id_t source_value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  const loom_value_t* source_value =
      loom_module_value(match_context->module, source_value_id);
  if (!loom_value_has_single_use(source_value)) return false;

  const loom_use_t use = loom_value_uses(source_value)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  if (!user_op || !loom_index_assume_isa(user_op)) return false;

  const uint16_t operand_index = loom_use_operand_index(use);
  loom_value_slice_t assumed_values = loom_index_assume_values(user_op);
  loom_value_slice_t assumed_results = loom_index_assume_results(user_op);
  if (operand_index >= assumed_values.count ||
      operand_index >= assumed_results.count ||
      assumed_values.values[operand_index] != source_value_id) {
    return false;
  }

  return loom_low_lower_rule_integer_element_range_facts(
      match_context->module, match_context->fact_table,
      assumed_results.values[operand_index], out_facts);
}

static bool loom_low_lower_rule_value_integer_element_range_facts(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_facts_t* out_facts) {
  if (loom_low_lower_rule_result_index_assume_facts(
          match_context, rule_set, source_op, value_ref_index, out_facts)) {
    return true;
  }
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  return loom_low_lower_rule_integer_element_range_facts(
      match_context->module, match_context->fact_table, value_id, out_facts);
}

static bool loom_low_lower_rule_float_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

static double loom_low_lower_rule_attr_copy_exact_f64(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, attr_copy->value_ref_index);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t facts = loom_value_facts_unknown();
  const bool has_float_facts = loom_low_lower_rule_float_immediate_facts(
      loom_low_lower_context_module(context), fact_table, source_value_id,
      &facts);
  IREE_ASSERT(has_float_facts);
  IREE_ASSERT(loom_value_facts_is_exact(facts));
  IREE_ASSERT(loom_value_facts_is_float(facts));
  return loom_value_facts_as_f64(facts);
}

static void loom_low_lower_rule_set_projected_bits_attr(
    const loom_low_lower_attr_copy_t* attr_copy, uint64_t bit_pattern,
    loom_named_attr_t* attr) {
  if (attr_copy->target_bit_offset != 0) {
    IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
    IREE_ASSERT_LE(bit_pattern,
                   (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
    bit_pattern <<= attr_copy->target_bit_offset;
  }
  attr->value = loom_attr_i64((int64_t)bit_pattern);
}

static int64_t loom_low_lower_rule_i64_source_attr(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    uint16_t source_attr_index) {
  IREE_ASSERT_LT(source_attr_index, source_op->attribute_count);
  loom_attribute_t source_attr = source_attrs[source_attr_index];
  IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64);
  return source_attr.i64;
}

static uint32_t loom_low_lower_rule_u32_low_bit_mask(uint32_t width) {
  IREE_ASSERT_GT(width, 0u);
  IREE_ASSERT_LE(width, 32u);
  return width == 32u ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static uint32_t loom_low_lower_rule_attr_copy_u32_bit_mask(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const int64_t width_i64 = loom_low_lower_rule_i64_source_attr(
      source_op, source_attrs, attr_copy->source_attr_index);
  IREE_ASSERT_GE(width_i64, 1);
  IREE_ASSERT_LE(width_i64, 32);
  uint32_t mask = loom_low_lower_rule_u32_low_bit_mask((uint32_t)width_i64);
  switch (attr_copy->kind) {
    case LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK:
      return mask;
    case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK:
    case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK: {
      const int64_t offset_i64 = loom_low_lower_rule_i64_source_attr(
          source_op, source_attrs, attr_copy->other_source_attr_index);
      IREE_ASSERT_GE(offset_i64, 0);
      IREE_ASSERT_LE(offset_i64, 31);
      IREE_ASSERT_LE(offset_i64 + width_i64, 32);
      mask <<= (uint32_t)offset_i64;
      return attr_copy->kind ==
                     LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK
                 ? ~mask
                 : mask;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated bit-mask attr copy kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static int64_t loom_low_lower_rule_attr_copy_literal_minus_i64_attrs(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy) {
  int64_t projected_value = attr_copy->literal_i64;
  const int64_t source_value = loom_low_lower_rule_i64_source_attr(
      source_op, source_attrs, attr_copy->source_attr_index);
  IREE_ASSERT_GE(source_value, 0);
  IREE_ASSERT_LE(source_value, projected_value);
  projected_value -= source_value;
  if (attr_copy->kind == LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS) {
    const int64_t other_source_value = loom_low_lower_rule_i64_source_attr(
        source_op, source_attrs, attr_copy->other_source_attr_index);
    IREE_ASSERT_GE(other_source_value, 0);
    IREE_ASSERT_LE(other_source_value, projected_value);
    projected_value -= other_source_value;
  } else {
    IREE_ASSERT_EQ(attr_copy->kind,
                   LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR);
  }
  return projected_value;
}

static iree_status_t loom_low_lower_rule_value_symbolically_fits_bit_count(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint8_t bit_count, bool is_signed_domain,
    bool* out_matches) {
  *out_matches = false;
  if (bit_count == 0 || bit_count > 64) return iree_ok_status();
  loom_symbolic_expr_context_t* expression_context =
      match_context->symbolic_expr_context;
  if (expression_context == NULL) return iree_ok_status();

  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_symbolic_expr_t value_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expression_context, value_id, &value_expression));

  loom_symbolic_expr_t minimum_expression = {0};
  loom_symbolic_expr_constant(0, &minimum_expression);
  loom_symbolic_proof_result_t minimum_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (is_signed_domain) {
    int64_t minimum_value =
        bit_count >= 64 ? INT64_MIN : -(INT64_C(1) << (bit_count - 1));
    loom_symbolic_expr_constant(minimum_value, &minimum_expression);
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le(expression_context, &minimum_expression,
                                  &value_expression, &minimum_proof));
  if (minimum_proof != LOOM_SYMBOLIC_PROOF_TRUE) {
    return iree_ok_status();
  }

  if (!is_signed_domain && bit_count >= 63) {
    *out_matches = true;
    return iree_ok_status();
  }
  if (is_signed_domain && bit_count >= 64) {
    *out_matches = true;
    return iree_ok_status();
  }

  uint64_t unsigned_maximum =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
  int64_t maximum_value = is_signed_domain ? (INT64_C(1) << (bit_count - 1)) - 1
                                           : (int64_t)unsigned_maximum;
  loom_symbolic_expr_t maximum_expression = {0};
  loom_symbolic_expr_constant(maximum_value, &maximum_expression);
  loom_symbolic_proof_result_t maximum_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le(expression_context, &value_expression,
                                  &maximum_expression, &maximum_proof));
  *out_matches = maximum_proof == LOOM_SYMBOLIC_PROOF_TRUE;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_value_facts_fit_bit_count(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint64_t bit_count, bool is_signed_domain,
    bool* out_matches) {
  *out_matches = false;
  if (bit_count > UINT8_MAX) {
    return iree_ok_status();
  }
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_value_integer_element_range_facts(
          match_context, rule_set, source_op, value_ref_index, &facts)) {
    return iree_ok_status();
  }
  if (is_signed_domain) {
    *out_matches =
        loom_value_facts_fit_signed_bit_count(facts, (uint8_t)bit_count);
  } else {
    *out_matches =
        loom_value_facts_fit_unsigned_bit_count(facts, (uint8_t)bit_count);
  }
  if (*out_matches) return iree_ok_status();
  return loom_low_lower_rule_value_symbolically_fits_bit_count(
      match_context, rule_set, source_op, value_ref_index, (uint8_t)bit_count,
      is_signed_domain, out_matches);
}

static bool loom_low_lower_rule_value_facts_exact_i64(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(facts, &exact_value);
}

static bool loom_low_lower_rule_value_facts_exact_power_of_two_i64(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(facts, &exact_value) &&
         loom_is_power_of_two_i64(exact_value);
}

static bool loom_low_lower_rule_value_facts_u32_divisor_magic_info(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id,
    loom_low_lower_u32_divisor_magic_info_t* out_info) {
  *out_info = (loom_low_lower_u32_divisor_magic_info_t){0};
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(module, fact_table, value_id,
                                                   &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value) || exact_value < 2 ||
      exact_value > UINT32_MAX) {
    return false;
  }
  *out_info = loom_low_lower_u32_divisor_magic_info((uint32_t)exact_value);
  return true;
}

static bool loom_low_lower_rule_value_facts_u32_divisor_magic_is_add(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, bool expected_is_add) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_low_lower_u32_divisor_magic_info_t info = {0};
  return loom_low_lower_rule_value_facts_u32_divisor_magic_info(
             match_context->module, match_context->fact_table, value_id,
             &info) &&
         info.is_add == expected_is_add;
}

static bool loom_low_lower_rule_value_facts_exact_f64(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_float_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  return loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts);
}

static bool loom_low_lower_rule_value_facts_f64_equals(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint64_t expected_bits) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_float_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  return loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts) &&
         (uint64_t)facts.range_lo == expected_bits;
}

static bool loom_low_lower_rule_value_facts_i64_range(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, int64_t minimum_i64, int64_t maximum_i64) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  return facts.range_lo >= minimum_i64 && facts.range_hi <= maximum_i64;
}

static bool loom_low_lower_rule_value_facts_i64_range_le(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint16_t other_value_ref_index) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_value_id_t other_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, other_value_ref_index);
  loom_value_facts_t other_facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, other_value_id,
          &other_facts)) {
    return false;
  }
  return facts.range_hi <= other_facts.range_lo;
}

static bool loom_low_lower_rule_value_facts_i64_range_ge(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint16_t other_value_ref_index) {
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_value_id_t other_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, other_value_ref_index);
  loom_value_facts_t other_facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, other_value_id,
          &other_facts)) {
    return false;
  }
  return facts.range_lo >= other_facts.range_hi;
}

static bool loom_low_lower_rule_value_storage_element_format(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_value_fact_numeric_format_flags_t expected_format) {
  if (expected_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN) {
    return false;
  }
  const loom_value_id_t value_id =
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
  const loom_type_t type =
      loom_module_value_type(match_context->module, value_id);
  const loom_fact_context_t* fact_context =
      match_context->fact_table != NULL ? &match_context->fact_table->context
                                        : NULL;
  loom_value_fact_storage_schema_t storage_schema = {0};
  return loom_encoding_query_type_storage_schema(
             fact_context, match_context->module, type, &storage_schema) &&
         storage_schema.encoded_operand.element_format == expected_format;
}

static bool loom_low_lower_source_memory_space_matches(
    loom_low_lower_source_memory_space_mask_t memory_space_mask,
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_UNKNOWN);
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GLOBAL);
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_WORKGROUP);
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_PRIVATE);
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_CONSTANT);
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_HOST);
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_DESCRIPTOR);
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return iree_any_bit_set(memory_space_mask,
                              LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GENERIC);
    default:
      return false;
  }
}

static bool loom_low_lower_source_memory_dynamic_index_source_matches(
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

static bool loom_low_lower_source_memory_dynamic_terms_match(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  if (source_memory->dynamic_term_count ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_TERM_COUNT_ANY) {
    return true;
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
    if (!loom_low_lower_source_memory_dynamic_index_source_matches(
            source_memory->dynamic_index_source, term->source) ||
        (!any_byte_stride &&
         term->byte_stride != source_memory->dynamic_byte_stride)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_source_memory_root_matches(
    const loom_module_t* module,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  switch (source_memory->root_kind) {
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY:
      return true;
    case LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT:
      return loom_low_source_memory_value_is_block_argument(
          module, access->root_value_id);
    default:
      return false;
  }
}

static bool loom_low_lower_source_memory_dynamic_offset_matches(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  const uint8_t bit_count = source_memory->dynamic_offset_unsigned_bit_count;
  if (bit_count == 0) {
    return true;
  }
  return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      access, access->static_byte_offset, bit_count);
}

static bool loom_low_lower_source_memory_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    loom_low_source_memory_access_plan_t* out_access,
    uint16_t* out_diagnostic_index) {
  if (out_diagnostic_index != NULL) {
    *out_diagnostic_index = source_memory->diagnostic_index;
  }
  if (match_context->fact_table == NULL) {
    return false;
  }
  loom_low_source_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(
          match_context->module, match_context->fact_table, source_op, &access,
          &diagnostic)) {
    return false;
  }
  if (access.operation_kind != source_memory->operation_kind ||
      access.root_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_low_lower_source_memory_root_matches(match_context->module,
                                                 source_memory, &access) ||
      !loom_low_lower_source_memory_space_matches(
          source_memory->memory_space_mask, access.memory_space) ||
      access.element_byte_count != source_memory->element_byte_count ||
      access.vector_lane_count != source_memory->vector_lane_count ||
      (source_memory->vector_lane_count > 1 &&
       access.vector_lane_byte_stride !=
           source_memory->vector_lane_byte_stride) ||
      access.static_byte_offset < source_memory->static_byte_offset_minimum ||
      access.static_byte_offset > source_memory->static_byte_offset_maximum ||
      (source_memory->minimum_alignment != 0 &&
       access.minimum_alignment < source_memory->minimum_alignment) ||
      access.cache_policy.build_flags !=
          source_memory->cache_policy_build_flags ||
      !loom_low_lower_source_memory_dynamic_terms_match(source_memory,
                                                        &access)) {
    return false;
  }
  if (!loom_low_lower_source_memory_dynamic_offset_matches(source_memory,
                                                           &access)) {
    if (out_diagnostic_index != NULL) {
      *out_diagnostic_index = source_memory->dynamic_offset_diagnostic_index;
    }
    return false;
  }
  if (out_access != NULL) {
    *out_access = access;
  }
  return true;
}

static bool loom_low_lower_rule_storage_width(
    const loom_op_t* source_op, const loom_low_lower_guard_t* guard,
    uint32_t* out_storage_unit_bit_count, uint32_t* out_width) {
  *out_storage_unit_bit_count = 0;
  *out_width = 0;
  if (guard->attr_index >= source_op->attribute_count ||
      loom_op_const_attrs(source_op)[guard->attr_index].kind != LOOM_ATTR_I64 ||
      guard->minimum_i64 <= 0 || guard->minimum_i64 > UINT32_MAX) {
    return false;
  }
  const int64_t width_i64 =
      loom_op_const_attrs(source_op)[guard->attr_index].i64;
  if (width_i64 <= 0 || width_i64 > UINT32_MAX) {
    return false;
  }
  const uint32_t storage_unit_bit_count = (uint32_t)guard->minimum_i64;
  const uint32_t width = (uint32_t)width_i64;
  if ((storage_unit_bit_count % width) != 0) {
    return false;
  }
  *out_storage_unit_bit_count = storage_unit_bit_count;
  *out_width = width;
  return true;
}

static bool loom_low_lower_rule_packed_integer_payload_from_lanes_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  if (guard->u64 == 0 || guard->u64 > UINT32_MAX) {
    return false;
  }
  uint32_t storage_unit_bit_count = 0;
  uint32_t width = 0;
  if (!loom_low_lower_rule_storage_width(source_op, guard,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->other_value_ref_index);
  loom_vector_packed_integer_payload_from_lanes_match_t match = {0};
  if (!loom_vector_packed_integer_payload_from_lanes_match(
          loom_module_value_type(match_context->module, lane_value_id),
          loom_module_value_type(match_context->module, storage_value_id),
          width, storage_unit_bit_count, UINT32_MAX, &match)) {
    return false;
  }

  return (match.result_shape.payload_bit_count % (uint32_t)guard->u64) == 0;
}

static bool loom_low_lower_rule_packed_integer_lanes_from_payload_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  if (guard->u64 == 0 || guard->u64 > UINT32_MAX || guard->maximum_i64 <= 0 ||
      guard->maximum_i64 > UINT32_MAX) {
    return false;
  }
  uint32_t storage_unit_bit_count = 0;
  uint32_t width = 0;
  if (!loom_low_lower_rule_storage_width(source_op, guard,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      rule_set, source_op, guard->other_value_ref_index);
  if (!loom_vector_packed_integer_lanes_from_payload_match(
          loom_module_value_type(match_context->module, storage_value_id),
          loom_module_value_type(match_context->module, lane_value_id), width,
          storage_unit_bit_count, (uint32_t)guard->u64,
          (uint32_t)guard->maximum_i64, NULL)) {
    return false;
  }
  return true;
}

static iree_status_t loom_low_lower_rule_guard_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard, bool* out_matches) {
  *out_matches = false;
  switch (guard->kind) {
    case LOOM_LOW_LOWER_GUARD_VALUE_TYPE: {
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      loom_type_t type =
          loom_module_value_type(match_context->module, value_id);
      *out_matches = loom_low_lower_rule_type_matches(
          &rule_set->type_patterns[guard->type_pattern_index], type);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_KIND:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].kind ==
                     guard->attr_kind;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches =
          loom_op_const_attrs(source_op)[guard->attr_index].kind ==
              LOOM_ATTR_ENUM &&
          loom_op_const_attrs(source_op)[guard->attr_index].raw == guard->u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].i64 >=
                         guard->minimum_i64 &&
                     loom_op_const_attrs(source_op)[guard->attr_index].i64 <=
                         guard->maximum_i64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches =
          (uint64_t)loom_op_const_attrs(source_op)[guard->attr_index].count ==
          guard->u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      if (guard->u64 >=
          loom_op_const_attrs(source_op)[guard->attr_index].count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index]
                             .i64_array[guard->u64] >= guard->minimum_i64 &&
                     loom_op_const_attrs(source_op)[guard->attr_index]
                             .i64_array[guard->u64] <= guard->maximum_i64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE: {
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches = true;
      loom_attribute_t attr = loom_op_const_attrs(source_op)[guard->attr_index];
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (attr.i64_array[i] < guard->minimum_i64 ||
            attr.i64_array[i] > guard->maximum_i64) {
          *out_matches = false;
          break;
        }
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE:
      return loom_low_lower_rule_descriptor_available(
          match_context, rule_set, guard->descriptor_ref, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE:
      *out_matches = loom_low_lower_rule_can_materialize_value(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->value_ref_index,
          &mapped_value));
      return loom_low_lower_rule_mapped_value_register_class_matches(
          match_context, mapped_value, guard->register_class_id, out_matches);
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->value_ref_index,
          &mapped_value));
      *out_matches = mapped_value.is_register &&
                     mapped_value.register_unit_count == guard->u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE: {
      IREE_ASSERT_GT(guard->u64, 0);
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      loom_type_t type =
          loom_module_value_type(match_context->module, value_id);
      if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
        return iree_ok_status();
      }
      const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
      *out_matches =
          static_dim0 >= 0 && ((uint64_t)static_dim0 % guard->u64) == 0;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ: {
      loom_low_lower_rule_mapped_value_t lhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_mapped_value(match_context, rule_set, source_op,
                                           guard->value_ref_index, &lhs_value));
      if (!lhs_value.is_register) {
        return iree_ok_status();
      }
      loom_low_lower_rule_mapped_value_t rhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->other_value_ref_index,
          &rhs_value));
      if (!rhs_value.is_register) {
        return iree_ok_status();
      }
      *out_matches =
          lhs_value.register_unit_count == rhs_value.register_unit_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_ELEMENT_COUNT_EQ: {
      const loom_value_id_t lhs_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      const loom_value_id_t rhs_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->other_value_ref_index);
      uint64_t lhs_count = 0;
      uint64_t rhs_count = 0;
      if (!loom_type_static_element_count(
              loom_module_value_type(match_context->module, lhs_id),
              &lhs_count) ||
          !loom_type_static_element_count(
              loom_module_value_type(match_context->module, rhs_id),
              &rhs_count)) {
        return iree_ok_status();
      }
      *out_matches = lhs_count == rhs_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_OPERAND_SEGMENT_COUNT_EQ: {
      if (guard->attr_index > source_op->operand_count) {
        return iree_ok_status();
      }
      const loom_op_vtable_t* vtable = loom_context_resolve_op(
          match_context->module->context, source_op->kind);
      if (vtable == NULL) {
        return iree_ok_status();
      }
      uint16_t segment_count = 0;
      if (guard->attr_index < vtable->fixed_operand_count) {
        segment_count = 1;
      } else if (guard->attr_index == vtable->fixed_operand_count &&
                 iree_any_bit_set(vtable->vtable_flags,
                                  LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
        segment_count =
            (uint16_t)(source_op->operand_count - guard->attr_index);
      }
      *out_matches = segment_count == guard->u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->u64, /*is_signed_domain=*/true, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->u64, /*is_signed_domain=*/false, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_i64(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_power_of_two_i64(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_IS_ADD:
      *out_matches = loom_low_lower_rule_value_facts_u32_divisor_magic_is_add(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->u64 != 0);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_F64:
      *out_matches = loom_low_lower_rule_value_facts_exact_f64(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->minimum_i64, guard->maximum_i64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_LE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_le(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_GE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_ge(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_F64_EQUALS:
      *out_matches = loom_low_lower_rule_value_facts_f64_equals(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT:
      *out_matches = loom_low_lower_rule_value_storage_element_format(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES:
      *out_matches =
          loom_low_lower_rule_packed_integer_payload_from_lanes_matches(
              match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD:
      *out_matches =
          loom_low_lower_rule_packed_integer_lanes_from_payload_matches(
              match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_NO_USES: {
      const loom_value_id_t value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      *out_matches = loom_value_has_no_uses(
          loom_module_value(match_context->module, value_id));
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE:
      *out_matches = loom_low_lower_rule_vector_extract_shape_matches(
          match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL:
      *out_matches = iree_all_bits_set(source_op->instance_flags, guard->u64);
      return iree_ok_status();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated lower guard kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule, bool* out_matches,
    uint16_t* out_diagnostic_index, uint16_t* out_matched_guard_count) {
  *out_matches = false;
  *out_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  *out_matched_guard_count = 0;
  for (uint16_t i = 0; i < rule->guard_count; ++i) {
    uint16_t guard_index = (uint16_t)(rule->guard_start + i);
    const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
    bool guard_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_matches(
        match_context, rule_set, source_op, guard, &guard_matches));
    if (!guard_matches) {
      *out_diagnostic_index = guard->diagnostic_index;
      *out_matched_guard_count = i;
      return iree_ok_status();
    }
  }
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    if (emit->source_memory_ordinal == 0) {
      continue;
    }
    const uint16_t source_memory_index =
        (uint16_t)(emit->source_memory_ordinal - 1);
    const loom_low_lower_source_memory_t* source_memory =
        &rule_set->source_memories[source_memory_index];
    uint16_t source_memory_diagnostic_index = source_memory->diagnostic_index;
    if (!loom_low_lower_source_memory_matches(
            match_context, source_op, source_memory, NULL,
            &source_memory_diagnostic_index)) {
      *out_diagnostic_index = source_memory_diagnostic_index;
      *out_matched_guard_count = rule->guard_count;
      return iree_ok_status();
    }
  }
  *out_matches = true;
  *out_matched_guard_count = rule->guard_count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select_rule_range_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  *out_selection = (loom_low_lower_rule_selection_t){
      .rule = NULL,
      .rule_index = UINT16_MAX,
      .has_source_op_span = false,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
      .matched_guard_count = 0,
  };

  if (rule_count == 0) {
    return iree_ok_status();
  }

  uint16_t best_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  uint16_t best_matched_guard_count = 0;
  for (uint16_t i = 0; i < rule_count; ++i) {
    uint16_t rule_index = (uint16_t)(rule_start + i);
    const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
    if (iree_all_bits_set(rule->flags,
                          LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY) &&
        !iree_all_bits_set(match_context->flags,
                           LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY)) {
      continue;
    }
    out_selection->has_source_op_span = true;
    bool rule_matches = false;
    uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    uint16_t matched_guard_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_matches(
        match_context, rule_set, source_op, rule, &rule_matches,
        &diagnostic_index, &matched_guard_count));
    if (rule_matches) {
      out_selection->rule = rule;
      out_selection->rule_index = rule_index;
      return iree_ok_status();
    }
    if (best_diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
        matched_guard_count > best_matched_guard_count) {
      best_diagnostic_index = diagnostic_index;
      best_matched_guard_count = matched_guard_count;
    }
  }

  out_selection->diagnostic_index = best_diagnostic_index;
  out_selection->matched_guard_count = best_matched_guard_count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  const uint16_t rule_start = span ? span->rule_start : 0;
  const uint16_t rule_count = span ? span->rule_count : 0;
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

static iree_status_t loom_low_lower_rule_match_map_value_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, &low_type));
  if (!loom_low_type_is_register(low_type)) {
    return iree_ok_status();
  }
  *out_mapped_value = (loom_low_lower_rule_mapped_value_t){
      .is_register = true,
      .descriptor_register_class_id = loom_low_register_type_class_id(low_type),
      .register_unit_count = loom_low_register_type_unit_count(low_type),
  };
  return iree_ok_status();
}

static bool loom_low_lower_rule_match_can_materialize_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id) {
  (void)match_context;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const loom_low_lower_value_materializer_t* materializer =
      loom_low_lower_rule_value_materializer(rule_set, value_ref);
  return materializer->can_materialize(context, source_value_id);
}

static const loom_low_lower_rule_descriptor_map_t*
loom_low_lower_rule_descriptor_map_find(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set) {
  for (uint16_t i = 0; i < context->lowering.rule_descriptor_map_count; ++i) {
    const loom_low_lower_rule_descriptor_map_t* map =
        &context->lowering.rule_descriptor_maps[i];
    if (map->rule_set == rule_set) {
      return map;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_descriptor_maps_initialize(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set) {
  IREE_ASSERT(descriptor_set != NULL);
  if (context->lowering.rule_descriptor_map_set == descriptor_set)
    return iree_ok_status();

  context->lowering.rule_descriptor_map_set = descriptor_set;
  context->lowering.rule_descriptor_maps = NULL;
  context->lowering.rule_descriptor_map_count = 0;

  const loom_low_lower_rule_set_list_t rule_sets = context->policy->rule_sets;
  if (rule_sets.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, rule_sets.count,
      sizeof(*context->lowering.rule_descriptor_maps),
      (void**)&context->lowering.rule_descriptor_maps));
  context->lowering.rule_descriptor_map_count = rule_sets.count;

  for (uint16_t i = 0; i < rule_sets.count; ++i) {
    const loom_low_lower_rule_set_t* rule_set = rule_sets.values[i];
    loom_low_lower_rule_descriptor_map_t* map =
        &context->lowering.rule_descriptor_maps[i];
    *map = (loom_low_lower_rule_descriptor_map_t){
        .rule_set = rule_set,
        .descriptors = NULL,
        .descriptor_count = rule_set->descriptor_ref_count,
    };
    if (rule_set->descriptor_ref_count == 0) {
      continue;
    }
    IREE_ASSERT(rule_set->descriptor_refs != NULL);
    const loom_low_descriptor_t** descriptors = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, rule_set->descriptor_ref_count, sizeof(*descriptors),
        (void**)&descriptors));
    map->descriptors = descriptors;
    for (uint16_t j = 0; j < rule_set->descriptor_ref_count; ++j) {
      descriptors[j] = NULL;
      const iree_string_view_t key = rule_set->descriptor_refs[j].key;
      const uint32_t descriptor_ordinal =
          loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      descriptors[j] = loom_low_descriptor_set_descriptor_at(
          descriptor_set, descriptor_ordinal);
      IREE_ASSERT(descriptors[j] != NULL);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_match_descriptor_ref_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_maps_initialize(
      context, match_context->descriptor_set));
  const loom_low_lower_rule_descriptor_map_t* map =
      loom_low_lower_rule_descriptor_map_find(context, rule_set);
  IREE_ASSERT(map != NULL);
  IREE_ASSERT_LT(descriptor_ref, map->descriptor_count);
  *out_descriptor = map->descriptors[descriptor_ref];
  return iree_ok_status();
}

static loom_symbolic_expr_context_t*
loom_low_lower_rule_symbolic_expr_context_from_lowering(
    loom_low_lower_context_t* context) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) return NULL;

  loom_low_lowering_frame_t* lowering = &context->lowering;
  if (!lowering->expression_context_initialized ||
      lowering->expression_context_fact_table != fact_table) {
    loom_symbolic_expr_context_initialize(
        loom_low_lower_context_module(context), fact_table, &context->arena,
        &lowering->expression_context);
    lowering->expression_context_fact_table = fact_table;
    lowering->expression_context_initialized = true;
  }
  return &lowering->expression_context;
}

static loom_low_lower_rule_match_context_t
loom_low_lower_rule_match_context_from_lowering(
    loom_low_lower_context_t* context) {
  return (loom_low_lower_rule_match_context_t){
      .module = loom_low_lower_context_module(context),
      .function = loom_low_lower_context_source_function(context),
      .bundle = loom_low_lower_context_bundle(context),
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .feature_bits =
          loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      .map_value =
          {
              .fn = loom_low_lower_rule_match_map_value_from_lowering,
              .user_data = context,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_rule_match_can_materialize_from_lowering,
              .user_data = context,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .fact_table = loom_low_lower_context_fact_table(context),
      .symbolic_expr_context =
          loom_low_lower_rule_symbolic_expr_context_from_lowering(context),
  };
}

void loom_low_lower_rule_materialize_diagnostic_params(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_diagnostic_t* diagnostic,
    loom_diagnostic_param_t* out_params) {
  const loom_low_lower_diagnostic_param_t* param_rows =
      diagnostic->param_count == 0
          ? NULL
          : &rule_set->diagnostic_params[diagnostic->param_start];
  for (uint8_t i = 0; i < diagnostic->param_count; ++i) {
    const loom_low_lower_diagnostic_param_t* row = &param_rows[i];
    switch (row->kind) {
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY:
        out_params[i] = loom_param_string(
            loom_low_lower_rule_target_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME:
        out_params[i] = loom_param_string(
            loom_low_lower_rule_export_name(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY:
        out_params[i] = loom_param_string(
            loom_low_lower_rule_config_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME:
        out_params[i] =
            loom_param_string(loom_low_lower_rule_function_name(match_context));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME:
        out_params[i] =
            loom_param_string(loom_op_name(match_context->module, source_op));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL:
        out_params[i] = loom_param_string(row->string_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE: {
        const loom_value_id_t value_id = loom_low_lower_rule_source_value(
            rule_set, source_op, row->value_ref_index);
        out_params[i] = loom_param_type(
            loom_module_value_type(match_context->module, value_id));
        break;
      }
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL:
        out_params[i] = loom_param_i64(row->i64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL:
        out_params[i] = loom_param_u32(row->u32_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL:
        out_params[i] = loom_param_u64(row->u64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL:
        out_params[i] = loom_param_bool(row->bool_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_MEMORY_MINIMUM_ALIGNMENT:
        out_params[i] =
            loom_param_u32(loom_low_lower_rule_source_memory_minimum_alignment(
                match_context, source_op));
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated diagnostic param kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
}

static iree_status_t loom_low_lower_rule_emit_diagnostic(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t diagnostic_index) {
  if (diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      diagnostic_index >= rule_set->diagnostic_count) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  const loom_low_lower_diagnostic_t* diagnostic =
      &rule_set->diagnostics[diagnostic_index];
  loom_diagnostic_param_t params[LOOM_LOW_LOWER_MAX_DIAGNOSTIC_PARAMS] = {0};
  IREE_ASSERT_LE(diagnostic->param_count, IREE_ARRAYSIZE(params));
  const loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  loom_low_lower_rule_materialize_diagnostic_params(
      &match_context, rule_set, source_op, diagnostic, params);
  return loom_low_lower_emit_error_ref(context, source_op,
                                       diagnostic->error_ref, params,
                                       diagnostic->param_count);
}

iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  return loom_low_lower_rule_set_select_with_match_context(
      &match_context, rule_set, source_op, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_contract(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  match_context.flags |= LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY;
  return loom_low_lower_rule_set_select_with_match_context(
      &match_context, rule_set, source_op, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_rule_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      &match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

const loom_low_lower_diagnostic_t* loom_low_lower_rule_set_selection_diagnostic(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t selection) {
  if (selection.rule != NULL ||
      selection.diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      selection.diagnostic_index >= rule_set->diagnostic_count) {
    return NULL;
  }
  return &rule_set->diagnostics[selection.diagnostic_index];
}

loom_low_lower_descriptor_ref_t loom_low_lower_rule_first_descriptor_ref(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_descriptor_ref_t descriptor_ref =
        rule_set->emits[emit_index].descriptor_ref;
    if (descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
      return descriptor_ref;
    }
  }
  return LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE;
}

iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t selection) {
  IREE_ASSERT(selection.rule == NULL);
  if (!selection.has_source_op_span) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  return loom_low_lower_rule_emit_diagnostic(context, rule_set, source_op,
                                             selection.diagnostic_index);
}

iree_status_t loom_low_lower_rule_set_select_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t** out_rule) {
  *out_rule = NULL;
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_set_select(context, rule_set, source_op, &selection));
  if (selection.rule == NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(context, rule_set,
                                                          source_op, selection);
  }
  *out_rule = selection.rule;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_resolve_emit_program(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t** out_resolved_emits) {
  *out_resolved_emits = NULL;
  if (rule->emit_count == 0) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->emit_count, sizeof(*resolved_emits),
      (void**)&resolved_emits));
  const loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    resolved_emits[i].emit = emit;
    IREE_ASSERT_NE(emit->descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
    const loom_low_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
        &match_context, rule_set, emit->descriptor_ref, &descriptor));
    if (descriptor == NULL) {
      const iree_string_view_t key =
          rule_set->descriptor_refs[emit->descriptor_ref].key;
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "generated target-low rule references missing descriptor '%.*s'",
          (int)key.size, key.data);
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
        context, descriptor, &resolved_emits[i].descriptor));
  }
  *out_resolved_emits = resolved_emits;
  return iree_ok_status();
}

static void loom_low_lower_rule_source_memory_access(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit,
    loom_low_source_memory_access_plan_t* out_access) {
  if (emit->source_memory_ordinal == 0) {
    IREE_ASSERT_UNREACHABLE("emitted source memory ordinal must be present");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint16_t source_memory_index =
      (uint16_t)(emit->source_memory_ordinal - 1);
  const loom_low_lower_source_memory_t* source_memory =
      &rule_set->source_memories[source_memory_index];
  const loom_low_lower_rule_match_context_t match_context =
      loom_low_lower_rule_match_context_from_lowering(context);
  if (!loom_low_lower_source_memory_matches(&match_context, source_op,
                                            source_memory, out_access, NULL)) {
    IREE_ASSERT_UNREACHABLE("selected source memory must still match");
    IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_build_attrs(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (emit->attr_copy_count == 0) {
    return iree_ok_status();
  }
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->attr_copy_count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < emit->attr_copy_count; ++i) {
    uint16_t attr_copy_index = (uint16_t)(emit->attr_copy_start + i);
    const loom_low_lower_attr_copy_t* attr_copy =
        &rule_set->attr_copies[attr_copy_index];
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(loom_low_lower_context_module(context),
                                  attr_copy->target_name, &attrs[i].name_id));
    switch (attr_copy->kind) {
      case LOOM_LOW_LOWER_ATTR_COPY_DIRECT:
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        attrs[i].value = source_attrs[attr_copy->source_attr_index];
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_ENUM_ORDINAL: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_ENUM);
        attrs[i].value = loom_attr_i64(loom_attr_as_enum(source_attr));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_LT(attr_copy->source_element_index, source_attr.count);
        const int64_t source_value =
            source_attr.i64_array[attr_copy->source_element_index];
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(source_value);
          break;
        }
        IREE_ASSERT_GE(source_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)source_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)source_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        IREE_ASSERT_GT(attr_copy->source_element_bit_width, 0);
        const uint32_t packed_bit_count =
            (uint32_t)attr_copy->source_element_count *
            attr_copy->source_element_bit_width;
        IREE_ASSERT_LE(packed_bit_count + attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint32_t)attr_copy->source_element_index +
                           attr_copy->source_element_count,
                       source_attr.count);
        const uint64_t element_mask =
            (UINT64_C(1) << attr_copy->source_element_bit_width) - 1u;
        uint64_t packed_value = 0;
        for (uint16_t j = 0; j < attr_copy->source_element_count; ++j) {
          const int64_t source_value =
              source_attr.i64_array[attr_copy->source_element_index + j];
          IREE_ASSERT_GE(source_value, 0);
          IREE_ASSERT_LE((uint64_t)source_value, element_mask);
          packed_value |= (uint64_t)source_value
                          << (j * attr_copy->source_element_bit_width);
        }
        packed_value <<= attr_copy->target_bit_offset;
        attrs[i].value = loom_attr_i64((int64_t)packed_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ATTRS_PACK_CONSECUTIVE: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        IREE_ASSERT_GT(attr_copy->source_element_bit_width, 0);
        const uint32_t packed_bit_count =
            (uint32_t)attr_copy->source_element_count *
            attr_copy->source_element_bit_width;
        IREE_ASSERT_LE(packed_bit_count + attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint32_t)attr_copy->source_attr_index +
                           attr_copy->source_element_count,
                       source_op->attribute_count);
        const uint64_t element_mask =
            (UINT64_C(1) << attr_copy->source_element_bit_width) - 1u;
        uint64_t packed_value = 0;
        for (uint16_t j = 0; j < attr_copy->source_element_count; ++j) {
          loom_attribute_t source_attr =
              source_attrs[attr_copy->source_attr_index + j];
          IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64);
          const int64_t source_value = source_attr.i64;
          IREE_ASSERT_GE(source_value, 0);
          IREE_ASSERT_LE((uint64_t)source_value, element_mask);
          packed_value |= (uint64_t)source_value
                          << (j * attr_copy->source_element_bit_width);
        }
        packed_value <<= attr_copy->target_bit_offset;
        attrs[i].value = loom_attr_i64((int64_t)packed_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_LANE_BYTE: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_LT(attr_copy->source_element_index, source_attr.count);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        const int64_t source_lane =
            source_attr.i64_array[attr_copy->source_element_index];
        IREE_ASSERT_GE(source_lane, 0);
        const int64_t byte_lane =
            source_lane * attr_copy->source_element_count +
            attr_copy->literal_i64;
        attrs[i].value = loom_attr_i64(byte_lane);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK:
        IREE_ASSERT_EQ(attr_copy->target_bit_offset, 0);
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_u32_bit_mask(
                source_op, source_attrs, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS:
        IREE_ASSERT_EQ(attr_copy->target_bit_offset, 0);
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_literal_minus_i64_attrs(
                source_op, source_attrs, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL:
        attrs[i].value = loom_attr_i64(attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(source_value);
          break;
        }
        IREE_ASSERT_GE(source_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)source_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)source_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_NEGATE: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GT(source_value, INT64_MIN);
        const int64_t projected_value = -source_value;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(projected_value);
          break;
        }
        IREE_ASSERT_GE(projected_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)projected_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value =
            loom_attr_i64((int64_t)((uint64_t)projected_value
                                    << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_LOG2: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT(loom_is_power_of_two_i64(source_value));
        const int64_t log2_value = loom_ilog2_i64(source_value);
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(log2_value);
          break;
        }
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)log2_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)log2_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GT(source_value, INT64_MIN);
        const int64_t projected_value = source_value - 1;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(projected_value);
          break;
        }
        IREE_ASSERT_GE(projected_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)projected_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value =
            loom_attr_i64((int64_t)((uint64_t)projected_value
                                    << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        loom_low_lower_u32_divisor_magic_info_t info = {0};
        const bool has_magic_info =
            loom_low_lower_rule_value_facts_u32_divisor_magic_info(
                loom_low_lower_context_module(context),
                loom_low_lower_context_fact_table(context), source_value_id,
                &info);
        IREE_ASSERT(has_magic_info);
        uint64_t projected_value =
            attr_copy->kind ==
                    LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER
                ? info.multiplier
                : info.post_shift;
        if (attr_copy->target_bit_offset != 0) {
          IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
          IREE_ASSERT_LE(projected_value,
                         (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
          projected_value <<= attr_copy->target_bit_offset;
        }
        attrs[i].value = loom_attr_i64((int64_t)projected_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_I32_AS_U32_BITS: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(rule_set, source_op,
                                             attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GE(source_value, INT32_MIN);
        IREE_ASSERT_LE(source_value, INT32_MAX);
        const uint32_t bit_pattern = (uint32_t)(int32_t)source_value;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(bit_pattern);
          break;
        }
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)bit_pattern,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)bit_pattern << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F16_BITS: {
        const float f32_value = (float)loom_low_lower_rule_attr_copy_exact_f64(
            context, rule_set, source_op, attr_copy);
        const uint16_t bit_pattern = iree_math_f32_to_f16(f32_value);
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_BF16_BITS: {
        const float f32_value = (float)loom_low_lower_rule_attr_copy_exact_f64(
            context, rule_set, source_op, attr_copy);
        const uint16_t bit_pattern = iree_math_f32_to_bf16(f32_value);
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F32_BITS: {
        const float f32_value = (float)loom_low_lower_rule_attr_copy_exact_f64(
            context, rule_set, source_op, attr_copy);
        uint32_t bit_pattern = 0;
        memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F64_BITS: {
        const double f64_value = loom_low_lower_rule_attr_copy_exact_f64(
            context, rule_set, source_op, attr_copy);
        uint64_t bit_pattern = 0;
        memcpy(&bit_pattern, &f64_value, sizeof(bit_pattern));
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET:
        attrs[i].value =
            loom_attr_i64(source_memory_access->static_byte_offset);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value = loom_attr_i64(
            source_memory_access->static_byte_offset / attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value = loom_attr_i64(
            source_memory_access->static_byte_offset % attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE:
        attrs[i].value = loom_attr_i64(
            source_memory_access->dynamic_terms[attr_copy->dynamic_term_index]
                .byte_stride);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_OP_INSTANCE_FLAGS:
        attrs[i].value = loom_attr_i64(source_op->instance_flags);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated attr copy kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  *out_attrs = loom_make_named_attr_slice(attrs, emit->attr_copy_count);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_map_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, out_type));
  IREE_ASSERT(loom_low_type_is_register(*out_type));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_low_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t** out_operands) {
  *out_operands = NULL;
  if (emit->operand_ref_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*low_operands),
      (void**)&low_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_low_value(
        context, rule_set, source_op, state, source_memory,
        source_memory_access, (uint16_t)(emit->operand_ref_start + i),
        &low_operands[i]));
  }
  *out_operands = low_operands;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_copy_low_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (emit->copy_operand_mask == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(emit->operand_ref_count, 16);
  const uint16_t valid_operand_mask =
      emit->operand_ref_count == 16
          ? UINT16_MAX
          : (uint16_t)(((uint16_t)1u << emit->operand_ref_count) - 1u);
  IREE_ASSERT_FALSE(
      iree_any_bit_set(emit->copy_operand_mask, (uint16_t)~valid_operand_mask));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(emit->copy_operand_mask, operand_bit)) {
      continue;
    }
    loom_type_t copy_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), low_operands[i], copy_type,
        source_op->location, &copy_op));
    low_operands[i] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static void loom_low_lower_rule_apply_operand_flags(
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1)) {
    IREE_ASSERT_GE(emit->operand_ref_count, 2);
    const loom_value_id_t temporary = low_operands[0];
    low_operands[0] = low_operands[1];
    low_operands[1] = temporary;
  }
}

static const loom_low_operand_t* loom_low_lower_rule_descriptor_result_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT_LT(result_index, descriptor->result_count);
  IREE_ASSERT((uint64_t)descriptor->operand_start + result_index <
              descriptor_set->operand_count);
  return &descriptor_set->operands[descriptor->operand_start + result_index];
}

static iree_status_t loom_low_lower_rule_build_result_types(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  if (emit->result_ref_count == 0) {
    return iree_ok_status();
  }
  const bool use_type_patterns = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN);
  const bool use_descriptor_types = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE);
  IREE_ASSERT_FALSE(use_type_patterns && use_descriptor_types);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count, sizeof(*result_types),
      (void**)&result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    if (use_type_patterns) {
      const uint16_t type_pattern_index =
          (uint16_t)(emit->result_type_pattern_start + i);
      const loom_type_t exact_type =
          loom_low_lower_rule_type_pattern_exact_type(
              &rule_set->type_patterns[type_pattern_index]);
      IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
          context, source_op, exact_type, &result_types[i]));
      IREE_ASSERT(loom_low_type_is_register(result_types[i]));
    } else if (use_descriptor_types) {
      const loom_low_descriptor_set_t* descriptor_set =
          loom_low_lower_context_descriptor_set(context);
      const loom_low_operand_t* operand =
          loom_low_lower_rule_descriptor_result_operand(
              descriptor_set, resolved_emit->descriptor.descriptor, i);
      IREE_ASSERT_EQ(operand->reg_class_alt_count, 1);
      const uint32_t alt_index = operand->reg_class_alt_start;
      IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      IREE_ASSERT_NE(alt->reg_class_id, LOOM_LOW_REG_CLASS_NONE);
      IREE_ASSERT_FALSE(
          iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE));
      IREE_ASSERT_GT(operand->unit_count, 0);
      IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
          context, alt->reg_class_id, operand->unit_count, &result_types[i]));
    } else {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, (uint16_t)(emit->result_ref_start + i));
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
          context, source_op, source_value_id, &result_types[i]));
    }
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static uint16_t loom_low_lower_rule_emit_result_bind_ref_index(
    const loom_low_lower_emit_t* emit, uint16_t result_ordinal) {
  IREE_ASSERT_LT(result_ordinal, emit->result_ref_count);
  const uint16_t result_bind_ref_start =
      iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS)
          ? emit->result_bind_ref_start
          : emit->result_ref_start;
  return (uint16_t)(result_bind_ref_start + result_ordinal);
}

static iree_status_t loom_low_lower_rule_bind_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    const loom_value_id_t* low_results) {
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const uint16_t value_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    switch (value_ref->kind) {
      case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
        loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
            rule_set, source_op, value_ref_index);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_value_id,
                                                       low_results[i]));
        break;
      }
      case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
        IREE_ASSERT_LT(value_ref->index, state->temporary_count);
        IREE_ASSERT(state->temporaries != NULL);
        IREE_ASSERT(state->temporaries[value_ref->index] ==
                        LOOM_VALUE_ID_INVALID ||
                    state->temporaries[value_ref->index] == low_results[i]);
        state->temporaries[value_ref->index] = low_results[i];
        break;
      case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      default:
        IREE_ASSERT_UNREACHABLE("result binding must target result refs");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_elide_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->elide_ref_count; ++i) {
    const uint16_t value_ref_index = (uint16_t)(rule->elide_ref_start + i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    IREE_ASSERT_EQ(value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    loom_value_id_t source_value_id =
        loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index);
    IREE_RETURN_IF_ERROR(loom_low_lower_elide_value(context, source_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_bind_aliases(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->alias_ref_count; ++i) {
    const uint16_t source_ref_index = (uint16_t)(rule->alias_ref_start + i * 2);
    const uint16_t result_ref_index = (uint16_t)(source_ref_index + 1);
    const loom_low_lower_value_ref_t* source_ref =
        &rule_set->value_refs[source_ref_index];
    const loom_low_lower_value_ref_t* result_ref =
        &rule_set->value_refs[result_ref_index];
    IREE_ASSERT_EQ(source_ref->kind, LOOM_LOW_LOWER_VALUE_REF_OPERAND);
    IREE_ASSERT_EQ(result_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    loom_value_id_t source_value_id =
        loom_low_lower_rule_source_value(rule_set, source_op, source_ref_index);
    loom_value_id_t result_value_id =
        loom_low_lower_rule_source_value(rule_set, source_op, result_ref_index);
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
        context, source_value_id, result_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_record_source_memory(
    loom_low_lower_context_t* context, const loom_low_lower_emit_t* emit,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    const loom_op_t* low_op) {
  if (emit->source_memory_ordinal == 0) {
    return iree_ok_status();
  }
  return loom_low_lower_record_source_memory_access(context, low_op,
                                                    source_memory_access);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));

  loom_op_t* low_const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &resolved_emit->descriptor, attrs, result_types[0],
      source_op->location, &low_const_op));
  const loom_value_id_t low_result = loom_low_const_result(low_const_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  const loom_low_lower_source_memory_t* source_memory = NULL;
  loom_low_source_memory_access_plan_t source_memory_access = {0};
  const loom_low_source_memory_access_plan_t* source_memory_access_ptr = NULL;
  if (emit->source_memory_ordinal != 0) {
    const uint16_t source_memory_index =
        (uint16_t)(emit->source_memory_ordinal - 1);
    source_memory = &rule_set->source_memories[source_memory_index];
    loom_low_lower_rule_source_memory_access(context, rule_set, source_op, emit,
                                             &source_memory_access);
    source_memory_access_ptr = &source_memory_access;
  }

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, source_memory,
      source_memory_access_ptr, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));
  loom_low_lower_rule_apply_operand_flags(emit, low_operands);

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, source_memory_access_ptr, &attrs));

  const loom_tied_result_t* tied_results = NULL;
  if (emit->tied_result_count != 0) {
    tied_results = &rule_set->tied_results[emit->tied_result_start];
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_emit->descriptor, low_operands,
      emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
      tied_results, emit->tied_result_count, source_op->location, &low_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_record_source_memory(
      context, emit, source_memory_access_ptr, low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static const loom_tied_result_t* loom_low_lower_rule_emit_tied_results(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  return emit->tied_result_count == 0
             ? NULL
             : &rule_set->tied_results[emit->tied_result_start];
}

static const loom_low_operand_t* loom_low_lower_rule_descriptor_packet_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t packet_operand_index) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  uint16_t packet_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    if (packet_index == packet_operand_index) {
      return operand;
    }
    ++packet_index;
  }
  IREE_ASSERT_UNREACHABLE("descriptor packet operand index is out of range");
  return NULL;
}

static iree_status_t loom_low_lower_rule_slice_register_units(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, uint32_t unit_offset, uint32_t unit_count,
    loom_value_id_t* out_slice_value_id) {
  *out_slice_value_id = LOOM_VALUE_ID_INVALID;
  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  const uint32_t total_unit_count = loom_low_register_type_unit_count(low_type);
  IREE_ASSERT_GT(unit_count, 0);
  IREE_ASSERT_LE(unit_offset, total_unit_count);
  IREE_ASSERT_LE(unit_count, total_unit_count - unit_offset);
  if (unit_offset == 0 && unit_count == total_unit_count) {
    *out_slice_value_id = low_value_id;
    return iree_ok_status();
  }
  const loom_type_t slice_type =
      loom_low_register_type_with_unit_count(low_type, unit_count);
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id, unit_offset,
      slice_type, source_op->location, &slice_op));
  *out_slice_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_slice_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane_value_id) {
  *out_lane_value_id = LOOM_VALUE_ID_INVALID;
  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(low_type),
                 loom_low_register_type_descriptor_set_stable_id(lane_type));
  IREE_ASSERT_EQ(loom_low_register_type_class_id(low_type),
                 loom_low_register_type_class_id(lane_type));
  IREE_ASSERT_LT(lane_index, loom_low_register_type_unit_count(low_type));
  if (loom_low_register_type_unit_count(low_type) == 1) {
    *out_lane_value_id = low_value_id;
    return iree_ok_status();
  }
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id, lane_index,
      lane_type, source_op->location, &slice_op));
  *out_lane_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static loom_type_t loom_low_lower_rule_register_lane_type(
    const loom_module_t* module, loom_value_id_t low_value_id) {
  const loom_type_t low_type = loom_module_value_type(module, low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  IREE_ASSERT_GT(loom_low_register_type_unit_count(low_type), 0);
  return loom_low_register_type_with_unit_count(low_type, 1);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_first_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));
  IREE_ASSERT(loom_low_type_is_register(result_types[0]));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_types[0]), 1);

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    const loom_type_t operand_lane_type =
        loom_low_register_type_with_unit_count(operand_type, 1);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_slice_lane(context, source_op, low_operands[i], 0,
                                       operand_lane_type, &lane_operands[i]));
  }
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));

  loom_op_t* low_op = NULL;
  const loom_tied_result_t* tied_results =
      loom_low_lower_rule_emit_tied_results(rule_set, emit);
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_emit->descriptor, lane_operands,
      emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
      tied_results, emit->tied_result_count, source_op->location, &low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_per_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_GT(emit->result_ref_count, 0);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      resolved_emit->descriptor.descriptor;
  uint32_t lane_count = 0;
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const loom_low_operand_t* packet_operand =
        loom_low_lower_rule_descriptor_packet_operand(descriptor_set,
                                                      descriptor, i);
    IREE_ASSERT_GT(packet_operand->unit_count, 0);
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    const uint32_t operand_unit_count =
        loom_low_register_type_unit_count(operand_type);
    IREE_ASSERT_GT(operand_unit_count, 0);
    IREE_ASSERT_EQ(operand_unit_count % packet_operand->unit_count, 0);
    const uint32_t operand_lane_count =
        operand_unit_count / packet_operand->unit_count;
    if (lane_count == 0) {
      lane_count = operand_lane_count;
    } else {
      IREE_ASSERT_EQ(operand_lane_count, lane_count);
    }
  }
  IREE_ASSERT_GT(lane_count, 0);

  loom_type_t* lane_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count, sizeof(*lane_result_types),
      (void**)&lane_result_types));
  loom_type_t* aggregate_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count, sizeof(*aggregate_result_types),
      (void**)&aggregate_result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const loom_low_operand_t* result_operand =
        loom_low_lower_rule_descriptor_result_operand(descriptor_set,
                                                      descriptor, i);
    IREE_ASSERT_GT(result_operand->unit_count, 0);
    IREE_ASSERT(loom_low_type_is_register(result_types[i]));
    const uint32_t type_unit_count =
        loom_low_register_type_unit_count(result_types[i]);
    IREE_ASSERT(type_unit_count == result_operand->unit_count ||
                type_unit_count == result_operand->unit_count * lane_count);
    lane_result_types[i] = loom_low_register_type_with_unit_count(
        result_types[i], result_operand->unit_count);
    aggregate_result_types[i] =
        type_unit_count == result_operand->unit_count * lane_count
            ? result_types[i]
            : loom_low_register_type_with_unit_count(
                  result_types[i], result_operand->unit_count * lane_count);
  }

  const loom_tied_result_t* tied_results =
      loom_low_lower_rule_emit_tied_results(rule_set, emit);
  if (lane_count == 1) {
    loom_low_lower_rule_apply_operand_flags(emit, low_operands);
    loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
        context, rule_set, source_op, emit, NULL, &attrs));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, low_operands,
        emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
        tied_results, emit->tied_result_count, source_op->location, &low_op));
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit,
                                            loom_low_op_results(low_op).values);
  }

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));
  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count * lane_count, sizeof(*lane_results),
      (void**)&lane_results));
  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
         ++operand_index) {
      const loom_low_operand_t* packet_operand =
          loom_low_lower_rule_descriptor_packet_operand(
              descriptor_set, descriptor, operand_index);
      const uint32_t operand_unit_count = packet_operand->unit_count;
      const uint32_t register_offset = lane_index * operand_unit_count;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_register_units(
          context, source_op, low_operands[operand_index], register_offset,
          operand_unit_count, &lane_operands[operand_index]));
    }
    loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, lane_operands,
        emit->operand_ref_count, attrs, lane_result_types,
        emit->result_ref_count, tied_results, emit->tied_result_count,
        source_op->location, &lane_op));
    const loom_value_slice_t low_results = loom_low_op_results(lane_op);
    for (uint16_t result_index = 0; result_index < emit->result_ref_count;
         ++result_index) {
      lane_results[result_index * lane_count + lane_index] =
          loom_value_slice_get(low_results, result_index);
    }
  }

  loom_value_id_t* low_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count, sizeof(*low_results),
      (void**)&low_results));
  for (uint16_t result_index = 0; result_index < emit->result_ref_count;
       ++result_index) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context),
        &lane_results[result_index * lane_count], lane_count,
        aggregate_result_types[result_index], source_op->location, &concat_op));
    low_results[result_index] = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results);
}

static iree_status_t loom_low_lower_rule_build_lane_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit, uint32_t lane_count, uint32_t lane_index,
    loom_value_id_t* lane_operands) {
  for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
       ++operand_index) {
    loom_value_id_t low_operand = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_low_value(
        context, rule_set, source_op, state, NULL, NULL,
        (uint16_t)(emit->operand_ref_start + operand_index), &low_operand));
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operand);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    const uint32_t operand_unit_count =
        loom_low_register_type_unit_count(operand_type);
    if (operand_unit_count == 1) {
      lane_operands[operand_index] = low_operand;
      continue;
    }
    IREE_ASSERT_EQ(operand_unit_count, lane_count);
    const loom_type_t operand_lane_type =
        loom_low_register_type_with_unit_count(operand_type, 1);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
        context, source_op, low_operand, lane_index, operand_lane_type,
        &lane_operands[operand_index]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, lane_operands));
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
  return iree_ok_status();
}

static void loom_low_lower_rule_bind_per_lane_sequence_result(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    bool is_final_emit, loom_value_id_t low_result, uint32_t lane_index,
    loom_value_id_t* lane_results) {
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  const uint16_t value_ref_index =
      loom_low_lower_rule_emit_result_bind_ref_index(emit, 0);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_FALSE(is_final_emit);
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      state->temporaries[value_ref->index] = low_result;
      break;
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT(is_final_emit);
      lane_results[lane_index] = low_result;
      break;
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    default:
      IREE_ASSERT_UNREACHABLE(
          "per-lane sequence results must bind temporaries or results");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_per_lane_sequence(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t* resolved_emits) {
  IREE_ASSERT_GT(rule->emit_count, 1);
  const uint16_t final_emit_ordinal = (uint16_t)(rule->emit_count - 1);
  const loom_low_lower_emit_t* final_emit =
      resolved_emits[final_emit_ordinal].emit;
  IREE_ASSERT_EQ(final_emit->kind,
                 LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE);
  IREE_ASSERT_EQ(final_emit->result_ref_count, 1);

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->emit_count, sizeof(*result_types), (void**)&result_types));
  loom_type_t* lane_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->emit_count, sizeof(*lane_result_types),
      (void**)&lane_result_types));
  loom_named_attr_slice_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->emit_count, sizeof(*attrs), (void**)&attrs));
  uint16_t max_operand_ref_count = 0;
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const loom_low_lower_resolved_emit_t* resolved_emit =
        &resolved_emits[emit_ordinal];
    const loom_low_lower_emit_t* emit = resolved_emit->emit;
    IREE_ASSERT_EQ(emit->kind,
                   LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE);
    IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);
    IREE_ASSERT_EQ(emit->result_ref_count, 1);
    const uint16_t bind_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, 0);
    const loom_low_lower_value_ref_t* bind_ref =
        &rule_set->value_refs[bind_ref_index];
    if (emit_ordinal == final_emit_ordinal) {
      IREE_ASSERT_EQ(bind_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    } else {
      IREE_ASSERT_EQ(bind_ref->kind, LOOM_LOW_LOWER_VALUE_REF_TEMPORARY);
    }
    loom_type_t* emit_result_types = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
        context, rule_set, source_op, resolved_emit, &emit_result_types));
    result_types[emit_ordinal] = emit_result_types[0];
    IREE_ASSERT(loom_low_type_is_register(result_types[emit_ordinal]));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
        context, rule_set, source_op, emit, NULL, &attrs[emit_ordinal]));
    max_operand_ref_count =
        iree_max(max_operand_ref_count, emit->operand_ref_count);
  }

  const loom_type_t final_result_type = result_types[final_emit_ordinal];
  const uint32_t lane_count =
      loom_low_register_type_unit_count(final_result_type);
  IREE_ASSERT_GT(lane_count, 0);
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const loom_type_t result_type = result_types[emit_ordinal];
    const uint32_t result_unit_count =
        loom_low_register_type_unit_count(result_type);
    IREE_ASSERT(result_unit_count == 1 || result_unit_count == lane_count);
    lane_result_types[emit_ordinal] =
        loom_low_register_type_with_unit_count(result_type, 1);
  }

  loom_value_id_t* lane_operands = NULL;
  if (max_operand_ref_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
        context, max_operand_ref_count, sizeof(*lane_operands),
        (void**)&lane_operands));
  }
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, lane_count, sizeof(*lane_results), (void**)&lane_results));

  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
         ++emit_ordinal) {
      const loom_low_lower_resolved_emit_t* resolved_emit =
          &resolved_emits[emit_ordinal];
      const loom_low_lower_emit_t* emit = resolved_emit->emit;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_lane_operands(
          context, rule_set, source_op, state, emit, lane_count, lane_index,
          lane_operands));
      loom_op_t* lane_op = NULL;
      const loom_tied_result_t* tied_results =
          loom_low_lower_rule_emit_tied_results(rule_set, emit);
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &resolved_emit->descriptor, lane_operands,
          emit->operand_ref_count, attrs[emit_ordinal],
          &lane_result_types[emit_ordinal], 1, tied_results,
          emit->tied_result_count, source_op->location, &lane_op));
      loom_low_lower_rule_bind_per_lane_sequence_result(
          rule_set, state, emit, emit_ordinal == final_emit_ordinal,
          loom_value_slice_get(loom_low_op_results(lane_op), 0), lane_index,
          lane_results);
    }
  }

  if (lane_count == 1) {
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            final_emit, lane_results);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      final_result_type, source_op->location, &concat_op));
  const loom_value_id_t low_result = loom_low_concat_result(concat_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          final_emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_accumulate_lanes(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 1);
  IREE_ASSERT_LT(emit->accumulator_operand_index, emit->operand_ref_count);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  IREE_ASSERT_EQ(emit->attr_copy_count, 0);
  IREE_ASSERT_EQ(emit->tied_result_count, 0);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);
  const loom_low_lower_emit_flags_t supported_flags =
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE |
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED |
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE;
  IREE_ASSERT_EQ(emit->flags & ~supported_flags, 0);
  const bool seed_first_lane = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE);
  const bool balanced_tree = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED);
  const bool skip_first_lane = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE);
  IREE_ASSERT_FALSE(seed_first_lane && skip_first_lane);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));

  loom_value_id_t source_result = loom_low_lower_rule_source_value(
      rule_set, source_op, emit->result_ref_start);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), 1);

  uint32_t lane_count = 0;
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    if (i == emit->accumulator_operand_index) {
      continue;
    }
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    if (lane_count == 0) {
      lane_count = loom_low_register_type_unit_count(operand_type);
      IREE_ASSERT_GT(lane_count, 0);
    } else {
      IREE_ASSERT_EQ(loom_low_register_type_unit_count(operand_type),
                     lane_count);
    }
  }
  IREE_ASSERT_GT(lane_count, 0);

  const loom_type_t result_type_scalar = result_type;
  loom_value_id_t accumulator = low_operands[emit->accumulator_operand_index];
  const loom_type_t accumulator_type = loom_module_value_type(
      loom_low_lower_context_module(context), accumulator);
  IREE_ASSERT(loom_low_type_is_register(accumulator_type));
  uint32_t first_lane_index = 0;
  if (seed_first_lane) {
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(accumulator_type),
                   lane_count);
    const loom_type_t accumulator_lane_type =
        loom_low_register_type_with_unit_count(accumulator_type, 1);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_slice_lane(context, source_op, accumulator, 0,
                                       accumulator_lane_type, &accumulator));
    first_lane_index = 1;
  } else {
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(accumulator_type), 1);
    first_lane_index = skip_first_lane ? 1 : 0;
  }

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  if (balanced_tree) {
    IREE_ASSERT_EQ(emit->operand_ref_count, 2);
    const uint16_t term_operand_index =
        emit->accumulator_operand_index == 0 ? 1 : 0;
    const uint32_t term_count = (seed_first_lane || skip_first_lane)
                                    ? lane_count
                                    : (uint32_t)(lane_count + 1);
    loom_value_id_t* lane_terms = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
        context, term_count, sizeof(*lane_terms), (void**)&lane_terms));
    uint32_t term_index = 0;
    lane_terms[term_index++] = accumulator;
    const loom_type_t term_operand_lane_type =
        loom_low_lower_rule_register_lane_type(
            loom_low_lower_context_module(context),
            low_operands[term_operand_index]);
    for (uint32_t lane_index = first_lane_index; lane_index < lane_count;
         ++lane_index) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
          context, source_op, low_operands[term_operand_index], lane_index,
          term_operand_lane_type, &lane_terms[term_index++]));
    }
    IREE_ASSERT_EQ(term_index, term_count);

    for (uint32_t step = 1; step < term_count; step <<= 1) {
      for (uint32_t lane_index = 0; lane_index + step < term_count;
           lane_index += step << 1) {
        lane_operands[emit->accumulator_operand_index] = lane_terms[lane_index];
        lane_operands[term_operand_index] = lane_terms[lane_index + step];
        loom_op_t* lane_op = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
            context, &resolved_emit->descriptor, lane_operands,
            emit->operand_ref_count, loom_make_named_attr_slice(NULL, 0),
            &result_type_scalar, 1, NULL, 0, source_op->location, &lane_op));
        lane_terms[lane_index] =
            loom_value_slice_get(loom_low_op_results(lane_op), 0);
      }
    }
    accumulator = lane_terms[0];
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit, &accumulator);
  }

  for (uint32_t lane_index = first_lane_index; lane_index < lane_count;
       ++lane_index) {
    for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
         ++operand_index) {
      if (operand_index == emit->accumulator_operand_index) {
        lane_operands[operand_index] = accumulator;
        continue;
      }
      const loom_type_t operand_lane_type =
          loom_low_lower_rule_register_lane_type(
              loom_low_lower_context_module(context),
              low_operands[operand_index]);
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
          context, source_op, low_operands[operand_index], lane_index,
          operand_lane_type, &lane_operands[operand_index]));
    }
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, lane_operands,
        emit->operand_ref_count, loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, NULL, 0, source_op->location, &lane_op));
    accumulator = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &accumulator);
}

iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t* resolved_emits) {
  IREE_ASSERT(rule->emit_count == 0 || resolved_emits != NULL);

  loom_low_lower_rule_emit_state_t state = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_emit_state_initialize(context, rule, &state));
  if (rule->emit_count != 0 &&
      resolved_emits[0].emit->kind ==
          LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_emit_descriptor_op_per_lane_sequence(
            context, rule_set, source_op, &state, rule, resolved_emits));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_bind_aliases(context, rule_set, source_op, rule));
    return loom_low_lower_rule_elide_results(context, rule_set, source_op,
                                             rule);
  }
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    const loom_low_lower_resolved_emit_t* resolved_emit = &resolved_emits[i];
    IREE_ASSERT(resolved_emit->emit == emit);
    switch (emit->kind) {
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_const(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op_first_lane(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op_per_lane(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE:
        IREE_ASSERT_UNREACHABLE(
            "per-lane sequence emits must be the whole emit program");
        IREE_BUILTIN_UNREACHABLE();
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES: {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_rule_emit_descriptor_op_accumulate_lanes(
                context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated lower emit kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_bind_aliases(context, rule_set, source_op, rule));
  return loom_low_lower_rule_elide_results(context, rule_set, source_op, rule);
}
