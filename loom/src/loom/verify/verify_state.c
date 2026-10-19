// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_state.h"

#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/util/adaptive_sort.h"

static bool loom_verify_value_id_less(const loom_value_id_t* lhs,
                                      const loom_value_id_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_verify_value_id_sort, loom_value_id_t,
                          loom_verify_value_id_less)

void loom_verify_record_diagnostic_status(loom_verify_state_t* state,
                                          iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  if (iree_status_is_ok(state->diagnostic_status)) {
    state->diagnostic_status = status;
  } else {
    iree_status_free(status);
  }
}

iree_status_t loom_verify_take_diagnostic_status(loom_verify_state_t* state) {
  iree_status_t status = state->diagnostic_status;
  state->diagnostic_status = iree_ok_status();
  return status;
}

iree_status_t loom_verify_pending_diagnostic_status(
    loom_verify_state_t* state) {
  if (iree_status_is_ok(state->diagnostic_status)) {
    return iree_ok_status();
  }
  return loom_verify_take_diagnostic_status(state);
}

iree_status_t loom_verify_sorted_values_assign(loom_verify_state_t* state,
                                               const loom_value_id_t* values,
                                               iree_host_size_t count) {
  return loom_verify_sorted_values_assign_pair(state, values, count, NULL, 0);
}

iree_status_t loom_verify_sorted_values_assign_pair(
    loom_verify_state_t* state, const loom_value_id_t* first_values,
    iree_host_size_t first_count, const loom_value_id_t* second_values,
    iree_host_size_t second_count) {
  iree_host_size_t total_count = 0;
  if (!iree_host_size_checked_add(first_count, second_count, &total_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "verifier value-set size overflow");
  }
  if (total_count > state->sorted_values.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &state->arena, state->sorted_values.count, total_count,
        sizeof(loom_value_id_t), &state->sorted_values.capacity,
        (void**)&state->sorted_values.values));
  }
  state->sorted_values.count = 0;
  const loom_value_id_t* slices[] = {first_values, second_values};
  const iree_host_size_t counts[] = {first_count, second_count};
  for (iree_host_size_t slice = 0; slice < IREE_ARRAYSIZE(slices); ++slice) {
    for (iree_host_size_t i = 0; i < counts[slice]; ++i) {
      const loom_value_id_t value_id = slices[slice][i];
      if (value_id == LOOM_VALUE_ID_INVALID ||
          value_id >= state->module->values.count) {
        continue;
      }
      state->sorted_values.values[state->sorted_values.count++] = value_id;
    }
  }
  if (state->sorted_values.count > 1) {
    loom_verify_value_id_sort(state->sorted_values.values,
                              state->sorted_values.count);
  }
  return iree_ok_status();
}

static iree_host_size_t loom_verify_sorted_values_lower_bound(
    const loom_verify_state_t* state, loom_value_id_t value_id) {
  iree_host_size_t low = 0;
  iree_host_size_t high = state->sorted_values.count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    if (state->sorted_values.values[middle] < value_id) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

bool loom_verify_sorted_values_contains(const loom_verify_state_t* state,
                                        loom_value_id_t value_id) {
  const iree_host_size_t index =
      loom_verify_sorted_values_lower_bound(state, value_id);
  return index < state->sorted_values.count &&
         state->sorted_values.values[index] == value_id;
}

bool loom_verify_sorted_values_contain_duplicate(
    const loom_verify_state_t* state, loom_value_id_t value_id) {
  const iree_host_size_t index =
      loom_verify_sorted_values_lower_bound(state, value_id);
  return index + 1 < state->sorted_values.count &&
         state->sorted_values.values[index] == value_id &&
         state->sorted_values.values[index + 1] == value_id;
}

// Returns true if this op has a function signature scope whose arguments are
// defined by FuncArgs and may be referenced by result types/predicates/ties.
bool loom_verify_has_func_signature_scope(const loom_op_vtable_t* vtable) {
  return vtable->func_like != NULL &&
         iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
}

iree_status_t loom_verify_push_scope(loom_verify_state_t* state,
                                     bool isolated) {
  if (state->scope_depth >= LOOM_VERIFY_INLINE_SCOPE_DEPTH) {
    if (state->scope_depth >= LOOM_VERIFY_MAX_SCOPE_DEPTH) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "region nesting depth exceeds verifier limit (%d); the input IR "
          "has more nesting than the verifier can track",
          LOOM_VERIFY_MAX_SCOPE_DEPTH);
    }
    if (state->scope_watermark_storage == state->scope_watermarks) {
      const iree_host_size_t spill_storage_size =
          LOOM_VERIFY_MAX_SCOPE_DEPTH *
          (sizeof(iree_host_size_t) + sizeof(uint8_t));
      uint8_t* spill_storage = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &state->arena, spill_storage_size, (void**)&spill_storage));
      iree_host_size_t* spill_watermarks = (iree_host_size_t*)spill_storage;
      uint8_t* spill_minimum_depths =
          spill_storage +
          LOOM_VERIFY_MAX_SCOPE_DEPTH * sizeof(iree_host_size_t);
      memcpy(spill_watermarks, state->scope_watermarks,
             sizeof(state->scope_watermarks));
      memcpy(spill_minimum_depths, state->visibility.scope_minimum_depths,
             sizeof(state->visibility.scope_minimum_depths));
      state->scope_watermark_storage = spill_watermarks;
      state->visibility.scope_minimum_depth_storage = spill_minimum_depths;
    }
  }
  state->scope_watermark_storage[state->scope_depth] =
      state->defined_stack_count;
  state->visibility.scope_minimum_depth_storage[state->scope_depth] =
      state->visibility.minimum_depth;
  ++state->scope_depth;
  if (isolated) {
    state->visibility.minimum_depth = (uint8_t)(state->scope_depth + 1);
  }
  return iree_ok_status();
}

void loom_verify_restore_definitions(loom_verify_state_t* state,
                                     iree_host_size_t watermark) {
  // Remove only definitions introduced after the retained watermark.
  for (iree_host_size_t i = watermark; i < state->defined_stack_count; ++i) {
    state->visibility.definition_depths[state->defined_stack[i]] = 0;
  }
  state->defined_stack_count = watermark;
}

void loom_verify_pop_scope(loom_verify_state_t* state) {
  if (state->scope_depth == 0) {
    return;
  }
  --state->scope_depth;
  loom_verify_restore_definitions(
      state, state->scope_watermark_storage[state->scope_depth]);
  state->visibility.minimum_depth =
      state->visibility.scope_minimum_depth_storage[state->scope_depth];
}

iree_status_t loom_verify_define_value(loom_verify_state_t* state,
                                       loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  state->visibility.definition_depths[value_id] =
      (uint8_t)(state->scope_depth + 1);
  // Push onto defined stack for scope cleanup. Grow dynamically if
  // the initial capacity heuristic was too small.
  if (state->defined_stack_count >= state->defined_stack_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &state->arena, state->defined_stack_count,
        state->defined_stack_count + 1, sizeof(uint32_t),
        &state->defined_stack_capacity, (void**)&state->defined_stack));
  }
  state->defined_stack[state->defined_stack_count++] = value_id;
  return iree_ok_status();
}

iree_string_view_t loom_verify_value_name(const loom_verify_state_t* state,
                                          loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return iree_make_cstring_view("<unnamed>");
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (value->name_id != LOOM_STRING_ID_INVALID &&
      value->name_id < state->module->strings.count) {
    return loom_string_table_get(&state->module->strings, value->name_id);
  }
  return iree_make_cstring_view("<unnamed>");
}

// Returns the symbol name for a symbol ref, or "<unnamed>" if unavailable.
iree_string_view_t loom_verify_symbol_name(const loom_verify_state_t* state,
                                           loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) ||
      ref.symbol_id >= state->module->symbols.count) {
    return iree_make_cstring_view("<unnamed>");
  }
  const loom_symbol_t* symbol = &state->module->symbols.entries[ref.symbol_id];
  if (symbol->name_id != LOOM_STRING_ID_INVALID &&
      symbol->name_id < state->module->strings.count) {
    return loom_string_table_get(&state->module->strings, symbol->name_id);
  }
  return iree_make_cstring_view("<unnamed>");
}

iree_string_view_t loom_verify_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) {
    return IREE_SV("unresolved");
  }
  return loom_symbol_definition_descriptor_name(symbol->definition);
}
bool loom_verify_at_error_limit(const loom_verify_state_t* state) {
  return state->max_errors > 0 &&
         state->result->error_count >= state->max_errors;
}

const loom_op_vtable_t* loom_verify_lookup_vtable(
    const loom_verify_state_t* state, loom_op_kind_t kind) {
  return loom_context_resolve_op(state->module->context, kind);
}

loom_type_t loom_verify_value_type(const loom_verify_state_t* state,
                                   loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    loom_type_t none = {0};
    return none;
  }
  if (value_id >= state->module->values.count) {
    loom_type_t none = {0};
    return none;
  }
  return loom_module_value_type(state->module, value_id);
}

// Resolves a field reference on an op to a value_id. For operand and
// result categories, returns the value_id at the given index. For
// attr and region categories, returns LOOM_VALUE_ID_INVALID (these
// fields are not values).
loom_value_id_t loom_verify_resolve_value_field(const loom_op_t* op,
                                                const loom_op_vtable_t* vtable,
                                                uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND: {
      if (loom_op_vtable_has_segmented_operands(vtable)) {
        loom_value_slice_t span = loom_op_operand_field_span(vtable, op, index);
        return span.count > 0 ? span.values[0] : LOOM_VALUE_ID_INVALID;
      }
      if (index < op->operand_count) {
        return loom_op_const_operands(op)[index];
      }
      break;
    }
    case LOOM_FIELD_RESULT:
      if (index < op->result_count) {
        return loom_op_const_results(op)[index];
      }
      break;
    default:
      break;
  }
  return LOOM_VALUE_ID_INVALID;
}

// Returns true if a field reference points to a variadic field.
// A variadic operand has index == fixed_operand_count when the vtable
// has LOOM_OP_VTABLE_VARIADIC_OPERANDS; similarly for results.
bool loom_verify_is_variadic_field(const loom_op_vtable_t* vtable,
                                   uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND: {
      if (loom_op_vtable_has_segmented_operands(vtable)) {
        uint8_t descriptor_count =
            loom_op_vtable_operand_descriptor_count(vtable);
        return index < descriptor_count &&
               iree_any_bit_set(vtable->operand_descriptors[index].flags,
                                LOOM_OPERAND_VARIADIC);
      }
      return (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) &&
             index == vtable->fixed_operand_count;
    }
    case LOOM_FIELD_RESULT:
      return (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) &&
             index == vtable->fixed_result_count;
    default:
      return false;
  }
}

// Returns the count of a variadic field. For operands, this is
// op->operand_count - vtable->fixed_operand_count. For results, similar.
uint16_t loom_verify_variadic_count(const loom_op_t* op,
                                    const loom_op_vtable_t* vtable,
                                    uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      if (loom_op_vtable_has_segmented_operands(vtable)) {
        return loom_op_operand_field_span(vtable, op, index).count;
      }
      if (index <= vtable->fixed_operand_count) {
        return (uint16_t)(op->operand_count - index);
      }
      break;
    case LOOM_FIELD_RESULT:
      if (index <= vtable->fixed_result_count) {
        return (uint16_t)(op->result_count - index);
      }
      break;
    default:
      break;
  }
  return 0;
}

iree_string_view_t loom_verify_field_name(const loom_op_vtable_t* vtable,
                                          uint8_t field_ref, char* buffer,
                                          iree_host_size_t buffer_size) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  if (category == LOOM_FIELD_OPERAND && vtable->operand_descriptors) {
    uint8_t descriptor_count = loom_op_vtable_operand_descriptor_count(vtable);
    if (index < descriptor_count) {
      return loom_bstring_view(vtable->operand_descriptors[index].name);
    }
  } else if (category == LOOM_FIELD_RESULT && vtable->result_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
    uint8_t descriptor_count =
        (uint8_t)(vtable->fixed_result_count + (has_variadic ? 1 : 0));
    if (index < descriptor_count) {
      return loom_bstring_view(vtable->result_descriptors[index].name);
    }
  } else if (category == LOOM_FIELD_ATTR && vtable->attr_descriptors &&
             index < vtable->attribute_count) {
    return loom_bstring_view(vtable->attr_descriptors[index].name);
  }
  const char* prefix = "field";
  if (category == LOOM_FIELD_OPERAND) {
    prefix = "operand";
  } else if (category == LOOM_FIELD_RESULT) {
    prefix = "result";
  } else if (category == LOOM_FIELD_ATTR) {
    prefix = "attribute";
  } else if (category == LOOM_FIELD_REGION) {
    prefix = "region";
  }
  iree_snprintf(buffer, buffer_size, "%s %u", prefix, index);
  return iree_make_cstring_view(buffer);
}

// Like loom_verify_field_name but for an indexed element of a variadic field.
// Produces names like "inputs[2]" or "results[0]".
iree_string_view_t loom_verify_indexed_field_name(
    const loom_op_vtable_t* vtable, uint8_t field_ref, uint16_t element_index,
    char* buffer, iree_host_size_t buffer_size) {
  char field_name_buffer[64];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                field_name.data, element_index);
  return iree_make_cstring_view(buffer);
}

// Resolves an ordinary operand/result value index to the declared field name.
// Variadic values use their declaring field name plus an element index, such as
// "inputs[2]".
iree_string_view_t loom_verify_value_field_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op, uint8_t category,
    uint16_t value_index, char* buffer, iree_host_size_t buffer_size) {
  if (category == LOOM_FIELD_OPERAND && vtable->operand_descriptors) {
    if (loom_op_vtable_has_segmented_operands(vtable)) {
      const loom_operand_descriptor_t* descriptor = NULL;
      uint8_t field_index = 0;
      uint16_t element_index = 0;
      if (loom_op_operand_descriptor_at(vtable, op, value_index, &descriptor,
                                        &field_index, &element_index)) {
        iree_string_view_t field_name = loom_bstring_view(descriptor->name);
        if (!iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
          return field_name;
        }
        iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                      field_name.data, element_index);
        return iree_make_cstring_view(buffer);
      }
    }
    uint8_t descriptor_count = loom_op_vtable_operand_descriptor_count(vtable);
    if (value_index < descriptor_count &&
        !iree_any_bit_set(vtable->vtable_flags,
                          LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
      return loom_bstring_view(vtable->operand_descriptors[value_index].name);
    }
    if (value_index < vtable->fixed_operand_count) {
      return loom_bstring_view(vtable->operand_descriptors[value_index].name);
    }
    if (iree_any_bit_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
      uint16_t element_index = value_index - vtable->fixed_operand_count;
      iree_string_view_t field_name = loom_bstring_view(
          vtable->operand_descriptors[vtable->fixed_operand_count].name);
      iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                    field_name.data, element_index);
      return iree_make_cstring_view(buffer);
    }
  } else if (category == LOOM_FIELD_RESULT && vtable->result_descriptors) {
    if (value_index < vtable->fixed_result_count) {
      return loom_bstring_view(vtable->result_descriptors[value_index].name);
    }
    if (iree_any_bit_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_VARIADIC_RESULTS)) {
      uint16_t element_index = value_index - vtable->fixed_result_count;
      iree_string_view_t field_name = loom_bstring_view(
          vtable->result_descriptors[vtable->fixed_result_count].name);
      iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                    field_name.data, element_index);
      return iree_make_cstring_view(buffer);
    }
  }
  const char* prefix = category == LOOM_FIELD_RESULT ? "result" : "operand";
  iree_snprintf(buffer, buffer_size, "%s %u", prefix, value_index);
  return iree_make_cstring_view(buffer);
}
// Resolves a variadic value field reference to its value array and
// element count. Returns NULL with |*out_count| = 0 if the field is
// not a value-bearing variadic (e.g., it points at an attr or region,
// or its start index is past the op's operand/result range).
const loom_value_id_t* loom_verify_resolve_variadic_field(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint8_t field_ref,
    uint16_t* out_count) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t start_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (category == LOOM_FIELD_OPERAND &&
      loom_op_vtable_has_segmented_operands(vtable)) {
    loom_value_slice_t span =
        loom_op_operand_field_span(vtable, op, start_index);
    *out_count = span.count;
    return span.values;
  }
  if (category == LOOM_FIELD_OPERAND && start_index <= op->operand_count) {
    *out_count = (uint16_t)(op->operand_count - start_index);
    return loom_op_const_operands(op) + start_index;
  }
  if (category == LOOM_FIELD_RESULT && start_index <= op->result_count) {
    *out_count = (uint16_t)(op->result_count - start_index);
    return loom_op_const_results(op) + start_index;
  }
  *out_count = 0;
  return NULL;
}
