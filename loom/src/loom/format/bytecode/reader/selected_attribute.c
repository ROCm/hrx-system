// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_attribute.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/selected_tables.h"

typedef enum loom_bytecode_selected_attribute_value_domain_e {
  LOOM_BYTECODE_SELECTED_ATTRIBUTE_VALUE_DOMAIN_NAMED = 0,
  LOOM_BYTECODE_SELECTED_ATTRIBUTE_VALUE_DOMAIN_SSA = 1,
} loom_bytecode_selected_attribute_value_domain_t;

typedef struct loom_bytecode_selected_attribute_scope_t {
  // Concrete wire namespace used by predicate VALUE arguments.
  loom_bytecode_selected_attribute_value_domain_t value_domain;
  // SSA value map when |value_domain| selects SSA numbers.
  loom_bytecode_attribute_ssa_materialization_scope_t ssa;
} loom_bytecode_selected_attribute_scope_t;

static iree_status_t loom_bytecode_selected_attribute_source_string(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_source_string) {
  if (source_string_id >= materializer->metadata->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(source_string_id),
        loom_param_u64(materializer->metadata->strings.count),
    };
    return loom_bytecode_reader_emit_error(materializer->decoder,
                                           LOOM_ERR_BYTECODE_010, params,
                                           IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_source_string = materializer->metadata->strings.values[source_string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_attribute_target_string(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_source_string,
    loom_string_id_t* out_target_string_id) {
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_source_string(
      materializer, source_string_id, field_name, offset, out_source_string));
  return loom_bytecode_selected_table_intern_string(
      materializer, (uint32_t)source_string_id, out_target_string_id);
}

static iree_status_t loom_bytecode_selected_attribute_emit_invalid_ssa_value(
    loom_bytecode_selected_table_materializer_t* materializer,
    iree_string_view_t symbol_name, uint64_t offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
      loom_param_u64(offset),
      loom_param_string(
          IREE_SV("predicate value reference must target a previously defined "
                  "value")),
  };
  return loom_bytecode_reader_emit_error(materializer->decoder,
                                         LOOM_ERR_BYTECODE_016, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_selected_attribute_decode_at_depth(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_selected_attribute_scope_t* scope,
    uint8_t aggregate_depth,
    loom_bytecode_selected_attribute_state_t* out_state);

static iree_status_t loom_bytecode_selected_attribute_decode_predicate_list(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_selected_attribute_scope_t* scope,
    loom_attribute_t* out_attr) {
  uint16_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_predicate_count(
      materializer->decoder, cursor, &predicate_count));
  loom_predicate_t* predicates = NULL;
  if (predicate_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(materializer->scratch_arena, predicate_count,
                                  sizeof(*predicates), (void**)&predicates));
  }
  for (uint16_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    loom_bytecode_wire_predicate_t wire_predicate;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_predicate(
        materializer->decoder, cursor, predicate_index, &wire_predicate));
    predicates[predicate_index] = wire_predicate.value;
    for (uint8_t i = 0; i < wire_predicate.value.arg_count; ++i) {
      if (wire_predicate.value.arg_tags[i] != LOOM_PRED_ARG_VALUE) {
        continue;
      }
      const uint64_t value_number = wire_predicate.value_numbers[i];
      if (scope->value_domain ==
          LOOM_BYTECODE_SELECTED_ATTRIBUTE_VALUE_DOMAIN_NAMED) {
        iree_string_view_t source_name = iree_string_view_empty();
        loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_target_string(
            materializer, value_number, IREE_SV("predicate_value_name"),
            wire_predicate.argument_offsets[i], &source_name, &target_name_id));
        predicates[predicate_index].args[i] = (int64_t)target_name_id;
      } else {
        if (value_number >= scope->ssa.value_count) {
          return loom_bytecode_selected_attribute_emit_invalid_ssa_value(
              materializer, scope->ssa.symbol_name,
              wire_predicate.argument_offsets[i]);
        }
        predicates[predicate_index].args[i] =
            (int64_t)scope->ssa.values[value_number];
      }
    }
  }
  *out_attr = loom_attr_predicate_list(predicates, predicate_count);
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_attribute_decode_parameterized(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    loom_parameterized_attr_kind_t expected_family_kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_selected_attribute_scope_t* scope,
    uint8_t aggregate_depth,
    loom_bytecode_selected_attribute_state_t* out_state) {
  if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("aggregate_depth"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
  }

  const uint64_t family_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_family_name_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &source_family_name_id));
  iree_string_view_t family_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_source_string(
      materializer, source_family_name_id, IREE_SV("parameterized_family"),
      family_offset, &family_name));
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_lookup_parameterized_attr_by_name(materializer->context,
                                                     family_name);
  if (family_descriptor == NULL) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("family_name"), family_offset,
        IREE_SV("parameterized_attribute_family_is_not_registered"));
  }
  if (expected_family_kind != LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      expected_family_kind != family_descriptor->kind) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("family_name"), family_offset,
        IREE_SV("parameterized_attribute_family_does_not_match_field_"
                "contract"));
  }

  const uint64_t present_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t present_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &present_count));
  if (present_count > family_descriptor->parameter_count) {
    return loom_bytecode_reader_emit_count_exceeds(
        materializer->decoder, IREE_SV("parameterized_parameters"),
        present_count, family_descriptor->parameter_count,
        present_count_offset);
  }

  loom_attribute_t* slots = NULL;
  if (family_descriptor->parameter_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        materializer->scratch_arena, family_descriptor->parameter_count,
        sizeof(*slots), (void**)&slots));
    memset(slots, 0, family_descriptor->parameter_count * sizeof(*slots));
  }
  uint8_t next_parameter_index = 0;
  for (uint64_t i = 0; i < present_count; ++i) {
    const uint64_t parameter_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t source_parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, cursor, &source_parameter_name_id));
    iree_string_view_t parameter_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_source_string(
        materializer, source_parameter_name_id,
        IREE_SV("parameterized_parameter"), parameter_offset, &parameter_name));
    const uint8_t parameter_index =
        loom_bytecode_attribute_find_parameter_index(
            family_descriptor->parameter_descriptors,
            family_descriptor->parameter_count, parameter_name,
            next_parameter_index);
    if (parameter_index == LOOM_ATTR_INDEX_NONE) {
      const bool declared_before_cursor =
          loom_bytecode_attribute_find_parameter_index(
              family_descriptor->parameter_descriptors,
              family_descriptor->parameter_count, parameter_name,
              /*start_index=*/0) != LOOM_ATTR_INDEX_NONE;
      return loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, cursor->range_name, IREE_SV("attribute"), i,
          IREE_SV("parameter_name"), parameter_offset,
          declared_before_cursor
              ? IREE_SV("parameterized_attribute_parameters_are_not_in_"
                        "declaration_order")
              : IREE_SV("parameterized_attribute_parameter_is_not_declared"));
    }
    next_parameter_index = parameter_index + 1;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        materializer->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_at_depth(
        materializer, cursor,
        &family_descriptor->parameter_descriptors[parameter_index], value_kind,
        &slots[parameter_index], available_type_count, scope,
        aggregate_depth + 1, out_state));
  }

  *out_attr = loom_make_parameterized_attr(family_descriptor->kind, slots,
                                           family_descriptor->parameter_count);
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_attribute_decode_at_depth(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_selected_attribute_scope_t* scope,
    uint8_t aggregate_depth,
    loom_bytecode_selected_attribute_state_t* out_state) {
  switch (kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
          materializer->decoder, cursor, &value));
      *out_attr = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_F64: {
      uint64_t bits = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
          materializer->decoder, cursor, &bits));
      double value = 0.0;
      memcpy(&value, &bits, sizeof(value));
      *out_attr = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_STRING: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t source_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &source_string_id));
      iree_string_view_t source_string = iree_string_view_empty();
      loom_string_id_t target_string_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_target_string(
          materializer, source_string_id, IREE_SV("attribute_string"), offset,
          &source_string, &target_string_id));
      *out_attr = loom_attr_string(target_string_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(materializer->decoder, cursor, &value));
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(
            materializer->decoder, IREE_SV("bool_attribute"), value, 2, offset);
      }
      *out_attr = loom_attr_bool(value != 0);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENUM: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(materializer->decoder, cursor, &value));
      *out_attr = loom_attr_enum(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_I64_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &count));
      if (count > UINT16_MAX || count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("i64_array"), count, UINT16_MAX,
            count_offset);
      }
      int64_t* values = NULL;
      if (count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)count,
            sizeof(*values), (void**)&values));
      }
      for (uint64_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
            materializer->decoder, cursor, &values[i]));
      }
      *out_attr = loom_attr_i64_array(values, (uint16_t)count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENUM_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (descriptor == NULL || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("enum_array"), count_offset,
            IREE_SV("enum_array_requires_a_descriptor_backed_field"));
      }
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &count));
      if (count > UINT16_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("enum_array"), count, UINT16_MAX,
            count_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, cursor, count, &span));
      *out_attr = loom_attr_enum_array(span.data, (uint16_t)span.data_length);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET: {
      loom_bytecode_wire_signed_enum_set_t set;
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_signed_enum_set(
          materializer->decoder, cursor, descriptor, &set));
      uint64_t* words = NULL;
      if (set.word_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)set.word_count * 2,
            sizeof(*words), (void**)&words));
        memcpy(words, set.words,
               (iree_host_size_t)set.word_count * 2 * sizeof(*words));
      }
      *out_attr = loom_attr_signed_enum_set(words, set.word_count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BYTES: {
      const uint64_t length_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t byte_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &byte_length));
      if (byte_length > UINT32_MAX || byte_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("bytes_attribute"), byte_length,
            UINT32_MAX, length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, cursor, byte_length, &span));
      *out_attr = loom_attr_bytes(span.data, (uint32_t)span.data_length);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      const uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t source_name_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &source_name_id));
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_source_string(
          materializer, source_name_id, IREE_SV("attribute_symbol"),
          name_offset, &source_name));
      loom_symbol_ref_t target_ref = loom_symbol_ref_null();
      if (!loom_bytecode_selected_table_lookup_symbol(
              materializer, (uint32_t)source_name_id, &target_ref)) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("symbol"), name_offset,
            IREE_SV("symbol_attribute_references_an_unknown_symbol"));
      }
      *out_attr = loom_attr_symbol(target_ref);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SYMBOL_ARRAY:
    case LOOM_BYTECODE_ATTR_SYMBOL_SET: {
      const bool is_set = kind == LOOM_BYTECODE_ATTR_SYMBOL_SET;
      const loom_attr_kind_t expected_kind =
          is_set ? LOOM_ATTR_SYMBOL_SET : LOOM_ATTR_SYMBOL_ARRAY;
      const iree_string_view_t collection_name =
          is_set ? IREE_SV("symbol_set") : IREE_SV("symbol_array");
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (descriptor == NULL || descriptor->attr_kind != expected_kind) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            collection_name, count_offset,
            IREE_SV("symbol_collection_requires_a_matching_descriptor"));
      }
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &count));
      if (count > UINT16_MAX || count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, collection_name, count, UINT16_MAX,
            count_offset);
      }
      loom_symbol_ref_t* values = NULL;
      if (count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)count,
            sizeof(*values), (void**)&values));
      }
      iree_string_view_t previous_name = iree_string_view_empty();
      for (uint64_t i = 0; i < count; ++i) {
        const uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t source_name_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, cursor, &source_name_id));
        iree_string_view_t source_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_source_string(
            materializer, source_name_id, IREE_SV("symbol_collection_element"),
            name_offset, &source_name));
        if (is_set && i > 0 &&
            iree_string_view_compare(previous_name, source_name) >= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, collection_name, i,
              IREE_SV("symbol"), name_offset,
              IREE_SV("symbol_set_elements_are_not_sorted_and_unique"));
        }
        previous_name = source_name;
        if (!loom_bytecode_selected_table_lookup_symbol(
                materializer, (uint32_t)source_name_id, &values[i])) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, collection_name, i,
              IREE_SV("symbol"), name_offset,
              IREE_SV("symbol_collection_references_an_unknown_symbol"));
        }
      }
      *out_attr = is_set ? loom_attr_symbol_set(values, (uint16_t)count)
                         : loom_attr_symbol_array(values, (uint16_t)count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      const uint64_t type_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t source_type_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &source_type_id));
      if (source_type_id >= available_type_count) {
        return loom_bytecode_reader_emit_table_ref(
            materializer->decoder, IREE_SV("TYPES"), source_type_id,
            available_type_count, type_offset);
      }
      loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
      loom_bytecode_selected_reference_state_t reference_state =
          LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_type(
          materializer, (loom_type_id_t)source_type_id, &target_type_id,
          &reference_state));
      if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
        *out_state = LOOM_BYTECODE_SELECTED_ATTRIBUTE_WAITING;
        return iree_ok_status();
      }
      *out_attr = loom_attr_type(target_type_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST:
      return loom_bytecode_selected_attribute_decode_predicate_list(
          materializer, cursor, scope, out_attr);
    case LOOM_BYTECODE_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("aggregate_depth"),
            loom_bytecode_reader_cursor_absolute_position(cursor),
            IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
      }
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t entry_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &entry_count));
      if (entry_count > UINT16_MAX || entry_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("dict_entries"), entry_count,
            UINT16_MAX, count_offset);
      }
      loom_named_attr_t* entries = NULL;
      if (entry_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)entry_count,
            sizeof(*entries), (void**)&entries));
      }
      iree_string_view_t previous_key = iree_string_view_empty();
      for (uint64_t i = 0; i < entry_count; ++i) {
        const uint64_t key_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t source_key_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, cursor, &source_key_id));
        iree_string_view_t source_key = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_target_string(
            materializer, source_key_id, IREE_SV("dict_key"), key_offset,
            &source_key, &entries[i].name_id));
        if (i > 0 && iree_string_view_compare(source_key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, IREE_SV("dict"), i,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary_keys_are_not_in_canonical_order"));
        }
        previous_key = source_key;
        entries[i].reserved = 0;
        loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
            materializer->decoder, cursor, &value_kind));
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_at_depth(
            materializer, cursor, /*descriptor=*/NULL, value_kind,
            &entries[i].value, available_type_count, scope, aggregate_depth + 1,
            out_state));
      }
      *out_attr = loom_make_canonical_attr_dict(entries, entry_count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      const uint64_t encoding_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t source_encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &source_encoding_id));
      if (source_encoding_id == 0 ||
          source_encoding_id > materializer->metadata->encodings.count) {
        return loom_bytecode_reader_emit_table_ref(
            materializer->decoder, IREE_SV("ENCODINGS"), source_encoding_id,
            materializer->metadata->encodings.count, encoding_offset);
      }
      uint16_t target_encoding_id = 0;
      loom_bytecode_selected_reference_state_t reference_state =
          LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_encoding(
          materializer, (uint16_t)source_encoding_id, &target_encoding_id,
          &reference_state));
      if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
        *out_state = LOOM_BYTECODE_SELECTED_ATTRIBUTE_WAITING;
        return iree_ok_status();
      }
      *out_attr = loom_attr_encoding(target_encoding_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED: {
      if (descriptor != NULL &&
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("family_name"),
            loom_bytecode_reader_cursor_absolute_position(cursor),
            IREE_SV("parameterized_attribute_does_not_match_field_kind"));
      }
      const loom_parameterized_attr_kind_t expected_family_kind =
          descriptor != NULL ? descriptor->reference.parameterized_attr_kind
                             : LOOM_PARAMETERIZED_ATTR_KIND_ANY;
      return loom_bytecode_selected_attribute_decode_parameterized(
          materializer, cursor, expected_family_kind, out_attr,
          available_type_count, scope, aggregate_depth, out_state);
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (descriptor == NULL ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("element_count"), count_offset,
            IREE_SV("parameterized_attribute_array_requires_a_descriptor_"
                    "backed_field"));
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("aggregate_depth"), count_offset,
            IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
      }
      uint64_t element_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &element_count));
      if (element_count > UINT16_MAX || element_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("parameterized_attribute_array"),
            element_count, UINT16_MAX, count_offset);
      }
      loom_attribute_t* attributes = NULL;
      if (element_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)element_count,
            sizeof(*attributes), (void**)&attributes));
      }
      for (uint64_t i = 0; i < element_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_selected_attribute_decode_parameterized(
                materializer, cursor,
                descriptor->reference.parameterized_attr_kind, &attributes[i],
                available_type_count, scope, aggregate_depth + 1, out_state));
      }
      *out_attr = loom_attr_parameterized_array(attributes, element_count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SCOPED_ENUM:
    case LOOM_BYTECODE_ATTR_COUNT:
      break;
  }
  return loom_bytecode_reader_emit_enum_value(
      materializer->decoder, IREE_SV("attribute_kind"), kind,
      LOOM_BYTECODE_ATTR_COUNT,
      loom_bytecode_reader_cursor_absolute_position(cursor));
}

iree_status_t loom_bytecode_selected_attribute_decode_named(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    loom_bytecode_selected_attribute_state_t* out_state) {
  *out_attr = loom_attr_absent();
  *out_state = LOOM_BYTECODE_SELECTED_ATTRIBUTE_READY;
  const loom_bytecode_selected_attribute_scope_t scope = {
      .value_domain = LOOM_BYTECODE_SELECTED_ATTRIBUTE_VALUE_DOMAIN_NAMED,
  };
  return loom_bytecode_selected_attribute_decode_at_depth(
      materializer, cursor, descriptor, kind, out_attr, available_type_count,
      &scope, /*aggregate_depth=*/0, out_state);
}

iree_status_t loom_bytecode_selected_attribute_decode_ssa(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope,
    loom_bytecode_selected_attribute_state_t* out_state) {
  *out_attr = loom_attr_absent();
  *out_state = LOOM_BYTECODE_SELECTED_ATTRIBUTE_READY;
  const loom_bytecode_selected_attribute_scope_t scope = {
      .value_domain = LOOM_BYTECODE_SELECTED_ATTRIBUTE_VALUE_DOMAIN_SSA,
      .ssa = *ssa_scope,
  };
  return loom_bytecode_selected_attribute_decode_at_depth(
      materializer, cursor, descriptor, kind, out_attr, available_type_count,
      &scope, /*aggregate_depth=*/0, out_state);
}
