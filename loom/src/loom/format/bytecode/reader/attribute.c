// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/attribute.h"

#include <string.h>

#include "loom/error/error_catalog.h"

typedef enum loom_bytecode_attribute_value_domain_e {
  LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_NAMED = 0,
  LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_SSA = 1,
} loom_bytecode_attribute_value_domain_t;

typedef struct loom_bytecode_attribute_validation_scope_t {
  // Concrete wire namespace used by predicate VALUE arguments.
  loom_bytecode_attribute_value_domain_t value_domain;
  // SSA validation bounds when |value_domain| selects SSA numbers.
  loom_bytecode_attribute_ssa_validation_scope_t ssa;
} loom_bytecode_attribute_validation_scope_t;

typedef struct loom_bytecode_attribute_materialization_scope_t {
  // Concrete wire namespace used by predicate VALUE arguments.
  loom_bytecode_attribute_value_domain_t value_domain;
  // SSA value map when |value_domain| selects SSA numbers.
  loom_bytecode_attribute_ssa_materialization_scope_t ssa;
} loom_bytecode_attribute_materialization_scope_t;

typedef struct loom_bytecode_wire_predicate_t {
  // Decoded predicate header, tags, and immediate arguments.
  loom_predicate_t value;
  // Raw wire ordinals for VALUE arguments.
  uint64_t value_numbers[IREE_ARRAYSIZE(((loom_predicate_t*)0)->args)];
  // Absolute wire offsets for argument diagnostics.
  uint64_t argument_offsets[IREE_ARRAYSIZE(((loom_predicate_t*)0)->args)];
} loom_bytecode_wire_predicate_t;

typedef struct loom_bytecode_wire_signed_enum_set_t {
  // Positive words followed by the equally sized negative word span.
  uint64_t words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT * 2];
  // Number of words in each signed half of |words|.
  uint8_t word_count;
} loom_bytecode_wire_signed_enum_set_t;

static iree_status_t loom_bytecode_attribute_validate_string_ref(
    const loom_bytecode_attribute_validator_t* validator, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  const loom_bytecode_reader_module_view_t* module_view =
      validator->module_view;
  if (string_id >= module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(validator->decoder,
                                           LOOM_ERR_BYTECODE_010, params,
                                           IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = module_view->strings.values[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_validate_type_ref(
    const loom_bytecode_attribute_validator_t* validator, uint64_t type_id,
    iree_host_size_t available_type_count, uint64_t offset) {
  if (type_id >= available_type_count) {
    return loom_bytecode_reader_emit_table_ref(validator->decoder,
                                               IREE_SV("TYPES"), type_id,
                                               available_type_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_validate_encoding_ref(
    const loom_bytecode_attribute_validator_t* validator, uint64_t encoding_id,
    uint64_t offset) {
  const iree_host_size_t encoding_count =
      validator->module_view->encodings.count;
  if (encoding_id == 0 || encoding_id > encoding_count) {
    return loom_bytecode_reader_emit_table_ref(
        validator->decoder, IREE_SV("ENCODINGS"), encoding_id, encoding_count,
        offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_emit_invalid_ssa_value(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t symbol_name,
    uint64_t offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
      loom_param_u64(offset),
      loom_param_string(
          IREE_SV("predicate value reference must target a previously defined "
                  "value")),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_016, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

static uint8_t loom_bytecode_attribute_find_parameter_index(
    const loom_attr_descriptor_t* parameter_descriptors,
    uint8_t parameter_count, iree_string_view_t parameter_name,
    uint8_t start_index) {
  uint8_t relative_index = LOOM_ATTR_INDEX_NONE;
  const loom_attr_descriptor_t* descriptor = loom_attr_descriptor_find_by_name(
      parameter_descriptors + start_index,
      (uint8_t)(parameter_count - start_index), parameter_name,
      &relative_index);
  return descriptor ? (uint8_t)(start_index + relative_index)
                    : LOOM_ATTR_INDEX_NONE;
}

static iree_status_t loom_bytecode_attribute_read_predicate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t predicate_index,
    loom_bytecode_wire_predicate_t* out_predicate) {
  loom_bytecode_wire_predicate_t predicate = {0};
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(decoder, cursor, &predicate.value.kind));
  if (predicate.value.kind >= LOOM_PREDICATE_COUNT_) {
    return loom_bytecode_reader_emit_enum_value(
        decoder, IREE_SV("predicate_kind"), predicate.value.kind,
        LOOM_PREDICATE_COUNT_, kind_offset);
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(
      decoder, cursor, &predicate.value.arg_count));
  const uint8_t expected_arg_count =
      loom_predicate_kind_argument_count(predicate.value.kind);
  if (predicate.value.arg_count != expected_arg_count ||
      predicate.value.arg_count > IREE_ARRAYSIZE(predicate.value.args)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("predicate"), predicate_index,
        IREE_SV("arg_count"), kind_offset + 1,
        IREE_SV("predicate_arity_does_not_match_its_kind"));
  }
  for (uint8_t i = 0; i < predicate.value.arg_count; ++i) {
    const uint64_t tag_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(
        decoder, cursor, &predicate.value.arg_tags[i]));
    predicate.argument_offsets[i] =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    switch (predicate.value.arg_tags[i]) {
      case LOOM_PRED_ARG_VALUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            decoder, cursor, &predicate.value_numbers[i]));
        break;
      }
      case LOOM_PRED_ARG_CONST: {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
            decoder, cursor, &predicate.value.args[i]));
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            decoder, IREE_SV("predicate_arg_tag"), predicate.value.arg_tags[i],
            LOOM_PRED_ARG_COUNT_, tag_offset);
    }
  }
  *out_predicate = predicate;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_read_predicate_count(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint16_t* out_predicate_count) {
  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &predicate_count));
  if (predicate_count > UINT16_MAX || predicate_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("predicate_list"), predicate_count, UINT16_MAX,
        count_offset);
  }
  *out_predicate_count = (uint16_t)predicate_count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_read_signed_enum_set(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor,
    loom_bytecode_wire_signed_enum_set_t* out_set) {
  const uint64_t payload_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_SIGNED_ENUM_SET ||
      iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("signed_enum_set"), payload_offset,
        IREE_SV("signed_enum_set_requires_a_closed_descriptor_backed_field"));
  }

  loom_bytecode_wire_signed_enum_set_t set = {0};
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(decoder, cursor, &set.word_count));
  if (set.word_count > LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("signed_enum_set_word_count"), set.word_count,
        LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT, payload_offset);
  }
  for (iree_host_size_t i = 0; i < (iree_host_size_t)set.word_count * 2; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(decoder, cursor, &set.words[i]));
  }

  const uint64_t* negative_words = set.words + set.word_count;
  for (iree_host_size_t i = 0; i < set.word_count; ++i) {
    if ((set.words[i] & negative_words[i]) != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          decoder, cursor->range_name, IREE_SV("attribute"), 0,
          IREE_SV("signed_enum_set"), payload_offset,
          IREE_SV("signed_enum_set_contains_contradictory_assertions"));
    }
  }
  if (set.word_count > 0 && set.words[set.word_count - 1] == 0 &&
      negative_words[set.word_count - 1] == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("signed_enum_set"), payload_offset,
        IREE_SV("signed_enum_set_is_not_canonically_trimmed"));
  }

  const loom_signed_enum_set_t value =
      loom_make_signed_enum_set(set.words, set.word_count);
  for (iree_host_size_t i = 0; i < 256; ++i) {
    if (!loom_signed_enum_set_contains_positive(value, (uint8_t)i) &&
        !loom_signed_enum_set_contains_negative(value, (uint8_t)i)) {
      continue;
    }
    if (!loom_attr_descriptor_has_enum_case(descriptor, (uint8_t)i)) {
      return loom_bytecode_reader_emit_invalid_field(
          decoder, cursor->range_name, IREE_SV("attribute"), 0,
          IREE_SV("signed_enum_set"), payload_offset,
          IREE_SV("signed_enum_set_value_is_not_declared"));
    }
  }

  *out_set = set;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_validate_predicate_list(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_validation_scope_t* scope,
    uint16_t* out_predicate_count) {
  uint16_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_predicate_count(
      validator->decoder, cursor, &predicate_count));
  for (uint16_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    loom_bytecode_wire_predicate_t predicate;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_predicate(
        validator->decoder, cursor, predicate_index, &predicate));
    for (uint8_t i = 0; i < predicate.value.arg_count; ++i) {
      if (predicate.value.arg_tags[i] != LOOM_PRED_ARG_VALUE) {
        continue;
      }
      const uint64_t value_number = predicate.value_numbers[i];
      if (scope->value_domain == LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_NAMED) {
        iree_string_view_t unused = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            validator, value_number, IREE_SV("predicate_value_name"),
            predicate.argument_offsets[i], &unused));
      } else if (value_number >= scope->ssa.value_count) {
        return loom_bytecode_attribute_emit_invalid_ssa_value(
            validator->decoder, scope->ssa.symbol_name,
            predicate.argument_offsets[i]);
      }
    }
  }
  *out_predicate_count = predicate_count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_attribute_validate_at_depth(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count,
    const loom_bytecode_attribute_validation_scope_t* scope,
    uint8_t aggregate_depth) {
  switch (kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      int64_t value = 0;
      return loom_bytecode_reader_read_svarint(validator->decoder, cursor,
                                               &value);
    }
    case LOOM_BYTECODE_ATTR_F64: {
      uint64_t bits = 0;
      return loom_bytecode_reader_read_u64_le(validator->decoder, cursor,
                                              &bits);
    }
    case LOOM_BYTECODE_ATTR_STRING:
    case LOOM_BYTECODE_ATTR_SCOPED_ENUM: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &string_id));
      iree_string_view_t unused = iree_string_view_empty();
      return loom_bytecode_attribute_validate_string_ref(
          validator, string_id, IREE_SV("attribute_string"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_ENUM: {
      uint8_t value = 0;
      return loom_bytecode_reader_read_u8(validator->decoder, cursor, &value);
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(validator->decoder, cursor, &value));
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(
            validator->decoder, IREE_SV("bool_attribute"), value, 2, offset);
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_I64_ARRAY: {
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(validator->decoder,
                                                             cursor, &count));
      for (uint64_t i = 0; i < count; ++i) {
        int64_t value = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
            validator->decoder, cursor, &value));
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENUM_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("enum_array"), count_offset,
            IREE_SV("enum_array_requires_a_descriptor_backed_field"));
      }
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(validator->decoder,
                                                             cursor, &count));
      if (count > UINT16_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("enum_array"), count, UINT16_MAX,
            count_offset);
      }
      iree_const_byte_span_t unused = iree_const_byte_span_empty();
      return loom_bytecode_reader_read_span(validator->decoder, cursor, count,
                                            &unused);
    }
    case LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET: {
      loom_bytecode_wire_signed_enum_set_t set;
      return loom_bytecode_attribute_read_signed_enum_set(
          validator->decoder, cursor, descriptor, &set);
    }
    case LOOM_BYTECODE_ATTR_BYTES: {
      uint64_t byte_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &byte_length));
      iree_const_byte_span_t unused = iree_const_byte_span_empty();
      return loom_bytecode_reader_read_span(validator->decoder, cursor,
                                            byte_length, &unused);
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t symbol_name_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &symbol_name_id));
      iree_string_view_t unused = iree_string_view_empty();
      return loom_bytecode_attribute_validate_string_ref(
          validator, symbol_name_id, IREE_SV("attribute_symbol"), offset,
          &unused);
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
      if (!descriptor || descriptor->attr_kind != expected_kind) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            collection_name, count_offset,
            IREE_SV("symbol_collection_requires_a_matching_descriptor"));
      }
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(validator->decoder,
                                                             cursor, &count));
      if (count > UINT16_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, collection_name, count, UINT16_MAX,
            count_offset);
      }
      iree_string_view_t previous_name = iree_string_view_empty();
      for (uint64_t i = 0; i < count; ++i) {
        const uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t name_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, cursor, &name_id));
        iree_string_view_t name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            validator, name_id, IREE_SV("symbol_collection_element"),
            name_offset, &name));
        if (is_set && i > 0 &&
            iree_string_view_compare(previous_name, name) >= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              validator->decoder, cursor->range_name, collection_name, i,
              IREE_SV("symbol"), name_offset,
              IREE_SV("symbol_set_elements_are_not_sorted_and_unique"));
        }
        previous_name = name;
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t type_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(validator->decoder,
                                                             cursor, &type_id));
      return loom_bytecode_attribute_validate_type_ref(
          validator, type_id, available_type_count, offset);
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST: {
      uint16_t unused_predicate_count = 0;
      return loom_bytecode_attribute_validate_predicate_list(
          validator, cursor, scope, &unused_predicate_count);
    }
    case LOOM_BYTECODE_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("aggregate_depth"),
            loom_bytecode_reader_cursor_absolute_position(cursor),
            IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
      }
      uint64_t entry_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &entry_count));
      iree_string_view_t previous_key = iree_string_view_empty();
      for (uint64_t i = 0; i < entry_count; ++i) {
        const uint64_t key_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t key_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, cursor, &key_id));
        iree_string_view_t key = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            validator, key_id, IREE_SV("dict_key"), key_offset, &key));
        if (i > 0 && iree_string_view_compare(key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              validator->decoder, cursor->range_name, IREE_SV("dict"), i,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary_keys_are_not_in_canonical_order"));
        }
        previous_key = key;
        loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
            validator->decoder, cursor, &value_kind));
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_at_depth(
            validator, cursor, /*descriptor=*/NULL, value_kind,
            available_type_count, scope, aggregate_depth + 1));
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      const uint64_t offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &encoding_id));
      return loom_bytecode_attribute_validate_encoding_ref(validator,
                                                           encoding_id, offset);
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("aggregate_depth"),
            loom_bytecode_reader_cursor_absolute_position(cursor),
            IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
      }
      const uint64_t family_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t family_name_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &family_name_id));
      iree_string_view_t family_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
          validator, family_name_id, IREE_SV("parameterized_family"),
          family_offset, &family_name));
      const loom_parameterized_attr_descriptor_t* family_descriptor =
          loom_context_lookup_parameterized_attr_by_name(validator->context,
                                                         family_name);
      if (!family_descriptor) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("family_name"), family_offset,
            IREE_SV("parameterized_attribute_family_is_not_registered"));
      }
      if (descriptor && descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("family_name"), family_offset,
            IREE_SV("parameterized_attribute_does_not_match_field_kind"));
      }
      if (descriptor &&
          descriptor->reference.parameterized_attr_kind !=
              LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
          descriptor->reference.parameterized_attr_kind !=
              family_descriptor->kind) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("family_name"), family_offset,
            IREE_SV("parameterized_attribute_family_does_not_match_field_"
                    "contract"));
      }
      const uint64_t present_count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t present_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &present_count));
      if (present_count > family_descriptor->parameter_count) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("parameterized_parameters"),
            present_count, family_descriptor->parameter_count,
            present_count_offset);
      }
      uint8_t next_parameter_index = 0;
      for (uint64_t i = 0; i < present_count; ++i) {
        const uint64_t parameter_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t parameter_name_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, cursor, &parameter_name_id));
        iree_string_view_t parameter_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            validator, parameter_name_id, IREE_SV("parameterized_parameter"),
            parameter_offset, &parameter_name));
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
              validator->decoder, cursor->range_name, IREE_SV("attribute"), i,
              IREE_SV("parameter_name"), parameter_offset,
              declared_before_cursor
                  ? IREE_SV("parameterized_attribute_parameters_are_not_in_"
                            "declaration_order")
                  : IREE_SV(
                        "parameterized_attribute_parameter_is_not_declared"));
        }
        next_parameter_index = parameter_index + 1;
        loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
            validator->decoder, cursor, &value_kind));
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_at_depth(
            validator, cursor,
            &family_descriptor->parameter_descriptors[parameter_index],
            value_kind, available_type_count, scope, aggregate_depth + 1));
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (!descriptor ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("element_count"), count_offset,
            IREE_SV("parameterized_attribute_array_requires_a_descriptor_"
                    "backed_field"));
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("aggregate_depth"), count_offset,
            IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
      }
      uint64_t element_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, cursor, &element_count));
      if (element_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("parameterized_attribute_array"),
            element_count, UINT16_MAX, count_offset);
      }
      loom_attr_descriptor_t element_descriptor = *descriptor;
      element_descriptor.attr_kind = LOOM_ATTR_PARAMETERIZED;
      for (uint64_t i = 0; i < element_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_at_depth(
            validator, cursor, &element_descriptor,
            LOOM_BYTECODE_ATTR_PARAMETERIZED, available_type_count, scope,
            aggregate_depth + 1));
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_COUNT:
      break;
  }
  return loom_bytecode_reader_emit_enum_value(
      validator->decoder, IREE_SV("attribute_kind"), kind,
      LOOM_BYTECODE_ATTR_COUNT,
      loom_bytecode_reader_cursor_absolute_position(cursor));
}

iree_status_t loom_bytecode_attribute_validate_named(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count) {
  const loom_bytecode_attribute_validation_scope_t scope = {
      .value_domain = LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_NAMED,
  };
  return loom_bytecode_attribute_validate_at_depth(
      validator, cursor, descriptor, kind, available_type_count, &scope,
      /*aggregate_depth=*/0);
}

iree_status_t loom_bytecode_attribute_validate_ssa(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope) {
  const loom_bytecode_attribute_validation_scope_t scope = {
      .value_domain = LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_SSA,
      .ssa = *ssa_scope,
  };
  return loom_bytecode_attribute_validate_at_depth(
      validator, cursor, descriptor, kind, available_type_count, &scope,
      /*aggregate_depth=*/0);
}

iree_status_t loom_bytecode_attribute_validate_predicate_list_ssa(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope,
    uint16_t* out_predicate_count) {
  const loom_bytecode_attribute_validation_scope_t scope = {
      .value_domain = LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_SSA,
      .ssa = *ssa_scope,
  };
  return loom_bytecode_attribute_validate_predicate_list(
      validator, cursor, &scope, out_predicate_count);
}

static loom_bytecode_attribute_validator_t
loom_bytecode_attribute_materializer_validator(
    loom_bytecode_attribute_materializer_t* materializer) {
  return (loom_bytecode_attribute_validator_t){
      .decoder = materializer->decoder,
      .context = materializer->context,
      .module_view = materializer->module_view,
  };
}

static iree_status_t loom_bytecode_attribute_materialize_predicate_list(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_materialization_scope_t* scope,
    loom_attribute_t* out_attr) {
  uint16_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_predicate_count(
      materializer->decoder, cursor, &predicate_count));
  loom_predicate_t* predicates = NULL;
  if (predicate_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &materializer->output_module->arena, predicate_count,
        sizeof(*predicates), (void**)&predicates));
  }
  loom_bytecode_attribute_validator_t validator =
      loom_bytecode_attribute_materializer_validator(materializer);
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
      if (scope->value_domain == LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_NAMED) {
        iree_string_view_t unused = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            &validator, value_number, IREE_SV("predicate_value_name"),
            wire_predicate.argument_offsets[i], &unused));
        predicates[predicate_index].args[i] = (int64_t)value_number;
      } else {
        if (value_number >= scope->ssa.value_count) {
          return loom_bytecode_attribute_emit_invalid_ssa_value(
              materializer->decoder, scope->ssa.symbol_name,
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

static iree_status_t loom_bytecode_attribute_materialize_at_depth(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_materialization_scope_t* scope,
    uint8_t aggregate_depth);

static iree_status_t loom_bytecode_attribute_materialize_parameterized(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    loom_parameterized_attr_kind_t expected_family_kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_materialization_scope_t* scope,
    uint8_t aggregate_depth) {
  if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
        IREE_SV("aggregate_depth"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("aggregate_attribute_nesting_exceeds_maximum_depth"));
  }
  const uint64_t family_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t family_name_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &family_name_id));
  loom_bytecode_attribute_validator_t validator =
      loom_bytecode_attribute_materializer_validator(materializer);
  iree_string_view_t family_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
      &validator, family_name_id, IREE_SV("parameterized_family"),
      family_offset, &family_name));
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_lookup_parameterized_attr_by_name(materializer->context,
                                                     family_name);
  if (!family_descriptor) {
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

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(materializer->scratch_arena);
  loom_attribute_t* slots = NULL;
  iree_status_t status = iree_ok_status();
  if (family_descriptor->parameter_count > 0) {
    status = iree_arena_allocate_array(materializer->scratch_arena,
                                       family_descriptor->parameter_count,
                                       sizeof(*slots), (void**)&slots);
    if (iree_status_is_ok(status)) {
      memset(slots, 0, family_descriptor->parameter_count * sizeof(*slots));
    }
  }
  uint8_t next_parameter_index = 0;
  for (uint64_t i = 0; i < present_count && iree_status_is_ok(status); ++i) {
    const uint64_t parameter_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t parameter_name_id = 0;
    status = loom_bytecode_reader_read_uvarint(materializer->decoder, cursor,
                                               &parameter_name_id);
    if (!iree_status_is_ok(status)) {
      break;
    }
    iree_string_view_t parameter_name = iree_string_view_empty();
    status = loom_bytecode_attribute_validate_string_ref(
        &validator, parameter_name_id, IREE_SV("parameterized_parameter"),
        parameter_offset, &parameter_name);
    if (!iree_status_is_ok(status)) {
      break;
    }
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
      status = loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, cursor->range_name, IREE_SV("attribute"), i,
          IREE_SV("parameter_name"), parameter_offset,
          declared_before_cursor
              ? IREE_SV("parameterized_attribute_parameters_are_not_in_"
                        "declaration_order")
              : IREE_SV("parameterized_attribute_parameter_is_not_declared"));
      break;
    }
    next_parameter_index = parameter_index + 1;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    status = loom_bytecode_attribute_read_kind(materializer->decoder, cursor,
                                               &value_kind);
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_bytecode_attribute_materialize_at_depth(
        materializer, cursor,
        &family_descriptor->parameter_descriptors[parameter_index], value_kind,
        &slots[parameter_index], available_type_count, scope,
        aggregate_depth + 1);
  }
  if (iree_status_is_ok(status)) {
    status = loom_module_make_parameterized_attr(
        materializer->output_module, family_descriptor->kind, slots,
        family_descriptor->parameter_count, out_attr);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

static iree_status_t loom_bytecode_attribute_materialize_at_depth(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_materialization_scope_t* scope,
    uint8_t aggregate_depth) {
  loom_bytecode_attribute_validator_t validator =
      loom_bytecode_attribute_materializer_validator(materializer);
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
      const uint64_t string_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &string_id));
      iree_string_view_t unused = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
          &validator, string_id, IREE_SV("attribute_string"), string_offset,
          &unused));
      *out_attr = loom_attr_string((loom_string_id_t)string_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      const uint64_t value_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(materializer->decoder, cursor, &value));
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(materializer->decoder,
                                                    IREE_SV("bool_attribute"),
                                                    value, 2, value_offset);
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
            &materializer->output_module->arena, (iree_host_size_t)count,
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
      const uint64_t payload_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("enum_array"), payload_offset,
            IREE_SV("enum_array_requires_a_descriptor_backed_field"));
      }
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &count));
      if (count > UINT16_MAX || count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("enum_array"), count, UINT16_MAX,
            payload_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, cursor, count, &span));
      uint8_t* values = NULL;
      if (span.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                span.data_length, (void**)&values));
        memcpy(values, span.data, span.data_length);
      }
      *out_attr = loom_attr_enum_array(values, (uint16_t)span.data_length);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET: {
      loom_bytecode_wire_signed_enum_set_t set;
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_signed_enum_set(
          materializer->decoder, cursor, descriptor, &set));
      uint64_t* owned_words = NULL;
      if (set.word_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &materializer->output_module->arena,
            (iree_host_size_t)set.word_count * 2, sizeof(*owned_words),
            (void**)&owned_words));
        memcpy(owned_words, set.words,
               (iree_host_size_t)set.word_count * 2 * sizeof(*owned_words));
      }
      *out_attr = loom_attr_signed_enum_set(owned_words, set.word_count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BYTES: {
      const uint64_t byte_length_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t byte_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &byte_length));
      if (byte_length > UINT32_MAX || byte_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("bytes_attribute"), byte_length,
            UINT32_MAX, byte_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, cursor, byte_length, &span));
      uint8_t* bytes = NULL;
      if (span.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                span.data_length, (void**)&bytes));
        memcpy(bytes, span.data, span.data_length);
      }
      *out_attr = loom_attr_bytes(bytes, (uint32_t)span.data_length);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      const uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &name_id));
      iree_string_view_t unused = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
          &validator, name_id, IREE_SV("attribute_symbol"), name_offset,
          &unused));
      const uint16_t symbol_id = loom_symbol_map_find(
          &materializer->module_view->symbols.map, (loom_string_id_t)name_id);
      if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("symbol"), name_offset,
            IREE_SV("symbol_attribute_references_an_unknown_symbol"));
      }
      *out_attr = loom_attr_symbol((loom_symbol_ref_t){0, symbol_id});
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
      if (!descriptor || descriptor->attr_kind != expected_kind) {
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
            &materializer->output_module->arena, (iree_host_size_t)count,
            sizeof(*values), (void**)&values));
      }
      iree_string_view_t previous_name = iree_string_view_empty();
      for (uint64_t i = 0; i < count; ++i) {
        const uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t name_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, cursor, &name_id));
        iree_string_view_t name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            &validator, name_id, IREE_SV("symbol_collection_element"),
            name_offset, &name));
        if (is_set && i > 0 &&
            iree_string_view_compare(previous_name, name) >= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, collection_name, i,
              IREE_SV("symbol"), name_offset,
              IREE_SV("symbol_set_elements_are_not_sorted_and_unique"));
        }
        previous_name = name;
        const uint16_t symbol_id = loom_symbol_map_find(
            &materializer->module_view->symbols.map, (loom_string_id_t)name_id);
        if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, collection_name, i,
              IREE_SV("symbol"), name_offset,
              IREE_SV("symbol_collection_references_an_unknown_symbol"));
        }
        values[i] = (loom_symbol_ref_t){0, symbol_id};
      }
      *out_attr = is_set ? loom_attr_symbol_set(values, (uint16_t)count)
                         : loom_attr_symbol_array(values, (uint16_t)count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      const uint64_t type_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t type_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &type_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_type_ref(
          &validator, type_id, available_type_count, type_offset));
      *out_attr = loom_attr_type((loom_type_id_t)type_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST:
      return loom_bytecode_attribute_materialize_predicate_list(
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
        uint64_t key_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, cursor, &key_id));
        iree_string_view_t key = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_string_ref(
            &validator, key_id, IREE_SV("dict_key"), key_offset, &key));
        if (i > 0 && iree_string_view_compare(key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, cursor->range_name, IREE_SV("dict"), i,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary_keys_are_not_in_canonical_order"));
        }
        previous_key = key;
        loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
            materializer->decoder, cursor, &value_kind));
        entries[i].name_id = (loom_string_id_t)key_id;
        entries[i].reserved = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_attribute_materialize_at_depth(
            materializer, cursor, /*descriptor=*/NULL, value_kind,
            &entries[i].value, available_type_count, scope,
            aggregate_depth + 1));
      }
      return loom_module_make_canonical_attr_dict(
          materializer->output_module,
          loom_make_named_attr_slice(entries, (iree_host_size_t)entry_count),
          out_attr);
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      const uint64_t encoding_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, cursor, &encoding_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_encoding_ref(
          &validator, encoding_id, encoding_offset));
      if (encoding_id > materializer->output_module->encodings.count) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("encoding"), encoding_offset,
            IREE_SV("encoding_attribute_references_an_unavailable_encoding"));
      }
      *out_attr = loom_attr_encoding((uint16_t)encoding_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED: {
      if (descriptor && descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("family_name"),
            loom_bytecode_reader_cursor_absolute_position(cursor),
            IREE_SV("parameterized_attribute_does_not_match_field_kind"));
      }
      const loom_parameterized_attr_kind_t expected_family_kind =
          descriptor ? descriptor->reference.parameterized_attr_kind
                     : LOOM_PARAMETERIZED_ATTR_KIND_ANY;
      return loom_bytecode_attribute_materialize_parameterized(
          materializer, cursor, expected_family_kind, out_attr,
          available_type_count, scope, aggregate_depth);
    }
    case LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY: {
      const uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      if (!descriptor ||
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
      if (element_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("parameterized_attribute_array"),
            element_count, UINT16_MAX, count_offset);
      }
      const iree_arena_checkpoint_t checkpoint =
          iree_arena_checkpoint_save(materializer->scratch_arena);
      loom_attribute_t* attributes = NULL;
      iree_status_t status = iree_ok_status();
      if (element_count > 0) {
        status = iree_arena_allocate_array(materializer->scratch_arena,
                                           element_count, sizeof(*attributes),
                                           (void**)&attributes);
      }
      for (uint64_t i = 0; i < element_count && iree_status_is_ok(status);
           ++i) {
        status = loom_bytecode_attribute_materialize_parameterized(
            materializer, cursor, descriptor->reference.parameterized_attr_kind,
            &attributes[i], available_type_count, scope, aggregate_depth + 1);
      }
      if (iree_status_is_ok(status)) {
        status = loom_module_make_parameterized_attr_array(
            materializer->output_module,
            loom_make_parameterized_attr_array(attributes, element_count),
            out_attr);
      }
      iree_arena_checkpoint_restore(&checkpoint);
      return status;
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

iree_status_t loom_bytecode_attribute_materialize_named(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count) {
  const loom_bytecode_attribute_materialization_scope_t scope = {
      .value_domain = LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_NAMED,
  };
  return loom_bytecode_attribute_materialize_at_depth(
      materializer, cursor, descriptor, kind, out_attr, available_type_count,
      &scope, /*aggregate_depth=*/0);
}

iree_status_t loom_bytecode_attribute_materialize_ssa(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope) {
  const loom_bytecode_attribute_materialization_scope_t scope = {
      .value_domain = LOOM_BYTECODE_ATTRIBUTE_VALUE_DOMAIN_SSA,
      .ssa = *ssa_scope,
  };
  return loom_bytecode_attribute_materialize_at_depth(
      materializer, cursor, descriptor, kind, out_attr, available_type_count,
      &scope, /*aggregate_depth=*/0);
}
