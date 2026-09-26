// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/format_tables.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/types.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Generated-format table payloads
//===----------------------------------------------------------------------===//

#define LOOM_OPERAND_DICT_INLINE_ENTRIES 16
#define LOOM_PARSED_I64_INLINE_VALUES 16

typedef struct loom_parsed_operand_dict_entry_t {
  // Interned key spelling for this operand dictionary entry.
  loom_string_id_t name_id;
  // SSA value referenced by this entry.
  loom_value_id_t value_id;
  // Source token for the entry key.
  loom_token_t key_token;
  // Source token for the entry SSA value.
  loom_token_t value_token;
} loom_parsed_operand_dict_entry_t;

typedef struct loom_parsed_operand_dict_entries_t {
  // Mutable entry storage, initially pointing at inline_entries.
  loom_parsed_operand_dict_entry_t* entries;
  // Number of populated entries.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
  // Hash slots containing entry ordinals plus one, or zero for an empty slot.
  // Allocated only after inline storage spills, with twice the entry capacity.
  uint16_t* slots;
  // Inline storage for small operand dictionaries.
  loom_parsed_operand_dict_entry_t
      inline_entries[LOOM_OPERAND_DICT_INLINE_ENTRIES];
} loom_parsed_operand_dict_entries_t;

typedef struct loom_parsed_i64_values_t {
  // Mutable value storage, initially pointing at inline_values.
  int64_t* values;
  // Number of populated values.
  iree_host_size_t count;
  // Allocated value capacity.
  iree_host_size_t capacity;
  // Inline storage for small value lists.
  int64_t inline_values[LOOM_PARSED_I64_INLINE_VALUES];
} loom_parsed_i64_values_t;

static const loom_attr_descriptor_t* loom_parse_format_find_attr_descriptor(
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name,
    uint8_t* out_attr_index) {
  if (!vtable->attr_descriptors) {
    return NULL;
  }
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                attr_name)) {
      continue;
    }
    *out_attr_index = i;
    return descriptor;
  }
  return NULL;
}

static bool loom_parse_format_parsed_attr_present(
    const loom_parsed_op_t* parsed, uint8_t attr_index) {
  if (attr_index >= parsed->attribute_count) {
    return false;
  }
  return !loom_attr_is_absent(parsed->attributes[attr_index]);
}

static bool loom_parse_format_element_covers_attr(
    const loom_format_element_t* element, uint16_t attr_index) {
  switch (element->kind) {
    case LOOM_FORMAT_KIND_ATTR_VALUE:
    case LOOM_FORMAT_KIND_SYMBOL_REF:
    case LOOM_FORMAT_KIND_KEY_REF:
    case LOOM_FORMAT_KIND_SCOPED_ENUM_REF:
    case LOOM_FORMAT_KIND_ATTR_PARAMS:
    case LOOM_FORMAT_KIND_TEMPLATE_PARAM:
    case LOOM_FORMAT_KIND_PREDICATE_LIST:
      return element->field_index == attr_index;
    case LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS:
      return element->field_index == attr_index;
    case LOOM_FORMAT_KIND_STABLE_KEY_REF:
      return element->field_index == attr_index || element->data == attr_index;
    case LOOM_FORMAT_KIND_INDEX_LIST:
      return LOOM_FORMAT_INDEX_LIST_STATIC_ATTR_INDEX(element->data) ==
             attr_index;
    case LOOM_FORMAT_KIND_OPERAND_DICT:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_ATTR_TABLE:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_ALIGNED_REFS:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_FUNC_ARGS:
      return LOOM_FORMAT_FUNC_ARGS_START_ATTR_INDEX(element->data) ==
                 attr_index ||
             LOOM_FORMAT_FUNC_ARGS_END_ATTR_INDEX(element->data) == attr_index;
    case LOOM_FORMAT_KIND_BLOCK_ARGS:
      return LOOM_FORMAT_BLOCK_ARGS_START_ATTR_INDEX(element->data) ==
                 attr_index ||
             LOOM_FORMAT_BLOCK_ARGS_END_ATTR_INDEX(element->data) == attr_index;
    case LOOM_FORMAT_KIND_ATTR_DICT:
      if (iree_any_bit_set(element->data, LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS)) {
        return false;
      }
      return element->field_index == attr_index;
    default:
      return false;
  }
}

static bool loom_parse_format_inline_attr_covers_attr(
    loom_format_t format, const loom_format_element_t* inline_element,
    uint16_t attr_index) {
  const loom_format_element_t* elements = format.elements;
  for (uint16_t i = 0; i < format.count; ++i) {
    const loom_format_element_t* element = &elements[i];
    if (element == inline_element) {
      continue;
    }
    if (loom_parse_format_element_covers_attr(element, attr_index)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_parse_format_apply_elided_attr_defaults(
    loom_parser_t* parser, const loom_op_vtable_t* vtable, loom_format_t format,
    const loom_format_element_t* inline_element, loom_parsed_op_t* parsed) {
  if (!vtable->attr_descriptors) {
    return iree_ok_status();
  }
  for (uint8_t attr_index = 0; attr_index < vtable->attribute_count;
       ++attr_index) {
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    if (!iree_any_bit_set(descriptor->flags, LOOM_ATTR_ELIDE_DEFAULT) ||
        loom_parse_format_parsed_attr_present(parsed, attr_index) ||
        !loom_parse_format_inline_attr_covers_attr(format, inline_element,
                                                   attr_index)) {
      continue;
    }
    loom_attribute_t default_value =
        loom_attr_descriptor_default_value(descriptor);
    IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
        parsed, &parser->parser_arena, attr_index, default_value));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_format_emit_unknown_attr_name(
    loom_parser_t* parser, loom_token_t token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("attribute")),
      loom_param_string(token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                          IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parse_format_emit_duplicate_attr_name(
    loom_parser_t* parser, loom_token_t token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_020, params,
                          IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parse_format_emit_duplicate_operand_dict_key(
    loom_parser_t* parser, loom_token_t key_token,
    loom_token_t previous_key_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(key_token.text),
  };
  return loom_parser_emit_related(
      parser, LOOM_ERR_PARSE_027, params, IREE_ARRAYSIZE(params), key_token,
      IREE_SV("previously defined here"), previous_key_token);
}

static void loom_parsed_operand_dict_entries_initialize(
    loom_parsed_operand_dict_entries_t* entries) {
  entries->entries = entries->inline_entries;
  entries->count = 0;
  entries->capacity = LOOM_OPERAND_DICT_INLINE_ENTRIES;
  entries->slots = NULL;
}

static iree_host_size_t loom_parsed_operand_dict_slot(
    const loom_parsed_operand_dict_entries_t* entries,
    loom_string_id_t name_id) {
  const iree_host_size_t mask = entries->capacity * 2 - 1;
  uint32_t hash = (uint32_t)name_id * 2654435769u;
  hash ^= hash >> 16;
  iree_host_size_t slot = hash & mask;
  while (entries->slots[slot] &&
         entries->entries[entries->slots[slot] - 1].name_id != name_id) {
    slot = (slot + 1) & mask;
  }
  return slot;
}

static const loom_parsed_operand_dict_entry_t* loom_parsed_operand_dict_find(
    const loom_parsed_operand_dict_entries_t* entries,
    loom_string_id_t name_id) {
  if (!entries->slots) {
    for (iree_host_size_t i = 0; i < entries->count; ++i) {
      if (entries->entries[i].name_id == name_id) {
        return &entries->entries[i];
      }
    }
    return NULL;
  }
  uint16_t ordinal =
      entries->slots[loom_parsed_operand_dict_slot(entries, name_id)];
  return ordinal ? &entries->entries[ordinal - 1] : NULL;
}

static iree_status_t loom_parsed_operand_dict_entries_add(
    iree_arena_allocator_t* arena, loom_parsed_operand_dict_entries_t* entries,
    loom_parsed_operand_dict_entry_t entry) {
  if (entries->count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "operand dictionary has more than %u entries",
                            (unsigned)UINT16_MAX);
  }
  if (entries->count >= entries->capacity) {
    iree_host_size_t capacity = entries->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, entries->count, 0, sizeof(loom_parsed_operand_dict_entry_t),
        &capacity, (void**)&entries->entries));
    entries->capacity = capacity;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity * 2, sizeof(*entries->slots), (void**)&entries->slots));
    memset(entries->slots, 0, capacity * 2 * sizeof(*entries->slots));
    for (iree_host_size_t i = 0; i < entries->count; ++i) {
      entries->slots[loom_parsed_operand_dict_slot(
          entries, entries->entries[i].name_id)] = (uint16_t)(i + 1);
    }
  }
  if (entries->slots) {
    entries->slots[loom_parsed_operand_dict_slot(entries, entry.name_id)] =
        (uint16_t)(entries->count + 1);
  }
  entries->entries[entries->count++] = entry;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_emit_operand_dict_type_mismatch(
    loom_parser_t* parser, loom_token_t key_token, loom_token_t value_token,
    loom_type_t actual_type, loom_type_t annotated_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(key_token.text),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("type annotation")),
      loom_param_type(annotated_type),
  };
  return loom_parser_emit(parser, LOOM_ERR_TYPE_001, params,
                          IREE_ARRAYSIZE(params), value_token);
}

static void loom_parsed_i64_values_initialize(
    loom_parsed_i64_values_t* values) {
  values->values = values->inline_values;
  values->count = 0;
  values->capacity = LOOM_PARSED_I64_INLINE_VALUES;
}

static iree_status_t loom_parsed_i64_values_add(
    loom_parser_t* parser, loom_parsed_i64_values_t* values, int64_t value) {
  if (values->count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parsed i64 list has more than %u values",
                            (unsigned)UINT16_MAX);
  }
  if (values->count == values->capacity) {
    iree_host_size_t old_capacity = values->capacity;
    iree_host_size_t new_capacity = old_capacity * 2;
    int64_t* new_values = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->parser_arena, new_capacity,
                                  sizeof(*new_values), (void**)&new_values));
    memcpy(new_values, values->values, old_capacity * sizeof(*new_values));
    values->values = new_values;
    values->capacity = new_capacity;
  }
  values->values[values->count++] = value;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_i64(loom_parser_t* parser,
                                           loom_token_t* out_token,
                                           int64_t* out_value) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
  int64_t value = 0;
  if (!iree_string_view_atoi_int64(token.text, &value)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                            IREE_ARRAYSIZE(params), token);
  }
  *out_token = token;
  *out_value = value;
  return iree_ok_status();
}

iree_status_t loom_parse_format_aligned_refs(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACKET, NULL);

  loom_parsed_i64_values_t alignments;
  loom_parsed_i64_values_initialize(&alignments);
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (alignments.count > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }
    IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, LOOM_KW_ALIGN));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
    loom_token_t alignment_token = loom_token_none();
    int64_t alignment = 0;
    IREE_RETURN_IF_ERROR(
        loom_parse_format_i64(parser, &alignment_token, &alignment));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);

    loom_token_t value_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &value_token);
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_VALUE(parser, value_token, &value_id);

    uint16_t operand_index = 0;
    if (loom_op_vtable_has_segmented_operands(vtable)) {
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
          parsed, &parser->parser_arena, element->field_index, value_id,
          &operand_index));
    } else {
      operand_index = (uint16_t)(element->field_index + alignments.count);
      IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
          parsed, &parser->parser_arena, operand_index, value_id));
    }
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
        parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
        operand_index, value_token, value_token.line, value_token.end_column));
    IREE_RETURN_IF_ERROR(
        loom_parsed_i64_values_add(parser, &alignments, alignment));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);

  int64_t* arena_alignments = NULL;
  if (alignments.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, alignments.count, sizeof(*arena_alignments),
        (void**)&arena_alignments));
    memcpy(arena_alignments, alignments.values,
           alignments.count * sizeof(*arena_alignments));
  }
  uint8_t alignments_attr_index = (uint8_t)element->data;
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, alignments_attr_index,
      loom_attr_i64_array(arena_alignments, (uint16_t)alignments.count)));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          alignments_attr_index, start_token);
}

static iree_status_t loom_parse_format_attr_table_row(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed,
    iree_host_size_t* value_count, uint16_t* out_row_width) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  uint16_t row_width = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (row_width > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }

    loom_token_t value_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &value_token);
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_VALUE(parser, value_token, &value_id);

    if (row_width == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "attribute table row width exceeds %u",
                              (unsigned)UINT16_MAX);
    }
    iree_host_size_t operand_index =
        (iree_host_size_t)element->field_index + *value_count;
    if (operand_index > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "attribute table operand index exceeds max operand count %u",
          (unsigned)UINT16_MAX);
    }
    if (loom_op_vtable_has_segmented_operands(vtable)) {
      uint16_t parsed_operand_index = 0;
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
          parsed, &parser->parser_arena, element->field_index, value_id,
          &parsed_operand_index));
      operand_index = parsed_operand_index;
    } else {
      IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
          parsed, &parser->parser_arena, (uint16_t)operand_index, value_id));
    }
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
        parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
        (uint16_t)operand_index, value_token, value_token.line,
        value_token.end_column));
    ++*value_count;
    ++row_width;
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  *out_row_width = row_width;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_emit_attr_table_row_width_mismatch(
    loom_parser_t* parser, loom_token_t token, uint16_t actual_width,
    uint16_t expected_width) {
  loom_diagnostic_param_t params[] = {
      loom_param_u32(actual_width),
      loom_param_u32(expected_width),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_030, params,
                          IREE_ARRAYSIZE(params), token);
}

iree_status_t loom_parse_format_attr_table(loom_parser_t* parser,
                                           const loom_op_vtable_t* vtable,
                                           const loom_format_element_t* element,
                                           loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  loom_parsed_i64_values_t keys;
  loom_parsed_i64_values_initialize(&keys);
  iree_host_size_t value_count = 0;
  bool has_case_row_width = false;
  uint16_t case_row_width = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (keys.count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }

    loom_token_t key_token = loom_token_none();
    int64_t key = 0;
    IREE_RETURN_IF_ERROR(loom_parse_format_i64(parser, &key_token, &key));
    IREE_RETURN_IF_ERROR(loom_parsed_i64_values_add(parser, &keys, key));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    uint16_t row_width = 0;
    IREE_RETURN_IF_ERROR(loom_parse_format_attr_table_row(
        parser, vtable, element, parsed, &value_count, &row_width));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    if (!has_case_row_width) {
      case_row_width = row_width;
      has_case_row_width = true;
    } else if (row_width != case_row_width) {
      IREE_RETURN_IF_ERROR(loom_parse_format_emit_attr_table_row_width_mismatch(
          parser, key_token, row_width, case_row_width));
    }
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, LOOM_KW_DEFAULT));

  loom_token_t default_token = loom_tokenizer_peek(&parser->tokenizer);
  uint16_t default_row_width = 0;
  IREE_RETURN_IF_ERROR(loom_parse_format_attr_table_row(
      parser, vtable, element, parsed, &value_count, &default_row_width));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  if (has_case_row_width && default_row_width != case_row_width) {
    IREE_RETURN_IF_ERROR(loom_parse_format_emit_attr_table_row_width_mismatch(
        parser, default_token, default_row_width, case_row_width));
  }

  int64_t* arena_keys = NULL;
  if (keys.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->module->arena, keys.count,
                                  sizeof(*arena_keys), (void**)&arena_keys));
    memcpy(arena_keys, keys.values, keys.count * sizeof(*arena_keys));
  }
  loom_attribute_t key_attr =
      loom_attr_i64_array(arena_keys, (uint16_t)keys.count);
  if (element->data > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format ATTR_TABLE attr index %u out of range",
                            element->data);
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, (uint8_t)element->data, key_attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          (uint8_t)element->data, start_token);
}

iree_status_t loom_parse_format_region_table(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  uint8_t keys_attr_index =
      LOOM_FORMAT_REGION_TABLE_KEYS_ATTR_INDEX(element->data);
  uint8_t default_region_index =
      LOOM_FORMAT_REGION_TABLE_DEFAULT_REGION_INDEX(element->data);
  if (keys_attr_index >= vtable->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE attr index %u out of range (op has %u "
        "attributes)",
        keys_attr_index, vtable->attribute_count);
  }
  if (default_region_index >= vtable->region_count ||
      element->field_index >= vtable->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE region indices default=%u cases=%u out of range "
        "(vtable has %u region descriptors)",
        default_region_index, element->field_index, vtable->region_count);
  }

  const loom_region_descriptor_t* case_descriptor =
      loom_op_vtable_region_descriptor(vtable, element->field_index);
  const loom_region_descriptor_t* default_descriptor =
      loom_op_vtable_region_descriptor(vtable, default_region_index);

  loom_parsed_i64_values_t keys;
  loom_parsed_i64_values_initialize(&keys);
  while (
      loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("case"))) {
    loom_token_t key_token = loom_token_none();
    int64_t key = 0;
    IREE_RETURN_IF_ERROR(loom_parse_format_i64(parser, &key_token, &key));
    IREE_RETURN_IF_ERROR(loom_parsed_i64_values_add(parser, &keys, key));

    loom_region_t* case_region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_parse_region(parser, case_descriptor, &case_region));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    uint8_t region_index = (uint8_t)(element->field_index + keys.count - 1);
    IREE_RETURN_IF_ERROR(loom_parsed_op_set_region(
        parsed, &parser->parser_arena, region_index, case_region));
    IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
        parser, parsed, LOOM_LOCATION_FIELD_REGION, region_index, key_token));
  }

  IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, LOOM_KW_DEFAULT));
  loom_token_t default_token = loom_tokenizer_peek(&parser->tokenizer);
  loom_region_t* default_region = NULL;
  IREE_RETURN_IF_ERROR(
      loom_parse_region(parser, default_descriptor, &default_region));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_region(
      parsed, &parser->parser_arena, default_region_index, default_region));
  IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_REGION, default_region_index,
      default_token));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);

  int64_t* arena_keys = NULL;
  if (keys.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->module->arena, keys.count,
                                  sizeof(*arena_keys), (void**)&arena_keys));
    memcpy(arena_keys, keys.values, keys.count * sizeof(*arena_keys));
  }
  loom_attribute_t key_attr =
      loom_attr_i64_array(arena_keys, (uint16_t)keys.count);
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, keys_attr_index, key_attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          keys_attr_index, start_token);
}

static iree_status_t loom_parse_format_operand_dict_entries(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed,
    iree_arena_allocator_t* scratch_arena) {
  uint32_t errors_before = parser->error_count;
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  loom_parsed_operand_dict_entries_t entries;
  loom_parsed_operand_dict_entries_initialize(&entries);
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (entries.count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }

    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, key_token.text, &key_id));
    const loom_parsed_operand_dict_entry_t* previous =
        loom_parsed_operand_dict_find(&entries, key_id);
    if (previous) {
      IREE_RETURN_IF_ERROR(loom_parse_format_emit_duplicate_operand_dict_key(
          parser, key_token, previous->key_token));
      return iree_ok_status();
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_token_t value_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &value_token);
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_VALUE(parser, value_token, &value_id);

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);

    loom_type_t annotated_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &annotated_type));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }

    loom_type_t actual_type = loom_module_value_type(parser->module, value_id);
    if (!loom_type_equal(actual_type, annotated_type)) {
      IREE_RETURN_IF_ERROR(loom_parse_format_emit_operand_dict_type_mismatch(
          parser, key_token, value_token, actual_type, annotated_type));
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(
        loom_parsed_operand_dict_entries_add(scratch_arena, &entries,
                                             (loom_parsed_operand_dict_entry_t){
                                                 .name_id = key_id,
                                                 .value_id = value_id,
                                                 .key_token = key_token,
                                                 .value_token = value_token,
                                             }));
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);

  if (entries.count == 0) {
    return iree_ok_status();
  }
  if (element->data > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format OPERAND_DICT attr index %u out of range",
                            element->data);
  }
  if ((iree_host_size_t)element->field_index + entries.count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "operand dictionary storage range exceeds max operand count %u",
        (unsigned)UINT16_MAX);
  }

  loom_named_attr_t* name_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&parser->module->arena, entries.count,
                                sizeof(*name_entries), (void**)&name_entries));
  // Carry original entry indices through sorting so SSA values and source
  // tokens stay paired without sorting the larger token-bearing scratch.
  for (iree_host_size_t i = 0; i < entries.count; ++i) {
    name_entries[i] = (loom_named_attr_t){
        .name_id = entries.entries[i].name_id,
        .reserved = 0,
        .value = loom_attr_i64((int64_t)i),
    };
  }
  loom_module_sort_attr_dict_entries(parser->module, name_entries,
                                     entries.count);
  for (iree_host_size_t i = 0; i < entries.count; ++i) {
    const loom_parsed_operand_dict_entry_t* entry =
        &entries.entries[name_entries[i].value.i64];
    uint16_t operand_index = (uint16_t)(element->field_index + i);
    if (loom_op_vtable_has_segmented_operands(vtable)) {
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
          parsed, &parser->parser_arena, element->field_index, entry->value_id,
          &operand_index));
    } else {
      IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
          parsed, &parser->parser_arena, operand_index, entry->value_id));
    }
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
        parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
        operand_index, entry->value_token, entry->value_token.line,
        entry->value_token.end_column));
    name_entries[i].value = loom_attr_i64((int64_t)i);
  }

  loom_attribute_t names_attr =
      loom_make_canonical_attr_dict(name_entries, entries.count);
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, (uint8_t)element->data, names_attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          (uint8_t)element->data, start_token);
}

iree_status_t loom_parse_format_operand_dict(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return iree_ok_status();
  }
  // Nested type parsing can retain parser-owned state. Only dictionary entries
  // and their membership index belong to this short-lived arena.
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(parser->parser_arena.block_pool, &scratch_arena);
  iree_status_t status = loom_parse_format_operand_dict_entries(
      parser, vtable, element, parsed, &scratch_arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

iree_status_t loom_parse_format_inline_attr_dict(loom_parser_t* parser,
                                                 const loom_op_vtable_t* vtable,
                                                 loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  uint16_t entry_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (entry_count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }

    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    uint8_t attr_index = 0;
    const loom_attr_descriptor_t* descriptor =
        loom_parse_format_find_attr_descriptor(vtable, key_token.text,
                                               &attr_index);
    if (!descriptor) {
      IREE_RETURN_IF_ERROR(
          loom_parse_format_emit_unknown_attr_name(parser, key_token));
      return iree_ok_status();
    }
    if (loom_parse_format_parsed_attr_present(parsed, attr_index)) {
      IREE_RETURN_IF_ERROR(
          loom_parse_format_emit_duplicate_attr_name(parser, key_token));
      return iree_ok_status();
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);
    loom_attribute_t attr = {0};
    uint32_t attr_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
    if (parser->error_count > attr_errors_before) {
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
        parsed, &parser->parser_arena, attr_index, attr));
    IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
        parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, attr_index, key_token));
    ++entry_count;
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  return iree_ok_status();
}
