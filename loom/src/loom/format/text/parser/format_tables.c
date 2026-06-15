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
#define LOOM_ATTR_TABLE_INLINE_KEYS 16

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
  // Inline storage for small operand dictionaries.
  loom_parsed_operand_dict_entry_t
      inline_entries[LOOM_OPERAND_DICT_INLINE_ENTRIES];
} loom_parsed_operand_dict_entries_t;

typedef struct loom_parsed_attr_table_keys_t {
  // Mutable key storage, initially pointing at inline_keys.
  int64_t* keys;
  // Number of populated keys.
  iree_host_size_t count;
  // Allocated key capacity.
  iree_host_size_t capacity;
  // Inline storage for small attribute-keyed tables.
  int64_t inline_keys[LOOM_ATTR_TABLE_INLINE_KEYS];
} loom_parsed_attr_table_keys_t;

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
    case LOOM_FORMAT_KIND_OP_REF:
    case LOOM_FORMAT_KIND_TEMPLATE_PARAM:
    case LOOM_FORMAT_KIND_PREDICATE_LIST:
      return element->field_index == attr_index;
    case LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS:
      return element->field_index == attr_index;
    case LOOM_FORMAT_KIND_DESCRIPTOR_REF:
    case LOOM_FORMAT_KIND_STABLE_KEY_REF:
      return element->field_index == attr_index || element->data == attr_index;
    case LOOM_FORMAT_KIND_INDEX_LIST:
      return LOOM_FORMAT_INDEX_LIST_STATIC_ATTR_INDEX(element->data) ==
             attr_index;
    case LOOM_FORMAT_KIND_OPERAND_DICT:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_ATTR_TABLE:
      return element->data == attr_index;
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
    const loom_op_vtable_t* vtable, const loom_format_element_t* inline_element,
    uint16_t attr_index) {
  const loom_format_element_t* elements = vtable->format_elements;
  for (uint16_t i = 0; i < vtable->format_element_count; ++i) {
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
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
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
        !loom_parse_format_inline_attr_covers_attr(vtable, inline_element,
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
}

static iree_status_t loom_parsed_operand_dict_entries_add(
    loom_parser_t* parser, loom_parsed_operand_dict_entries_t* entries,
    loom_parsed_operand_dict_entry_t entry) {
  if (entries->count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "operand dictionary has more than %u entries",
                            (unsigned)UINT16_MAX);
  }
  if (entries->count >= entries->capacity) {
    iree_host_size_t capacity = entries->capacity;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(&parser->parser_arena, entries->count, 0,
                              sizeof(loom_parsed_operand_dict_entry_t),
                              &capacity, (void**)&entries->entries));
    entries->capacity = capacity;
  }
  entries->entries[entries->count++] = entry;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_compare_string_ids(
    const loom_module_t* module, loom_string_id_t lhs_id,
    loom_string_id_t rhs_id, int* out_comparison) {
  if (lhs_id == LOOM_STRING_ID_INVALID || lhs_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand dictionary key string id %u is out of range (module has "
        "%" PRIhsz " strings)",
        lhs_id, module->strings.count);
  }
  if (rhs_id == LOOM_STRING_ID_INVALID || rhs_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand dictionary key string id %u is out of range (module has "
        "%" PRIhsz " strings)",
        rhs_id, module->strings.count);
  }
  *out_comparison = iree_string_view_compare(module->strings.entries[lhs_id],
                                             module->strings.entries[rhs_id]);
  return iree_ok_status();
}

static iree_status_t loom_parse_format_sort_operand_dict_entries(
    loom_parser_t* parser, loom_parsed_operand_dict_entries_t* entries) {
  for (iree_host_size_t i = 1; i < entries->count; ++i) {
    loom_parsed_operand_dict_entry_t entry = entries->entries[i];
    iree_host_size_t insert_index = i;
    while (insert_index > 0) {
      int comparison = 0;
      IREE_RETURN_IF_ERROR(loom_parse_format_compare_string_ids(
          parser->module, entry.name_id,
          entries->entries[insert_index - 1].name_id, &comparison));
      if (comparison > 0) {
        break;
      }
      entries->entries[insert_index] = entries->entries[insert_index - 1];
      --insert_index;
    }
    entries->entries[insert_index] = entry;
  }
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

static void loom_parsed_attr_table_keys_initialize(
    loom_parsed_attr_table_keys_t* keys) {
  keys->keys = keys->inline_keys;
  keys->count = 0;
  keys->capacity = LOOM_ATTR_TABLE_INLINE_KEYS;
}

static iree_status_t loom_parsed_attr_table_keys_add(
    loom_parser_t* parser, loom_parsed_attr_table_keys_t* keys, int64_t key) {
  if (keys->count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "attribute table has more than %u keys",
                            (unsigned)UINT16_MAX);
  }
  if (keys->count == keys->capacity) {
    iree_host_size_t old_capacity = keys->capacity;
    iree_host_size_t new_capacity = old_capacity * 2;
    int64_t* new_keys = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->parser_arena, new_capacity,
                                  sizeof(*new_keys), (void**)&new_keys));
    memcpy(new_keys, keys->keys, old_capacity * sizeof(*new_keys));
    keys->keys = new_keys;
    keys->capacity = new_capacity;
  }
  keys->keys[keys->count++] = key;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_i64_attr_table_key(
    loom_parser_t* parser, loom_token_t* out_token, int64_t* out_key) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
  int64_t key = 0;
  if (!iree_string_view_atoi_int64(token.text, &key)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                            IREE_ARRAYSIZE(params), token);
  }
  *out_token = token;
  *out_key = key;
  return iree_ok_status();
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

  loom_parsed_attr_table_keys_t keys;
  loom_parsed_attr_table_keys_initialize(&keys);
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
    IREE_RETURN_IF_ERROR(
        loom_parse_format_i64_attr_table_key(parser, &key_token, &key));
    IREE_RETURN_IF_ERROR(loom_parsed_attr_table_keys_add(parser, &keys, key));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    uint16_t row_width = 0;
    IREE_RETURN_IF_ERROR(loom_parse_format_attr_table_row(
        parser, vtable, element, parsed, &value_count, &row_width));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_brace(parser);
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
    memcpy(arena_keys, keys.keys, keys.count * sizeof(*arena_keys));
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

  loom_parsed_attr_table_keys_t keys;
  loom_parsed_attr_table_keys_initialize(&keys);
  while (
      loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("case"))) {
    loom_token_t key_token = loom_token_none();
    int64_t key = 0;
    IREE_RETURN_IF_ERROR(
        loom_parse_format_i64_attr_table_key(parser, &key_token, &key));
    IREE_RETURN_IF_ERROR(loom_parsed_attr_table_keys_add(parser, &keys, key));

    loom_region_t* case_region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_parse_region(parser, case_descriptor, &case_region));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_brace(parser);
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
    memcpy(arena_keys, keys.keys, keys.count * sizeof(*arena_keys));
  }
  loom_attribute_t key_attr =
      loom_attr_i64_array(arena_keys, (uint16_t)keys.count);
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, keys_attr_index, key_attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          keys_attr_index, start_token);
}

iree_status_t loom_parse_format_operand_dict(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return iree_ok_status();
  }

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
    for (iree_host_size_t i = 0; i < entries.count; ++i) {
      if (entries.entries[i].name_id == key_id) {
        IREE_RETURN_IF_ERROR(loom_parse_format_emit_duplicate_operand_dict_key(
            parser, key_token, entries.entries[i].key_token));
        loom_parser_sync_to_brace(parser);
        return iree_ok_status();
      }
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
      loom_parser_sync_to_brace(parser);
      return iree_ok_status();
    }

    loom_type_t actual_type = loom_module_value_type(parser->module, value_id);
    if (!loom_type_equal(actual_type, annotated_type)) {
      IREE_RETURN_IF_ERROR(loom_parse_format_emit_operand_dict_type_mismatch(
          parser, key_token, value_token, actual_type, annotated_type));
      loom_parser_sync_to_brace(parser);
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(
        loom_parsed_operand_dict_entries_add(parser, &entries,
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

  IREE_RETURN_IF_ERROR(
      loom_parse_format_sort_operand_dict_entries(parser, &entries));

  loom_named_attr_t* name_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&parser->parser_arena, entries.count,
                                sizeof(*name_entries), (void**)&name_entries));
  for (iree_host_size_t i = 0; i < entries.count; ++i) {
    uint16_t operand_index = (uint16_t)(element->field_index + i);
    if (loom_op_vtable_has_segmented_operands(vtable)) {
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
          parsed, &parser->parser_arena, element->field_index,
          entries.entries[i].value_id, &operand_index));
    } else {
      IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
          parsed, &parser->parser_arena, operand_index,
          entries.entries[i].value_id));
    }
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
        parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
        operand_index, entries.entries[i].value_token,
        entries.entries[i].value_token.line,
        entries.entries[i].value_token.end_column));
    name_entries[i] = (loom_named_attr_t){
        .name_id = entries.entries[i].name_id,
        .reserved = 0,
        .value = loom_attr_i64((int64_t)i),
    };
  }

  loom_attribute_t names_attr = {0};
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      parser->module, loom_make_named_attr_slice(name_entries, entries.count),
      &names_attr));
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, (uint8_t)element->data, names_attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          (uint8_t)element->data, start_token);
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
      loom_parser_sync_to_brace(parser);
      return iree_ok_status();
    }
    if (loom_parse_format_parsed_attr_present(parsed, attr_index)) {
      IREE_RETURN_IF_ERROR(
          loom_parse_format_emit_duplicate_attr_name(parser, key_token));
      loom_parser_sync_to_brace(parser);
      return iree_ok_status();
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);
    loom_attribute_t attr = {0};
    uint32_t attr_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
    if (parser->error_count > attr_errors_before) {
      loom_parser_sync_to_brace(parser);
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
