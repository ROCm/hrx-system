// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/attrs.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/types.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

static bool loom_parse_special_f64_spelling(iree_string_view_t text,
                                            double* out_value) {
  if (iree_string_view_equal(text, IREE_SV("nan")) ||
      iree_string_view_equal(text, IREE_SV("-nan"))) {
    *out_value = NAN;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("inf"))) {
    *out_value = INFINITY;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("-inf"))) {
    *out_value = -INFINITY;
    return true;
  }
  return false;
}

static iree_status_t loom_parse_f64_token(loom_parser_t* parser,
                                          loom_token_t token,
                                          double* out_value) {
  if (loom_parse_special_f64_spelling(token.text, out_value)) {
    return iree_ok_status();
  }
  if (!iree_string_view_atod(token.text, out_value)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_016, params,
                            IREE_ARRAYSIZE(params), token);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_i64_array_attr(loom_parser_t* parser,
                                               loom_attribute_t* out_attr) {
  // [1, 2, 3]
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  int64_t inline_values[32];
  int64_t* values = inline_values;
  iree_host_size_t capacity = IREE_ARRAYSIZE(inline_values);
  iree_host_size_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    loom_token_t token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
    if (count == UINT16_MAX) {
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               token);
    }
    if (count >= capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(&parser->parser_arena, count,
                                                 count + 1, sizeof(*values),
                                                 &capacity, (void**)&values));
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(token.text, &value)) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(token.text),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                              IREE_ARRAYSIZE(params), token);
    }
    values[count++] = value;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  int64_t* arena_values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(int64_t), (void**)&arena_values));
    memcpy(arena_values, values, count * sizeof(int64_t));
  }
  *out_attr = loom_attr_i64_array(arena_values, (uint16_t)count);
  return iree_ok_status();
}

static iree_status_t loom_parse_enum_attr_value(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_value) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_LANGLE) {
    loom_token_t opening_token = token;
    (void)loom_tokenizer_next(&parser->tokenizer);
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
    uint32_t value = 0;
    bool value_in_range =
        iree_string_view_atoi_uint32(token.text, &value) && value <= UINT8_MAX;
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
    if (!iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
      *out_value = 0;
      return loom_parser_emit_unexpected_token(
          parser, opening_token, IREE_SV("a declared enum keyword"));
    }
    if (!value_in_range) {
      *out_value = 0;
      return loom_parser_emit_unexpected_token(
          parser, token, IREE_SV("an integer in [0, 255]"));
    }
    *out_value = (uint8_t)value;
    return iree_ok_status();
  }

  if (token.kind != LOOM_TOKEN_BARE_IDENT && token.kind != LOOM_TOKEN_OP_NAME) {
    return loom_parser_emit_unexpected_token(parser, token,
                                             IREE_SV("enum keyword"));
  }
  (void)loom_tokenizer_next(&parser->tokenizer);
  if (!descriptor->enum_case_names) {
    IREE_ASSERT_UNREACHABLE("enum attribute has no case name table");
    IREE_BUILTIN_UNREACHABLE();
  }
  iree_host_size_t case_span = loom_attr_descriptor_enum_case_span(descriptor);
  for (iree_host_size_t i = 0; i < case_span; ++i) {
    if (descriptor->enum_case_names[i] &&
        loom_bstring_equal(descriptor->enum_case_names[i], token.text)) {
      *out_value = (uint8_t)i;
      return iree_ok_status();
    }
  }

  iree_string_view_t enum_name = descriptor->name
                                     ? loom_attr_descriptor_name(descriptor)
                                     : IREE_SV("enum");
  loom_diagnostic_param_t params[] = {
      loom_param_string(enum_name),
      loom_param_string(token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_017, params,
                          IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parse_enum_array_attr(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t* out_attr) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  uint8_t inline_values[32];
  uint8_t* values = inline_values;
  iree_host_size_t capacity = IREE_ARRAYSIZE(inline_values);
  iree_host_size_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }
    if (count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(
          parser, peek, IREE_SV("at most 65535 enum values"));
    }
    if (count >= capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(&parser->parser_arena, count,
                                                 count + 1, sizeof(*values),
                                                 &capacity, (void**)&values));
    }
    IREE_RETURN_IF_ERROR(
        loom_parse_enum_attr_value(parser, descriptor, &values[count]));
    ++count;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  uint8_t* arena_values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena,
                                                   count, sizeof(*arena_values),
                                                   (void**)&arena_values));
    memcpy(arena_values, values, count * sizeof(*arena_values));
  }
  *out_attr = loom_attr_enum_array(arena_values, (uint16_t)count);
  return iree_ok_status();
}

static iree_status_t loom_parse_signed_enum_set_attr(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t* out_attr) {
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_SIGNED_ENUM_SET ||
      iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parsing a signed enum set requires a closed descriptor-backed field");
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    return loom_parser_emit_unexpected_token(
        parser, loom_tokenizer_peek(&parser->tokenizer), IREE_SV("'['"));
  }

  uint64_t words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT * 2] = {0};
  bool has_value = false;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (has_value &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }
    loom_token_t value_token = loom_tokenizer_peek(&parser->tokenizer);
    bool is_negative =
        loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_MINUS);
    uint8_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_parse_enum_attr_value(parser, descriptor, &value));
    const iree_host_size_t word_index = value / 64u;
    const uint64_t bit = UINT64_C(1) << (value % 64u);
    uint64_t* positive_word = &words[word_index];
    uint64_t* negative_word =
        &words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT + word_index];
    if (iree_any_bit_set(*positive_word | *negative_word, bit)) {
      return loom_parser_emit_unexpected_token(
          parser, value_token, IREE_SV("each signed enum value at most once"));
    }
    *(is_negative ? negative_word : positive_word) |= bit;
    has_value = true;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    return loom_parser_emit_unexpected_token(
        parser, loom_tokenizer_peek(&parser->tokenizer), IREE_SV("']'"));
  }

  iree_host_size_t word_count = LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT;
  while (word_count > 0 && words[word_count - 1] == 0 &&
         words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT + word_count - 1] == 0) {
    --word_count;
  }
  uint64_t* arena_words = NULL;
  if (word_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->module->arena, word_count * 2,
                                  sizeof(*arena_words), (void**)&arena_words));
    memcpy(arena_words, words, word_count * sizeof(*arena_words));
    memcpy(arena_words + word_count,
           words + LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT,
           word_count * sizeof(*arena_words));
  }
  *out_attr = loom_attr_signed_enum_set(arena_words, (uint16_t)word_count);
  return iree_ok_status();
}

static iree_status_t loom_parse_symbol_collection_attr(
    loom_parser_t* parser, loom_attr_kind_t kind, loom_attribute_t* out_attr) {
  const bool is_set = kind == LOOM_ATTR_SYMBOL_SET;
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  loom_symbol_ref_t inline_values[32];
  loom_symbol_ref_t* values = inline_values;
  iree_host_size_t value_capacity = IREE_ARRAYSIZE(inline_values);
  loom_token_t inline_tokens[32];
  loom_token_t* tokens = inline_tokens;
  iree_host_size_t token_capacity = IREE_ARRAYSIZE(inline_tokens);
  iree_host_size_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }
    if (count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(
          parser, peek, IREE_SV("at most 65535 symbol references"));
    }
    if (count >= value_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, count, count + 1, sizeof(*values),
          &value_capacity, (void**)&values));
    }
    if (is_set && count >= token_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, count, count + 1, sizeof(*tokens),
          &token_capacity, (void**)&tokens));
    }
    if (is_set) tokens[count] = loom_tokenizer_peek(&parser->tokenizer);
    loom_attribute_t value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_parse_symbol_ref_attr(parser, &value));
    values[count++] = loom_attr_as_symbol(value);
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  if (is_set) {
    loom_symbol_ref_t duplicate_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_module_try_make_symbol_set(
        parser->module, loom_make_symbol_ref_array(values, count),
        &duplicate_ref, out_attr));
    if (loom_symbol_ref_is_valid(duplicate_ref)) {
      loom_string_id_t duplicate_name_id =
          parser->module->symbols.entries[duplicate_ref.symbol_id].name_id;
      iree_string_view_t duplicate_name =
          parser->module->strings.entries[duplicate_name_id];
      loom_token_t first_token = loom_token_none();
      loom_token_t duplicate_token = loom_token_none();
      bool found_first = false;
      for (iree_host_size_t i = 0; i < count; ++i) {
        if (!iree_string_view_equal(tokens[i].text, duplicate_name)) continue;
        if (!found_first) {
          first_token = tokens[i];
          found_first = true;
        } else {
          duplicate_token = tokens[i];
          break;
        }
      }
      IREE_ASSERT(found_first);
      IREE_ASSERT(duplicate_token.kind != LOOM_TOKEN_NONE);
      loom_diagnostic_param_t params[] = {
          loom_param_string(duplicate_name),
      };
      return loom_parser_emit_related(
          parser, LOOM_ERR_PARSE_035, params, IREE_ARRAYSIZE(params),
          duplicate_token, IREE_SV("previously listed here"), first_token);
    }
    return iree_ok_status();
  }

  loom_symbol_ref_t* arena_values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena,
                                                   count, sizeof(*arena_values),
                                                   (void**)&arena_values));
    memcpy(arena_values, values, count * sizeof(*arena_values));
  }
  *out_attr = loom_attr_symbol_array(arena_values, (uint16_t)count);
  return iree_ok_status();
}

static int8_t loom_parse_hex_nibble(uint8_t c) {
  if (c >= '0' && c <= '9') return (int8_t)(c - '0');
  if (c >= 'a' && c <= 'f') return (int8_t)(10 + c - 'a');
  if (c >= 'A' && c <= 'F') return (int8_t)(10 + c - 'A');
  return -1;
}

static iree_status_t loom_parse_bytes_attr_hex(loom_parser_t* parser,
                                               loom_token_t hex_token,
                                               loom_attribute_t* out_attr) {
  iree_string_view_t hex = hex_token.text;
  if ((hex.size & 1) != 0 || hex.size > ((uint64_t)UINT32_MAX * 2u)) {
    return loom_parser_emit_unexpected_token(
        parser, hex_token, IREE_SV("an even-length hex byte string"));
  }
  for (iree_host_size_t i = 0; i < hex.size; ++i) {
    if (loom_parse_hex_nibble((uint8_t)hex.data[i]) < 0) {
      return loom_parser_emit_unexpected_token(
          parser, hex_token, IREE_SV("an even-length hex byte string"));
    }
  }

  const uint32_t byte_length = (uint32_t)(hex.size / 2u);
  uint8_t* bytes = NULL;
  if (byte_length != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, byte_length, sizeof(uint8_t), (void**)&bytes));
    for (uint32_t i = 0; i < byte_length; ++i) {
      int8_t high = loom_parse_hex_nibble((uint8_t)hex.data[i * 2u]);
      int8_t low = loom_parse_hex_nibble((uint8_t)hex.data[i * 2u + 1u]);
      bytes[i] = (uint8_t)((high << 4) | low);
    }
  }
  *out_attr = loom_attr_bytes(bytes, byte_length);
  return iree_ok_status();
}

static iree_status_t loom_parse_bytes_attr(loom_parser_t* parser,
                                           loom_attribute_t* out_attr) {
  if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                          IREE_SV("bytes"))) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'bytes'"));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  loom_token_t hex_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &hex_token);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return loom_parse_bytes_attr_hex(parser, hex_token, out_attr);
}

static bool loom_parse_next_generic_attr_is_bytes(loom_parser_t* parser) {
  loom_tokenizer_t lookahead = parser->tokenizer;
  if (!loom_tokenizer_try_consume_keyword(&lookahead, IREE_SV("bytes"))) {
    return false;
  }
  return loom_tokenizer_at(&lookahead, LOOM_TOKEN_LPAREN);
}

static iree_status_t loom_parse_present_attr_dict(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr);

// Parameter fields may recursively contain parameterized attributes or
// dictionaries. The shared aggregate depth bound keeps this recursion at no
// more than LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH frames.
static iree_status_t loom_parse_attr_value_at_depth(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    uint16_t nesting_depth, loom_type_parse_mode_t type_mode,
    loom_attribute_t* out_attr);

static iree_status_t loom_parse_generic_attr_value_with_type_mode(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr);

static iree_status_t loom_parse_next_parameter_is_named(loom_parser_t* parser,
                                                        bool* out_is_named) {
  *out_is_named = false;
  loom_tokenizer_t lookahead = parser->tokenizer;
  if (loom_tokenizer_at(&lookahead, LOOM_TOKEN_BARE_IDENT)) {
    (void)loom_tokenizer_next(&lookahead);
    *out_is_named = loom_tokenizer_at(&lookahead, LOOM_TOKEN_EQUALS);
  }
  return loom_tokenizer_consume_status(&lookahead);
}

static iree_status_t loom_parse_parameterized_attr_parameters_impl(
    loom_parser_t* parser,
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    uint16_t nesting_depth, loom_type_parse_mode_t type_mode,
    loom_attribute_t* out_attr) {
  loom_token_t opening_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, &opening_token);
  if (nesting_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_021, params,
                            IREE_ARRAYSIZE(params), opening_token);
  }

  loom_attribute_t parameter_slots[UINT8_MAX] = {0};
  uint64_t present_bits[4] = {0};
  uint16_t parsed_parameter_count = 0;
  uint32_t errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (parsed_parameter_count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }

    loom_token_t parameter_token = loom_tokenizer_peek(&parser->tokenizer);
    uint8_t parameter_index = 0;
    const loom_attr_descriptor_t* parameter_descriptor = NULL;
    bool is_named = true;
    if (parsed_parameter_count == 0 &&
        family_descriptor->primary_parameter_index !=
            LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER) {
      IREE_RETURN_IF_ERROR(
          loom_parse_next_parameter_is_named(parser, &is_named));
      if (!is_named) {
        parameter_index = family_descriptor->primary_parameter_index;
        parameter_descriptor =
            &family_descriptor->parameter_descriptors[parameter_index];
      }
    }
    if (is_named) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &parameter_token);
      parameter_descriptor = loom_attr_descriptor_find_by_name(
          family_descriptor->parameter_descriptors,
          family_descriptor->parameter_count, parameter_token.text,
          &parameter_index);
      if (!parameter_descriptor) {
        char expected[320];
        iree_string_view_t family_name =
            loom_bstring_view(family_descriptor->name);
        int expected_length = iree_snprintf(
            expected, sizeof(expected), "a parameter declared by '#%.*s'",
            (int)family_name.size, family_name.data);
        return loom_parser_emit_unexpected_token(
            parser, parameter_token,
            iree_make_string_view(expected, (iree_host_size_t)expected_length));
      }
    }

    uint64_t parameter_bit = UINT64_C(1) << (parameter_index & 63u);
    uint64_t* present_word = &present_bits[parameter_index >> 6];
    if ((*present_word & parameter_bit) != 0) {
      return loom_parser_emit_unexpected_token(
          parser, parameter_token, IREE_SV("each parameter at most once"));
    }
    *present_word |= parameter_bit;

    if (is_named) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);
    }
    IREE_RETURN_IF_ERROR(loom_parse_attr_value_at_depth(
        parser, parameter_descriptor, (uint16_t)(nesting_depth + 1), type_mode,
        &parameter_slots[parameter_index]));
    if (parser->error_count > errors_before) return iree_ok_status();
    ++parsed_parameter_count;
  }

  loom_token_t closing_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, &closing_token);
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    if (!loom_attr_is_absent(parameter_slots[i]) ||
        iree_any_bit_set(parameter_descriptor->flags, LOOM_ATTR_OPTIONAL)) {
      continue;
    }
    char expected[320];
    iree_string_view_t parameter_name =
        loom_attr_descriptor_name(parameter_descriptor);
    int expected_length =
        iree_snprintf(expected, sizeof(expected), "required parameter '%.*s'",
                      (int)parameter_name.size, parameter_name.data);
    return loom_parser_emit_unexpected_token(
        parser, closing_token,
        iree_make_string_view(expected, (iree_host_size_t)expected_length));
  }

  return loom_module_make_parameterized_attr(
      parser->module, family_descriptor->kind, parameter_slots,
      family_descriptor->parameter_count, out_attr);
}

iree_status_t loom_parse_parameterized_attr_parameters(
    loom_parser_t* parser, loom_parameterized_attr_kind_t family_kind,
    loom_attribute_t* out_attr) {
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_resolve_parameterized_attr(parser->context, family_kind);
  if (!family_descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown parameterized attribute family kind %u",
                            (unsigned)family_kind);
  }
  return loom_parse_parameterized_attr_parameters_impl(
      parser, family_descriptor, /*nesting_depth=*/0, LOOM_TYPE_PARSE_BODY,
      out_attr);
}

static iree_status_t loom_parse_parameterized_attr(
    loom_parser_t* parser, loom_parameterized_attr_kind_t expected_family_kind,
    uint16_t nesting_depth, loom_type_parse_mode_t type_mode,
    loom_attribute_t* out_attr) {
  loom_token_t family_token = loom_tokenizer_peek(&parser->tokenizer);
  if (family_token.kind != LOOM_TOKEN_HASH_ATTR) {
    return loom_parser_emit_unexpected_token(
        parser, family_token, IREE_SV("a parameterized attribute family"));
  }
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_lookup_parameterized_attr_by_name(parser->context,
                                                     family_token.text);
  if (!family_descriptor) {
    return loom_parser_emit_unexpected_token(
        parser, family_token,
        IREE_SV("a registered parameterized attribute family"));
  }
  if (expected_family_kind != LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      family_descriptor->kind != expected_family_kind) {
    const loom_parameterized_attr_descriptor_t* expected_family_descriptor =
        loom_context_resolve_parameterized_attr(parser->context,
                                                expected_family_kind);
    char expected[272];
    int expected_length = iree_snprintf(
        expected, sizeof(expected), "'#%.*s'",
        (int)loom_bstring_view(expected_family_descriptor->name).size,
        loom_bstring_view(expected_family_descriptor->name).data);
    return loom_parser_emit_unexpected_token(
        parser, family_token,
        iree_make_string_view(expected, (iree_host_size_t)expected_length));
  }
  (void)loom_tokenizer_next(&parser->tokenizer);
  return loom_parse_parameterized_attr_parameters_impl(
      parser, family_descriptor, nesting_depth, type_mode, out_attr);
}

static iree_status_t loom_parse_parameterized_attr_array(
    loom_parser_t* parser, loom_parameterized_attr_kind_t expected_family_kind,
    uint16_t nesting_depth, loom_type_parse_mode_t type_mode,
    loom_attribute_t* out_attr) {
  loom_token_t opening_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACKET, &opening_token);
  if (nesting_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_021, params,
                            IREE_ARRAYSIZE(params), opening_token);
  }

  loom_attribute_t inline_attributes[16];
  loom_attribute_t* attributes = inline_attributes;
  iree_host_size_t capacity = IREE_ARRAYSIZE(inline_attributes);
  iree_host_size_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }
    if (count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(
          parser, peek, IREE_SV("at most 65535 parameterized attributes"));
    }
    if (count >= capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, count, count + 1, sizeof(*attributes),
          &capacity, (void**)&attributes));
    }
    const uint32_t element_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_parameterized_attr(
        parser, expected_family_kind, (uint16_t)(nesting_depth + 1), type_mode,
        &attributes[count]));
    if (parser->error_count > element_errors_before) return iree_ok_status();
    ++count;
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);
  return loom_module_make_parameterized_attr_array(
      parser->module, loom_make_parameterized_attr_array(attributes, count),
      out_attr);
}

static iree_status_t loom_parse_attr_value_at_depth(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    uint16_t nesting_depth, loom_type_parse_mode_t type_mode,
    loom_attribute_t* out_attr) {
  switch (descriptor->attr_kind) {
    case LOOM_ATTR_I64: {
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
      *out_attr = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_F64: {
      loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind == LOOM_TOKEN_BARE_IDENT) {
        double special_value = 0.0;
        if (!loom_parse_special_f64_spelling(token.text, &special_value)) {
          return loom_parser_emit_unexpected_token(parser, token,
                                                   IREE_SV("FLOAT"));
        }
      } else if (token.kind != LOOM_TOKEN_FLOAT) {
        return loom_parser_emit_unexpected_token(parser, token,
                                                 IREE_SV("FLOAT"));
      }
      token = loom_tokenizer_next(&parser->tokenizer);
      double value = 0.0;
      IREE_RETURN_IF_ERROR(loom_parse_f64_token(parser, token, &value));
      *out_attr = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      loom_token_t token = loom_token_none();
      if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_BARE_IDENTIFIER)) {
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &token);
      } else {
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &token);
      }
      // String token text is decoded and bare identifiers are already sliced
      // to their payload, so both forms can be interned directly.
      loom_string_id_t string_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_string(parser->module, token.text, &string_id));
      *out_attr = loom_attr_string(string_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_BOOL: {
      if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                             IREE_SV("true"))) {
        *out_attr = loom_attr_bool(true);
      } else if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                                    IREE_SV("false"))) {
        *out_attr = loom_attr_bool(false);
      } else {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek,
                                                 IREE_SV("'true' or 'false'"));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL: {
      return loom_parse_symbol_ref_attr(parser, out_attr);
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET:
      return loom_parse_symbol_collection_attr(
          parser, (loom_attr_kind_t)descriptor->attr_kind, out_attr);
    case LOOM_ATTR_ENUM: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_parse_enum_attr_value(parser, descriptor, &value));
      *out_attr = loom_attr_enum(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      return loom_parse_enum_array_attr(parser, descriptor, out_attr);
    }
    case LOOM_ATTR_SIGNED_ENUM_SET:
      return loom_parse_signed_enum_set_attr(parser, descriptor, out_attr);
    case LOOM_ATTR_I64_ARRAY: {
      return loom_parse_i64_array_attr(parser, out_attr);
    }
    case LOOM_ATTR_BYTES: {
      return loom_parse_bytes_attr(parser, out_attr);
    }
    case LOOM_ATTR_TYPE: {
      loom_type_t type = {0};
      IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
      loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_type_id(parser->module, type, &type_id));
      *out_attr = loom_attr_type(type_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_ENCODING: {
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
          parser, LOOM_STRING_ID_INVALID, &encoding_id));
      *out_attr = loom_attr_encoding(encoding_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT:
      return loom_parse_present_attr_dict(parser, nesting_depth, type_mode,
                                          out_attr);
    case LOOM_ATTR_PARAMETERIZED:
      return loom_parse_parameterized_attr(
          parser, descriptor->reference.parameterized_attr_kind, nesting_depth,
          type_mode, out_attr);
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      return loom_parse_parameterized_attr_array(
          parser, descriptor->reference.parameterized_attr_kind, nesting_depth,
          type_mode, out_attr);
    case LOOM_ATTR_ANY:
      return loom_parse_generic_attr_value_with_type_mode(parser, nesting_depth,
                                                          type_mode, out_attr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported attribute kind %d",
                              (int)descriptor->attr_kind);
  }
}

iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr) {
  return loom_parse_attr_value_at_depth(parser, descriptor,
                                        /*nesting_depth=*/0,
                                        LOOM_TYPE_PARSE_BODY, out_attr);
}

iree_status_t loom_parse_attr_value_with_type_mode(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr) {
  return loom_parse_attr_value_at_depth(parser, descriptor,
                                        /*nesting_depth=*/0, type_mode,
                                        out_attr);
}

iree_status_t loom_parse_symbol_ref_attr(loom_parser_t* parser,
                                         loom_attribute_t* out_attr) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SYMBOL, &token);
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, token.text, &name_id));
  loom_symbol_ref_t ref = {.module_id = 0};
  ref.symbol_id = loom_symbol_map_find(&parser->symbol_lookup, name_id);
  if (ref.symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(parser->module, name_id, &ref.symbol_id));
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &parser->symbol_lookup, &parser->parser_arena, name_id, ref.symbol_id));
    loom_parser_symbol_origins_t* origins = &parser->symbol_origins;
    if (origins->count >= origins->capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, origins->count, origins->count + 1,
          sizeof(loom_parser_symbol_origin_t), &origins->capacity,
          (void**)&origins->entries));
    }
    origins->entries[origins->count++] = (loom_parser_symbol_origin_t){
        .symbol_id = ref.symbol_id,
        .token = token,
    };
  }
  *out_attr = loom_attr_symbol(ref);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Predicate parsing
//===----------------------------------------------------------------------===//

// Predicate kind names for lookup.
static const struct {
  loom_bstring_t name;
  uint8_t kind;
} loom_predicate_names[] = {
    {(const uint8_t*)"\x02"
                     "eq",
     LOOM_PREDICATE_EQ},
    {(const uint8_t*)"\x02"
                     "ne",
     LOOM_PREDICATE_NE},
    {(const uint8_t*)"\x02"
                     "lt",
     LOOM_PREDICATE_LT},
    {(const uint8_t*)"\x02"
                     "le",
     LOOM_PREDICATE_LE},
    {(const uint8_t*)"\x02"
                     "gt",
     LOOM_PREDICATE_GT},
    {(const uint8_t*)"\x02"
                     "ge",
     LOOM_PREDICATE_GE},
    {(const uint8_t*)"\x03"
                     "mul",
     LOOM_PREDICATE_MUL},
    {(const uint8_t*)"\x03"
                     "min",
     LOOM_PREDICATE_MIN},
    {(const uint8_t*)"\x03"
                     "max",
     LOOM_PREDICATE_MAX},
    {(const uint8_t*)"\x04"
                     "pow2",
     LOOM_PREDICATE_POW2},
    {(const uint8_t*)"\x05"
                     "range",
     LOOM_PREDICATE_RANGE},
    {(const uint8_t*)"\x07"
                     "not_nan",
     LOOM_PREDICATE_NOT_NAN},
    {(const uint8_t*)"\x07"
                     "not_inf",
     LOOM_PREDICATE_NOT_INF},
    {(const uint8_t*)"\x06"
                     "finite",
     LOOM_PREDICATE_FINITE},
};

static iree_status_t loom_parse_predicate(loom_parser_t* parser,
                                          loom_predicate_t* out_predicate) {
  // Parse predicate kind name.
  loom_token_t name_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &name_token);

  uint8_t pred_kind = UINT8_MAX;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_predicate_names); ++i) {
    if (loom_bstring_equal(loom_predicate_names[i].name, name_token.text)) {
      pred_kind = loom_predicate_names[i].kind;
      break;
    }
  }
  if (pred_kind == UINT8_MAX) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(name_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_013, params,
                            IREE_ARRAYSIZE(params), name_token);
  }

  // Expect '('.
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  loom_predicate_t predicate = {
      .kind = pred_kind,
  };

  // Parse arguments.
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (predicate.arg_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (predicate.arg_count >= 3) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }

    loom_token_t arg_token = loom_tokenizer_peek(&parser->tokenizer);
    if (arg_token.kind == LOOM_TOKEN_SSA_VALUE) {
      // SSA value reference.
      loom_tokenizer_next(&parser->tokenizer);
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      LOOM_PARSE_RESOLVE_VALUE(parser, arg_token, &value_id);
      predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_VALUE;
      predicate.args[predicate.arg_count] = (int64_t)value_id;
    } else if (arg_token.kind == LOOM_TOKEN_INTEGER) {
      // Constant.
      loom_tokenizer_next(&parser->tokenizer);
      int64_t value = 0;
      if (!iree_string_view_atoi_int64(arg_token.text, &value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(arg_token.text),
        };
        return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                                IREE_ARRAYSIZE(params), arg_token);
      }
      predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_CONST;
      predicate.args[predicate.arg_count] = value;
    } else {
      return loom_parser_emit_unexpected_token(parser, arg_token,
                                               IREE_SV("a predicate argument"));
    }
    ++predicate.arg_count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  uint8_t expected_argument_count =
      loom_predicate_kind_argument_count(pred_kind);
  if (predicate.arg_count != expected_argument_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(name_token.text),
        loom_param_u32(expected_argument_count),
        loom_param_u32(predicate.arg_count),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_031, params,
                            IREE_ARRAYSIZE(params), name_token);
  }
  *out_predicate = predicate;
  return iree_ok_status();
}

static iree_status_t loom_parse_predicate_array_attr(
    loom_parser_t* parser, const loom_predicate_t* predicates,
    iree_host_size_t count, loom_attribute_t* out_attr) {
  loom_predicate_t* arena_predicates = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(*arena_predicates),
        (void**)&arena_predicates));
    memcpy(arena_predicates, predicates, count * sizeof(*arena_predicates));
  }
  *out_attr = loom_attr_predicate_list(arena_predicates, count);
  return iree_ok_status();
}

iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_attribute_t* out_attr) {
  // [pred(args), ...]
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  loom_predicate_t inline_predicates[16];
  loom_predicate_t* predicates = inline_predicates;
  iree_host_size_t capacity = IREE_ARRAYSIZE(inline_predicates);
  iree_host_size_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      break;
    }
    if (count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(
          parser, peek, IREE_SV("at most 65535 predicates"));
    }
    if (count >= capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, count, count + 1, sizeof(*predicates),
          &capacity, (void**)&predicates));
    }
    IREE_RETURN_IF_ERROR(loom_parse_predicate(parser, &predicates[count++]));
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  return loom_parse_predicate_array_attr(parser, predicates, count, out_attr);
}

//===----------------------------------------------------------------------===//
// Dict attribute parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_parser_emit_duplicate_attr_dict_key(
    loom_parser_t* parser, loom_token_t key_token,
    loom_token_t previous_key_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(key_token.text),
  };
  return loom_parser_emit_related(
      parser, LOOM_ERR_PARSE_020, params, IREE_ARRAYSIZE(params), key_token,
      IREE_SV("previously defined here"), previous_key_token);
}

static iree_string_view_t loom_parsed_attr_dict_entry_name(
    const loom_module_t* module, const loom_parsed_attr_dict_entry_t* entry) {
  return module->strings.entries[entry->attr.name_id];
}

static iree_status_t loom_parse_generic_attr_value_with_type_mode(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr) {
  loom_token_t value_token = loom_tokenizer_peek(&parser->tokenizer);
  switch (value_token.kind) {
    case LOOM_TOKEN_INTEGER: {
      loom_tokenizer_next(&parser->tokenizer);
      int64_t int_value = 0;
      if (!iree_string_view_atoi_int64(value_token.text, &int_value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(value_token.text),
        };
        return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                                IREE_ARRAYSIZE(params), value_token);
      }
      *out_attr = loom_attr_i64(int_value);
      return iree_ok_status();
    }
    case LOOM_TOKEN_FLOAT: {
      loom_tokenizer_next(&parser->tokenizer);
      double float_value = 0.0;
      IREE_RETURN_IF_ERROR(
          loom_parse_f64_token(parser, value_token, &float_value));
      *out_attr = loom_attr_f64(float_value);
      return iree_ok_status();
    }
    case LOOM_TOKEN_STRING: {
      loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, value_token.text, &string_id));
      *out_attr = loom_attr_string(string_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_SYMBOL:
      return loom_parse_symbol_ref_attr(parser, out_attr);
    case LOOM_TOKEN_BARE_IDENT: {
      if (loom_parse_next_generic_attr_is_bytes(parser)) {
        return loom_parse_bytes_attr(parser, out_attr);
      }
      double special_value = 0.0;
      if (loom_parse_special_f64_spelling(value_token.text, &special_value)) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_f64(special_value);
        return iree_ok_status();
      }
      if (iree_string_view_equal(value_token.text, IREE_SV("true"))) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_bool(true);
        return iree_ok_status();
      }
      if (iree_string_view_equal(value_token.text, IREE_SV("false"))) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_bool(false);
        return iree_ok_status();
      }
      loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t ident_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, value_token.text, &ident_id));
      *out_attr = loom_attr_string(ident_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_LBRACKET:
      return loom_parse_i64_array_attr(parser, out_attr);
    case LOOM_TOKEN_HASH_ATTR: {
      if (loom_context_lookup_parameterized_attr_by_name(parser->context,
                                                         value_token.text)) {
        return loom_parse_parameterized_attr(
            parser, LOOM_PARAMETERIZED_ATTR_KIND_ANY, nesting_depth, type_mode,
            out_attr);
      }
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
          parser, LOOM_STRING_ID_INVALID, &encoding_id));
      *out_attr = loom_attr_encoding(encoding_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_LBRACE:
      return loom_parse_present_attr_dict(parser, (uint16_t)(nesting_depth + 1),
                                          type_mode, out_attr);
    default:
      break;
  }
  return loom_parser_emit_unexpected_token(parser, value_token,
                                           IREE_SV("an attribute value"));
}

iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
                                            loom_attribute_t* out_attr) {
  return loom_parse_generic_attr_value_with_type_mode(
      parser, nesting_depth, LOOM_TYPE_PARSE_BODY, out_attr);
}

void loom_parser_sort_attr_dict_entries(const loom_module_t* module,
                                        loom_parsed_attr_dict_entry_t* entries,
                                        uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    loom_parsed_attr_dict_entry_t entry = entries[i];
    iree_string_view_t key_name =
        loom_parsed_attr_dict_entry_name(module, &entry);
    uint16_t slot = i;
    while (slot > 0) {
      iree_string_view_t previous_key_name =
          loom_parsed_attr_dict_entry_name(module, &entries[slot - 1]);
      if (iree_string_view_compare(previous_key_name, key_name) <= 0) {
        break;
      }
      entries[slot] = entries[slot - 1];
      --slot;
    }
    entries[slot] = entry;
  }
}

static iree_status_t loom_parse_present_attr_dict(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr) {
  loom_token_t open_brace_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, &open_brace_token);
  if (nesting_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_021, params,
                            IREE_ARRAYSIZE(params), open_brace_token);
  }

  uint32_t errors_before = parser->error_count;
  loom_parsed_attr_dict_entry_t stack_entries[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= IREE_ARRAYSIZE(stack_entries)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }

    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, key_token.text, &key_id));
    for (uint16_t i = 0; i < count; ++i) {
      if (stack_entries[i].attr.name_id == key_id) {
        return loom_parser_emit_duplicate_attr_dict_key(
            parser, key_token, stack_entries[i].key_token);
      }
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(loom_parse_generic_attr_value_with_type_mode(
        parser, nesting_depth, type_mode, &value));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }

    stack_entries[count].attr.name_id = key_id;
    stack_entries[count].attr.reserved = 0;
    stack_entries[count].attr.value = value;
    stack_entries[count].key_token = key_token;
    ++count;
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);

  if (count == 0) {
    *out_attr = loom_make_canonical_attr_dict(NULL, 0);
    return iree_ok_status();
  }

  loom_parser_sort_attr_dict_entries(parser->module, stack_entries, count);

  loom_named_attr_t* arena_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, count,
                                                 sizeof(*arena_entries),
                                                 (void**)&arena_entries));
  for (uint16_t i = 0; i < count; ++i) {
    arena_entries[i] = stack_entries[i].attr;
  }
  *out_attr = loom_make_canonical_attr_dict(arena_entries, count);
  return iree_ok_status();
}

iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr) {
  // {key = value, key = value, ...}
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    // Empty dict; no brace means absent.
    memset(out_attr, 0, sizeof(*out_attr));
    return iree_ok_status();
  }
  return loom_parse_present_attr_dict(parser, /*nesting_depth=*/0,
                                      LOOM_TYPE_PARSE_BODY, out_attr);
}
