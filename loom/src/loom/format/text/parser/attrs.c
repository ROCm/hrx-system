// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/attrs.h"

#include <math.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/types.h"

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

  int64_t stack_values[32];
  uint16_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    loom_token_t token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
    if (count >= IREE_ARRAYSIZE(stack_values)) {
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               token);
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(token.text, &value)) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(token.text),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                              IREE_ARRAYSIZE(params), token);
    }
    stack_values[count++] = value;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  int64_t* arena_values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(int64_t), (void**)&arena_values));
    memcpy(arena_values, stack_values, count * sizeof(int64_t));
  }
  *out_attr = loom_attr_i64_array(arena_values, count);
  return iree_ok_status();
}

iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
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
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &token);
      // Token text is the decoded content without surrounding quotes, so
      // intern it directly.
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
    case LOOM_ATTR_ENUM: {
      loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind != LOOM_TOKEN_BARE_IDENT &&
          token.kind != LOOM_TOKEN_OP_NAME) {
        return loom_parser_emit_unexpected_token(parser, token,
                                                 IREE_SV("identifier"));
      }
      (void)loom_tokenizer_next(&parser->tokenizer);
      // Look up the enum case name.
      if (!descriptor->enum_case_names) {
        // Internal bug: enum attribute declared without case names.
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "enum attribute has no case name table");
      }
      // Linear scan through case names. Enum case lists are short.
      for (uint8_t i = 0; i < descriptor->enum_case_count; ++i) {
        if (descriptor->enum_case_names[i] &&
            loom_bstring_equal(descriptor->enum_case_names[i], token.text)) {
          *out_attr = loom_attr_enum(i);
          return iree_ok_status();
        }
      }
      {
        iree_string_view_t enum_name =
            descriptor->name ? loom_attr_descriptor_name(descriptor)
                             : IREE_SV("enum");
        loom_diagnostic_param_t params[] = {
            loom_param_string(enum_name),
            loom_param_string(token.text),
        };
        return loom_parser_emit(parser, LOOM_ERR_PARSE_017, params,
                                IREE_ARRAYSIZE(params), token);
      }
    }
    case LOOM_ATTR_I64_ARRAY: {
      return loom_parse_i64_array_attr(parser, out_attr);
    }
    case LOOM_ATTR_TYPE: {
      loom_type_t type = {0};
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
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
    case LOOM_ATTR_ANY:
      return loom_parse_generic_attr_value(parser, /*nesting_depth=*/0,
                                           out_attr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported attribute kind %d",
                              (int)descriptor->attr_kind);
  }
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
};

iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_attribute_t* out_attr) {
  // [pred(args), ...]
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  loom_predicate_t stack_predicates[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= 16) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }

    // Parse predicate kind name.
    loom_token_t name_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &name_token);

    uint8_t pred_kind = UINT8_MAX;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_predicate_names);
         ++i) {
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
        return loom_parser_emit_unexpected_token(
            parser, arg_token, IREE_SV("a predicate argument"));
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

    stack_predicates[count++] = predicate;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  loom_predicate_t* arena_predicates = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(loom_predicate_t),
        (void**)&arena_predicates));
    memcpy(arena_predicates, stack_predicates,
           count * sizeof(loom_predicate_t));
  }
  *out_attr = loom_attr_predicate_list(arena_predicates, count);
  return iree_ok_status();
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

static iree_status_t loom_parse_present_attr_dict(loom_parser_t* parser,
                                                  uint16_t nesting_depth,
                                                  loom_attribute_t* out_attr);

iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
                                            loom_attribute_t* out_attr) {
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
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
          parser, LOOM_STRING_ID_INVALID, &encoding_id));
      *out_attr = loom_attr_encoding(encoding_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_LBRACE:
      return loom_parse_present_attr_dict(parser, (uint16_t)(nesting_depth + 1),
                                          out_attr);
    default:
      break;
  }
  return loom_parser_emit_unexpected_token(parser, value_token,
                                           IREE_SV("an attribute value"));
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

static iree_status_t loom_parse_present_attr_dict(loom_parser_t* parser,
                                                  uint16_t nesting_depth,
                                                  loom_attribute_t* out_attr) {
  loom_token_t open_brace_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, &open_brace_token);
  if (nesting_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(LOOM_ATTR_DICT_MAX_NESTING_DEPTH),
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
    IREE_RETURN_IF_ERROR(
        loom_parse_generic_attr_value(parser, nesting_depth, &value));
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
  return loom_parse_present_attr_dict(parser, /*nesting_depth=*/0, out_attr);
}
