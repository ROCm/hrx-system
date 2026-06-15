// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/format.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/format_signatures.h"
#include "loom/format/text/parser/format_tables.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/scope.h"
#include "loom/format/text/parser/types.h"
#include "loom/ops/op_defs.h"
#include "loom/util/stable_id.h"

//===----------------------------------------------------------------------===//
// Format element parsers
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_format_add_field_span(loom_parser_t* parser,
                                               loom_parsed_op_t* parsed,
                                               loom_location_field_kind_t kind,
                                               uint16_t index,
                                               loom_token_t start_token) {
  return loom_parsed_op_add_field_span(parsed, &parser->parser_arena, kind,
                                       index, start_token,
                                       parser->tokenizer.consumed_end_line,
                                       parser->tokenizer.consumed_end_column);
}

// Parses a mixed static/dynamic index list: [0, %x, 4].
// Static values become an i64_array attribute. Dynamic values
// (INT64_MIN sentinels) generate operand references.
static iree_status_t loom_parse_format_index_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  uint16_t static_attr_index =
      LOOM_FORMAT_INDEX_LIST_STATIC_ATTR_INDEX(element->data);
  loom_token_t list_start_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  int64_t static_values[32];
  uint16_t value_count = 0;
  uint16_t dynamic_value_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (value_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (value_count >= 32) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }

    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      // Dynamic index.
      loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
      static_values[value_count] = INT64_MIN;  // Sentinel.
      uint16_t operand_index = element->field_index + dynamic_value_count++;
      if (loom_op_vtable_has_segmented_operands(vtable)) {
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
            parsed, &parser->parser_arena, element->field_index, value_id,
            &operand_index));
      } else {
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
            parsed, &parser->parser_arena, operand_index, value_id));
      }
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
          parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
          operand_index, token, token.line, token.end_column));
    } else {
      // Static index.
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
      static_values[value_count] = value;
    }
    ++value_count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  // Build the static i64 array attribute.
  int64_t* arena_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena,
                                                 value_count, sizeof(int64_t),
                                                 (void**)&arena_values));
  memcpy(arena_values, static_values, value_count * sizeof(int64_t));
  loom_attribute_t attr = loom_attr_i64_array(arena_values, value_count);
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, static_attr_index, attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          static_attr_index, list_start_token);
}

static iree_status_t loom_parse_format_keyword_is_present(
    loom_parser_t* parser, const loom_format_element_t* keyword_element,
    bool* out_present) {
  *out_present = false;
  if (keyword_element->data >= LOOM_KW_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format KEYWORD data %u out of range (max %u)",
                            keyword_element->data, (uint16_t)LOOM_KW_COUNT_);
  }
  loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
  loom_token_kind_t expected_kind =
      loom_keyword_token_kind(keyword_element->data);
  if (expected_kind == LOOM_TOKEN_BARE_IDENT) {
    loom_bstring_t keyword_bstring =
        loom_keyword_bstring((loom_keyword_id_t)keyword_element->data);
    if (!keyword_bstring) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "keyword id %u is out of range",
                              keyword_element->data);
    }
    *out_present = peek.kind == LOOM_TOKEN_BARE_IDENT &&
                   loom_bstring_equal(keyword_bstring, peek.text);
  } else {
    *out_present = peek.kind == expected_kind;
  }
  return iree_ok_status();
}

static const loom_format_element_t* loom_format_next_non_glue(
    const loom_op_vtable_t* vtable, uint16_t start_index) {
  uint16_t index = start_index;
  while (index < vtable->format_element_count &&
         vtable->format_elements[index].kind == LOOM_FORMAT_KIND_GLUE) {
    ++index;
  }
  return index < vtable->format_element_count ? &vtable->format_elements[index]
                                              : NULL;
}

static bool loom_format_keyword_starts_clause(const loom_op_vtable_t* vtable,
                                              uint16_t keyword_index) {
  const loom_format_element_t* next =
      loom_format_next_non_glue(vtable, (uint16_t)(keyword_index + 1));
  return next && next->kind == LOOM_FORMAT_KIND_KEYWORD &&
         next->data == LOOM_KW_LPAREN;
}

static iree_status_t loom_parse_format_keyword_clause_is_present(
    loom_parser_t* parser, const loom_format_element_t* keyword_element,
    bool* out_present) {
  IREE_RETURN_IF_ERROR(loom_parse_format_keyword_is_present(
      parser, keyword_element, out_present));
  if (!*out_present) return iree_ok_status();

  loom_tokenizer_t lookahead = parser->tokenizer;
  (void)loom_tokenizer_next(&lookahead);
  *out_present = loom_tokenizer_at(&lookahead, LOOM_TOKEN_LPAREN);
  return loom_tokenizer_consume_status(&lookahead);
}

// Evaluates an optional group anchor and returns the number of
// format elements to skip if the group is absent. When the group
// is present, |*out_skip_count| is set to 0.
static iree_status_t loom_parse_format_optional_group(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, uint16_t element_index,
    const loom_parsed_op_t* parsed, uint16_t* out_skip_count) {
  uint16_t skip_count = element->data >> 2;
  uint8_t anchor_category = element->data & 3;
  bool present = false;

  switch (anchor_category) {
    case LOOM_ANCHOR_OPERAND: {
      // The probe token depends on the first inner element of the
      // optional group:
      //   KEYWORD       — peek for that specific keyword text/kind
      //                   (e.g., "iter_args" or "where" before the
      //                   list it introduces).
      //   BINDING_LIST  — peek for '(' since the binding list opens
      //                   with a paren when the introducing keyword
      //                   has been dropped.
      //   anything else — fall back to peeking for an SSA value (the
      //                   common case for variadic operand groups).
      uint16_t first_inner_index = element_index + 1;
      while (first_inner_index < vtable->format_element_count &&
             vtable->format_elements[first_inner_index].kind ==
                 LOOM_FORMAT_KIND_GLUE) {
        ++first_inner_index;
      }
      const loom_format_element_t* first_inner =
          (first_inner_index < vtable->format_element_count)
              ? &vtable->format_elements[first_inner_index]
              : NULL;
      if (first_inner && first_inner->kind == LOOM_FORMAT_KIND_KEYWORD) {
        if (loom_format_keyword_starts_clause(vtable, first_inner_index)) {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_clause_is_present(
              parser, first_inner, &present));
        } else {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_is_present(
              parser, first_inner, &present));
        }
      } else if (first_inner &&
                 (first_inner->kind == LOOM_FORMAT_KIND_BINDING_LIST ||
                  first_inner->kind == LOOM_FORMAT_KIND_BLOCK_ARGS)) {
        present = loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LPAREN);
      } else {
        present = loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE);
      }
      break;
    }
    case LOOM_ANCHOR_ATTR: {
      // When the first inner element is a KEYWORD (e.g., "where"
      // before a predicate list), refine the check to match that
      // specific keyword. The generic token set would false-positive
      // on '{' (LBRACE), which is ambiguous between attr_dict and
      // region.
      //
      // When the first inner element is an ATTR_VALUE with ENUM kind,
      // refine the check to match a specific enum case name. Without
      // this, consecutive optional enum groups (visibility, purity)
      // all probe positive on any bare ident and the wrong group
      // consumes the token.
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      uint16_t first_inner_index = element_index + 1;
      while (first_inner_index < vtable->format_element_count &&
             vtable->format_elements[first_inner_index].kind ==
                 LOOM_FORMAT_KIND_GLUE) {
        ++first_inner_index;
      }
      const loom_format_element_t* first_inner =
          (first_inner_index < vtable->format_element_count)
              ? &vtable->format_elements[first_inner_index]
              : NULL;
      if (first_inner && first_inner->kind == LOOM_FORMAT_KIND_KEYWORD) {
        if (loom_format_keyword_starts_clause(vtable, first_inner_index)) {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_clause_is_present(
              parser, first_inner, &present));
        } else {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_is_present(
              parser, first_inner, &present));
        }
      } else if (first_inner &&
                 first_inner->kind == LOOM_FORMAT_KIND_ATTR_VALUE) {
        const loom_attr_descriptor_t* descriptor =
            &vtable->attr_descriptors[first_inner->field_index];
        if (descriptor->attr_kind == LOOM_ATTR_ENUM &&
            descriptor->enum_case_names) {
          present = false;
          if (peek.kind == LOOM_TOKEN_BARE_IDENT ||
              peek.kind == LOOM_TOKEN_OP_NAME) {
            for (uint8_t c = 0; c < descriptor->enum_case_count; ++c) {
              if (descriptor->enum_case_names[c] &&
                  loom_bstring_equal(descriptor->enum_case_names[c],
                                     peek.text)) {
                present = true;
                break;
              }
            }
          }
        } else {
          present = (peek.kind == LOOM_TOKEN_INTEGER ||
                     peek.kind == LOOM_TOKEN_FLOAT ||
                     peek.kind == LOOM_TOKEN_STRING ||
                     peek.kind == LOOM_TOKEN_BARE_IDENT ||
                     peek.kind == LOOM_TOKEN_LBRACKET ||
                     peek.kind == LOOM_TOKEN_LBRACE);
        }
      } else if (first_inner &&
                 first_inner->kind == LOOM_FORMAT_KIND_SYMBOL_REF) {
        present = peek.kind == LOOM_TOKEN_SYMBOL;
      } else if (first_inner &&
                 (first_inner->kind == LOOM_FORMAT_KIND_OP_REF ||
                  first_inner->kind == LOOM_FORMAT_KIND_DESCRIPTOR_REF ||
                  first_inner->kind == LOOM_FORMAT_KIND_STABLE_KEY_REF ||
                  first_inner->kind == LOOM_FORMAT_KIND_TEMPLATE_PARAM ||
                  first_inner->kind == LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS)) {
        present = peek.kind == LOOM_TOKEN_LANGLE;
      } else {
        present =
            (peek.kind == LOOM_TOKEN_INTEGER || peek.kind == LOOM_TOKEN_FLOAT ||
             peek.kind == LOOM_TOKEN_STRING ||
             peek.kind == LOOM_TOKEN_BARE_IDENT ||
             peek.kind == LOOM_TOKEN_LBRACKET ||
             peek.kind == LOOM_TOKEN_LBRACE);
      }
      break;
    }
    case LOOM_ANCHOR_REGION: {
      uint16_t first_inner_index = element_index + 1;
      while (first_inner_index < vtable->format_element_count &&
             vtable->format_elements[first_inner_index].kind ==
                 LOOM_FORMAT_KIND_GLUE) {
        ++first_inner_index;
      }
      const loom_format_element_t* first_inner =
          first_inner_index < vtable->format_element_count
              ? &vtable->format_elements[first_inner_index]
              : NULL;
      if (first_inner && first_inner->kind == LOOM_FORMAT_KIND_KEYWORD) {
        if (loom_format_keyword_starts_clause(vtable, first_inner_index)) {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_clause_is_present(
              parser, first_inner, &present));
        } else {
          IREE_RETURN_IF_ERROR(loom_parse_format_keyword_is_present(
              parser, first_inner, &present));
        }
      } else {
        present = loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE);
      }
      break;
    }
    case LOOM_ANCHOR_RESULTS:
      present = parsed->result_count > 0 ||
                loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_ARROW);
      break;
  }

  *out_skip_count = present ? 0 : skip_count;
  return iree_ok_status();
}

// Parses an instance flag list: flag1|flag2.
// Matches flag names against the vtable's flag case names.
static iree_status_t loom_parse_format_instance_flag_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable, uint8_t* out_flags) {
  uint8_t flags = 0;
  bool first = true;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (!first) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_PIPE)) {
        break;
      }
    }
    loom_token_t flag_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &flag_token);
    bool found = false;
    for (uint8_t bit = 0; bit < vtable->instance_flags_case_count; ++bit) {
      if (loom_bstring_equal(vtable->instance_flags_case_names[bit],
                             flag_token.text)) {
        flags |= (1u << bit);
        found = true;
        break;
      }
    }
    if (!found) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("instance flag")),
          loom_param_string(flag_token.text),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                              IREE_ARRAYSIZE(params), flag_token);
    }
    first = false;
  }
  *out_flags = flags;
  return iree_ok_status();
}

// Parses instance flags: <flag1|flag2>.
static iree_status_t loom_parse_format_flags(loom_parser_t* parser,
                                             const loom_op_vtable_t* vtable,
                                             loom_parsed_op_t* parsed) {
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    IREE_RETURN_IF_ERROR(loom_parse_format_instance_flag_list(
        parser, vtable, &parsed->instance_flags));
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
    }
  }
  return iree_ok_status();
}

// Parses an op or descriptor key reference: <op.name> or <descriptor-key>.
static iree_status_t loom_parse_format_op_ref(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  loom_token_t op_ref_start_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  loom_token_t name_token = loom_tokenizer_peek(&parser->tokenizer);
  if (name_token.kind != LOOM_TOKEN_OP_NAME &&
      name_token.kind != LOOM_TOKEN_BARE_IDENT) {
    return loom_parser_emit_unexpected_token(parser, name_token,
                                             IREE_SV("identifier"));
  }
  name_token = loom_tokenizer_next(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));
  loom_attribute_t attr = loom_attr_string(name_id);
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->field_index, attr));
  return loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->field_index,
      op_ref_start_token);
}

// Parses a descriptor key reference whose row ordinal is resolved later after
// target binding selects the descriptor set: <target.descriptor>.
static iree_status_t loom_parse_format_descriptor_ref(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  loom_token_t descriptor_ref_start_token =
      loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  loom_token_t key_token = loom_tokenizer_peek(&parser->tokenizer);
  if (key_token.kind != LOOM_TOKEN_OP_NAME &&
      key_token.kind != LOOM_TOKEN_BARE_IDENT) {
    return loom_parser_emit_unexpected_token(parser, key_token,
                                             IREE_SV("descriptor key"));
  }
  key_token = loom_tokenizer_next(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  loom_string_id_t key_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, key_token.text, &key_id));
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->field_index,
      loom_attr_string(key_id)));
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->data, loom_attr_i64(-1)));
  IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->field_index,
      descriptor_ref_start_token));
  return loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->data,
      descriptor_ref_start_token);
}

// Parses a stable symbolic key reference and materializes both text key and
// numeric identity: <target.key>.
static iree_status_t loom_parse_format_stable_key_ref(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  loom_token_t descriptor_ref_start_token =
      loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  loom_token_t key_token = loom_tokenizer_peek(&parser->tokenizer);
  if (key_token.kind != LOOM_TOKEN_OP_NAME &&
      key_token.kind != LOOM_TOKEN_BARE_IDENT) {
    return loom_parser_emit_unexpected_token(parser, key_token,
                                             IREE_SV("descriptor key"));
  }
  key_token = loom_tokenizer_next(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  loom_string_id_t key_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, key_token.text, &key_id));
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->field_index,
      loom_attr_string(key_id)));
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->data,
      loom_attr_i64((int64_t)loom_stable_id_from_string(key_token.text))));
  IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->field_index,
      descriptor_ref_start_token));
  return loom_parse_format_add_field_span(
      parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->data,
      descriptor_ref_start_token);
}

// Parses a required template parameter attribute: <attr-value>.
static iree_status_t loom_parse_format_template_param(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[element->field_index];
  loom_attribute_t attr = {0};
  uint32_t attr_errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
  if (parser->error_count > attr_errors_before) return iree_ok_status();
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->field_index, attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          element->field_index, start_token);
}

// Parses a required template parameter plus optional instance flags:
// <attr-value> or <attr-value, flag1|flag2>.
static iree_status_t loom_parse_format_template_param_flags(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed) {
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[element->field_index];
  loom_attribute_t attr = {0};
  uint32_t attr_errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
  if (parser->error_count > attr_errors_before) return iree_ok_status();
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    IREE_RETURN_IF_ERROR(loom_parse_format_instance_flag_list(
        parser, vtable, &parsed->instance_flags));
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
      parsed, &parser->parser_arena, element->field_index, attr));
  return loom_parse_format_add_field_span(parser, parsed,
                                          LOOM_LOCATION_FIELD_ATTRIBUTE,
                                          element->field_index, start_token);
}

static iree_status_t loom_parse_format_emit_operand_ref_type_mismatch(
    loom_parser_t* parser, loom_token_t value_token, loom_type_t actual_type,
    loom_type_t annotated_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(value_token.text),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("type annotation")),
      loom_param_type(annotated_type),
  };
  return loom_parser_emit(parser, LOOM_ERR_TYPE_001, params,
                          IREE_ARRAYSIZE(params), value_token);
}

//===----------------------------------------------------------------------===//
// Format walker
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_format_project_func_args(
    loom_parser_t* parser, uint16_t pending_func_arg_start, bool clone_values) {
  for (uint16_t i = pending_func_arg_start; i < parser->pending_func_args.count;
       ++i) {
    loom_value_id_t value_id = parser->pending_func_args.entries[i].value_id;
    if (clone_values) {
      loom_type_t value_type = loom_module_value_type(parser->module, value_id);
      IREE_RETURN_IF_ERROR(
          loom_module_define_value(parser->module, value_type, &value_id));
    }
    IREE_RETURN_IF_ERROR(loom_parser_add_pending_block_arg(
        parser, value_id, parser->pending_func_args.entries[i].name_token));
  }
  return iree_ok_status();
}

static bool loom_parse_format_symbol_ref_targets_register_context(
    const loom_op_vtable_t* vtable, uint16_t attr_index) {
  if (!vtable->func_like || !vtable->attr_descriptors ||
      attr_index >= vtable->attribute_count) {
    return false;
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[attr_index];
  return descriptor->symbol_ref &&
         iree_any_bit_set(descriptor->symbol_ref->interfaces,
                          LOOM_SYMBOL_INTERFACE_TARGET);
}

static iree_status_t loom_parse_format_update_register_context_from_target(
    loom_parser_t* parser, const loom_op_vtable_t* vtable, uint16_t attr_index,
    loom_attribute_t attr) {
  if (!loom_parse_format_symbol_ref_targets_register_context(vtable,
                                                             attr_index)) {
    return iree_ok_status();
  }
  if (!parser->low_asm_environment.vtable ||
      !parser->low_asm_environment.vtable->lookup_target_descriptor_set) {
    return iree_ok_status();
  }
  const loom_text_low_asm_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      parser->low_asm_environment.vtable->lookup_target_descriptor_set(
          parser->low_asm_environment.state, parser->module, attr,
          &descriptor_set));
  if (descriptor_set != NULL) {
    parser->low_register_descriptor_set = descriptor_set;
  }
  return iree_ok_status();
}

iree_status_t loom_parser_walk_format(loom_parser_t* parser,
                                      const loom_op_vtable_t* vtable,
                                      loom_token_t op_name_token,
                                      loom_parsed_op_t* parsed,
                                      uint16_t pending_func_arg_start,
                                      bool* out_func_args_consumed_by_region) {
  *out_func_args_consumed_by_region = false;
  const loom_format_element_t* elements = vtable->format_elements;
  uint16_t element_count = vtable->format_element_count;
  bool is_symbol_definition =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);

  uint32_t errors_before = parser->error_count;
  for (uint16_t i = 0; i < element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    switch (element->kind) {
      case LOOM_FORMAT_KIND_OPERAND_REF: {
        if (element->field_index == 0xFF) {
          // Induction variable reference (e.g., %iv in test.loop).
          // Consume the SSA token, create an index-typed value, and queue it
          // as a pending block arg. REGION defines the loop IV name in the
          // child scope when seeding the entry block.
          loom_token_t iv_token = loom_token_none();
          LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &iv_token);
          loom_value_id_t iv_value_id = 0;
          IREE_RETURN_IF_ERROR(loom_module_define_value(
              parser->module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
              &iv_value_id));
          IREE_RETURN_IF_ERROR(
              loom_parser_add_pending_block_arg(parser, iv_value_id, iv_token));
          break;
        }
        loom_token_t token = loom_token_none();
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &token);
        loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
        LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
        uint16_t operand_index = element->field_index;
        if (loom_op_vtable_has_segmented_operands(vtable)) {
          IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
              parsed, &parser->parser_arena, element->field_index, value_id,
              &operand_index));
        } else {
          IREE_RETURN_IF_ERROR(loom_parsed_op_set_operand(
              parsed, &parser->parser_arena, operand_index, value_id));
        }
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
            parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
            operand_index, token, token.line, token.end_column));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_REFS: {
        // Variadic operands: %a, %b, %c. Parse until we hit something
        // that isn't an SSA value or comma.
        bool first = true;
        while (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE) ||
               (!first &&
                loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA))) {
          if (!first) {
            loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA);
          }
          if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
            break;
          }
          loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
          loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
          LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
          uint16_t operand_index = parsed->operand_count;
          if (loom_op_vtable_has_segmented_operands(vtable)) {
            IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
                parsed, &parser->parser_arena, element->field_index, value_id,
                &operand_index));
          } else {
            IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
                parsed, &parser->parser_arena, value_id));
          }
          IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
              parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
              operand_index, token, token.line, token.end_column));
          first = false;
        }
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_TYPED_REFS: {
        // Variadic typed operands: %a: type, %b: type. Parse until the next
        // token is no longer an SSA value.
        bool first = true;
        while (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE) ||
               (!first &&
                loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA))) {
          if (!first) {
            loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA);
          }
          if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
            break;
          }
          loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
          loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
          LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
          LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);

          uint32_t type_errors_before = parser->error_count;
          loom_type_t annotated_type = {0};
          IREE_RETURN_IF_ERROR(
              loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &annotated_type));
          if (parser->error_count > type_errors_before) {
            return iree_ok_status();
          }
          loom_type_t actual_type =
              loom_module_value_type(parser->module, value_id);
          if (!loom_type_equal(actual_type, annotated_type)) {
            IREE_RETURN_IF_ERROR(
                loom_parse_format_emit_operand_ref_type_mismatch(
                    parser, token, actual_type, annotated_type));
            return iree_ok_status();
          }
          uint16_t operand_index = parsed->operand_count;
          if (loom_op_vtable_has_segmented_operands(vtable)) {
            IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
                parsed, &parser->parser_arena, element->field_index, value_id,
                &operand_index));
          } else {
            IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
                parsed, &parser->parser_arena, value_id));
          }
          IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
              parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
              operand_index, token, token.line, token.end_column));
          first = false;
        }
        break;
      }

      case LOOM_FORMAT_KIND_SUCCESSOR_REF: {
        loom_token_t token = loom_token_none();
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BLOCK_LABEL, &token);
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_successor(
            parsed, &parser->parser_arena, element->field_index,
            /*block=*/NULL, token));
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
            parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_SUCCESSOR,
            element->field_index, token, token.line, token.end_column));
        break;
      }

      case LOOM_FORMAT_KIND_ATTR_VALUE: {
        loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
        const loom_attr_descriptor_t* descriptor =
            &vtable->attr_descriptors[element->field_index];
        loom_attribute_t attr = {0};
        uint32_t attr_errors_before = parser->error_count;
        IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
        if (parser->error_count > attr_errors_before) return iree_ok_status();
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
            parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->field_index,
            start_token));
        break;
      }

      case LOOM_FORMAT_KIND_SYMBOL_REF: {
        loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
        loom_attribute_t attr = {0};
        uint32_t attr_errors_before = parser->error_count;
        IREE_RETURN_IF_ERROR(loom_parse_symbol_ref_attr(parser, &attr));
        if (parser->error_count > attr_errors_before) return iree_ok_status();
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
            parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_ATTRIBUTE,
            element->field_index, token, token.line, token.end_column));
        IREE_RETURN_IF_ERROR(
            loom_parse_format_update_register_context_from_target(
                parser, vtable, element->field_index, attr));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_TYPE: {
        // Type of an operand — parse and verify (or just consume).
        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_TYPES: {
        // Variadic operand types. Parse types separated by commas.
        bool first = true;
        for (;;) {
          loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
          if (peek.kind == LOOM_TOKEN_EOF || peek.kind == LOOM_TOKEN_RPAREN ||
              peek.kind == LOOM_TOKEN_RBRACE) {
            break;
          }
          if (!first) {
            if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                            LOOM_TOKEN_COMMA)) {
              break;
            }
          }
          loom_type_t type = {0};
          IREE_RETURN_IF_ERROR(
              loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
          first = false;
        }
        break;
      }

      case LOOM_FORMAT_KIND_RESULT_TYPE:
      case LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE: {
        IREE_RETURN_IF_ERROR(loom_parse_format_result_type(
            parser, vtable, op_name_token, element, parsed,
            is_symbol_definition));
        break;
      }

      case LOOM_FORMAT_KIND_RESULT_TYPE_LIST: {
        IREE_RETURN_IF_ERROR(loom_parse_format_result_type_list(
            parser, vtable, op_name_token, element, parsed,
            is_symbol_definition));
        break;
      }

      case LOOM_FORMAT_KIND_KEYWORD: {
        IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, element->data));
        break;
      }

      case LOOM_FORMAT_KIND_REGION: {
        loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
        const loom_region_descriptor_t* region_descriptor =
            loom_op_vtable_region_descriptor(vtable, element->field_index);
        if (!region_descriptor) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format REGION field_index %u out of range (op has %u regions)",
              element->field_index, vtable->region_count);
        }
        bool has_pending_func_args =
            parser->pending_func_args.count > pending_func_arg_start;
        bool is_func_like_body =
            vtable->func_like &&
            vtable->func_like->body_region_index == element->field_index;
        bool projects_func_args = iree_any_bit_set(
            region_descriptor->flags, LOOM_REGION_PROJECT_FUNC_ARGS);
        if (has_pending_func_args &&
            (projects_func_args || is_func_like_body)) {
          IREE_RETURN_IF_ERROR(loom_parse_format_project_func_args(
              parser, pending_func_arg_start,
              /*clone_values=*/!is_func_like_body));
        }
        if (has_pending_func_args && is_func_like_body) {
          *out_func_args_consumed_by_region = true;
        }
        loom_region_t* region = NULL;
        IREE_RETURN_IF_ERROR(loom_parse_region_with_syntax(
            parser, region_descriptor, (loom_region_syntax_t)element->data,
            &region));
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_region(
            parsed, &parser->parser_arena, element->field_index, region));
        IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
            parser, parsed, LOOM_LOCATION_FIELD_REGION, element->field_index,
            start_token));
        break;
      }

      case LOOM_FORMAT_KIND_INDEX_LIST: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_index_list(parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_BINDING_LIST: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_binding_list(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_BLOCK_ARGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_block_args(parser));
        break;
      }

      case LOOM_FORMAT_KIND_FUNC_ARGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_func_args(parser, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_PREDICATE_LIST: {
        loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
        loom_attribute_t attr = {0};
        IREE_RETURN_IF_ERROR(loom_parse_predicate_list(parser, &attr));
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
            parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE, element->field_index,
            start_token));
        break;
      }

      case LOOM_FORMAT_KIND_OPTIONAL_GROUP: {
        uint16_t skip_count = 0;
        IREE_RETURN_IF_ERROR(loom_parse_format_optional_group(
            parser, vtable, element, i, parsed, &skip_count));
        i += skip_count;
        break;
      }

      case LOOM_FORMAT_KIND_GLUE:
        // No-op for the parser. Spacing is irrelevant when reading.
        break;

      case LOOM_FORMAT_KIND_FLAGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_flags(parser, vtable, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_OP_REF: {
        IREE_RETURN_IF_ERROR(loom_parse_format_op_ref(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_DESCRIPTOR_REF: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_descriptor_ref(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_STABLE_KEY_REF: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_stable_key_ref(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_TEMPLATE_PARAM: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_template_param(parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_template_param_flags(
            parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_ATTR_DICT: {
        if (iree_any_bit_set(element->data,
                             LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS)) {
          IREE_RETURN_IF_ERROR(
              loom_parse_format_inline_attr_dict(parser, vtable, parsed));
          IREE_RETURN_IF_ERROR(loom_parse_format_apply_elided_attr_defaults(
              parser, vtable, element, parsed));
        } else {
          loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
          loom_attribute_t attr = {0};
          IREE_RETURN_IF_ERROR(loom_parse_attr_dict(parser, &attr));
          IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
              parsed, &parser->parser_arena, element->field_index, attr));
          if (!loom_attr_is_absent(attr)) {
            IREE_RETURN_IF_ERROR(loom_parse_format_add_field_span(
                parser, parsed, LOOM_LOCATION_FIELD_ATTRIBUTE,
                element->field_index, start_token));
          }
        }
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_DICT: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_operand_dict(parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_ATTR_TABLE: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_attr_table(parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_REGION_TABLE: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_region_table(parser, vtable, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_SCOPE: {
        loom_parser_definition_scope_flags_t definition_scope_flags = 0;
        if (vtable->symbol_def &&
            loom_symbol_definition_implements(vtable->symbol_def,
                                              LOOM_SYMBOL_INTERFACE_GLOBAL)) {
          definition_scope_flags |=
              LOOM_PARSER_DEFINITION_SCOPE_FLAG_RESOLVE_PLACEHOLDERS_FROM_USE;
        }
        IREE_RETURN_IF_ERROR(loom_parser_definition_scope_push(
            parser, (uint16_t)(i + element->data), definition_scope_flags));
        break;
      }
    }
    // Bail out of the format walk if any element emitted a parse error.
    // The diagnostic is already emitted — the caller (loom_parse_op)
    // handles recovery.
    if (parser->error_count > errors_before) return iree_ok_status();
    // Pop definition scope after the last child element has been processed.
    IREE_RETURN_IF_ERROR(loom_parser_definition_scope_pop_if_needed(parser, i));
    if (parser->error_count > errors_before) return iree_ok_status();
  }
  return iree_ok_status();
}
