// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/format_signatures.h"

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/scope.h"
#include "loom/format/text/parser/types.h"

//===----------------------------------------------------------------------===//
// Generated-format signature payloads
//===----------------------------------------------------------------------===//

// Builds the parser-owned result overlay so result type annotations can
// reference co-results by name. The overlay is reset by the caller once the
// current RESULT_TYPE* element has been parsed.
static iree_status_t loom_parse_format_prepare_result_scope(
    loom_parser_t* parser, const loom_parsed_op_t* parsed) {
  iree_host_size_t named_result_count = 0;
  for (uint16_t result_index = 0; result_index < parsed->result_count;
       ++result_index) {
    loom_value_id_t value_id = parsed->result_ids[result_index];
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID &&
        name_id < parser->module->strings.count) {
      ++named_result_count;
    }
  }
  if (named_result_count == 0) {
    loom_parser_result_scope_reset(&parser->result_scope);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parser_result_scope_prepare(
      &parser->result_scope, named_result_count, &parser->parser_arena));
  for (uint16_t result_index = 0; result_index < parsed->result_count;
       ++result_index) {
    loom_value_id_t value_id = parsed->result_ids[result_index];
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id == LOOM_STRING_ID_INVALID ||
        name_id >= parser->module->strings.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_parser_result_scope_define(
        &parser->result_scope, name_id, value_id, /*out_duplicate=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_format_assign_lhs_result_type(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, loom_parsed_op_t* parsed, uint16_t result_index,
    loom_type_t type) {
  if (result_index >= parsed->result_count) {
    return loom_parser_emit_result_count_mismatch(parser, vtable, op_name_token,
                                                  (uint16_t)(result_index + 1),
                                                  parsed->result_count);
  }
  return loom_module_set_value_type(parser->module,
                                    parsed->result_ids[result_index], type);
}

static iree_status_t loom_parse_format_append_symbol_result(
    loom_parser_t* parser, loom_parsed_op_t* parsed, loom_type_t type,
    loom_token_t name_token) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_parser_define_value(parser, name_token, type, &value_id));
  if (parser->error_count > 0) {
    return iree_ok_status();
  }
  return loom_parsed_op_add_result(parsed, &parser->parser_arena, value_id,
                                   name_token);
}

static iree_status_t loom_parse_format_resolve_tied_result_operand(
    loom_parser_t* parser, const loom_parsed_op_t* parsed,
    loom_token_t ssa_token, uint16_t* out_operand_index) {
  loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_VALUE(parser, ssa_token, &operand_id);

  for (uint16_t operand_index = 0; operand_index < parsed->operand_count;
       ++operand_index) {
    if (parsed->operand_ids[operand_index] == operand_id) {
      *out_operand_index = operand_index;
      return iree_ok_status();
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(ssa_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_001, params,
                          IREE_ARRAYSIZE(params), ssa_token);
}

static iree_status_t loom_parse_format_resolve_symbol_tied_result_operand(
    loom_parser_t* parser, const loom_parsed_op_t* parsed,
    loom_token_t ssa_token, uint16_t* out_operand_index) {
  loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_VALUE(parser, ssa_token, &operand_id);

  for (uint16_t operand_index = 0; operand_index < parsed->operand_count;
       ++operand_index) {
    if (parsed->operand_ids[operand_index] == operand_id) {
      *out_operand_index = operand_index;
      return iree_ok_status();
    }
  }
  for (uint16_t arg_index = 0; arg_index < parser->pending_block_args.count;
       ++arg_index) {
    if (parser->pending_block_args.entries[arg_index].value_id == operand_id) {
      *out_operand_index = (uint16_t)(parsed->operand_count + arg_index);
      return iree_ok_status();
    }
  }
  uint16_t pending_operand_base =
      (uint16_t)(parsed->operand_count + parser->pending_block_args.count);
  for (uint16_t arg_index = 0; arg_index < parser->pending_func_args.count;
       ++arg_index) {
    if (parser->pending_func_args.entries[arg_index].value_id == operand_id) {
      *out_operand_index = (uint16_t)(pending_operand_base + arg_index);
      return iree_ok_status();
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(ssa_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_001, params,
                          IREE_ARRAYSIZE(params), ssa_token);
}

// Parses a body-op result type list: (type, %operand as type, ...).
// Result names are preallocated on the LHS and must match the number of
// parsed type entries.
static iree_status_t loom_parse_format_lhs_result_type_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
    }
  }

  loom_type_parse_mode_t type_mode = LOOM_TYPE_PARSE_BODY;
  uint16_t result_index = 0;
  while ((!use_parens ||
          !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (result_index > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      loom_token_t ssa_token = loom_tokenizer_peek(&parser->tokenizer);
      loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                             IREE_SV("as"))) {
        uint16_t operand_index = UINT16_MAX;
        IREE_RETURN_IF_ERROR(loom_parse_format_resolve_tied_result_operand(
            parser, parsed, ssa_token, &operand_index));
        if (parser->error_count > errors_before) {
          return iree_ok_status();
        }

        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));

        loom_value_id_t operand_id = parsed->operand_ids[operand_index];
        loom_type_t operand_type =
            loom_module_value_type(parser->module, operand_id);
        loom_tied_result_t tied = {
            .result_index = result_index,
            .operand_index = operand_index,
            .has_type_change = !loom_type_equal(operand_type, type),
        };
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_tied_result(
            parsed, &parser->parser_arena, tied));
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
            parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
            operand_index, ssa_token, ssa_token.line, ssa_token.end_column));

        IREE_RETURN_IF_ERROR(loom_parse_format_assign_lhs_result_type(
            parser, vtable, op_name_token, parsed, result_index, type));
      } else {
        return loom_parser_emit_unexpected_token(
            parser, ssa_token, IREE_SV("a result type or '%operand as type'"));
      }
    } else {
      loom_type_t type = {0};
      IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
      IREE_RETURN_IF_ERROR(loom_parse_format_assign_lhs_result_type(
          parser, vtable, op_name_token, parsed, result_index, type));
    }
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    ++result_index;
  }

  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
    }
  }
  if (result_index < parsed->result_count) {
    return loom_parser_emit_result_count_mismatch(
        parser, vtable, op_name_token, parsed->result_count, result_index);
  }
  return iree_ok_status();
}

// Parses a symbol-definition result type list:
//   (type, %name: type, %arg as type, ...)
//
// Result values are created as each type is parsed. Named result values are
// local to the surrounding Scope(...) and become visible only to subsequent
// result types / predicates in the same signature.
static iree_status_t loom_parse_format_symbol_result_type_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
    }
  }

  bool first = true;
  while ((!use_parens ||
          !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (!first) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      loom_token_t ssa_token = loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_ARG, &type));
        IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
            parser, parsed, type, ssa_token));
      } else if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                                    IREE_SV("as"))) {
        uint16_t operand_index = UINT16_MAX;
        IREE_RETURN_IF_ERROR(
            loom_parse_format_resolve_symbol_tied_result_operand(
                parser, parsed, ssa_token, &operand_index));
        if (parser->error_count > errors_before) {
          return iree_ok_status();
        }

        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));

        loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
        if (operand_index < parsed->operand_count) {
          operand_id = parsed->operand_ids[operand_index];
        } else {
          uint16_t pending_index =
              (uint16_t)(operand_index - parsed->operand_count);
          if (pending_index < parser->pending_block_args.count) {
            operand_id =
                parser->pending_block_args.entries[pending_index].value_id;
          } else {
            pending_index =
                (uint16_t)(pending_index - parser->pending_block_args.count);
            operand_id =
                parser->pending_func_args.entries[pending_index].value_id;
          }
        }
        loom_type_t operand_type =
            loom_module_value_type(parser->module, operand_id);
        loom_tied_result_t tied = {
            .result_index = parsed->result_count,
            .operand_index = operand_index,
            .has_type_change = !loom_type_equal(operand_type, type),
        };
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_tied_result(
            parsed, &parser->parser_arena, tied));
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
            parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
            operand_index, ssa_token, ssa_token.line, ssa_token.end_column));
        IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
            parser, parsed, type, loom_token_none()));
      } else {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek,
                                                 IREE_SV("':' or 'as'"));
      }
    } else {
      loom_type_t type = {0};
      loom_type_parse_mode_t type_mode = loom_parser_in_definition_scope(parser)
                                             ? LOOM_TYPE_PARSE_ARG
                                             : LOOM_TYPE_PARSE_BODY;
      IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
      IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
          parser, parsed, type, loom_token_none()));
    }
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    first = false;
  }

  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_parse_format_result_type(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool is_symbol_definition) {
  loom_type_parse_mode_t type_mode = loom_parser_in_definition_scope(parser)
                                         ? LOOM_TYPE_PARSE_ARG
                                         : LOOM_TYPE_PARSE_BODY;
  if (is_symbol_definition) {
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
    return loom_parse_format_append_symbol_result(parser, parsed, type,
                                                  loom_token_none());
  }

  IREE_RETURN_IF_ERROR(loom_parse_format_prepare_result_scope(parser, parsed));
  loom_type_t type = {0};
  iree_status_t status = loom_parse_type(parser, type_mode, &type);
  loom_parser_result_scope_reset(&parser->result_scope);
  IREE_RETURN_IF_ERROR(status);
  return loom_parse_format_assign_lhs_result_type(
      parser, vtable, op_name_token, parsed, element->field_index, type);
}

iree_status_t loom_parse_format_result_type_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool is_symbol_definition) {
  if (is_symbol_definition) {
    return loom_parse_format_symbol_result_type_list(parser, element, parsed);
  }

  IREE_RETURN_IF_ERROR(loom_parse_format_prepare_result_scope(parser, parsed));
  iree_status_t status = loom_parse_format_lhs_result_type_list(
      parser, vtable, op_name_token, element, parsed);
  loom_parser_result_scope_reset(&parser->result_scope);
  return status;
}

typedef struct loom_parsed_binding_t {
  // Region entry block argument name introduced by this binding.
  loom_token_t arg_token;
} loom_parsed_binding_t;

#define LOOM_PARSE_FORMAT_INLINE_BINDINGS 8

typedef struct loom_parsed_binding_list_t {
  // Parsed bindings in source order.
  loom_parsed_binding_t* bindings;
  // Number of parsed bindings.
  iree_host_size_t count;
  // Allocated binding capacity.
  iree_host_size_t capacity;
  // Inline storage for common small binding lists.
  loom_parsed_binding_t inline_bindings[LOOM_PARSE_FORMAT_INLINE_BINDINGS];
} loom_parsed_binding_list_t;

static void loom_parsed_binding_list_initialize(
    loom_parsed_binding_list_t* list) {
  list->bindings = list->inline_bindings;
  list->count = 0;
  list->capacity = IREE_ARRAYSIZE(list->inline_bindings);
}

static iree_status_t loom_parsed_binding_list_append(
    loom_parser_t* parser, loom_parsed_binding_list_t* list,
    loom_parsed_binding_t binding) {
  if (list->count >= list->capacity) {
    iree_host_size_t capacity = list->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, list->count, /*minimum_capacity=*/0,
        sizeof(*list->bindings), &capacity, (void**)&list->bindings));
    list->capacity = capacity;
  }
  list->bindings[list->count++] = binding;
  return iree_ok_status();
}

static loom_type_t loom_parse_format_binding_arg_type(
    const loom_format_element_t* element, loom_type_t type) {
  if (element->data != LOOM_BINDING_ELEMENT) {
    return type;
  }
  if (!loom_type_is_shaped(type)) {
    return type;
  }
  return loom_type_scalar(loom_type_element_type(type));
}

static iree_status_t loom_parse_format_define_binding_arg(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_binding_t binding, loom_type_t type) {
  loom_type_t arg_type = loom_parse_format_binding_arg_type(element, type);
  loom_value_id_t arg_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_define_value(parser->module, arg_type, &arg_value_id));
  return loom_parser_add_pending_block_arg(parser, arg_value_id,
                                           binding.arg_token);
}

static iree_status_t loom_parse_format_binding_entry(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, loom_parsed_binding_t* out_binding) {
  loom_token_t arg_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &arg_token);

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'='"));
  }

  loom_token_t operand_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &operand_token);
  loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_VALUE(parser, operand_token, &operand_id);

  uint16_t operand_index = parsed->operand_count;
  if (parsed->operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_segmented_operand(
        parsed, &parser->parser_arena, element->field_index, operand_id,
        &operand_index));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_parsed_op_add_operand(parsed, &parser->parser_arena, operand_id));
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
      parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND, operand_index,
      operand_token, operand_token.line, operand_token.end_column));

  *out_binding = (loom_parsed_binding_t){
      .arg_token = arg_token,
  };
  return iree_ok_status();
}

static iree_status_t loom_parse_format_grouped_binding_types(
    loom_parser_t* parser, const loom_format_element_t* element,
    const loom_parsed_binding_list_t* bindings) {
  for (iree_host_size_t i = 0; i < bindings->count; ++i) {
    if (i > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
    IREE_RETURN_IF_ERROR(loom_parse_format_define_binding_arg(
        parser, element, bindings->bindings[i], type));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_format_grouped_binding_list_tail(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, loom_parsed_binding_t first_binding) {
  loom_parsed_binding_list_t bindings;
  loom_parsed_binding_list_initialize(&bindings);
  IREE_RETURN_IF_ERROR(
      loom_parsed_binding_list_append(parser, &bindings, first_binding));

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parsed_binding_t binding;
    IREE_RETURN_IF_ERROR(
        loom_parse_format_binding_entry(parser, element, parsed, &binding));
    IREE_RETURN_IF_ERROR(
        loom_parsed_binding_list_append(parser, &bindings, binding));

    if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
      return loom_parse_format_grouped_binding_types(parser, element,
                                                     &bindings);
    }
    if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      continue;
    }
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek,
                                             IREE_SV("':' or ','"));
  }

  loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
  return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("':'"));
}

iree_status_t loom_parse_format_binding_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if ((parsed->operand_segment_count > 0 &&
         element->field_index < parsed->operand_segment_count &&
         parsed->operand_segment_counts[element->field_index] > 0) ||
        (parsed->operand_segment_count == 0 &&
         parsed->operand_count > element->field_index)) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    loom_parsed_binding_t binding;
    IREE_RETURN_IF_ERROR(
        loom_parse_format_binding_entry(parser, element, parsed, &binding));

    if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      IREE_RETURN_IF_ERROR(loom_parse_format_grouped_binding_list_tail(
          parser, element, parsed, binding));
      break;
    }
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("':'"));
    }

    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
    IREE_RETURN_IF_ERROR(
        loom_parse_format_define_binding_arg(parser, element, binding, type));
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  return iree_ok_status();
}

iree_status_t loom_parse_format_block_args(loom_parser_t* parser) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  iree_host_size_t parsed_arg_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (parsed_arg_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    loom_token_t arg_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &arg_token);
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);

    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));

    loom_value_id_t arg_value_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(parser->module, type, &arg_value_id));
    IREE_RETURN_IF_ERROR(
        loom_parser_add_pending_block_arg(parser, arg_value_id, arg_token));
    ++parsed_arg_count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  return iree_ok_status();
}

iree_status_t loom_parse_format_func_args(loom_parser_t* parser,
                                          loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  iree_host_size_t parsed_arg_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (parsed_arg_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    loom_token_t name_token = loom_token_none();
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      name_token = loom_tokenizer_next(&parser->tokenizer);
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("':'"));
      }
    }

    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_ARG, &type));

    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_parser_define_value(parser, name_token, type, &value_id));
    if (parser->error_count > 0) {
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(
        loom_parser_add_pending_func_arg(parser, value_id, name_token));
    ++parsed_arg_count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  return iree_ok_status();
}
