// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/low_asm.h"

//===----------------------------------------------------------------------===//
// Target-low structural asm parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_low_asm_expect_single_result(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t mnemonic_token) {
  if (result_names->count != 1) {
    return loom_parser_emit_low_asm_result_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/1,
        (uint32_t)result_names->count);
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_define_single_result(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_op_t* op) {
  if (op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural builder returned %u results",
                            op->result_count);
  }
  return loom_parser_define_value_name(parser, result_names->tokens[0],
                                       loom_op_results(op)[0]);
}

static iree_status_t loom_parse_low_asm_angle_key(loom_parser_t* parser,
                                                  loom_token_t mnemonic_token,
                                                  loom_token_t* out_key_token) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);
  loom_token_t key_token = loom_tokenizer_peek(&parser->tokenizer);
  if (key_token.kind != LOOM_TOKEN_STRING &&
      key_token.kind != LOOM_TOKEN_BARE_IDENT &&
      key_token.kind != LOOM_TOKEN_OP_NAME) {
    return loom_parser_emit_low_asm_error(
        parser, key_token, IREE_SV("expected structural intrinsic key"));
  }
  *out_key_token = loom_tokenizer_next(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  (void)mnemonic_token;
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_result_type(
    loom_parser_t* parser, loom_type_t* out_result_type) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  return loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, out_result_type);
}

static iree_status_t loom_parse_low_asm_arrow_result_type(
    loom_parser_t* parser, loom_type_t* out_result_type) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_ARROW, NULL);
  return loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, out_result_type);
}

static iree_status_t loom_parse_low_asm_validate_operand_type(
    loom_parser_t* parser, loom_token_t mnemonic_token, loom_value_id_t value,
    loom_type_t annotated_type) {
  loom_type_t actual_type = loom_module_value_type(parser->module, value);
  if (!loom_type_equal(actual_type, annotated_type)) {
    return loom_parser_emit_low_asm_error(
        parser, mnemonic_token,
        IREE_SV("operand type annotation does not match value type"));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_value_ref(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    loom_parsed_op_t* parsed_spans, uint16_t operand_index,
    loom_value_id_t* out_value) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    return loom_parser_emit_low_asm_operand_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/1, /*actual_count=*/0);
  }
  loom_token_t value_token = loom_tokenizer_next(&parser->tokenizer);
  uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(
      loom_parser_resolve_value(parser, value_token, out_value));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  return loom_parsed_op_add_field_span(
      parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
      operand_index, value_token, parser->tokenizer.consumed_end_line,
      parser->tokenizer.consumed_end_column);
}

static iree_status_t loom_parse_low_asm_value_list(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    loom_parsed_op_t* parsed_spans, loom_low_asm_value_list_t* values) {
  const uint32_t errors_before = parser->error_count;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    for (;;) {
      if (values->count > UINT16_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "low asm value list exceeds uint16_t range");
      }
      loom_value_id_t value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_parse_low_asm_value_ref(parser, mnemonic_token, parsed_spans,
                                       (uint16_t)values->count, &value));
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_low_asm_value_list_append(parser, values, value));
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_operand_type_list(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    const loom_low_asm_value_list_t* values) {
  const uint32_t errors_before = parser->error_count;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  for (iree_host_size_t i = 0; i < values->count; ++i) {
    if (i > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
        parser, mnemonic_token, values->values[i], type));
  }
  if (values->count == 0 &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    return loom_parser_emit_low_asm_operand_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/0, /*actual_count=*/1);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_structural_attr_dict(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    loom_token_t mnemonic_token, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_named_attr_slice_empty();
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return iree_ok_status();
  }

  loom_token_t open_brace_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, &open_brace_token);
  (void)open_brace_token;
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

    const loom_attr_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(
        parser->low_asm_environment.vtable->structural_attr_descriptor(
            parser->low_asm_environment.state, kind, key_token.text,
            &descriptor));
    if (descriptor == NULL) {
      return loom_parser_emit_low_asm_error(
          parser, key_token, IREE_SV("unexpected structural attribute"));
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &value));
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
    return iree_ok_status();
  }

  loom_parser_sort_attr_dict_entries(parser->module, stack_entries, count);
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &parser->parser_arena, count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < count; ++i) {
    attrs[i] = stack_entries[i].attr;
  }
  *out_attrs = loom_make_named_attr_slice(attrs, count);
  (void)mnemonic_token;
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_structural_location_and_build(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    iree_string_view_t key, int64_t offset, loom_type_t result_type,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_packet_location(
      parser, start_token, mnemonic_token, parsed_spans, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(parser->low_asm_environment.vtable->build_structural(
      parser->low_asm_environment.state, &parser->builder, kind, key, operands,
      operand_count, attrs, offset, result_type, location, &op));
  if (op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural builder returned no operation");
  }
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_define_single_result(parser, result_names, op));
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static iree_status_t loom_parse_low_asm_structural_resource_or_live_in(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_token_t key_token = loom_token_none();
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_angle_key(parser, mnemonic_token, &key_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE) {
    IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
        parser, kind, mnemonic_token, &attrs));
  } else {
    loom_attribute_t attr = {0};
    IREE_RETURN_IF_ERROR(loom_parse_attr_dict(parser, &attr));
    attrs = loom_attr_as_dict(attr);
  }
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, kind, result_names, start_token, mnemonic_token, comments,
      comment_count, NULL, 0, attrs, key_token.text, 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_concat(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_low_asm_value_list_t operands;
  loom_low_asm_value_list_initialize(&operands);
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_list(parser, mnemonic_token,
                                                     parsed_spans, &operands));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_operand_type_list(parser, mnemonic_token, &operands));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT, result_names, start_token,
      mnemonic_token, comments, comment_count, operands.values, operands.count,
      loom_named_attr_slice_empty(), iree_string_view_empty(), 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_slice(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &source));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACKET, NULL);
  loom_token_t offset_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &offset_token);
  int64_t offset = 0;
  if (!iree_string_view_atoi_int64(offset_token.text, &offset)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(offset_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                            IREE_ARRAYSIZE(params), offset_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t source_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &source_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, source, source_type));

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE, result_names, start_token,
      mnemonic_token, comments, comment_count, &source, 1,
      loom_named_attr_slice_empty(), iree_string_view_empty(), offset,
      result_type, parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_copy(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &source));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t source_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &source_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, source, source_type));

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY, result_names, start_token,
      mnemonic_token, comments, comment_count, &source, 1,
      loom_named_attr_slice_empty(), iree_string_view_empty(), 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_storage_reserve(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE, mnemonic_token,
      &attrs));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE, result_names,
      start_token, mnemonic_token, comments, comment_count, NULL, 0, attrs,
      iree_string_view_empty(), 0, result_type, parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_storage_address(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &storage));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t offset_attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS, mnemonic_token,
      &offset_attrs));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t storage_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &storage_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, storage, storage_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS, result_names,
      start_token, mnemonic_token, comments, comment_count, &storage, 1,
      offset_attrs, iree_string_view_empty(), 0, result_type, parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_storage_view(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &storage));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW, mnemonic_token,
      &attrs));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t storage_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &storage_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, storage, storage_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW, result_names,
      start_token, mnemonic_token, comments, comment_count, &storage, 1, attrs,
      iree_string_view_empty(), 0, result_type, parsed_spans);
}

bool loom_low_asm_structural_kind_from_token(
    loom_token_t token, loom_text_low_asm_structural_kind_t* out_kind) {
  *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_UNKNOWN;
  if (token.kind != LOOM_TOKEN_BARE_IDENT) {
    return false;
  }
  if (iree_string_view_equal(token.text, IREE_SV("resource"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("live_in"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("concat"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("slice"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("storage"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("storage_address"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("storage_view"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("copy"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY;
    return true;
  }
  return false;
}

iree_status_t loom_parse_low_asm_structural(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  switch (kind) {
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE:
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN:
      return loom_parse_low_asm_structural_resource_or_live_in(
          parser, kind, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT:
      return loom_parse_low_asm_structural_concat(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE:
      return loom_parse_low_asm_structural_slice(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE:
      return loom_parse_low_asm_structural_storage_reserve(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS:
      return loom_parse_low_asm_structural_storage_address(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW:
      return loom_parse_low_asm_structural_storage_view(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY:
      return loom_parse_low_asm_structural_copy(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    default:
      return loom_parser_emit_low_asm_error(
          parser, mnemonic_token, IREE_SV("unknown structural intrinsic"));
  }
}
