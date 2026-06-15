// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/pipeline.h"

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/ir/context.h"

//===----------------------------------------------------------------------===//
// Pass pipeline region parsing
//===----------------------------------------------------------------------===//

typedef struct loom_pipeline_attr_entry_t {
  loom_named_attr_t attr;
  loom_token_t key_token;
} loom_pipeline_attr_entry_t;

static iree_status_t loom_parse_pipeline_region_contents(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, bool* out_region_end_consumed);

static bool loom_pipeline_token_is_name(loom_token_t token) {
  return token.kind == LOOM_TOKEN_BARE_IDENT ||
         token.kind == LOOM_TOKEN_OP_NAME;
}

static bool loom_pipeline_token_is_keyword(loom_token_t token,
                                           iree_string_view_t keyword) {
  return token.kind == LOOM_TOKEN_BARE_IDENT &&
         iree_string_view_equal(token.text, keyword);
}

static iree_status_t loom_parse_pipeline_name(loom_parser_t* parser,
                                              iree_string_view_t expected,
                                              loom_token_t* out_token) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_pipeline_token_is_name(token)) {
    return loom_parser_emit_unexpected_token(parser, token, expected);
  }
  *out_token = loom_tokenizer_next(&parser->tokenizer);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_token_location(
    loom_parser_t* parser, loom_token_t token,
    loom_location_id_t* out_location) {
  *out_location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id == LOOM_SOURCE_ID_INVALID) return iree_ok_status();
  loom_location_entry_t entry = loom_location_file_range(
      parser->source_id, (uint16_t)token.line, (uint16_t)token.column,
      (uint16_t)token.line, (uint16_t)token.end_column);
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_pipeline_intern_token(loom_parser_t* parser,
                                                loom_token_t token,
                                                loom_string_id_t* out_id) {
  return loom_module_intern_string(parser->module, token.text, out_id);
}

static iree_status_t loom_pipeline_emit_duplicate_attr(
    loom_parser_t* parser, loom_token_t key_token,
    loom_token_t previous_key_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(key_token.text),
  };
  return loom_parser_emit_related(
      parser, LOOM_ERR_PARSE_020, params, IREE_ARRAYSIZE(params), key_token,
      IREE_SV("previously defined here"), previous_key_token);
}

static iree_status_t loom_parse_pipeline_attr_parens(
    loom_parser_t* parser, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    return iree_ok_status();
  }

  loom_pipeline_attr_entry_t stack_entries[16];
  uint16_t count = 0;
  const uint32_t errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
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
        return loom_pipeline_emit_duplicate_attr(parser, key_token,
                                                 stack_entries[i].key_token);
      }
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);
    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(
        loom_parse_generic_attr_value(parser, /*nesting_depth=*/1, &value));
    if (parser->error_count > errors_before) return iree_ok_status();

    stack_entries[count++] = (loom_pipeline_attr_entry_t){
        .attr =
            (loom_named_attr_t){
                .name_id = key_id,
                .value = value,
            },
        .key_token = key_token,
    };
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  if (count == 0) return iree_ok_status();

  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &parser->parser_arena, count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < count; ++i) {
    attrs[i] = stack_entries[i].attr;
  }
  *out_attrs = loom_make_named_attr_slice(attrs, count);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_lookup_op(
    loom_parser_t* parser, iree_string_view_t op_name, loom_token_t token,
    loom_op_kind_t* out_kind, const loom_op_vtable_t** out_vtable) {
  const loom_op_vtable_t* vtable =
      loom_context_lookup_op_by_name(parser->context, op_name, out_kind);
  if (vtable) {
    *out_vtable = vtable;
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_006, params,
                          IREE_ARRAYSIZE(params), token);
}

static const loom_attr_descriptor_t* loom_pipeline_find_attr(
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name,
    uint8_t* out_attr_index) {
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                               attr_name)) {
      *out_attr_index = i;
      return descriptor;
    }
  }
  return NULL;
}

static iree_status_t loom_pipeline_set_attr(loom_op_t* op,
                                            const loom_op_vtable_t* vtable,
                                            iree_string_view_t attr_name,
                                            loom_attribute_t attr) {
  uint8_t attr_index = 0;
  const loom_attr_descriptor_t* descriptor =
      loom_pipeline_find_attr(vtable, attr_name, &attr_index);
  if (!descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "op '%.*s' has no attr '%.*s'",
        (int)loom_op_vtable_name(vtable).size, loom_op_vtable_name(vtable).data,
        (int)attr_name.size, attr_name.data);
  }
  loom_op_attrs(op)[attr_index] = attr;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_set_string_attr(
    loom_parser_t* parser, loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name, loom_token_t value_token) {
  loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_intern_token(parser, value_token, &string_id));
  return loom_pipeline_set_attr(op, vtable, attr_name,
                                loom_attr_string(string_id));
}

static iree_status_t loom_pipeline_set_dict_attr(
    loom_parser_t* parser, loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name, loom_named_attr_slice_t attrs) {
  if (attrs.count == 0) return iree_ok_status();
  loom_attribute_t canonical_attr = {0};
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      parser->module, attrs, &canonical_attr));
  return loom_pipeline_set_attr(op, vtable, attr_name, canonical_attr);
}

static iree_status_t loom_pipeline_set_symbol_attr(
    loom_op_t* op, const loom_op_vtable_t* vtable, iree_string_view_t attr_name,
    loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_SYMBOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline symbol attr expected SYMBOL payload");
  }
  return loom_pipeline_set_attr(op, vtable, attr_name, attr);
}

static iree_status_t loom_pipeline_set_enum_attr(loom_parser_t* parser,
                                                 loom_op_t* op,
                                                 const loom_op_vtable_t* vtable,
                                                 iree_string_view_t attr_name,
                                                 loom_token_t value_token) {
  uint8_t attr_index = 0;
  const loom_attr_descriptor_t* descriptor =
      loom_pipeline_find_attr(vtable, attr_name, &attr_index);
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "op '%.*s' has no enum attr '%.*s'",
        (int)loom_op_vtable_name(vtable).size, loom_op_vtable_name(vtable).data,
        (int)attr_name.size, attr_name.data);
  }
  for (uint8_t i = 0; i < descriptor->enum_case_count; ++i) {
    if (iree_string_view_equal(
            value_token.text,
            loom_bstring_view(descriptor->enum_case_names[i]))) {
      loom_op_attrs(op)[attr_index] = loom_attr_enum(i);
      return iree_ok_status();
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_string(value_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_017, params,
                          IREE_ARRAYSIZE(params), value_token);
}

static iree_status_t loom_pipeline_alloc_op(
    loom_parser_t* parser, iree_string_view_t op_name, loom_token_t token,
    loom_location_id_t location, loom_op_t** out_op,
    const loom_op_vtable_t** out_vtable) {
  *out_op = NULL;
  *out_vtable = NULL;

  loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
  const loom_op_vtable_t* vtable = NULL;
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_lookup_op(parser, op_name, token, &kind, &vtable));
  if (parser->error_count > errors_before) return iree_ok_status();
  if (loom_op_vtable_operand_descriptor_count(vtable) != 0 ||
      vtable->fixed_result_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline op '%.*s' unexpectedly has operands or "
                            "results",
                            (int)op_name.size, op_name.data);
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &parser->builder, kind, /*operand_count=*/0, /*result_count=*/0,
      vtable->region_count, /*tied_result_count=*/0, vtable->attribute_count,
      location, &op));
  for (uint8_t i = 0; i < vtable->region_count; ++i) {
    loom_region_t* region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_module_allocate_region(parser->module, 1, &region));
    loom_op_regions(op)[i] = region;
  }

  *out_op = op;
  *out_vtable = vtable;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_finalize_statement(
    loom_parser_t* parser, loom_op_t* op, const iree_string_view_t* comments,
    iree_host_size_t comment_count) {
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&parser->builder, op));
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static const loom_region_descriptor_t* loom_pipeline_body_region_descriptor(
    const loom_op_vtable_t* vtable) {
  return loom_op_vtable_region_descriptor(vtable, 0);
}

static iree_status_t loom_parse_pipeline_nested_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_op_t* parent_op, loom_region_t* region) {
  uint32_t errors_before = parser->error_count;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  loom_parser_scope_t* outer_scope = parser->scope;
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, outer_scope, &parser->scope));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&parser->builder, parent_op, region);
  bool region_end_consumed = false;
  iree_status_t status = loom_parse_pipeline_region_contents(
      parser, region_descriptor, region, &region_end_consumed);
  if (parser->error_count > errors_before && !region_end_consumed &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parser_sync_to_brace(parser);
  }
  loom_parser_pending_block_args_clear(&parser->pending_func_args);
  loom_parser_pending_block_args_clear(&parser->pending_block_args);
  loom_builder_restore(&parser->builder, saved_ip);
  loom_parser_scope_pop(parser);
  return status;
}

static iree_status_t loom_parse_pipeline_repeat_options(
    loom_parser_t* parser, loom_token_t start_token,
    loom_named_attr_slice_t attrs, loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name = parser->module->strings.entries[entry->name_id];
    if (entry->value.kind != LOOM_ATTR_I64) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("pass repeat option value")),
          loom_param_string(name),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                              IREE_ARRAYSIZE(params), start_token);
    }
    if (iree_string_view_equal(name, IREE_SV("count")) ||
        iree_string_view_equal(name, IREE_SV("max_iterations"))) {
      IREE_RETURN_IF_ERROR(
          loom_pipeline_set_attr(op, vtable, name, entry->value));
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("pass repeat option")),
        loom_param_string(name),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                            IREE_ARRAYSIZE(params), start_token);
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_pipeline_statement(loom_parser_t* parser) {
  const uint32_t errors_before = parser->error_count;
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                       &comment_count);

  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_token_location(parser, start_token, &location));

  if (loom_pipeline_token_is_keyword(start_token, IREE_SV("for"))) {
    loom_tokenizer_next(&parser->tokenizer);
    loom_token_t anchor_token = loom_token_none();
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_name(
        parser, IREE_SV("pass anchor"), &anchor_token));

    loom_op_t* op = NULL;
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(
        parser, IREE_SV("pass.for"), start_token, location, &op, &vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_pipeline_set_enum_attr(
        parser, op, vtable, IREE_SV("anchor"), anchor_token));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(
        loom_pipeline_finalize_statement(parser, op, comments, comment_count));
    return loom_parse_pipeline_nested_region(
        parser, loom_pipeline_body_region_descriptor(vtable), op,
        loom_op_regions(op)[0]);
  }

  if (loom_pipeline_token_is_keyword(start_token, IREE_SV("where"))) {
    loom_tokenizer_next(&parser->tokenizer);
    loom_token_t predicate_token = loom_token_none();
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_name(
        parser, IREE_SV("pass predicate"), &predicate_token));
    loom_named_attr_slice_t attrs = {0};
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_attr_parens(parser, &attrs));
    if (parser->error_count > errors_before) return iree_ok_status();

    loom_op_t* op = NULL;
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(
        parser, IREE_SV("pass.where"), start_token, location, &op, &vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_pipeline_set_string_attr(
        parser, op, vtable, IREE_SV("predicate"), predicate_token));
    IREE_RETURN_IF_ERROR(loom_pipeline_set_dict_attr(parser, op, vtable,
                                                     IREE_SV("attrs"), attrs));
    IREE_RETURN_IF_ERROR(
        loom_pipeline_finalize_statement(parser, op, comments, comment_count));
    return loom_parse_pipeline_nested_region(
        parser, loom_pipeline_body_region_descriptor(vtable), op,
        loom_op_regions(op)[0]);
  }

  if (loom_pipeline_token_is_keyword(start_token, IREE_SV("repeat"))) {
    loom_tokenizer_next(&parser->tokenizer);
    loom_token_t mode_token = loom_token_none();
    IREE_RETURN_IF_ERROR(
        loom_parse_pipeline_name(parser, IREE_SV("repeat mode"), &mode_token));
    loom_named_attr_slice_t attrs = {0};
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_attr_parens(parser, &attrs));
    if (parser->error_count > errors_before) return iree_ok_status();

    loom_op_t* op = NULL;
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(
        parser, IREE_SV("pass.repeat"), start_token, location, &op, &vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_pipeline_set_enum_attr(
        parser, op, vtable, IREE_SV("mode"), mode_token));
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_repeat_options(parser, start_token,
                                                            attrs, op, vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(
        loom_pipeline_finalize_statement(parser, op, comments, comment_count));
    return loom_parse_pipeline_nested_region(
        parser, loom_pipeline_body_region_descriptor(vtable), op,
        loom_op_regions(op)[0]);
  }

  if (loom_pipeline_token_is_keyword(start_token, IREE_SV("call"))) {
    loom_tokenizer_next(&parser->tokenizer);
    loom_attribute_t callee_attr = {0};
    IREE_RETURN_IF_ERROR(loom_parse_symbol_ref_attr(parser, &callee_attr));
    if (parser->error_count > errors_before) return iree_ok_status();

    loom_op_t* op = NULL;
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(
        parser, IREE_SV("pass.call"), start_token, location, &op, &vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_pipeline_set_symbol_attr(
        op, vtable, IREE_SV("callee"), callee_attr));
    return loom_pipeline_finalize_statement(parser, op, comments,
                                            comment_count);
  }

  if (loom_pipeline_token_is_keyword(start_token, IREE_SV("fail")) ||
      loom_pipeline_token_is_keyword(start_token, IREE_SV("halt"))) {
    loom_token_t keyword_token = loom_tokenizer_next(&parser->tokenizer);
    loom_token_t message_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &message_token);
    iree_string_view_t op_name =
        iree_string_view_equal(keyword_token.text, IREE_SV("fail"))
            ? IREE_SV("pass.fail")
            : IREE_SV("pass.halt");

    loom_op_t* op = NULL;
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(parser, op_name, start_token,
                                                location, &op, &vtable));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_pipeline_set_string_attr(
        parser, op, vtable, IREE_SV("message"), message_token));
    return loom_pipeline_finalize_statement(parser, op, comments,
                                            comment_count);
  }

  loom_token_t key_token = loom_token_none();
  IREE_RETURN_IF_ERROR(
      loom_parse_pipeline_name(parser, IREE_SV("pass name"), &key_token));
  loom_named_attr_slice_t options = {0};
  IREE_RETURN_IF_ERROR(loom_parse_pipeline_attr_parens(parser, &options));
  if (parser->error_count > errors_before) return iree_ok_status();

  loom_op_t* op = NULL;
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_alloc_op(
      parser, IREE_SV("pass.run"), start_token, location, &op, &vtable));
  if (parser->error_count > errors_before) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_set_string_attr(
      parser, op, vtable, IREE_SV("key"), key_token));
  IREE_RETURN_IF_ERROR(loom_pipeline_set_dict_attr(
      parser, op, vtable, IREE_SV("options"), options));
  return loom_pipeline_finalize_statement(parser, op, comments, comment_count);
}

static iree_status_t loom_parse_pipeline_region_contents(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, bool* out_region_end_consumed) {
  *out_region_end_consumed = false;

  IREE_RETURN_IF_ERROR(loom_parser_seed_region_entry_block(parser, region));

  loom_block_t* entry_block = loom_region_entry_block(region);
  loom_builder_set_block(&parser->builder, entry_block);
  const uint32_t block_errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) break;

    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
      loom_token_t label_token = loom_tokenizer_next(&parser->tokenizer);
      IREE_RETURN_IF_ERROR(loom_parser_emit_unexpected_token(
          parser, label_token, IREE_SV("pass pipeline statement")));
      loom_parser_sync_to_newline(parser);
      continue;
    }

    const uint32_t statement_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_pipeline_statement(parser));
    if (parser->error_count > statement_errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }

  if (parser->error_count == block_errors_before) {
    IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
        parser, region_descriptor, entry_block));
  }

  loom_tokenizer_discard_pending_comments(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  *out_region_end_consumed = true;
  return iree_ok_status();
}

iree_status_t loom_parse_pipeline_prefixed_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  uint32_t errors_before = parser->error_count;

  if (!loom_tokenizer_at_keyword(&parser->tokenizer, IREE_SV("pipeline"))) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek,
                                             IREE_SV("'pipeline'"));
  }
  loom_tokenizer_next(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, NULL);

  loom_parser_scope_t* outer_scope = parser->scope;
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, outer_scope, &parser->scope));

  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(parser->module, 1, &region));

  loom_builder_ip_t saved_ip = loom_builder_save(&parser->builder);
  loom_builder_set_block(&parser->builder, loom_region_entry_block(region));

  iree_host_size_t pending_successor_start =
      parser->pending_successor_refs.count;
  bool region_end_consumed = false;
  iree_status_t status = loom_parse_pipeline_region_contents(
      parser, region_descriptor, region, &region_end_consumed);
  if (parser->error_count > errors_before && !region_end_consumed &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parser_sync_to_brace(parser);
  }
  loom_parser_pending_block_args_clear(&parser->pending_func_args);
  loom_parser_pending_block_args_clear(&parser->pending_block_args);
  if (parser->error_count > errors_before || !iree_status_is_ok(status)) {
    parser->pending_successor_refs.count = pending_successor_start;
  }

  loom_builder_restore(&parser->builder, saved_ip);
  loom_parser_scope_pop(parser);
  IREE_RETURN_IF_ERROR(status);
  if (parser->error_count > errors_before) return iree_ok_status();

  *out_region = region;
  return iree_ok_status();
}
