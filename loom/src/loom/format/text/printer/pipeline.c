// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/pipeline.h"

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static bool loom_print_pipeline_is_ident_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == '$';
}

static bool loom_print_pipeline_is_ident_continue(char c, bool allow_dot) {
  return loom_print_pipeline_is_ident_start(c) || (c >= '0' && c <= '9') ||
         c == '-' || (allow_dot && c == '.');
}

static bool loom_print_pipeline_is_printable_name(iree_string_view_t name,
                                                  bool allow_dot) {
  if (name.size == 0 || !loom_print_pipeline_is_ident_start(name.data[0])) {
    return false;
  }
  for (iree_host_size_t i = 1; i < name.size; ++i) {
    if (!loom_print_pipeline_is_ident_continue(name.data[i], allow_dot)) {
      return false;
    }
  }
  return true;
}

static bool loom_print_pipeline_is_elidable_terminator(
    const loom_region_descriptor_t* region_descriptor, const loom_op_t* op) {
  return region_descriptor->implicit_terminator != LOOM_OP_KIND_UNKNOWN && op &&
         op->kind == region_descriptor->implicit_terminator &&
         op->operand_count == 0 && op->result_count == 0 &&
         op->region_count == 0 && op->successor_count == 0 &&
         op->tied_result_count == 0 && op->attribute_count == 0 &&
         op->instance_flags == 0;
}

static const loom_op_vtable_t* loom_print_pipeline_op_vtable(
    const loom_print_context_t* ctx, const loom_op_t* op,
    iree_string_view_t expected_name) {
  const loom_op_vtable_t* vtable = loom_op_vtable(ctx->module, op);
  if (!vtable ||
      !iree_string_view_equal(loom_op_vtable_name(vtable), expected_name)) {
    return NULL;
  }
  if (op->operand_count != 0 || op->result_count != 0 ||
      op->successor_count != 0 || op->tied_result_count != 0 ||
      op->instance_flags != 0 ||
      op->attribute_count != vtable->attribute_count ||
      op->region_count != vtable->region_count) {
    return NULL;
  }
  if (vtable->attribute_count > 0 && !vtable->attr_descriptors) {
    return NULL;
  }
  return vtable;
}

static const loom_attribute_t* loom_print_pipeline_find_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                               attr_name)) {
      *out_descriptor = descriptor;
      return &loom_op_const_attrs(op)[i];
    }
  }
  return NULL;
}

static bool loom_print_pipeline_string_attr(const loom_print_context_t* ctx,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable,
                                            iree_string_view_t attr_name,
                                            iree_string_view_t* out_value) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr || attr->kind != LOOM_ATTR_STRING ||
      attr->string_id >= ctx->module->strings.count) {
    return false;
  }
  *out_value = ctx->module->strings.entries[attr->string_id];
  return true;
}

static bool loom_print_pipeline_enum_attr(const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          iree_string_view_t attr_name,
                                          iree_string_view_t* out_value) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || attr->kind != LOOM_ATTR_ENUM || !descriptor ||
      descriptor->attr_kind != LOOM_ATTR_ENUM || !descriptor->enum_case_names ||
      attr->raw >= descriptor->enum_case_count) {
    return false;
  }
  *out_value =
      loom_bstring_view(descriptor->enum_case_names[(uint8_t)attr->raw]);
  return true;
}

static bool loom_print_pipeline_symbol_attr(const loom_print_context_t* ctx,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable,
                                            iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr || attr->kind != LOOM_ATTR_SYMBOL || attr->symbol.module_id != 0 ||
      attr->symbol.symbol_id >= ctx->module->symbols.count) {
    return false;
  }
  loom_string_id_t name_id =
      ctx->module->symbols.entries[attr->symbol.symbol_id].name_id;
  return name_id < ctx->module->strings.count &&
         loom_print_pipeline_is_printable_name(
             ctx->module->strings.entries[name_id], /*allow_dot=*/false);
}

static bool loom_print_pipeline_attr_value_is_printable(
    const loom_print_context_t* ctx, const loom_attribute_t* attr,
    uint8_t nesting_depth) {
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
      return true;
    case LOOM_ATTR_STRING:
      return attr->string_id < ctx->module->strings.count;
    case LOOM_ATTR_I64_ARRAY:
      return attr->count == 0 || attr->i64_array != NULL;
    case LOOM_ATTR_SYMBOL: {
      if (attr->symbol.module_id != 0 ||
          attr->symbol.symbol_id >= ctx->module->symbols.count) {
        return false;
      }
      loom_string_id_t name_id =
          ctx->module->symbols.entries[attr->symbol.symbol_id].name_id;
      return name_id < ctx->module->strings.count &&
             loom_print_pipeline_is_printable_name(
                 ctx->module->strings.entries[name_id], /*allow_dot=*/false);
    }
    case LOOM_ATTR_TYPE:
      return attr->type_id < ctx->module->types.count;
    case LOOM_ATTR_ENCODING:
      return attr->encoding_id > 0 &&
             attr->encoding_id <= ctx->module->encodings.count;
    case LOOM_ATTR_DICT: {
      if (nesting_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
        return false;
      }
      if (attr->count > 0 && !attr->dict_entries) {
        return false;
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        if (entry->name_id >= ctx->module->strings.count ||
            !loom_print_pipeline_is_printable_name(
                ctx->module->strings.entries[entry->name_id],
                /*allow_dot=*/false) ||
            !loom_print_pipeline_attr_value_is_printable(
                ctx, &entry->value, (uint8_t)(nesting_depth + 1))) {
          return false;
        }
      }
      return true;
    }
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_PREDICATE_LIST:
    case LOOM_ATTR_ANY:
    case LOOM_ATTR_COUNT_:
      return false;
  }
  return false;
}

static bool loom_print_pipeline_dict_attr(const loom_print_context_t* ctx,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          iree_string_view_t attr_name,
                                          loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || loom_attr_is_absent(*attr)) {
    return true;
  }
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_DICT ||
      attr->kind != LOOM_ATTR_DICT ||
      !loom_print_pipeline_attr_value_is_printable(ctx, attr,
                                                   /*nesting_depth=*/0)) {
    return false;
  }
  *out_attrs = loom_attr_as_dict(*attr);
  return true;
}

static bool loom_print_pipeline_optional_i64_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name, const loom_attribute_t** out_attr) {
  *out_attr = NULL;
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || loom_attr_is_absent(*attr)) {
    return true;
  }
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_I64 ||
      attr->kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_attr = attr;
  return true;
}

static const loom_region_descriptor_t*
loom_print_pipeline_body_region_descriptor(const loom_op_vtable_t* vtable) {
  return vtable->region_count == 1 ? loom_op_vtable_region_descriptor(vtable, 0)
                                   : NULL;
}

static bool loom_print_pipeline_statement_is_friendly(
    const loom_print_context_t* ctx, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  iree_string_view_t value = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.run"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("key"),
                                           &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/true) &&
           loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("options"),
                                         &attrs);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.for"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_enum_attr(op, vtable, IREE_SV("anchor"),
                                         &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/false) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.where"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_string_attr(ctx, op, vtable,
                                           IREE_SV("predicate"), &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/true) &&
           loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("attrs"),
                                         &attrs) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.repeat"));
  if (vtable) {
    const loom_attribute_t* count_attr = NULL;
    const loom_attribute_t* max_iterations_attr = NULL;
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_enum_attr(op, vtable, IREE_SV("mode"), &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/false) &&
           loom_print_pipeline_optional_i64_attr(op, vtable, IREE_SV("count"),
                                                 &count_attr) &&
           loom_print_pipeline_optional_i64_attr(
               op, vtable, IREE_SV("max_iterations"), &max_iterations_attr) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.call"));
  if (vtable) {
    return loom_print_pipeline_symbol_attr(ctx, op, vtable, IREE_SV("callee"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.fail"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("message"),
                                           &value);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.halt"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("message"),
                                           &value);
  }

  return false;
}

bool loom_print_pipeline_region_is_friendly(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_LOCATIONS) || !region ||
      !region_descriptor ||
      region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN ||
      region->block_count != 1 || region->flags != 0) {
    return false;
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  iree_host_size_t block_comment_count = 0;
  (void)loom_module_block_comments(ctx->module, block, &block_comment_count);
  if (block->arg_count != 0 || block->op_count == 0 ||
      block_comment_count != 0 || loom_print_block_has_label(ctx, block) ||
      !loom_print_pipeline_is_elidable_terminator(region_descriptor,
                                                  block->last_op)) {
    return false;
  }

  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == block->last_op) {
      break;
    }
    if (!loom_print_pipeline_statement_is_friendly(ctx, op)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_print_pipeline_attr_parens(
    loom_print_context_t* ctx, loom_named_attr_slice_t attrs) {
  if (attrs.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
  for (uint16_t i = 0; i < attrs.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    const loom_named_attr_t* entry = &attrs.entries[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, ctx->module->strings.entries[entry->name_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &entry->value, ctx->module, NULL));
  }
  return loom_output_stream_write_char(ctx->stream, ')');
}

static iree_status_t loom_print_pipeline_string_attr_value(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing pass pipeline string attribute");
  }
  return loom_print_attr(ctx->stream, attr, ctx->module, NULL);
}

static iree_status_t loom_print_pipeline_symbol_attr_value(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing pass pipeline symbol attribute");
  }
  return loom_print_attr(ctx->stream, attr, ctx->module, NULL);
}

static iree_status_t loom_print_pipeline_nested_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

static iree_status_t loom_print_pipeline_statement(loom_print_context_t* ctx,
                                                   const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  iree_string_view_t value = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.run"));
  if (vtable) {
    (void)loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("key"),
                                          &value);
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("options"),
                                        &attrs);
    return loom_print_pipeline_attr_parens(ctx, attrs);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.for"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    (void)loom_print_pipeline_enum_attr(op, vtable, IREE_SV("anchor"), &value);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "for "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.where"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    (void)loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("predicate"),
                                          &value);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "where "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("attrs"),
                                        &attrs);
    IREE_RETURN_IF_ERROR(loom_print_pipeline_attr_parens(ctx, attrs));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.repeat"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    const loom_attribute_t* count_attr = NULL;
    const loom_attribute_t* max_iterations_attr = NULL;
    (void)loom_print_pipeline_enum_attr(op, vtable, IREE_SV("mode"), &value);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "repeat "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_optional_i64_attr(op, vtable, IREE_SV("count"),
                                                &count_attr);
    (void)loom_print_pipeline_optional_i64_attr(
        op, vtable, IREE_SV("max_iterations"), &max_iterations_attr);
    if (count_attr || max_iterations_attr) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
      bool wrote_attr = false;
      if (count_attr) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, "count = "));
        IREE_RETURN_IF_ERROR(
            loom_print_attr(ctx->stream, count_attr, ctx->module, NULL));
        wrote_attr = true;
      }
      if (max_iterations_attr) {
        if (wrote_attr) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, "max_iterations = "));
        IREE_RETURN_IF_ERROR(loom_print_attr(ctx->stream, max_iterations_attr,
                                             ctx->module, NULL));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.call"));
  if (vtable) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "call "));
    return loom_print_pipeline_symbol_attr_value(ctx, op, vtable,
                                                 IREE_SV("callee"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.fail"));
  if (vtable) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "fail "));
    return loom_print_pipeline_string_attr_value(ctx, op, vtable,
                                                 IREE_SV("message"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.halt"));
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass pipeline statement");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "halt "));
  return loom_print_pipeline_string_attr_value(ctx, op, vtable,
                                               IREE_SV("message"));
}

static iree_status_t loom_print_pipeline_nested_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  ++ctx->indent;
  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == block->last_op &&
        loom_print_pipeline_is_elidable_terminator(region_descriptor, op)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, op));
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    IREE_RETURN_IF_ERROR(loom_print_pipeline_statement(ctx, op));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  --ctx->indent;
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  return loom_output_stream_write_char(ctx->stream, '}');
}

iree_status_t loom_print_pipeline_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "pipeline", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_pipeline_nested_region(ctx, region, region_descriptor));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

iree_status_t loom_print_pipeline_skipped_region(loom_print_context_t* ctx) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "pipeline", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}
