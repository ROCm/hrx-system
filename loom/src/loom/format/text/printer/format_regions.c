// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format_regions.h"

#include <inttypes.h>

#include "loom/format/text/printer/low_asm.h"
#include "loom/format/text/printer/pipeline.h"
#include "loom/format/text/printer/regions.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Generated-format region payloads
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_braced_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  if (region) {
    ++ctx->indent;
    IREE_RETURN_IF_ERROR(loom_print_region_body(ctx, region, region_descriptor,
                                                entry_args_declared_by_parent));
    --ctx->indent;
  }
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

static char loom_print_region_syntax_first_char(loom_print_context_t* ctx,
                                                loom_region_syntax_t syntax) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT:
      return '{';
    case LOOM_REGION_SYNTAX_TEST_DO:
      return 'd';
    case LOOM_REGION_SYNTAX_LOW_ASM:
      return 'a';
    case LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL:
      return loom_print_low_asm_is_requested(ctx) ? 'a' : '{';
    case LOOM_REGION_SYNTAX_PIPELINE:
      return 'p';
    default:
      return '?';
  }
}

static bool loom_print_region_entry_args_declared_by_parent(
    const loom_op_vtable_t* vtable,
    const loom_region_descriptor_t* region_descriptor, uint8_t region_index,
    const loom_format_element_t* region_element) {
  if (vtable->func_like &&
      vtable->func_like->body_region_index == region_index) {
    return true;
  }
  if (region_descriptor && iree_any_bit_set(region_descriptor->flags,
                                            LOOM_REGION_PROJECT_FUNC_ARGS)) {
    return true;
  }

  bool pending_entry_args = false;
  for (uint16_t i = 0; i < vtable->format_element_count; ++i) {
    const loom_format_element_t* element = &vtable->format_elements[i];
    if (element == region_element) {
      return pending_entry_args;
    }

    switch ((loom_format_kind_t)element->kind) {
      case LOOM_FORMAT_KIND_OPERAND_REF: {
        if (element->field_index == 0xFF) {
          pending_entry_args = true;
        }
        break;
      }
      case LOOM_FORMAT_KIND_BINDING_LIST:
      case LOOM_FORMAT_KIND_FUNC_ARGS: {
        pending_entry_args = true;
        break;
      }
      case LOOM_FORMAT_KIND_BLOCK_ARGS: {
        if (element->field_index == region_index) {
          pending_entry_args = true;
        }
        break;
      }
      case LOOM_FORMAT_KIND_REGION: {
        pending_entry_args = false;
        break;
      }
      default:
        break;
    }
  }
  return false;
}

static iree_status_t loom_print_region_body_with_syntax(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    loom_region_syntax_t syntax, bool entry_args_declared_by_parent) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT: {
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    case LOOM_REGION_SYNTAX_TEST_DO: {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "do", false));
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    case LOOM_REGION_SYNTAX_LOW_ASM:
      return loom_print_low_asm_region(ctx, region, region_descriptor,
                                       entry_args_declared_by_parent);
    case LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL:
      if (loom_print_low_asm_is_requested(ctx)) {
        if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_REQUIRE_LOW_ASM)) {
          return loom_print_low_asm_region(ctx, region, region_descriptor,
                                           entry_args_declared_by_parent);
        }
        bool printed = false;
        IREE_RETURN_IF_ERROR(loom_print_low_asm_optional_region(
            ctx, region, region_descriptor, entry_args_declared_by_parent,
            &printed));
        if (printed) {
          return iree_ok_status();
        }
      }
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    case LOOM_REGION_SYNTAX_PIPELINE: {
      if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
        return loom_print_pipeline_skipped_region(ctx);
      }
      if (!entry_args_declared_by_parent &&
          loom_print_pipeline_region_is_friendly(ctx, region,
                                                 region_descriptor)) {
        return loom_print_pipeline_region(ctx, region, region_descriptor);
      }
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported region syntax %u", (uint32_t)syntax);
  }

  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
  } else {
    IREE_RETURN_IF_ERROR(loom_print_braced_region(
        ctx, region, region_descriptor, entry_args_declared_by_parent));
  }
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

iree_status_t loom_print_region_element(loom_print_context_t* ctx,
                                        const loom_op_t* op,
                                        const loom_op_vtable_t* vtable,
                                        const loom_format_element_t* element) {
  if (element->field_index >= op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION field_index %u out of range (op has %u regions)",
        element->field_index, op->region_count);
  }
  const loom_region_descriptor_t* region_descriptor =
      loom_op_vtable_region_descriptor(vtable, element->field_index);
  if (!region_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION field_index %u out of range (vtable has %u region "
        "descriptors)",
        element->field_index, vtable->region_count);
  }
  loom_region_t* region = loom_op_regions(op)[element->field_index];
  loom_region_syntax_t syntax = (loom_region_syntax_t)element->data;
  const bool region_args_declared_by_parent =
      loom_print_region_entry_args_declared_by_parent(
          vtable, region_descriptor, (uint8_t)element->field_index, element);
  iree_host_size_t region_start = loom_print_next_token_start_offset(
      ctx, /*glue=*/false, loom_print_region_syntax_first_char(ctx, syntax));
  IREE_RETURN_IF_ERROR(loom_print_region_body_with_syntax(
      ctx, region, region_descriptor, syntax, region_args_declared_by_parent));
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_REGION, element->field_index),
      region_start, ctx->stream->offset);
  return iree_ok_status();
}

iree_status_t loom_print_region_table(loom_print_context_t* ctx,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable,
                                      const loom_format_element_t* element) {
  uint8_t keys_attr_index =
      LOOM_FORMAT_REGION_TABLE_KEYS_ATTR_INDEX(element->data);
  uint8_t default_region_index =
      LOOM_FORMAT_REGION_TABLE_DEFAULT_REGION_INDEX(element->data);
  uint8_t case_region_index = (uint8_t)element->field_index;
  if (keys_attr_index >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE attr index %u out of range (op has %u "
        "attributes)",
        keys_attr_index, op->attribute_count);
  }
  loom_attribute_t keys_attr = loom_op_attrs(op)[keys_attr_index];
  if (keys_attr.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE attr index %u expected I64_ARRAY attr but found "
        "%d",
        keys_attr_index, (int)keys_attr.kind);
  }
  if (keys_attr.count > 0 && !keys_attr.i64_array) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "REGION_TABLE keys attr has count %u but NULL values", keys_attr.count);
  }
  if (default_region_index >= op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE default region index %u out of range (op has %u "
        "regions)",
        default_region_index, op->region_count);
  }
  if (case_region_index > op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE case region index %u out of range (op has %u "
        "regions)",
        case_region_index, op->region_count);
  }
  iree_host_size_t expected_region_count =
      (iree_host_size_t)case_region_index + keys_attr.count;
  if (expected_region_count != op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE has %u case keys but %u case regions",
        keys_attr.count, (uint8_t)(op->region_count - case_region_index));
  }

  const loom_region_descriptor_t* default_descriptor =
      loom_op_vtable_region_descriptor(vtable, default_region_index);
  const loom_region_descriptor_t* case_descriptor =
      loom_op_vtable_region_descriptor(vtable, case_region_index);
  if (!default_descriptor || !case_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE region descriptors are unavailable");
  }

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t table_start = ctx->stream->offset;
  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
    loom_print_did_write(ctx);
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, keys_attr_index),
        table_start, ctx->stream->offset);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  ++ctx->indent;
  loom_region_t** regions = loom_op_regions(op);
  for (uint16_t row = 0; row < keys_attr.count; ++row) {
    uint8_t region_index = (uint8_t)(case_region_index + row);
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    iree_host_size_t region_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        ctx->stream, "case %" PRId64 " ", keys_attr.i64_array[row]));
    IREE_RETURN_IF_ERROR(
        loom_print_braced_region(ctx, regions[region_index], case_descriptor,
                                 /*entry_args_declared_by_parent=*/false));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_REGION, region_index),
        region_start, ctx->stream->offset);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }

  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  iree_host_size_t default_start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, "default "));
  IREE_RETURN_IF_ERROR(loom_print_braced_region(
      ctx, regions[default_region_index], default_descriptor,
      /*entry_args_declared_by_parent=*/false));
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_REGION, default_region_index),
      default_start, ctx->stream->offset);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  --ctx->indent;
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, keys_attr_index),
      table_start, ctx->stream->offset);
  return iree_ok_status();
}
