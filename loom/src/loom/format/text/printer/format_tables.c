// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format_tables.h"

#include <inttypes.h>

#include "loom/format/text/printer/atoms.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Generated-format table payloads
//===----------------------------------------------------------------------===//

static bool loom_print_format_element_covers_attr(
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

static bool loom_print_inline_attr_dict_attr_present(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element, uint16_t attr_index) {
  if (attr_index >= op->attribute_count) {
    return false;
  }
  const loom_attribute_t* attr = &loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(*attr)) {
    return false;
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[attr_index];
  if (loom_attr_descriptor_elides_value(descriptor, attr)) {
    return false;
  }
  const loom_format_element_t* elements = vtable->format_elements;
  for (uint16_t i = 0; i < vtable->format_element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    if (element == inline_element) {
      continue;
    }
    if (loom_print_format_element_covers_attr(element, attr_index)) {
      return false;
    }
  }
  return true;
}

static bool loom_print_find_next_inline_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element, bool has_previous_name,
    iree_string_view_t previous_name, uint16_t* out_attr_index,
    const loom_attr_descriptor_t** out_descriptor) {
  bool found = false;
  uint16_t best_index = 0;
  iree_string_view_t best_name = iree_string_view_empty();
  const loom_attr_descriptor_t* best_descriptor = NULL;

  uint16_t count = iree_min(vtable->attribute_count, op->attribute_count);
  for (uint16_t i = 0; i < count; ++i) {
    if (!loom_print_inline_attr_dict_attr_present(op, vtable, inline_element,
                                                  i)) {
      continue;
    }
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    iree_string_view_t name = loom_attr_descriptor_name(descriptor);
    if (has_previous_name &&
        iree_string_view_compare(name, previous_name) <= 0) {
      continue;
    }
    if (found && iree_string_view_compare(name, best_name) >= 0) {
      continue;
    }
    found = true;
    best_index = i;
    best_name = name;
    best_descriptor = descriptor;
  }

  if (!found) {
    return false;
  }
  *out_attr_index = best_index;
  *out_descriptor = best_descriptor;
  return true;
}

iree_status_t loom_print_inline_attr_dict(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element) {
  if (!vtable->attr_descriptors) {
    return iree_ok_status();
  }

  bool has_previous_name = false;
  iree_string_view_t previous_name = iree_string_view_empty();
  bool wrote_dict = false;
  for (;;) {
    uint16_t attr_index = 0;
    const loom_attr_descriptor_t* descriptor = NULL;
    if (!loom_print_find_next_inline_attr(op, vtable, inline_element,
                                          has_previous_name, previous_name,
                                          &attr_index, &descriptor)) {
      break;
    }

    if (!wrote_dict) {
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
      wrote_dict = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }

    iree_host_size_t attr_start = ctx->stream->offset;
    iree_string_view_t name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(loom_print_attr(
        ctx->stream, &loom_op_attrs(op)[attr_index], ctx->module, descriptor));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, attr_index),
        attr_start, ctx->stream->offset);

    previous_name = name;
    has_previous_name = true;
  }

  if (wrote_dict) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

iree_status_t loom_print_operand_dict(loom_print_context_t* ctx,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable,
                                      const loom_format_element_t* element) {
  loom_value_slice_t operand_span =
      loom_op_operand_field_span(vtable, op, element->field_index);
  const loom_value_id_t* operand_base = loom_op_const_operands(op);
  uint16_t start = operand_span.count > 0
                       ? (uint16_t)(operand_span.values - operand_base)
                       : 0;
  if (operand_span.count > 0 && (operand_span.values < operand_base ||
                                 operand_span.values + operand_span.count >
                                     operand_base + op->operand_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT field_index %u out of range (op has %u operands)",
        element->field_index, op->operand_count);
  }
  uint16_t operand_count = operand_span.count;
  if (element->data >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT attr index %u out of range (op has %u attributes)",
        element->data, op->attribute_count);
  }

  loom_attribute_t names_attr = loom_op_attrs(op)[element->data];
  if (loom_attr_is_absent(names_attr)) {
    if (operand_count == 0) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT has %u operands but names attr %u is absent",
        operand_count, element->data);
  }
  if (names_attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT attr index %u expected DICT attr but found %d",
        element->data, (int)names_attr.kind);
  }
  if (names_attr.count == 0) {
    if (operand_count == 0) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT has %u operands but names attr is empty",
        operand_count);
  }
  if (!names_attr.dict_entries) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OPERAND_DICT names attr has count %u but NULL "
                            "entries",
                            names_attr.count);
  }
  if (names_attr.count != operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format OPERAND_DICT has %u names but %u operands",
                            names_attr.count, operand_count);
  }

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t dict_start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
  for (uint16_t i = 0; i < names_attr.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    const loom_named_attr_t* entry = &names_attr.dict_entries[i];
    if (entry->name_id >= ctx->module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "OPERAND_DICT key string id %u out of range (module has %" PRIhsz
          " strings)",
          entry->name_id, ctx->module->strings.count);
    }
    if (i > 0) {
      int comparison = iree_string_view_compare(
          ctx->module->strings.entries[names_attr.dict_entries[i - 1].name_id],
          ctx->module->strings.entries[entry->name_id]);
      if (comparison >= 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "OPERAND_DICT names attr keys are not "
                                "canonical sorted unique keys");
      }
    }
    if (entry->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "OPERAND_DICT key attr expected I64 ordinal but found %d",
          (int)entry->value.kind);
    }
    int64_t ordinal = entry->value.i64;
    if (ordinal < 0 || ordinal >= operand_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "OPERAND_DICT key ordinal %" PRId64
                              " out of range for %u operands",
                              ordinal, operand_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (names_attr.dict_entries[j].value.kind == LOOM_ATTR_I64 &&
          names_attr.dict_entries[j].value.i64 == ordinal) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "OPERAND_DICT key ordinal %" PRId64 " is duplicated", ordinal);
      }
    }

    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, ctx->module->strings.entries[entry->name_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    uint16_t operand_index = (uint16_t)(start + (uint16_t)ordinal);
    iree_host_size_t value_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_print_value_ref(
        ctx->stream, ctx->module, loom_op_const_operands(op)[operand_index]));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index),
        value_start, ctx->stream->offset);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " : "));
    IREE_RETURN_IF_ERROR(
        loom_print_value_type(ctx, loom_op_const_operands(op)[operand_index]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
      dict_start, ctx->stream->offset);
  return iree_ok_status();
}

static iree_status_t loom_print_attr_table_row_values(loom_print_context_t* ctx,
                                                      const loom_op_t* op,
                                                      loom_value_slice_t values,
                                                      uint16_t row_index,
                                                      uint16_t row_width) {
  const loom_value_id_t* operand_base = loom_op_const_operands(op);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
  for (uint16_t column = 0; column < row_width; ++column) {
    if (column > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    uint16_t operand_index = (uint16_t)(row_index * row_width + column);
    const loom_value_id_t* operand = &values.values[operand_index];
    uint16_t flat_operand_index = (uint16_t)(operand - operand_base);
    iree_host_size_t value_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(
        loom_print_value_ref(ctx->stream, ctx->module, *operand));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, flat_operand_index),
        value_start, ctx->stream->offset);
  }
  return loom_output_stream_write_char(ctx->stream, ')');
}

iree_status_t loom_print_attr_table(loom_print_context_t* ctx,
                                    const loom_op_t* op,
                                    const loom_op_vtable_t* vtable,
                                    const loom_format_element_t* element) {
  loom_value_slice_t operand_span =
      loom_op_operand_field_span(vtable, op, element->field_index);
  const loom_value_id_t* operand_base = loom_op_const_operands(op);
  if (operand_span.count > 0 && (operand_span.values < operand_base ||
                                 operand_span.values + operand_span.count >
                                     operand_base + op->operand_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE field_index %u out of range (op has %u operands)",
        element->field_index, op->operand_count);
  }
  if (element->data >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE attr index %u out of range (op has %u attributes)",
        element->data, op->attribute_count);
  }

  loom_attribute_t keys_attr = loom_op_attrs(op)[element->data];
  if (keys_attr.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE attr index %u expected I64_ARRAY attr but found %d",
        element->data, (int)keys_attr.kind);
  }
  if (keys_attr.count > 0 && !keys_attr.i64_array) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ATTR_TABLE keys attr has count %u but NULL values",
                            keys_attr.count);
  }
  uint16_t operand_count = operand_span.count;
  iree_host_size_t row_count = (iree_host_size_t)keys_attr.count + 1;
  if (operand_count % row_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format ATTR_TABLE has %u operands for %" PRIhsz
                            " rows",
                            operand_count, row_count);
  }
  uint16_t row_width = (uint16_t)(operand_count / row_count);

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t table_start = ctx->stream->offset;

  if (keys_attr.count == 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{}"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
    ++ctx->indent;
    for (uint16_t row = 0; row < keys_attr.count; ++row) {
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          ctx->stream, "%" PRId64 " = ", keys_attr.i64_array[row]));
      IREE_RETURN_IF_ERROR(loom_print_attr_table_row_values(
          ctx, op, operand_span, row, row_width));
      if (row + 1 < keys_attr.count) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ','));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
    }
    --ctx->indent;
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, " default"));
  uint16_t default_row = keys_attr.count;
  IREE_RETURN_IF_ERROR(loom_print_attr_table_row_values(
      ctx, op, operand_span, default_row, row_width));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
      table_start, ctx->stream->offset);
  return iree_ok_status();
}
