// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format.h"

#include <inttypes.h>

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/format_regions.h"
#include "loom/format/text/printer/format_signatures.h"
#include "loom/format/text/printer/format_tables.h"
#include "loom/format/text/printer/regions.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Predicate printing
//===----------------------------------------------------------------------===//

// Prints a predicate argument based on its tag.
static iree_status_t loom_print_predicate_arg(loom_print_context_t* ctx,
                                              uint8_t tag, int64_t value) {
  switch (tag) {
    case LOOM_PRED_ARG_VALUE: {
      return loom_print_value_ref(ctx, (loom_value_id_t)value);
    }
    case LOOM_PRED_ARG_CONST:
      return loom_output_stream_write_format(ctx->stream, "%" PRId64, value);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown predicate arg tag %d", (int)tag);
  }
}

static iree_status_t loom_print_predicate(loom_print_context_t* ctx,
                                          const loom_predicate_t* predicate) {
  const char* predicate_name = loom_predicate_kind_name(predicate->kind);
  if (!predicate_name) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown predicate kind %d", (int)predicate->kind);
  }
  uint8_t expected_argument_count =
      loom_predicate_kind_argument_count(predicate->kind);
  if (predicate->arg_count != expected_argument_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate kind %s expects %u arguments, got %u",
                            predicate_name, expected_argument_count,
                            predicate->arg_count);
  }
  // Emit kind name and opening paren: "mul("
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, predicate_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
  // Emit arguments separated by ", ".
  for (uint8_t j = 0; j < predicate->arg_count; ++j) {
    if (j > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_print_predicate_arg(ctx, predicate->arg_tags[j],
                                                  predicate->args[j]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

// Prints a predicate list in the format: [pred(%name, 16), lt(%K, 1024)]
static iree_status_t loom_print_predicate_list(
    loom_print_context_t* ctx, const loom_predicate_t* predicates,
    uint16_t count) {
  if (count > 0 && !predicates) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate list has count %u but NULL predicates",
                            count);
  }
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "[", false));
  for (uint16_t i = 0; i < count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_predicate(ctx, &predicates[i]));
  }
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "]", false));
  return iree_ok_status();
}

static bool loom_print_optional_attr_present(const loom_op_t* op,
                                             uint16_t attr_index) {
  if (attr_index >= op->attribute_count) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static bool loom_print_attr_is_optional(const loom_op_vtable_t* vtable,
                                        uint16_t attr_index) {
  return vtable->attr_descriptors && attr_index < vtable->attribute_count &&
         iree_any_bit_set(vtable->attr_descriptors[attr_index].flags,
                          LOOM_ATTR_OPTIONAL);
}

static iree_status_t loom_print_bind_function_low_repr(
    loom_print_context_t* ctx, const loom_op_vtable_t* vtable,
    uint16_t attr_index, loom_attribute_t attr) {
  if (!vtable->func_like ||
      vtable->func_like->repr_contract_attr_index == LOOM_ATTR_INDEX_NONE ||
      vtable->func_like->repr_contract_attr_index != attr_index) {
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= ctx->module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function representation contract must be a valid string key");
  }
  ctx->low_repr = (loom_text_low_repr_context_t){
      .contract_key = ctx->module->strings.entries[attr.string_id],
  };
  if (!loom_text_low_asm_environment_supports_printing(
          &ctx->low_asm_environment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "printing a Low function requires a representation environment");
  }
  ctx->low_repr.descriptor_set = loom_low_repr_lookup_descriptor_set(
      &ctx->low_asm_environment.low_repr, ctx->low_repr.contract_key);
  if (!ctx->low_repr.descriptor_set) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Low representation contract '%.*s' was not found",
                            (int)ctx->low_repr.contract_key.size,
                            ctx->low_repr.contract_key.data);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Format element walk.
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_attr_with_field(
    loom_print_context_t* ctx, const loom_attribute_t* attr,
    const loom_attr_descriptor_t* descriptor,
    loom_print_field_ref_t field_ref) {
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(loom_print_attr(ctx, attr, descriptor));
  iree_host_size_t end = ctx->stream->offset;
  loom_print_did_write(ctx);
  loom_print_report_field(ctx, field_ref, start, end);
  return iree_ok_status();
}

static iree_status_t loom_print_instance_flag_list(
    loom_print_context_t* ctx, const loom_op_vtable_t* vtable, uint8_t flags) {
  bool first = true;
  for (uint8_t bit = 0; bit < vtable->instance_flags_case_count; ++bit) {
    if (flags & (1u << bit)) {
      if (!first) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '|'));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          ctx->stream,
          loom_bstring_view(vtable->instance_flags_case_names[bit])));
      first = false;
    }
  }
  return iree_ok_status();
}

static uint16_t loom_print_operand_flat_index(const loom_op_t* op,
                                              const loom_value_id_t* operand) {
  return (uint16_t)(operand - loom_op_const_operands(op));
}

iree_status_t loom_print_format_elements(loom_print_context_t* ctx,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  const loom_format_element_t* elements = vtable->format_elements;
  uint16_t element_count = vtable->format_element_count;

  for (uint16_t i = 0; i < element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    switch (element->kind) {
      case LOOM_FORMAT_KIND_OPERAND_REF: {
        loom_value_id_t value_id = 0;
        if (element->field_index == 0xFF) {
          loom_region_t** regions = loom_op_regions(op);
          const loom_block_t* entry_block =
              (op->region_count > 0 && regions[0] &&
               regions[0]->block_count > 0)
                  ? loom_region_const_entry_block(regions[0])
                  : NULL;
          value_id = (op->region_count > 0 && regions[0] &&
                      regions[0]->block_count > 0 && entry_block &&
                      entry_block->arg_count > 0)
                         ? loom_block_arg_id(entry_block, 0)
                         : LOOM_VALUE_ID_INVALID;
        } else {
          loom_value_slice_t span =
              loom_op_operand_field_span(vtable, op, element->field_index);
          if (span.count == 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "format OPERAND_REF field_index %u out of range (op has %u "
                "operands)",
                element->field_index, op->operand_count);
          }
          value_id = span.values[0];
        }
        uint16_t operand_index = element->field_index;
        if (value_id != LOOM_VALUE_ID_INVALID) {
          loom_value_slice_t span =
              loom_op_operand_field_span(vtable, op, element->field_index);
          if (span.count > 0) {
            operand_index = loom_print_operand_flat_index(op, span.values);
          }
        }
        IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
            ctx, value_id,
            loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_REFS: {
        loom_value_slice_t operands =
            loom_op_operand_field_span(vtable, op, element->field_index);
        for (uint16_t j = 0; j < operands.count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          uint16_t operand_index =
              loom_print_operand_flat_index(op, &operands.values[j]);
          IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
              ctx, operands.values[j],
              loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
        }
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_TYPED_REFS: {
        loom_value_slice_t operands =
            loom_op_operand_field_span(vtable, op, element->field_index);
        for (uint16_t j = 0; j < operands.count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          uint16_t operand_index =
              loom_print_operand_flat_index(op, &operands.values[j]);
          IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
              ctx, operands.values[j],
              loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, operands.values[j]));
          loom_print_did_write(ctx);
        }
        break;
      }
      case LOOM_FORMAT_KIND_SUCCESSOR_REF: {
        IREE_RETURN_IF_ERROR(
            loom_print_successor_ref(ctx, op, element->field_index));
        break;
      }
      case LOOM_FORMAT_KIND_SUCCESSOR_REFS: {
        if (element->field_index > op->successor_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format SUCCESSOR_REFS field_index %u out of range (op has %u "
              "successors)",
              element->field_index, op->successor_count);
        }
        for (uint16_t successor_index = element->field_index;
             successor_index < op->successor_count; ++successor_index) {
          if (successor_index > element->field_index) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          IREE_RETURN_IF_ERROR(
              loom_print_successor_ref(ctx, op, successor_index));
        }
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_VALUE: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_VALUE field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &loom_op_attrs(op)[element->field_index], descriptor,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_SYMBOL_REF: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format SYMBOL_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &loom_op_attrs(op)[element->field_index], NULL,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_TYPE: {
        loom_value_slice_t span =
            loom_op_operand_field_span(vtable, op, element->field_index);
        if (span.count == 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format OPERAND_TYPE field_index %u out of range (op has %u "
              "operands)",
              element->field_index, op->operand_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, span.values[0]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE: {
        if (element->field_index >= op->result_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format RESULT_TYPE field_index %u out of range (op has %u "
              "results)",
              element->field_index, op->result_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_result_value_type(
            ctx, loom_op_const_results(op)[element->field_index]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_TYPES: {
        loom_value_slice_t operands =
            loom_op_operand_field_span(vtable, op, element->field_index);
        for (uint16_t j = 0; j < operands.count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, operands.values[j]));
          loom_print_did_write(ctx);
        }
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE: {
        if (element->field_index >= op->result_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format RESULT_TYPE_SINGLE field_index %u out of range (op has "
              "%u results)",
              element->field_index, op->result_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_result_value_type(
            ctx, loom_op_const_results(op)[element->field_index]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE_LIST: {
        IREE_RETURN_IF_ERROR(
            loom_print_result_type_list(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_KEYWORD: {
        if (element->data >= LOOM_KW_COUNT_) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format KEYWORD data %u out of range (max %u)", element->data,
              (uint16_t)LOOM_KW_COUNT_);
        }
        loom_bstring_t keyword_bstring =
            loom_keyword_bstring((loom_keyword_id_t)element->data);
        IREE_RETURN_IF_ERROR(
            loom_print_emit(ctx, loom_bstring_view(keyword_bstring), false));
        break;
      }
      case LOOM_FORMAT_KIND_REGION: {
        IREE_RETURN_IF_ERROR(
            loom_print_region_element(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_INDEX_LIST: {
        uint16_t static_attr_index =
            LOOM_FORMAT_INDEX_LIST_STATIC_ATTR_INDEX(element->data);
        if (static_attr_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format INDEX_LIST attr index %u out of range (op has %u "
              "attributes)",
              static_attr_index, op->attribute_count);
        }
        loom_attribute_t static_attr = loom_op_attrs(op)[static_attr_index];
        loom_value_slice_t dynamic_operands =
            loom_op_operand_field_span(vtable, op, element->field_index);
        uint16_t dynamic_index = 0;
        bool glue =
            i > 0 && LOOM_FORMAT_INDEX_LIST_HAS_LEADING_GLUE(element->data);
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "[", glue));
        for (uint16_t j = 0; j < static_attr.count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          int64_t static_value = static_attr.i64_array[j];
          if (static_value == INT64_MIN) {
            uint16_t operand_index = dynamic_index++;
            if (operand_index >= dynamic_operands.count) {
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "format INDEX_LIST dynamic operand %u out of range (op has "
                  "%u operands)",
                  operand_index, dynamic_operands.count);
            }
            const loom_value_id_t* operand =
                &dynamic_operands.values[operand_index];
            uint16_t flat_operand_index =
                loom_print_operand_flat_index(op, operand);
            IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
                ctx, *operand,
                loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND,
                                     flat_operand_index)));
          } else {
            IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
            IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
                ctx->stream, "%" PRId64, static_value));
            loom_print_did_write(ctx);
          }
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "]", false));
        break;
      }
      case LOOM_FORMAT_KIND_BINDING_LIST: {
        IREE_RETURN_IF_ERROR(loom_print_binding_list(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_BLOCK_ARGS: {
        IREE_RETURN_IF_ERROR(loom_print_block_args(ctx, op, element));
        break;
      }
      case LOOM_FORMAT_KIND_FUNC_ARGS: {
        IREE_RETURN_IF_ERROR(loom_print_func_args(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_DICT: {
        if (iree_any_bit_set(element->data,
                             LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS)) {
          IREE_RETURN_IF_ERROR(
              loom_print_inline_attr_dict(ctx, op, vtable, element));
          break;
        }
        // AttrDict reads a LOOM_ATTR_DICT attribute at field_index and
        // emits its entries as {key = value, key = value, ...}.
        bool optional =
            loom_print_attr_is_optional(vtable, element->field_index);
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_DICT field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t dict_attr = loom_op_attrs(op)[element->field_index];
        if (loom_attr_is_absent(dict_attr)) {
          if (optional) {
            break;
          }
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "format ATTR_DICT field_index %u is absent",
                                  element->field_index);
        }
        if (dict_attr.kind != LOOM_ATTR_DICT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_DICT field_index %u expected DICT attr but found %d",
              element->field_index, (int)dict_attr.kind);
        }
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &dict_attr, NULL,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_DICT: {
        IREE_RETURN_IF_ERROR(loom_print_operand_dict(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_TABLE: {
        IREE_RETURN_IF_ERROR(loom_print_attr_table(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_ALIGNED_REFS: {
        IREE_RETURN_IF_ERROR(loom_print_aligned_refs(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_REGION_TABLE: {
        IREE_RETURN_IF_ERROR(loom_print_region_table(ctx, op, vtable, element));
        break;
      }
      case LOOM_FORMAT_KIND_PREDICATE_LIST: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format PREDICATE_LIST field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_PREDICATE_LIST) {
          if (attr.count > 0 && !attr.predicate_list) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "PREDICATE_LIST attr has count %u but NULL predicates",
                attr.count);
          }
          iree_host_size_t predicate_start =
              loom_print_next_token_start_offset(ctx, false, '[');
          IREE_RETURN_IF_ERROR(
              loom_print_predicate_list(ctx, attr.predicate_list, attr.count));
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              predicate_start, ctx->stream->offset);
        }
        break;
      }
      case LOOM_FORMAT_KIND_FLAGS: {
        // Per-instance flags in angle brackets, glued to the op name:
        // scalar.addi<nsw|nuw>. Walks set bits in instance_flags and
        // emits keywords from the vtable's instance_flags_case_names.
        uint8_t flags = op->instance_flags;
        if (flags != 0 && vtable->instance_flags_case_names != NULL) {
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          IREE_RETURN_IF_ERROR(
              loom_print_instance_flag_list(ctx, vtable, flags));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
        }
        break;
      }
      case LOOM_FORMAT_KIND_KEY_REF: {
        // Bare symbolic key in angle brackets, glued to the preceding token:
        // template.def<@tile.contract>. The field_index references a string
        // attribute holding the canonical key spelling.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format KEY_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_STRING &&
            attr.string_id != LOOM_STRING_ID_INVALID &&
            attr.string_id < ctx->module->strings.count) {
          iree_string_view_t key = ctx->module->strings.entries[attr.string_id];
          iree_host_size_t key_ref_start = ctx->stream->offset;
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          IREE_RETURN_IF_ERROR(loom_print_emit(ctx, key, true));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              key_ref_start, ctx->stream->offset);
        }
        IREE_RETURN_IF_ERROR(loom_print_bind_function_low_repr(
            ctx, vtable, element->field_index, attr));
        break;
      }
      case LOOM_FORMAT_KIND_STABLE_KEY_REF: {
        // Stable-key reference in angle brackets, glued to the op name. The
        // field_index references the spelling and data references its derived
        // numeric identity.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format STABLE_KEY_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        if (element->data >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format key-ref data field_index %u out of range "
              "(op has %u attributes)",
              element->data, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_STRING &&
            attr.string_id != LOOM_STRING_ID_INVALID &&
            attr.string_id < ctx->module->strings.count) {
          iree_string_view_t key = ctx->module->strings.entries[attr.string_id];
          iree_host_size_t ref_start = ctx->stream->offset;
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          IREE_RETURN_IF_ERROR(loom_print_emit(ctx, key, true));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              ref_start, ctx->stream->offset);
          loom_print_report_field(
              ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
              ref_start, ctx->stream->offset);
        }
        break;
      }
      case LOOM_FORMAT_KIND_SCOPED_ENUM_REF: {
        // Stable spelling of a representation-scoped enum, glued to the op
        // name: low.op<amdgpu.v_add_u32>.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format SCOPED_ENUM_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind != LOOM_ATTR_SCOPED_ENUM) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format SCOPED_ENUM_REF field %u is not a scoped enum",
              element->field_index);
        }
        if (!ctx->low_repr.descriptor_set) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "printing a scoped descriptor requires an enclosing Low "
              "representation contract");
        }
        if (!ctx->low_asm_environment.low_repr.vtable) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "printing a scoped descriptor requires a Low representation "
              "environment");
        }
        iree_string_view_t key = loom_low_repr_descriptor_key(
            &ctx->low_asm_environment.low_repr, ctx->low_repr.descriptor_set,
            loom_attr_as_scoped_enum(attr));
        if (iree_string_view_is_empty(key)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "scoped enum ordinal is outside the active Low representation "
              "contract");
        }
        iree_host_size_t ref_start = ctx->stream->offset;
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
        IREE_RETURN_IF_ERROR(loom_print_emit(ctx, key, true));
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
        loom_print_did_write(ctx);
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
            ref_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_TEMPLATE_PARAM: {
        // Required compile-time op parameter in angle brackets, glued to
        // the op name: vector.reduce<addf>.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format TEMPLATE_PARAM field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        iree_host_size_t param_start = ctx->stream->offset;
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
        IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attr, descriptor));
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
        loom_print_did_write(ctx);
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
            param_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_PARAMS: {
        // Known-family parameter payload glued to the op name:
        // encoding.matches<element_format = u4>.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_PARAMS field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        iree_host_size_t param_start = ctx->stream->offset;
        IREE_RETURN_IF_ERROR(
            loom_print_parameterized_attr_parameters(ctx, &attr, descriptor));
        loom_print_did_write(ctx);
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
            param_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS: {
        // Required compile-time op parameter plus optional instance flags:
        // vector.reduce<addf, reassoc|nnan|nsz>.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format TEMPLATE_PARAM_FLAGS field_index %u out of range (op "
              "has %u attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        iree_host_size_t param_start = ctx->stream->offset;
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
        IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attr, descriptor));
        uint8_t flags = op->instance_flags;
        if (flags != 0 && vtable->instance_flags_case_names != NULL) {
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ", ", false));
          IREE_RETURN_IF_ERROR(
              loom_print_instance_flag_list(ctx, vtable, flags));
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
        loom_print_did_write(ctx);
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
            param_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_GLUE:
        loom_print_set_glue(ctx);
        break;
      case LOOM_FORMAT_KIND_OPTIONAL_GROUP: {
        uint16_t skip_count = element->data >> 2;
        uint8_t anchor_category = element->data & 3;
        bool present = false;
        switch (anchor_category) {
          case LOOM_ANCHOR_OPERAND:
            present =
                loom_op_operand_field_present(vtable, op, element->field_index);
            break;
          case LOOM_ANCHOR_ATTR:
            present =
                loom_print_optional_attr_present(op, element->field_index);
            break;
          case LOOM_ANCHOR_REGION: {
            if (element->field_index < op->region_count) {
              loom_region_t** regions = loom_op_regions(op);
              present = regions[element->field_index] != NULL &&
                        regions[element->field_index]->block_count > 0;
            }
            break;
          }
          case LOOM_ANCHOR_RESULTS:
            present = op->result_count > 0;
            break;
        }
        if (!present) {
          i += skip_count;
        }
        break;
      }
      case LOOM_FORMAT_KIND_SCOPE:
        // Scope is transparent for printing; children follow inline.
        // No scope state needed; the printer reads names from the value
        // table directly.
        break;
    }
  }
  return iree_ok_status();
}
