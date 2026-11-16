// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format_signatures.h"

#include "loom/format/text/printer/atoms.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Generated-format signature payloads
//===----------------------------------------------------------------------===//

// Returns the signature argument IDs for a FuncArgs element. Bodyful func-like
// ops use the entry block args; bodyless declarations store signature args as
// op operands.
static const loom_value_id_t* loom_print_func_arg_ids(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    uint16_t* out_arg_count) {
  *out_arg_count = 0;
  if (!vtable->func_like) {
    return NULL;
  }
  return loom_func_like_arg_ids(
      (loom_func_like_t){
          .op = (loom_op_t*)op,
          .vtable = vtable->func_like,
      },
      out_arg_count);
}

static const loom_value_id_t* loom_print_region_entry_arg_ids(
    const loom_op_t* op, uint8_t region_index, uint16_t* out_arg_count) {
  *out_arg_count = 0;
  if (region_index >= op->region_count) {
    return NULL;
  }
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) {
    return NULL;
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  *out_arg_count = block->arg_count;
  return block->arg_ids;
}

// Returns the operand domain used for tied-result printing. Regular body ops
// tie to op operands; symbol-defining func-like ops tie to signature args.
static const loom_value_id_t* loom_print_tied_operand_ids(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    uint16_t* out_operand_count) {
  if (vtable->func_like &&
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return loom_print_func_arg_ids(op, vtable, out_operand_count);
  }
  *out_operand_count = op->operand_count;
  return loom_op_const_operands(op);
}

// Locally scoped signature results need a binder when their identity is
// referenced, even without an explicit source name. Construction and mutation
// retain operand, type, and attribute uses, so deciding this requires no
// signature traversal.
static bool loom_print_signature_result_needs_binding(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return (value->name_id != LOOM_STRING_ID_INVALID &&
          value->name_id < module->strings.count) ||
         !loom_value_has_no_uses(value) ||
         loom_value_has_attribute_uses(value) ||
         loom_module_value_has_type_uses(module, value_id);
}

iree_status_t loom_print_result_type_list(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_format_element_t* element) {
  bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
  bool uniform = (element->data & LOOM_RESULT_TYPE_LIST_UNIFORM) != 0;
  if (use_parens) {
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", false));
  }
  const bool prints_signature_results =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      loom_op_vtable_has_signature_only_results(vtable);
  uint16_t tied_operand_count = 0;
  const loom_value_id_t* tied_operand_ids =
      loom_print_tied_operand_ids(op, vtable, &tied_operand_count);
  uint16_t printed_result_count =
      uniform && op->result_count > 0 ? 1 : op->result_count;
  for (uint16_t j = 0; j < printed_result_count; ++j) {
    if (j > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }

    if (prints_signature_results &&
        loom_print_signature_result_needs_binding(
            ctx->module, loom_op_const_results(op)[j])) {
      IREE_RETURN_IF_ERROR(
          loom_print_value_name(ctx, loom_op_const_results(op)[j]));
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
    }
    const loom_tied_result_t* tied = loom_op_tied_results(op);
    for (uint16_t t = 0; t < op->tied_result_count; ++t) {
      if (tied[t].result_index == j) {
        if (tied[t].operand_index >= tied_operand_count || !tied_operand_ids) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "tied result %u references operand %u but op has %u operands",
              tied[t].result_index, tied[t].operand_index, tied_operand_count);
        }
        loom_value_id_t tied_operand_id =
            tied_operand_ids[tied[t].operand_index];
        IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, tied_operand_id));
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "as", false));
        break;
      }
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(
        loom_print_result_value_type(ctx, loom_op_const_results(op)[j]));
    loom_print_did_write(ctx);
  }
  if (use_parens) {
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", false));
  }
  return iree_ok_status();
}

iree_status_t loom_print_binding_list(loom_print_context_t* ctx,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable,
                                      const loom_format_element_t* element) {
  loom_value_slice_t operands =
      loom_op_operand_field_span(vtable, op, element->field_index);
  uint16_t binding_count = operands.count;
  loom_region_t** regions = loom_op_regions(op);
  const loom_block_t* block = NULL;
  if (op->region_count > 0 && regions[0] && regions[0]->block_count > 0) {
    block = loom_region_const_entry_block(regions[0]);
  }
  uint16_t block_arg_offset = 0;
  if (block && block->arg_count > binding_count) {
    block_arg_offset = block->arg_count - binding_count;
  }
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
  for (uint16_t j = 0; j < binding_count; ++j) {
    if (j > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    if (block && (block_arg_offset + j) < block->arg_count) {
      IREE_RETURN_IF_ERROR(loom_print_value_name(
          ctx, loom_block_arg_id(block, (uint16_t)(block_arg_offset + j))));
    }
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "=", false));
    const loom_value_id_t* operand_ptr = &operands.values[j];
    uint16_t operand_index =
        (uint16_t)(operand_ptr - loom_op_const_operands(op));
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, *operand_ptr,
        loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    loom_value_id_t type_value_id = *operand_ptr;
    if (iree_any_bit_set(element->data,
                         LOOM_BINDING_LIST_ANNOTATES_BLOCK_ARGS) &&
        block && (block_arg_offset + j) < block->arg_count) {
      type_value_id =
          loom_block_arg_id(block, (uint16_t)(block_arg_offset + j));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, type_value_id));
    loom_print_did_write(ctx);
  }
  return loom_print_emit_cstr(ctx, ")", false);
}

iree_status_t loom_print_block_args(loom_print_context_t* ctx,
                                    const loom_op_t* op,
                                    const loom_format_element_t* element) {
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_print_region_entry_arg_ids(op, element->field_index, &arg_count);
  const uint8_t start_attr_index =
      LOOM_FORMAT_BLOCK_ARGS_START_ATTR_INDEX(element->data);
  const uint8_t end_attr_index =
      LOOM_FORMAT_BLOCK_ARGS_END_ATTR_INDEX(element->data);
  int64_t start = 0;
  int64_t end = arg_count;
  if (start_attr_index != LOOM_ATTR_INDEX_NONE) {
    start = loom_attr_as_i64(loom_op_const_attrs(op)[start_attr_index]);
  }
  if (end_attr_index != LOOM_ATTR_INDEX_NONE) {
    end = loom_attr_as_i64(loom_op_const_attrs(op)[end_attr_index]);
  }
  if (start < 0 || end < start || end > arg_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region argument slice [%" PRId64 ", %" PRId64
                            ") is outside entry signature with %u arguments",
                            start, end, arg_count);
  }
  if (start > 0) {
    arg_ids += start;
  }
  arg_count = (uint16_t)(end - start);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
  for (uint16_t j = 0; j < arg_count; ++j) {
    if (j > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, arg_ids[j]));
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, arg_ids[j]));
    loom_print_did_write(ctx);
  }
  return loom_print_emit_cstr(ctx, ")", false);
}

iree_status_t loom_print_func_args(loom_print_context_t* ctx,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable,
                                   const loom_format_element_t* element) {
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = NULL;
  if (element->field_index != LOOM_OPERAND_INDEX_NONE) {
    loom_value_slice_t args =
        loom_op_operand_field_span(vtable, op, element->field_index);
    arg_ids = args.values;
    arg_count = args.count;
  } else {
    arg_ids = loom_print_func_arg_ids(op, vtable, &arg_count);
  }
  const iree_const_byte_span_t access = loom_func_like_signature_flags(
      (loom_func_like_t){.op = (loom_op_t*)op, .vtable = vtable->func_like});
  if (access.data_length &&
      access.data_length != (uint32_t)arg_count + op->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "signature flags count does not match signature");
  }
  const uint8_t start_attr_index =
      LOOM_FORMAT_FUNC_ARGS_START_ATTR_INDEX(element->data);
  const uint8_t end_attr_index =
      LOOM_FORMAT_FUNC_ARGS_END_ATTR_INDEX(element->data);
  int64_t start = 0;
  int64_t end = arg_count;
  if (start_attr_index != LOOM_ATTR_INDEX_NONE) {
    start = loom_attr_as_i64(loom_op_const_attrs(op)[start_attr_index]);
  }
  if (end_attr_index != LOOM_ATTR_INDEX_NONE) {
    end = loom_attr_as_i64(loom_op_const_attrs(op)[end_attr_index]);
  }
  if (start < 0 || end < start || end > arg_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function argument slice [%" PRId64 ", %" PRId64
                            ") is outside signature with %u arguments",
                            start, end, arg_count);
  }
  if (start > 0) {
    arg_ids += start;
  }
  arg_count = (uint16_t)(end - start);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
  for (uint16_t j = 0; j < arg_count; ++j) {
    if (j > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    if (element->field_index != LOOM_OPERAND_INDEX_NONE) {
      uint16_t operand_index =
          (uint16_t)(&arg_ids[j] - loom_op_const_operands(op));
      IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
          ctx, arg_ids[j],
          loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
    } else {
      IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, arg_ids[j]));
    }
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, arg_ids[j]));
    loom_print_did_write(ctx);
    const loom_argument_access_t bound =
        access.data_length ? loom_signature_flags_access(access.data[start + j])
                           : LOOM_ARGUMENT_ACCESS_UNSPECIFIED;
    if (bound != LOOM_ARGUMENT_ACCESS_UNSPECIFIED) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "{access =", false));
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      loom_attr_descriptor_t descriptor =
          vtable
              ->attr_descriptors[vtable->func_like->signature_flags_attr_index];
      descriptor.attr_kind = LOOM_ATTR_ENUM;
      const loom_attribute_t attribute = loom_attr_enum(bound);
      IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attribute, &descriptor));
      loom_print_did_write(ctx);
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "}", false));
    }
  }
  return loom_print_emit_cstr(ctx, ")", false);
}
