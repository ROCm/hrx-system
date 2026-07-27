// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/regions.h"

#include <stdio.h>

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/format.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static iree_status_t loom_print_block_comments(loom_print_context_t* ctx,
                                               const loom_block_t* block,
                                               uint16_t indent) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_block_comments(ctx->module, block, &comment_count);
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_print_indent_at(ctx, indent));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "//"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, comments[i]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block and successor labels.
//===----------------------------------------------------------------------===//

bool loom_print_block_has_label(const loom_print_context_t* ctx,
                                const loom_block_t* block) {
  return block->label_id != LOOM_STRING_ID_INVALID &&
         block->label_id < ctx->module->strings.count;
}

static bool loom_print_region_label_exists(const loom_print_context_t* ctx,
                                           const loom_region_t* region,
                                           iree_string_view_t label) {
  if (!region) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (!loom_print_block_has_label(ctx, block)) {
      continue;
    }
    if (iree_string_view_equal(ctx->module->strings.entries[block->label_id],
                               label)) {
      return true;
    }
  }
  return false;
}

static bool loom_print_region_find_block_index(const loom_region_t* region,
                                               const loom_block_t* block,
                                               uint16_t* out_block_index) {
  if (!region || !block) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (loom_region_const_block(region, block_index) == block) {
      *out_block_index = block_index;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_print_block_label_view(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_block_t* block, char* synthetic_buffer,
    iree_host_size_t synthetic_buffer_capacity, iree_string_view_t* out_label) {
  if (loom_print_block_has_label(ctx, block)) {
    *out_label = ctx->module->strings.entries[block->label_id];
    return iree_ok_status();
  }
  uint16_t block_index = 0;
  if (!loom_print_region_find_block_index(region, block, &block_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor target block is not in the printed "
                            "region and has no explicit label");
  }
  for (uint32_t suffix = 0;; ++suffix) {
    int length =
        suffix == 0
            ? snprintf(synthetic_buffer, synthetic_buffer_capacity, "_bb%u",
                       (unsigned)block_index)
            : snprintf(synthetic_buffer, synthetic_buffer_capacity, "_bb%u_%u",
                       (unsigned)block_index, (unsigned)suffix);
    if (length < 0 || (iree_host_size_t)length >= synthetic_buffer_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "synthetic block label exceeds buffer capacity");
    }
    iree_string_view_t candidate =
        iree_make_string_view(synthetic_buffer, (iree_host_size_t)length);
    if (!loom_print_region_label_exists(ctx, region, candidate)) {
      *out_label = candidate;
      return iree_ok_status();
    }
  }
}

bool loom_print_block_needs_synthetic_label(const loom_print_context_t* ctx,
                                            const loom_region_t* region,
                                            const loom_block_t* block) {
  if (!region || !block || loom_print_block_has_label(ctx, block)) {
    return false;
  }
  if (loom_region_const_entry_block(region) != block) {
    return true;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* source = loom_region_const_block(region, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(source, op) {
      loom_block_t* const* successors = loom_op_const_successors(op);
      for (uint8_t i = 0; i < op->successor_count; ++i) {
        if (successors[i] == block) {
          return true;
        }
      }
    }
  }
  return false;
}

iree_status_t loom_print_successor_ref(loom_print_context_t* ctx,
                                       const loom_op_t* op,
                                       uint8_t successor_index) {
  if (successor_index >= op->successor_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format SUCCESSOR_REF field_index %u out of range (op has %u "
        "successors)",
        successor_index, op->successor_count);
  }
  const loom_block_t* target = loom_op_const_successors(op)[successor_index];
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor %u target is NULL", successor_index);
  }
  const loom_region_t* region = op->parent_block
                                    ? op->parent_block->parent_region
                                    : target->parent_region;
  if (target->parent_region && region && target->parent_region != region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor %u target belongs to a different region",
                            successor_index);
  }
  char label_buffer[64];
  iree_string_view_t label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_print_block_label_view(
      ctx, region, target, label_buffer, sizeof(label_buffer), &label));

  iree_host_size_t successor_start =
      loom_print_next_token_start_offset(ctx, false, '^');
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, label));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_SUCCESSOR, successor_index),
      successor_start, ctx->stream->offset);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op and region printing.
//===----------------------------------------------------------------------===//

iree_status_t loom_print_op(loom_print_context_t* ctx, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  if (ctx->module->context) {
    vtable = loom_context_resolve_op(ctx->module->context, op->kind);
  }
  if (!vtable) {
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    return loom_output_stream_write_cstring(ctx->stream, "<unknown op>\n");
  }

  // Reset token state for this op.
  ctx->has_previous_token = false;
  ctx->glue_next = false;
  ctx->last_char = 0;

  iree_status_t status = iree_ok_status();

  // Print results on the LHS. Symbol-defining ops (functions, globals)
  // carry result values for type information but don't produce SSA values.
  bool print_results =
      op->result_count > 0 &&
      !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  if (iree_status_is_ok(status) && print_results) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t j = 0; j < op->result_count && iree_status_is_ok(status);
         ++j) {
      if (j > 0) {
        status = loom_print_emit_cstr(ctx, ",", false);
      }
      if (iree_status_is_ok(status)) {
        status = loom_print_value_name_with_field(
            ctx, results[j], loom_print_field_ref(LOOM_PRINT_FIELD_RESULT, j));
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_print_emit_cstr(ctx, "=", false);
    }
  }

  // Print op name.
  if (iree_status_is_ok(status)) {
    status = loom_print_emit(ctx, loom_op_vtable_name(vtable), false);
  }

  // Walk format elements. Regions are printed inline when their REGION format
  // element is encountered, properly interleaving tokens with region bodies.
  const loom_text_low_asm_descriptor_set_t* previous_descriptor_set =
      ctx->low_register_descriptor_set;
  if (iree_status_is_ok(status)) {
    status = loom_print_format_elements(ctx, op, vtable);
  }
  ctx->low_register_descriptor_set = previous_descriptor_set;

  // Location annotation (omitted for LOOM_LOCATION_UNKNOWN).
  if (iree_status_is_ok(status) && (ctx->flags & LOOM_TEXT_PRINT_LOCATIONS)) {
    status = loom_print_location(ctx->stream, ctx->module, op->location);
  }

  // Terminate the op line.
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_char(ctx->stream, '\n');
  }

  return status;
}

bool loom_print_should_elide_implicit_terminator(
    const loom_region_descriptor_t* region_descriptor, const loom_op_t* op) {
  if (region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN) {
    return false;
  }
  return op->kind == region_descriptor->implicit_terminator &&
         op->operand_count == 0 && op->result_count == 0 &&
         op->region_count == 0 && op->tied_result_count == 0 &&
         op->attribute_count == 0 && op->instance_flags == 0;
}

iree_status_t loom_print_block_label_line_with_options(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_block_t* block, bool print_block_args) {
  uint16_t label_indent = ctx->indent > 0 ? (uint16_t)(ctx->indent - 1) : 0;
  char label_buffer[64];
  iree_string_view_t label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_print_block_label_view(
      ctx, region, block, label_buffer, sizeof(label_buffer), &label));
  IREE_RETURN_IF_ERROR(loom_print_block_comments(ctx, block, label_indent));
  IREE_RETURN_IF_ERROR(loom_print_indent_at(ctx, label_indent));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, label));
  if (print_block_args && block->arg_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      if (arg_index > 0) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, ", "));
      }
      loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
      IREE_RETURN_IF_ERROR(
          loom_print_value_ref(ctx->stream, ctx->module, arg_id));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ": "));
      IREE_RETURN_IF_ERROR(
          loom_print_type(ctx, loom_module_value_type(ctx->module, arg_id)));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
  }
  return loom_output_stream_write_cstring(ctx->stream, ":\n");
}

iree_status_t loom_print_block_label_line(loom_print_context_t* ctx,
                                          const loom_region_t* region,
                                          const loom_block_t* block) {
  return loom_print_block_label_line_with_options(ctx, region, block,
                                                  /*print_block_args=*/true);
}

iree_status_t loom_print_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    const bool entry_block = block_index == 0;
    const bool block_args_declared_by_parent =
        entry_block && entry_args_declared_by_parent;
    const bool needs_label =
        loom_print_block_has_label(ctx, block) ||
        loom_print_block_needs_synthetic_label(ctx, region, block) ||
        (block->arg_count != 0 && !block_args_declared_by_parent);

    if (needs_label) {
      IREE_RETURN_IF_ERROR(loom_print_block_label_line_with_options(
          ctx, region, block, !block_args_declared_by_parent));
    }

    bool printed_any = false;
    const loom_op_t* last_live_op = block->last_op;
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      if (current_op == last_live_op &&
          loom_print_should_elide_implicit_terminator(region_descriptor,
                                                      current_op)) {
        continue;
      }
      // Blank line between top-level symbol definitions (func.def, func.decl,
      // etc.) in the module body.
      if (printed_any && ctx->indent == 0) {
        const loom_op_vtable_t* current_vtable =
            loom_op_vtable(ctx->module, current_op);
        if (current_vtable &&
            (current_vtable->traits & LOOM_TRAIT_SYMBOL_DEFINE)) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_char(ctx->stream, '\n'));
        }
      }
      printed_any = true;
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                     const loom_region_t* region) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    if (loom_print_block_has_label(ctx, block) ||
        loom_print_block_needs_synthetic_label(ctx, region, block)) {
      IREE_RETURN_IF_ERROR(loom_print_block_label_line(ctx, region, block));
    }

    bool printed_any = false;
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      if (printed_any) {
        const loom_op_vtable_t* current_vtable =
            loom_op_vtable(ctx->module, current_op);
        if (current_vtable &&
            (current_vtable->traits & LOOM_TRAIT_SYMBOL_DEFINE)) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_char(ctx->stream, '\n'));
        }
      }
      printed_any = true;
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
    }
  }
  return iree_ok_status();
}
