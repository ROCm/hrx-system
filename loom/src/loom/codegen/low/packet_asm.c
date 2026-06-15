// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_asm.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/util/stream.h"

typedef struct loom_low_packet_asm_state_t {
  // Schedule table being rendered.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying locations for SSA values.
  const loom_low_allocation_table_t* allocation;
  // Optional per-packet selected asm-form table.
  const loom_low_packet_asm_form_table_t* selected_asm_forms;
  // Formatter options supplied by the caller.
  const loom_low_packet_asm_options_t* options;
  // Text destination.
  iree_string_builder_t* builder;
} loom_low_packet_asm_state_t;

static iree_string_view_t loom_low_packet_asm_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_low_packet_asm_append_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_builder_t* builder) {
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low packet asm descriptor string is empty");
  }
  return iree_string_builder_append_string(builder, value);
}

static iree_status_t loom_low_packet_asm_append_value(
    loom_low_packet_asm_state_t* state, loom_value_id_t value_id) {
  uint32_t assignment_index = UINT32_MAX;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_map_active_value_assignment(
          state->allocation, value_id, &assignment_index);
  return state->options->format_value.fn(
      state->options->format_value.user_data, state->allocation, value_id,
      assignment, assignment_index, state->builder);
}

static iree_status_t loom_low_packet_asm_append_value_list(
    loom_low_packet_asm_state_t* state, const loom_value_id_t* values,
    iree_host_size_t value_count) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(state->builder, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_value(state, values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_asm_append_block_label(
    loom_low_packet_asm_state_t* state, const loom_block_t* block) {
  const uint32_t block_index =
      loom_low_packet_block_index(state->schedule, block);
  if (block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low packet asm block is outside function");
  }
  return iree_string_builder_append_format(state->builder, "^bb%" PRIu32,
                                           block_index);
}

static iree_status_t loom_low_packet_asm_append_block_arguments(
    loom_low_packet_asm_state_t* state, const loom_block_t* block) {
  if (block->arg_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(state->builder, "("));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(state->builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_packet_asm_append_value(state, loom_block_arg_id(block, i)));
  }
  return iree_string_builder_append_cstring(state->builder, ")");
}

static const loom_named_attr_t* loom_low_packet_asm_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_low_packet_asm_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_low_packet_asm_append_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_attribute(attr, module, &stream);
}

static iree_status_t loom_low_packet_asm_append_immediates(
    loom_low_packet_asm_state_t* state, const loom_low_descriptor_t* descriptor,
    const loom_low_asm_form_t* asm_form, loom_named_attr_slice_t attrs) {
  if (asm_form->immediate_count == 0) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  const loom_module_t* module = state->schedule->module;
  iree_host_size_t printed_count = 0;
  for (uint16_t i = 0; i < asm_form->immediate_count; ++i) {
    const uint32_t asm_immediate_index = asm_form->immediate_start + i;
    if (asm_immediate_index >= descriptor_set->asm_immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low packet asm immediate row is out of range");
    }
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_immediate_index];
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low packet asm immediate references an invalid descriptor field");
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    asm_immediate->immediate_index];
    iree_string_view_t field_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    iree_string_view_t spelling = field_name;
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      spelling = loom_low_descriptor_set_string(
          descriptor_set, asm_immediate->name_string_offset);
    }
    if (iree_string_view_is_empty(spelling)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low packet asm immediate spelling is empty");
    }
    const loom_named_attr_t* attr =
        loom_low_packet_asm_find_attr(module, attrs, field_name);
    if (attr == NULL) {
      if (iree_any_bit_set(immediate->flags,
                           LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low packet asm descriptor immediate '%.*s' is missing",
          (int)field_name.size, field_name.data);
    }
    if (printed_count == 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(state->builder, " {"));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(state->builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(state->builder, spelling));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(state->builder, " = "));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_asm_append_attr(module, &attr->value, state->builder));
    ++printed_count;
  }
  if (printed_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_cstring(state->builder, "}");
}

static iree_status_t loom_low_packet_asm_append_asm_form_value(
    loom_low_packet_asm_state_t* state, const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index, bool is_result,
    const loom_value_id_t* results, iree_host_size_t result_count,
    const loom_value_id_t* operands, iree_host_size_t operand_count) {
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low packet asm descriptor operand index is out "
                            "of range");
  }
  if (is_result) {
    if (descriptor_operand_index >= descriptor->result_count ||
        descriptor_operand_index >= result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low packet asm result field does not name an emitted result");
    }
    return loom_low_packet_asm_append_value(state,
                                            results[descriptor_operand_index]);
  }
  if (descriptor_operand_index < descriptor->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low packet asm operand field unexpectedly names a descriptor result");
  }
  const uint16_t operand_index =
      descriptor_operand_index - descriptor->result_count;
  if (operand_index >= operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low packet asm operand field does not name an emitted operand");
  }
  return loom_low_packet_asm_append_value(state, operands[operand_index]);
}

static iree_status_t loom_low_packet_asm_append_asm_form_values(
    loom_low_packet_asm_state_t* state, const loom_low_descriptor_t* descriptor,
    uint32_t start, uint16_t count, bool is_result,
    const loom_value_id_t* results, iree_host_size_t result_count,
    const loom_value_id_t* operands, iree_host_size_t operand_count) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(state->builder, ", "));
    }
    const uint32_t asm_operand_index = start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low packet asm operand row is out of range");
    }
    IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_asm_form_value(
        state, descriptor,
        descriptor_set->asm_operand_indices[asm_operand_index], is_result,
        results, result_count, operands, operand_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_asm_append_descriptor_packet(
    loom_low_packet_asm_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_packet_lookup_asm_form(
      state->schedule, state->selected_asm_forms, packet, &asm_form_ordinal));
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low packet asm canonical asm form is out of "
                            "range");
  }

  const loom_op_t* op = packet->node->op;
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const iree_host_size_t result_count = op->result_count;
  const iree_host_size_t operand_count = op->operand_count;
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  if (loom_low_op_isa(op)) {
    attrs = loom_low_op_attrs(op);
  } else if (loom_low_const_isa(op)) {
    attrs = loom_low_const_attrs(op);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low packet asm descriptor node is not low.op or "
                            "low.const");
  }

  if (asm_form->result_operand_index_count > 0) {
    IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_asm_form_values(
        state, packet->descriptor, asm_form->result_operand_index_start,
        asm_form->result_operand_index_count, /*is_result=*/true, results,
        result_count, operands, operand_count));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(state->builder, " = "));
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_descriptor_string(
      descriptor_set, asm_form->mnemonic_string_offset, state->builder));
  if (asm_form->operand_index_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(state->builder, " "));
    IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_asm_form_values(
        state, packet->descriptor, asm_form->operand_index_start,
        asm_form->operand_index_count, /*is_result=*/false, results,
        result_count, operands, operand_count));
  }
  return loom_low_packet_asm_append_immediates(state, packet->descriptor,
                                               asm_form, attrs);
}

static iree_status_t loom_low_packet_asm_append_return(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, "return"));
  loom_value_slice_t values = loom_low_return_values(op);
  if (values.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(state->builder, " "));
  return loom_low_packet_asm_append_value_list(state, values.values,
                                               values.count);
}

static iree_status_t loom_low_packet_asm_append_copy(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value(state, loom_low_copy_result(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, " = copy "));
  return loom_low_packet_asm_append_value(state, loom_low_copy_source(op));
}

static iree_status_t loom_low_packet_asm_append_concat(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value(state, loom_low_concat_result(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, " = concat("));
  loom_value_slice_t sources = loom_low_concat_sources(op);
  IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_value_list(
      state, sources.values, sources.count));
  return iree_string_builder_append_cstring(state->builder, ")");
}

static iree_status_t loom_low_packet_asm_append_slice(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value(state, loom_low_slice_result(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, " = slice "));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value(state, loom_low_slice_source(op)));
  return iree_string_builder_append_format(state->builder, "[%" PRId64 "]",
                                           loom_low_slice_offset(op));
}

static iree_status_t loom_low_packet_asm_append_branch_args(
    loom_low_packet_asm_state_t* state, const loom_value_id_t* values,
    iree_host_size_t value_count) {
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(state->builder, "("));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value_list(state, values, value_count));
  return iree_string_builder_append_cstring(state->builder, ")");
}

static iree_status_t loom_low_packet_asm_append_br(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, "br "));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_block_label(state, loom_low_br_dest(op)));
  loom_value_slice_t args = loom_low_br_args(op);
  return loom_low_packet_asm_append_branch_args(state, args.values, args.count);
}

static iree_status_t loom_low_packet_asm_append_cond_br(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, "cond_br "));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_value(state, loom_low_cond_br_condition(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_block_label(
      state, loom_low_cond_br_true_dest(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(state->builder, ", "));
  return loom_low_packet_asm_append_block_label(
      state, loom_low_cond_br_false_dest(op));
}

typedef iree_status_t (*loom_low_packet_asm_append_structural_fn_t)(
    loom_low_packet_asm_state_t* state, const loom_op_t* op);

typedef struct loom_low_packet_asm_structural_dispatch_t {
  // Structural op handled by this row.
  loom_op_kind_t op_kind;
  // Formatter for the structural packet.
  loom_low_packet_asm_append_structural_fn_t append;
} loom_low_packet_asm_structural_dispatch_t;

static const loom_low_packet_asm_structural_dispatch_t
    kLoomLowPacketAsmStructuralDispatch[] = {
        {LOOM_OP_LOW_RETURN, loom_low_packet_asm_append_return},
        {LOOM_OP_LOW_COPY, loom_low_packet_asm_append_copy},
        {LOOM_OP_LOW_SLICE, loom_low_packet_asm_append_slice},
        {LOOM_OP_LOW_CONCAT, loom_low_packet_asm_append_concat},
        {LOOM_OP_LOW_BR, loom_low_packet_asm_append_br},
        {LOOM_OP_LOW_COND_BR, loom_low_packet_asm_append_cond_br},
};

static iree_status_t loom_low_packet_asm_append_structural_packet(
    loom_low_packet_asm_state_t* state, const loom_op_t* op) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomLowPacketAsmStructuralDispatch); ++i) {
    const loom_low_packet_asm_structural_dispatch_t* row =
        &kLoomLowPacketAsmStructuralDispatch[i];
    if (op->kind != row->op_kind) {
      continue;
    }
    return row->append(state, op);
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(state->schedule->module, op);
  iree_string_view_t op_name =
      vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "low packet asm structural op %.*s is unsupported",
                          (int)op_name.size, op_name.data);
}

static iree_status_t loom_low_packet_asm_append_packet(
    loom_low_packet_asm_state_t* state, const loom_low_packet_view_t* packet) {
  if (packet->descriptor != NULL) {
    return loom_low_packet_asm_append_descriptor_packet(state, packet);
  }
  return loom_low_packet_asm_append_structural_packet(state, packet->node->op);
}

static iree_status_t loom_low_packet_asm_append_block(
    loom_low_packet_asm_state_t* state, iree_host_size_t block_index) {
  const loom_low_schedule_table_t* schedule = state->schedule;
  iree_string_builder_t* builder = state->builder;
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "^bb%" PRIhsz, block_index));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_asm_append_block_arguments(state, block->block));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":\n"));
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start + i;
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(schedule, state->allocation,
                                                 packet_index, &packet));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "  "));
    IREE_RETURN_IF_ERROR(loom_low_packet_asm_append_packet(state, &packet));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_asm_format(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_asm_options_t* options,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  if (options->selected_asm_forms != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_packet_validate_asm_form_table(
        schedule, options->selected_asm_forms));
  }

  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch));
  loom_low_packet_asm_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .selected_asm_forms = options->selected_asm_forms,
      .options = options,
      .builder = builder,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t block_index = 0;
       block_index < schedule->block_count && iree_status_is_ok(status);
       ++block_index) {
    status = loom_low_packet_asm_append_block(&state, block_index);
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
