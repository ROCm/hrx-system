// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/assembly.h"

#include <inttypes.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"

iree_string_view_t loom_native_assembly_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_string_ref_t string_ref) {
  return loom_low_descriptor_set_string(descriptor_set, string_ref);
}

iree_status_t loom_native_assembly_append_block_label(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block,
    iree_string_builder_t* builder) {
  const uint32_t block_index = loom_low_packet_block_index(schedule, block);
  return iree_string_builder_append_format(builder, ".Lbb%" PRIu32,
                                           block_index);
}

iree_status_t loom_native_assembly_emit_unsupported_operation(
    iree_string_view_t format,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(format),
      loom_param_string(
          loom_op_vtable_name(loom_op_vtable(context->schedule->module, op))),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_BACKEND_062,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  *context->accepted = false;
  return iree_diagnostic_emit(context->emitter, &emission);
}

static iree_status_t loom_native_assembly_append_packet(
    const loom_native_assembly_format_options_t* options,
    const loom_native_assembly_packet_context_t* context) {
  if (context->packet->descriptor != NULL) {
    return options->append_descriptor_packet.fn(
        options->append_descriptor_packet.user_data, context);
  }
  return options->append_structural_packet.fn(
      options->append_structural_packet.user_data, context);
}

static void loom_native_assembly_truncate_builder(
    iree_string_builder_t* builder, iree_host_size_t size) {
  IREE_ASSERT(size <= iree_string_builder_size(builder));
  builder->size = size;
  if (builder->buffer != NULL) {
    builder->buffer[size] = 0;
  }
}

static bool loom_native_assembly_op_branches_to(const loom_op_t* op,
                                                const loom_block_t* block) {
  if (loom_direct_branch_isa(op)) {
    return loom_direct_branch_dest(op) == block;
  }
  if (loom_low_cond_br_isa(op)) {
    return loom_low_cond_br_true_dest(op) == block ||
           loom_low_cond_br_false_dest(op) == block;
  }
  if (loom_low_op_cond_br_isa(op)) {
    return loom_low_op_cond_br_true_dest(op) == block ||
           loom_low_op_cond_br_false_dest(op) == block;
  }
  return false;
}

static bool loom_native_assembly_block_is_branch_target(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block) {
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* scheduled_block =
        &schedule->blocks[block_index];
    for (uint32_t i = 0; i < scheduled_block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)scheduled_block->scheduled_node_start + i;
      const loom_low_packet_view_t packet =
          loom_low_packet_at(schedule, packet_index);
      if (loom_native_assembly_op_branches_to(packet.node->op, block)) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_native_assembly_should_print_block_label(
    const loom_low_schedule_table_t* schedule, iree_host_size_t block_index) {
  if (block_index != 0) {
    return true;
  }
  return loom_native_assembly_block_is_branch_target(
      schedule, schedule->blocks[block_index].block);
}

static iree_status_t loom_native_assembly_append_block(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_native_assembly_format_options_t* options,
    iree_string_builder_t* builder, iree_host_size_t block_index,
    iree_diagnostic_emitter_t emitter, bool* accepted) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  if (loom_native_assembly_should_print_block_label(schedule, block_index)) {
    IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
        schedule, block->block, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":\n"));
  }
  for (uint32_t i = 0; i < block->scheduled_node_count && *accepted; ++i) {
    const iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start + i;
    const loom_low_packet_view_t packet =
        loom_low_packet_at(schedule, packet_index);
    const bool packet_is_visible =
        !loom_low_packet_is_compile_time_only(&packet) &&
        !loom_low_live_in_isa(packet.node->op) &&
        !loom_low_storage_reserve_isa(packet.node->op) &&
        !loom_low_storage_view_isa(packet.node->op);
    loom_native_assembly_packet_context_t context = {
        .schedule = schedule,
        .allocation = allocation,
        .packet = &packet,
        .builder = builder,
        .emitter = emitter,
        .accepted = accepted,
    };
    if (options->append_before_packet.fn != NULL) {
      IREE_RETURN_IF_ERROR(options->append_before_packet.fn(
          options->append_before_packet.user_data, &context));
    }
    if (!*accepted || !packet_is_visible) {
      continue;
    }
    const iree_host_size_t line_start = iree_string_builder_size(builder);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "  "));
    const iree_host_size_t content_start = iree_string_builder_size(builder);
    IREE_RETURN_IF_ERROR(loom_native_assembly_append_packet(options, &context));
    if (iree_string_builder_size(builder) == content_start) {
      loom_native_assembly_truncate_builder(builder, line_start);
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

iree_status_t loom_native_assembly_format_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_native_assembly_format_options_t* options,
    iree_string_builder_t* builder, iree_diagnostic_emitter_t emitter,
    bool* out_accepted) {
  *out_accepted = true;
  const iree_host_size_t fragment_start = iree_string_builder_size(builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t block_index = 0;
       block_index < schedule->block_count && *out_accepted &&
       iree_status_is_ok(status);
       ++block_index) {
    status = loom_native_assembly_append_block(schedule, allocation, options,
                                               builder, block_index, emitter,
                                               out_accepted);
  }
  if (!*out_accepted) {
    loom_native_assembly_truncate_builder(builder, fragment_start);
  }
  return status;
}
