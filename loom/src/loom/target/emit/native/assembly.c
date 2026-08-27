// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/assembly.h"

#include <inttypes.h>

#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"

iree_string_view_t loom_native_assembly_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t loom_native_assembly_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset) {
  return loom_low_descriptor_set_string(descriptor_set, string_offset);
}

const loom_named_attr_t* loom_native_assembly_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_native_assembly_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

iree_status_t loom_native_assembly_read_i64_attr(const loom_module_t* module,
                                                 loom_named_attr_slice_t attrs,
                                                 iree_string_view_t name,
                                                 int64_t* out_value) {
  const loom_named_attr_t* attr =
      loom_native_assembly_find_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly requires attribute '%.*s'",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly attribute '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

iree_status_t loom_native_assembly_append_block_label(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block,
    iree_string_builder_t* builder) {
  const uint32_t block_index = loom_low_packet_block_index(schedule, block);
  if (block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native assembly block is outside the scheduled "
                            "function");
  }
  return iree_string_builder_append_format(builder, ".Lbb%" PRIu32,
                                           block_index);
}

iree_status_t loom_native_assembly_make_unsupported_structural_packet_status(
    iree_string_view_t target_name,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_op_vtable_t* vtable =
      loom_op_vtable(context->schedule->module, op);
  iree_string_view_t op_name =
      vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "%.*s assembly emitter does not support structural "
                          "op %.*s",
                          (int)target_name.size, target_name.data,
                          (int)op_name.size, op_name.data);
}

static iree_status_t loom_native_assembly_append_with_callback(
    const loom_native_assembly_append_packet_callback_t* callback,
    iree_string_view_t op_name,
    const loom_native_assembly_packet_context_t* context) {
  if (callback->fn == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native assembly emitter does not support %.*s",
                            (int)op_name.size, op_name.data);
  }
  return callback->fn(callback->user_data, context);
}

static iree_status_t loom_native_assembly_append_structural_packet(
    const loom_native_assembly_format_options_t* options,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_op_vtable_t* vtable =
      loom_op_vtable(context->schedule->module, op);
  iree_string_view_t op_name =
      vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return loom_native_assembly_append_with_callback(
      &options->append_structural_packet, op_name, context);
}

static iree_status_t loom_native_assembly_append_packet(
    const loom_native_assembly_format_options_t* options,
    const loom_native_assembly_packet_context_t* context) {
  if (context->packet->descriptor != NULL) {
    return options->append_descriptor_packet.fn(
        options->append_descriptor_packet.user_data, context);
  }
  return loom_native_assembly_append_structural_packet(options, context);
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
  if (loom_low_br_isa(op)) {
    return loom_low_br_dest(op) == block;
  }
  if (loom_low_cond_br_isa(op)) {
    return loom_low_cond_br_true_dest(op) == block ||
           loom_low_cond_br_false_dest(op) == block;
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
    iree_string_builder_t* builder, iree_host_size_t block_index) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  if (loom_native_assembly_should_print_block_label(schedule, block_index)) {
    IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
        schedule, block->block, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":\n"));
  }
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start + i;
    const loom_low_packet_view_t packet =
        loom_low_packet_at(schedule, packet_index);
    if (loom_low_packet_is_compile_time_only(&packet)) {
      continue;
    }
    const bool packet_is_visible =
        !loom_low_live_in_isa(packet.node->op) &&
        !loom_low_storage_reserve_isa(packet.node->op) &&
        !loom_low_storage_view_isa(packet.node->op);
    loom_native_assembly_packet_context_t context = {
        .schedule = schedule,
        .allocation = allocation,
        .packet = &packet,
        .builder = builder,
    };
    if (options->append_before_packet.fn != NULL) {
      IREE_RETURN_IF_ERROR(options->append_before_packet.fn(
          options->append_before_packet.user_data, &context));
    }
    if (!packet_is_visible) continue;
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
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  (void)scratch_arena;
  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t block_index = 0;
       block_index < schedule->block_count && iree_status_is_ok(status);
       ++block_index) {
    status = loom_native_assembly_append_block(schedule, allocation, options,
                                               builder, block_index);
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
