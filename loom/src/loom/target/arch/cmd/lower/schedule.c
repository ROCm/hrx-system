// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/schedule.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/composition.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"

enum {
  LOOM_CMD_SCHEDULE_INITIAL_CAPACITY = 32,
};

typedef struct loom_cmd_schedule_frame_t {
  // Next operation awaiting traversal in this region.
  loom_op_t* next_op;
  // First wave available to children in this region.
  iree_host_size_t base_wave;
  // Number of waves occupied by children visited so far.
  iree_host_size_t span;
  // Composition rule applied to direct children.
  loom_command_composition_mode_t mode;
} loom_cmd_schedule_frame_t;

typedef struct loom_cmd_schedule_command_build_t {
  // Classified issue command.
  loom_cmd_schedule_command_t command;
  // Wave assigned while traversing the structured schedule.
  iree_host_size_t wave_index;
} loom_cmd_schedule_command_build_t;

typedef struct loom_cmd_schedule_build_t {
  // Source module used to inspect operation traits and names.
  const loom_module_t* module;
  // Arena owning traversal and command tables.
  iree_arena_allocator_t* arena;
  // Non-recursive structured-region traversal stack.
  loom_cmd_schedule_frame_t* frames;
  // Number of active traversal frames.
  iree_host_size_t frame_count;
  // Number of allocated traversal frames.
  iree_host_size_t frame_capacity;
  // Commands accumulated in source traversal order.
  loom_cmd_schedule_command_build_t* commands;
  // Number of accumulated commands.
  iree_host_size_t command_count;
  // Number of allocated command rows.
  iree_host_size_t command_capacity;
  // Total number of device-ABI argument values across all commands.
  iree_host_size_t argument_value_count;
  // Allocation definitions accumulated in source traversal order.
  loom_cmd_schedule_allocation_t* allocations;
  // Number of accumulated allocation definitions.
  iree_host_size_t allocation_count;
  // Number of allocated allocation rows.
  iree_host_size_t allocation_capacity;
} loom_cmd_schedule_build_t;

static iree_status_t loom_cmd_schedule_reserve_frames(
    loom_cmd_schedule_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count = build->frame_count + additional_count;
  if (required_count <= build->frame_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(build->arena, build->frame_count, required_count,
                               sizeof(*build->frames), &build->frame_capacity,
                               (void**)&build->frames);
}

static iree_status_t loom_cmd_schedule_reserve_commands(
    loom_cmd_schedule_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      build->command_count + additional_count;
  if (required_count <= build->command_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(build->arena, build->command_count,
                               required_count, sizeof(*build->commands),
                               &build->command_capacity,
                               (void**)&build->commands);
}

static iree_status_t loom_cmd_schedule_reserve_allocations(
    loom_cmd_schedule_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      build->allocation_count + additional_count;
  if (required_count <= build->allocation_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(build->arena, build->allocation_count,
                               required_count, sizeof(*build->allocations),
                               &build->allocation_capacity,
                               (void**)&build->allocations);
}

static iree_status_t loom_cmd_schedule_push_frame(
    loom_cmd_schedule_build_t* build, loom_region_t* region,
    iree_host_size_t base_wave, loom_command_composition_mode_t mode) {
  if (region == NULL || region->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "portable command scheduling requires single-block regions");
  }
  IREE_RETURN_IF_ERROR(loom_cmd_schedule_reserve_frames(build, 1));
  build->frames[build->frame_count++] = (loom_cmd_schedule_frame_t){
      .next_op = loom_region_entry_block(region)->first_op,
      .base_wave = base_wave,
      .span = 0,
      .mode = mode,
  };
  return iree_ok_status();
}

static iree_host_size_t loom_cmd_schedule_child_base_wave(
    const loom_cmd_schedule_frame_t* frame) {
  return frame->mode == LOOM_COMMAND_COMPOSITION_MODE_SERIAL
             ? frame->base_wave + frame->span
             : frame->base_wave;
}

static void loom_cmd_schedule_commit_child_span(
    loom_cmd_schedule_frame_t* frame, iree_host_size_t child_span) {
  if (frame->mode == LOOM_COMMAND_COMPOSITION_MODE_SERIAL) {
    frame->span += child_span;
  } else {
    frame->span = iree_max(frame->span, child_span);
  }
}

static loom_cmd_schedule_value_slice_t loom_cmd_schedule_value_slice(
    loom_value_slice_t source) {
  return (loom_cmd_schedule_value_slice_t){
      .values = source.values,
      .count = source.count,
  };
}

static iree_status_t loom_cmd_schedule_append_command(
    loom_cmd_schedule_build_t* build, loom_cmd_schedule_frame_t* frame,
    const loom_op_t* op) {
  loom_cmd_schedule_command_t command = {
      .source_op = op,
  };
  if (loom_command_timestamp_isa(op)) {
    command.callee = loom_symbol_ref_null();
    command.kind = LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP;
    command.arguments = (loom_cmd_schedule_value_slice_t){
        .values = loom_op_const_operands(op),
        .count = 1,
    };
    command.source_arguments = command.arguments;
  } else if (loom_kernel_launch_isa(op)) {
    command.callee = loom_kernel_launch_callee(op);
    command.arguments =
        loom_cmd_schedule_value_slice(loom_kernel_launch_arguments(op));
    command.source_arguments = (loom_cmd_schedule_value_slice_t){
        .values = loom_op_operands((loom_op_t*)op),
        .count = op->operand_count,
    };
    command.kind = LOOM_CMD_SCHEDULE_COMMAND_KIND_PIPELINE_LAUNCH;
  } else {
    IREE_ASSERT(loom_kernel_dispatch_isa(op));
    command.callee = loom_kernel_dispatch_callee(op);
    command.arguments =
        loom_cmd_schedule_value_slice(loom_kernel_dispatch_arguments(op));
    command.workgroup_counts = loom_cmd_schedule_value_slice(
        loom_kernel_dispatch_workgroup_counts(op));
    IREE_ASSERT_GT(command.workgroup_counts.count, 0u);
    const loom_type_t first_count_type = loom_module_value_type(
        build->module, command.workgroup_counts.values[0]);
    command.kind = loom_type_is_view(first_count_type)
                       ? LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_INDIRECT
                       : LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT;
    command.source_arguments = command.arguments;
  }
  if (command.kind != LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP &&
      !iree_host_size_checked_add(build->argument_value_count,
                                  command.arguments.count,
                                  &build->argument_value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command schedule argument table is too large");
  }
  IREE_RETURN_IF_ERROR(loom_cmd_schedule_reserve_commands(build, 1));
  build->commands[build->command_count++] = (loom_cmd_schedule_command_build_t){
      .command = command,
      .wave_index = loom_cmd_schedule_child_base_wave(frame),
  };
  loom_cmd_schedule_commit_child_span(frame, 1);
  return iree_ok_status();
}

static iree_status_t loom_cmd_schedule_append_allocation(
    loom_cmd_schedule_build_t* build, const loom_cmd_schedule_frame_t* frame,
    const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_cmd_schedule_reserve_allocations(build, 1));
  build->allocations[build->allocation_count++] =
      (loom_cmd_schedule_allocation_t){
          .op = op,
          .definition_wave = loom_cmd_schedule_child_base_wave(frame),
      };
  return iree_ok_status();
}

static bool loom_cmd_schedule_is_common_terminator(const loom_op_t* op) {
  // Keep the three source terminators visible to this hot operation-kind
  // dispatch so Clang can form dense command and kernel jump tables. The
  // semantic trait fallback below admits target extensions.
  return loom_command_return_isa(op) || loom_command_yield_isa(op) ||
         loom_kernel_launch_yield_isa(op);
}

static iree_status_t loom_cmd_schedule_build_commands(
    loom_cmd_schedule_build_t* build, loom_region_t* program_body,
    iree_host_size_t* out_wave_count) {
  *out_wave_count = 0;
  build->frame_capacity = LOOM_CMD_SCHEDULE_INITIAL_CAPACITY;
  build->command_capacity = LOOM_CMD_SCHEDULE_INITIAL_CAPACITY;
  build->allocation_capacity = LOOM_CMD_SCHEDULE_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, build->frame_capacity, sizeof(*build->frames),
      (void**)&build->frames));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, build->command_capacity, sizeof(*build->commands),
      (void**)&build->commands));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, build->allocation_capacity, sizeof(*build->allocations),
      (void**)&build->allocations));
  IREE_RETURN_IF_ERROR(
      loom_cmd_schedule_push_frame(build, program_body, /*base_wave=*/0,
                                   LOOM_COMMAND_COMPOSITION_MODE_SERIAL));

  while (build->frame_count != 0) {
    loom_cmd_schedule_frame_t* frame = &build->frames[build->frame_count - 1];
    if (frame->next_op == NULL) {
      const iree_host_size_t completed_span = frame->span;
      --build->frame_count;
      if (build->frame_count == 0) {
        *out_wave_count = completed_span;
      } else {
        loom_cmd_schedule_commit_child_span(
            &build->frames[build->frame_count - 1], completed_span);
      }
      continue;
    }

    loom_op_t* op = frame->next_op;
    frame->next_op = op->next_op;
    if (op->flags & LOOM_OP_FLAG_DEAD) {
      continue;
    }
    if (loom_cmd_schedule_is_common_terminator(op)) {
      continue;
    }
    loom_region_t* child_region = NULL;
    loom_command_composition_mode_t child_mode =
        LOOM_COMMAND_COMPOSITION_MODE_SERIAL;
    if (loom_command_composition_is_serial_schedule(op)) {
      child_region = loom_op_regions(op)[0];
    } else if (loom_command_composition_is_concurrent_schedule(op)) {
      child_region = loom_op_regions(op)[0];
      child_mode = LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT;
    } else if (loom_kernel_dispatch_isa(op) || loom_kernel_launch_isa(op) ||
               loom_command_timestamp_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_cmd_schedule_append_command(build, frame, op));
      continue;
    } else if (loom_buffer_alloca_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_cmd_schedule_append_allocation(build, frame, op));
      continue;
    } else if (op->region_count == 0 &&
               iree_any_bit_set(loom_op_effective_traits(build->module, op),
                                LOOM_TRAIT_PURE | LOOM_TRAIT_TERMINATOR)) {
      // Pure leaf dataflow may feed dispatch metadata but does not itself emit
      // a command. Later placement consumes its resolved SSA facts directly.
      // The terminator bit admits regionless target extension terminators not
      // covered by the common source fast path above.
      continue;
    } else {
      const iree_string_view_t op_name = loom_op_name(build->module, op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "source operation `%.*s` is not representable in a portable "
          "command schedule",
          (int)op_name.size, op_name.data);
    }

    const iree_host_size_t child_base_wave =
        loom_cmd_schedule_child_base_wave(frame);
    IREE_RETURN_IF_ERROR(loom_cmd_schedule_push_frame(
        build, child_region, child_base_wave, child_mode));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_schedule_group_waves(
    loom_cmd_schedule_build_t* build, iree_host_size_t wave_count,
    loom_cmd_schedule_plan_t* out_plan) {
  loom_cmd_schedule_wave_t* waves = NULL;
  loom_cmd_schedule_command_t* commands = NULL;
  iree_host_size_t* write_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, wave_count, sizeof(*waves), (void**)&waves));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(build->arena, build->command_count,
                                sizeof(*commands), (void**)&commands));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(build->arena, wave_count,
                                                 sizeof(*write_offsets),
                                                 (void**)&write_offsets));
  memset(waves, 0, wave_count * sizeof(*waves));

  for (iree_host_size_t i = 0; i < build->command_count; ++i) {
    const iree_host_size_t wave_index = build->commands[i].wave_index;
    IREE_ASSERT_LT(wave_index, wave_count);
    ++waves[wave_index].command_count;
  }
  iree_host_size_t command_offset = 0;
  for (iree_host_size_t i = 0; i < wave_count; ++i) {
    waves[i].command_offset = command_offset;
    write_offsets[i] = command_offset;
    command_offset += waves[i].command_count;
  }
  IREE_ASSERT_EQ(command_offset, build->command_count);
  for (iree_host_size_t i = 0; i < build->command_count; ++i) {
    const loom_cmd_schedule_command_build_t command = build->commands[i];
    commands[write_offsets[command.wave_index]++] = command.command;
  }

  *out_plan = (loom_cmd_schedule_plan_t){
      .commands = commands,
      .command_count = build->command_count,
      .argument_value_count = build->argument_value_count,
      .waves = waves,
      .wave_count = wave_count,
      .allocations = build->allocations,
      .allocation_count = build->allocation_count,
  };
  return iree_ok_status();
}

iree_status_t loom_cmd_schedule_plan_build(const loom_module_t* module,
                                           loom_region_t* program_body,
                                           iree_arena_allocator_t* arena,
                                           loom_cmd_schedule_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(program_body);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_cmd_schedule_plan_t){0};

  loom_cmd_schedule_build_t build = {
      .module = module,
      .arena = arena,
  };
  iree_host_size_t wave_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_cmd_schedule_build_commands(&build, program_body, &wave_count));
  if (wave_count == 0) {
    return iree_ok_status();
  }
  return loom_cmd_schedule_group_waves(&build, wave_count, out_plan);
}
