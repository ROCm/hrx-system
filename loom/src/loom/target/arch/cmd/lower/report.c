// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/report.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static iree_string_view_t loom_cmd_report_function_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_symbol_ref_t ref =
      loom_func_like_callee(loom_func_like_const_cast(module, op));
  return loom_string_table_get(&module->strings,
                               module->symbols.entries[ref.symbol_id].name_id);
}

static loom_target_compile_report_command_kind_t loom_cmd_report_kind(
    loom_cmd_program_command_kind_t kind) {
  switch (loom_cmd_program_command_kind_base(kind)) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_DIRECT;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_STATIC;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_DYNAMIC;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP:
      return LOOM_TARGET_COMPILE_REPORT_COMMAND_TIMESTAMP;
    default:
      IREE_CHECK_UNREACHABLE("resolved command has an encoding kind");
  }
}

static loom_target_compile_report_command_buffer_role_t loom_cmd_report_role(
    loom_cmd_program_buffer_role_t role) {
  return role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED
             ? LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_FIXED
             : LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_REBINDABLE;
}

static loom_target_compile_report_command_buffer_t loom_cmd_report_buffer(
    const loom_cmd_program_buffer_ref_t* buffer) {
  return (loom_target_compile_report_command_buffer_t){
      .role = loom_cmd_report_role(buffer->role),
      .root_index = buffer->root_index,
      .byte_offset = buffer->byte_offset,
      .byte_length = buffer->byte_length,
  };
}

static void loom_cmd_report_command(const loom_cmd_program_plan_t* plan,
                                    const loom_cmd_program_root_t* root,
                                    const loom_cmd_program_tables_t* tables,
                                    const loom_cmd_program_command_t* command,
                                    loom_target_compile_report_command_t* row) {
  uint32_t entry_index = UINT32_MAX;
  switch (row->kind) {
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_DIRECT:
      entry_index = command->payload.dispatch_direct.entry_index;
      row->payload.workgroup_counts[0] =
          command->payload.dispatch_direct.workgroup_count_x;
      row->payload.workgroup_counts[1] =
          command->payload.dispatch_direct.workgroup_count_y;
      row->payload.workgroup_counts[2] =
          command->payload.dispatch_direct.workgroup_count_z;
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_STATIC:
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_DYNAMIC:
      entry_index = command->payload.dispatch_indirect.entry_index;
      row->payload.indirect_counts = loom_cmd_report_buffer(
          &tables->buffer_refs[command->payload.dispatch_indirect
                                   .workgroup_count_buffer_ref]);
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL:
      row->payload.fill.target = loom_cmd_report_buffer(
          &tables->buffer_refs[command->payload.fill.target_buffer_ref]);
      row->payload.fill.pattern = command->payload.fill.pattern;
      row->payload.fill.pattern_byte_count =
          command->payload.fill.pattern_length;
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY:
      row->payload.copy.source = loom_cmd_report_buffer(
          &tables->buffer_refs[command->payload.copy.source_buffer_ref]);
      row->payload.copy.target = loom_cmd_report_buffer(
          &tables->buffer_refs[command->payload.copy.target_buffer_ref]);
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_TIMESTAMP:
      row->payload.timestamp = loom_cmd_report_buffer(
          &tables->buffer_refs[command->payload.timestamp.target_buffer_ref]);
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER:
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT:
      IREE_ASSERT_UNREACHABLE("command kind is a diagnostic operation");
  }
  if (entry_index != UINT32_MAX) {
    row->entry_name = loom_cmd_report_function_name(
        plan->root_module,
        plan->entry_requirements[root->entry_requirement_indices[entry_index]]
            .declaration_op);
    row->argument_count =
        tables->entry_schemas[command->argument_schema_index].argument_count;
  }
}

iree_status_t loom_cmd_program_record_report(
    const loom_cmd_program_plan_t* plan, iree_host_size_t root_index,
    const loom_cmd_program_tables_t* tables, iree_arena_allocator_t* arena,
    loom_target_compile_report_t* report) {
  const loom_cmd_program_root_t* root = &plan->roots[root_index];
  const loom_cmd_program_requirements_t* requirements = &tables->requirements;
  loom_target_compile_report_command_plan_t observation = {
      .root_name =
          loom_cmd_report_function_name(plan->root_module, root->function_op),
      .fixed_buffer_count = requirements->fixed_buffer_count,
      .rebindable_binding_count = requirements->rebindable_binding_count,
      .executable_count = requirements->executable_count,
      .entry_count = requirements->entry_count,
      .transient = {requirements->transient.binding_index,
                    requirements->transient.required_byte_length,
                    requirements->transient.minimum_alignment},
      .launch_counts = {requirements->launch_counts.binding_index,
                        requirements->launch_counts.required_byte_length,
                        requirements->launch_counts.minimum_alignment},
      .command_count = tables->command_count,
      .binding_count = tables->launch_binding_count,
      .parameter_root_count = tables->parameter_root_count,
      .parameter_count = tables->parameter_count,
  };
  const bool details = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_COMMAND_PLAN_ROWS);
  loom_target_compile_report_command_t* commands = NULL;
  loom_target_compile_report_command_binding_t* bindings = NULL;
  loom_target_compile_report_command_storage_t* parameter_roots = NULL;
  loom_target_compile_report_command_parameter_t* parameters = NULL;
  if (details) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, observation.command_count,
                                  sizeof(*commands), (void**)&commands));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, observation.binding_count,
                                  sizeof(*bindings), (void**)&bindings));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, observation.parameter_root_count, sizeof(*parameter_roots),
        (void**)&parameter_roots));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, observation.parameter_count,
                                  sizeof(*parameters), (void**)&parameters));
  }
  for (uint32_t i = 0; i < tables->command_count; ++i) {
    const loom_cmd_program_command_t* command = &tables->commands[i];
    const loom_target_compile_report_command_kind_t kind =
        loom_cmd_report_kind(command->kind);
    ++observation.command_kind_counts[kind];
    const bool barrier =
        loom_cmd_program_command_kind_has_barrier(command->kind);
    observation.barrier_before_count += barrier;
    if (details) {
      commands[i] = (loom_target_compile_report_command_t){
          .kind = kind,
          .flags =
              barrier ? LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER_BEFORE : 0,
      };
      loom_cmd_report_command(plan, root, tables, command, &commands[i]);
    }
  }
  if (details) {
    for (uint32_t i = 0; i < tables->launch_binding_count; ++i) {
      bindings[i] = (loom_target_compile_report_command_binding_t){
          .role = loom_cmd_report_role(tables->launch_bindings[i].role),
          .root_index = tables->launch_bindings[i].root_index,
      };
    }
    for (uint32_t i = 0; i < tables->parameter_root_count; ++i) {
      const loom_cmd_program_parameter_root_t* source =
          &tables->parameter_roots[i];
      parameter_roots[i] = (loom_target_compile_report_command_storage_t){
          .root_index = source->fixed_buffer_index,
          .required_byte_length = source->required_byte_length,
          .minimum_alignment = source->minimum_alignment,
      };
    }
    for (uint32_t i = 0; i < tables->parameter_count; ++i) {
      const loom_cmd_program_parameter_t* source = &tables->parameters[i];
      parameters[i] = (loom_target_compile_report_command_parameter_t){
          .key = source->key,
          .root_index = source->fixed_buffer_index,
          .byte_offset = source->byte_offset,
          .byte_length = source->byte_length,
          .minimum_alignment = source->minimum_alignment,
      };
    }
  }
  observation.commands = commands;
  observation.bindings = bindings;
  observation.parameter_roots = parameter_roots;
  observation.parameters = parameters;
  return loom_target_compile_report_record_command_plan(report, &observation);
}
