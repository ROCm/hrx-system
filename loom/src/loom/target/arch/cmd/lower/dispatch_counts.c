// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/dispatch_counts.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/cmd/program.h"

typedef struct loom_cmd_dispatch_count_build_t {
  // Source module owning schedule operations and values.
  const loom_module_t* module;
  // Facts already computed for the command root.
  const loom_value_fact_table_t* facts;
  // Diagnostic sink for authored placement failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Cleared after the first authored placement failure.
  bool valid;
} loom_cmd_dispatch_count_build_t;

static iree_string_view_t loom_cmd_dispatch_count_symbol_name(
    const loom_cmd_dispatch_count_build_t* build,
    loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, build->module->symbols.count);
  const loom_string_id_t name_id =
      build->module->symbols.entries[symbol_ref.symbol_id].name_id;
  IREE_ASSERT_LT(name_id, build->module->strings.count);
  return build->module->strings.entries[name_id];
}

static bool loom_cmd_dispatch_count_exact_u32(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    uint32_t* out_value, bool* out_is_exact) {
  *out_value = 0;
  *out_is_exact = false;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(facts, value_id), &value)) {
    return true;
  }
  *out_is_exact = true;
  if (value < 0 || value > UINT32_MAX) return false;
  *out_value = (uint32_t)value;
  return true;
}

static iree_status_t loom_cmd_dispatch_count_emit_direct_error(
    loom_cmd_dispatch_count_build_t* build,
    const loom_cmd_schedule_command_t* command, uint32_t dimension,
    iree_string_view_t requirement) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_cmd_dispatch_count_symbol_name(build, command->callee)),
      loom_param_with_field_ref(
          loom_param_u32(dimension),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, dimension)),
      loom_param_string(requirement),
  };
  const loom_diagnostic_emission_t emission = {
      .module = build->module,
      .op = command->source_op,
      .error = LOOM_ERR_LOWERING_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  build->valid = false;
  return iree_diagnostic_emit(build->diagnostic_emitter, &emission);
}

static iree_status_t loom_cmd_dispatch_count_record_direct(
    loom_cmd_dispatch_count_build_t* build,
    const loom_cmd_schedule_command_t* command,
    loom_cmd_dispatch_count_t* out_count) {
  IREE_ASSERT_EQ(command->kind,
                 LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT);
  const loom_cmd_schedule_value_slice_t counts = command->workgroup_counts;
  IREE_ASSERT_GE(counts.count, 1u);
  IREE_ASSERT_LE(counts.count, 3u);
  loom_target_dispatch_workgroup_count_t direct = {
      .x = 1,
      .y = 1,
      .z = 1,
  };
  uint32_t* direct_values[] = {&direct.x, &direct.y, &direct.z};
  for (uint16_t i = 0; i < counts.count; ++i) {
    bool is_exact = false;
    if (!loom_cmd_dispatch_count_exact_u32(build->facts, counts.values[i],
                                           direct_values[i], &is_exact)) {
      return loom_cmd_dispatch_count_emit_direct_error(
          build, command, i, IREE_SV("within the unsigned 32-bit range"));
    }
    if (!is_exact) {
      return loom_cmd_dispatch_count_emit_direct_error(
          build, command, i, IREE_SV("an exact unsigned 32-bit value"));
    }
  }
  *out_count = (loom_cmd_dispatch_count_t){
      .kind = LOOM_CMD_DISPATCH_COUNT_KIND_DIRECT,
      .payload.direct = direct,
  };
  return iree_ok_status();
}

typedef enum loom_cmd_dispatch_count_indirect_origin_e {
  LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_UNRESOLVED = 0,
  LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_STATIC = 1,
  LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_DYNAMIC = 2,
} loom_cmd_dispatch_count_indirect_origin_t;

static loom_cmd_dispatch_count_indirect_origin_t
loom_cmd_dispatch_count_classify_indirect_origin(
    const loom_cmd_dispatch_count_build_t* build,
    loom_value_id_t source_value) {
  IREE_ASSERT_LT(source_value, build->module->values.count);
  const loom_value_t* source = loom_module_value(build->module, source_value);
  if (!loom_value_is_block_arg(source) &&
      loom_command_parameter_isa(loom_value_def_op(source))) {
    return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_STATIC;
  }

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(build->facts, source_value);
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(&build->facts->context, facts,
                                             &reference)) {
    return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_UNRESOLVED;
  }
  int64_t byte_offset = 0;
  int64_t byte_length = 0;
  if (!loom_value_facts_as_exact_i64(reference.base_byte_offset,
                                     &byte_offset) ||
      !loom_value_facts_as_exact_i64(reference.footprint_byte_length,
                                     &byte_length) ||
      byte_offset < 0 ||
      byte_offset % LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT != 0 ||
      byte_length != LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH) {
    return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_UNRESOLVED;
  }

  IREE_ASSERT_LT(reference.root_value_id, build->module->values.count);
  const loom_value_t* root_value =
      loom_module_value(build->module, reference.root_value_id);
  if (loom_value_is_block_arg(root_value)) {
    return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_STATIC;
  }
  const loom_op_t* defining_op = loom_value_def_op(root_value);
  if (defining_op && loom_buffer_alloca_isa(defining_op)) {
    return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_DYNAMIC;
  }
  return LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_UNRESOLVED;
}

static iree_status_t loom_cmd_dispatch_count_emit_indirect_error(
    loom_cmd_dispatch_count_build_t* build,
    const loom_cmd_schedule_command_t* command, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, uint8_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .module = build->module,
      .op = command->source_op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  build->valid = false;
  return iree_diagnostic_emit(build->diagnostic_emitter, &emission);
}

static iree_status_t loom_cmd_dispatch_count_record_indirect(
    loom_cmd_dispatch_count_build_t* build,
    const loom_cmd_schedule_command_t* command, bool has_preceding_wave,
    loom_cmd_dispatch_count_t* out_count) {
  IREE_ASSERT_EQ(command->kind,
                 LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_INDIRECT);
  const loom_cmd_schedule_value_slice_t counts = command->workgroup_counts;
  IREE_ASSERT_EQ(counts.count, 1u);
  const loom_value_id_t source_value = counts.values[0];
  const loom_cmd_dispatch_count_indirect_origin_t origin =
      loom_cmd_dispatch_count_classify_indirect_origin(build, source_value);
  const iree_string_view_t kernel_name =
      loom_cmd_dispatch_count_symbol_name(build, command->callee);
  if (origin == LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_UNRESOLVED) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(kernel_name),
        loom_param_with_field_ref(
            loom_param_string(IREE_SV("an exact aligned buffer range")),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
    };
    return loom_cmd_dispatch_count_emit_indirect_error(
        build, command, LOOM_ERR_LOWERING_054, params, IREE_ARRAYSIZE(params));
  }
  if (origin == LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_DYNAMIC &&
      !has_preceding_wave) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(kernel_name),
    };
    return loom_cmd_dispatch_count_emit_indirect_error(
        build, command, LOOM_ERR_LOWERING_053, params, IREE_ARRAYSIZE(params));
  }
  *out_count = (loom_cmd_dispatch_count_t){
      .kind = origin == LOOM_CMD_DISPATCH_COUNT_INDIRECT_ORIGIN_DYNAMIC
                  ? LOOM_CMD_DISPATCH_COUNT_KIND_INDIRECT_DYNAMIC
                  : LOOM_CMD_DISPATCH_COUNT_KIND_INDIRECT_STATIC,
      .payload.indirect_source_value = source_value,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_dispatch_count_emit_override_error(
    loom_cmd_dispatch_count_build_t* build,
    const loom_cmd_schedule_command_t* command) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_cmd_dispatch_count_symbol_name(build, command->callee)),
  };
  const loom_diagnostic_emission_t emission = {
      .module = build->module,
      .op = command->source_op,
      .error = LOOM_ERR_LOWERING_055,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  build->valid = false;
  return iree_diagnostic_emit(build->diagnostic_emitter, &emission);
}

iree_status_t loom_cmd_dispatch_count_table_build(
    const loom_module_t* module, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_cmd_dispatch_count_t** out_counts) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_valid);
  IREE_ASSERT_ARGUMENT(out_counts);
  *out_valid = false;
  *out_counts = NULL;

  loom_cmd_dispatch_count_t* counts = NULL;
  if (schedule->command_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->command_count, sizeof(*counts), (void**)&counts));
  }
  loom_cmd_dispatch_count_build_t build = {
      .module = module,
      .facts = facts,
      .diagnostic_emitter = diagnostic_emitter,
      .valid = true,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t wave_index = 0;
       wave_index < schedule->wave_count && iree_status_is_ok(status) &&
       build.valid;
       ++wave_index) {
    const loom_cmd_schedule_wave_t wave = schedule->waves[wave_index];
    for (iree_host_size_t i = 0;
         i < wave.command_count && iree_status_is_ok(status) && build.valid;
         ++i) {
      const iree_host_size_t command_index = wave.command_offset + i;
      const loom_cmd_schedule_command_t* command =
          &schedule->commands[command_index];
      IREE_ASSERT(loom_kernel_dispatch_isa(command->source_op));
      if (loom_kernel_dispatch_workgroup_size(command->source_op).count != 0) {
        status = loom_cmd_dispatch_count_emit_override_error(&build, command);
        continue;
      }
      switch (command->kind) {
        case LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT:
          status = loom_cmd_dispatch_count_record_direct(
              &build, command, &counts[command_index]);
          break;
        case LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_INDIRECT:
          status = loom_cmd_dispatch_count_record_indirect(
              &build, command, wave_index != 0, &counts[command_index]);
          break;
        default:
          IREE_ASSERT_UNREACHABLE("schedule contains only kernel dispatches");
          IREE_BUILTIN_UNREACHABLE();
      }
    }
  }

  if (iree_status_is_ok(status) && build.valid) {
    *out_valid = true;
    *out_counts = counts;
  }
  return status;
}
