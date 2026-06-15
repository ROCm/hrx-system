// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/archive_emitter.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/ireevm/descriptors/descriptors.h"
#include "loom/target/arch/ireevm/provider.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/emit/ireevm/module_plan.h"
#include "loom/target/entry_selection.h"
#include "loom/target/provider.h"
#include "loom/target/registers.h"

enum {
  LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS = 20,
};

typedef struct loom_ireevm_archive_emit_report_totals_t {
  // Total low schedule node count across emitted local functions.
  uint64_t schedule_node_count;
  // Total scheduled packet count across emitted local functions.
  uint64_t scheduled_node_count;
  // Total schedule dependency edge count across emitted local functions.
  uint64_t schedule_dependency_count;
  // Total descriptor resource-use count across emitted local functions.
  uint64_t schedule_resource_use_count;
  // Total required hazard-gap count across emitted local functions.
  uint64_t schedule_hazard_gap_count;
  // Total schedule model-summary count across emitted local functions.
  uint64_t schedule_model_summary_count;
  // Total register-pressure summary count across emitted local functions.
  uint64_t register_pressure_summary_count;
  // Maximum live register units observed across emitted local functions.
  uint64_t register_pressure_peak_live_units;
  // Total allocation assignment count across emitted local functions.
  uint64_t allocation_assignment_count;
  // Total spill assignment count across emitted local functions.
  uint64_t allocation_spill_count;
  // Total synthetic spill-plan count across emitted local functions.
  uint64_t allocation_spill_plan_count;
  // Total coalesced low.copy count across emitted local functions.
  uint64_t allocation_coalesced_copy_count;
  // Total materialized low.copy count across emitted local functions.
  uint64_t allocation_materialized_copy_count;
  // Total emitted VM bytecode opcode count across emitted local functions.
  uint64_t emitted_instruction_count;
  // Total semantic VM bytecode bytes across emitted local functions.
  uint64_t emitted_code_byte_count;
  // Total padded VM bytecode storage bytes across emitted local functions.
  uint64_t emitted_code_storage_byte_count;
} loom_ireevm_archive_emit_report_totals_t;

static iree_string_view_t loom_ireevm_archive_emit_module_name(
    const loom_ireevm_archive_emit_options_t* options) {
  if (options && !iree_string_view_is_empty(options->module_name)) {
    return options->module_name;
  }
  return IREE_SV("loom");
}

static iree_string_view_t loom_ireevm_archive_emit_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

typedef struct loom_ireevm_archive_emit_state_t {
  // Module being emitted.
  loom_module_t* module;
  // Caller-provided options, or NULL for defaults.
  const loom_ireevm_archive_emit_options_t* options;
  // Host allocator used for transient buffers and output ownership.
  iree_allocator_t allocator;
  // True after arena-backed state has been initialized.
  bool initialized;
  // Optional caller-owned structured compile report.
  loom_target_compile_report_t* report;
  // Target-neutral diagnostic options derived from |options|.
  loom_target_entry_options_t target_options;
  // Normalized diagnostic cap used by target phases.
  uint32_t max_errors;
  // Target provider environment used to compose target-owned registries.
  loom_target_environment_t target_environment;
  // Target-low descriptor registry used for verification and emission.
  loom_target_low_descriptor_registry_t low_registry;
  // Diagnostic materializer shared by target-low verification and emission.
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
  // Block pool backing all per-emission arena allocations.
  iree_arena_block_pool_t block_pool;
  // Arena for facts, frames, bytecode arrays, and archive table views.
  iree_arena_allocator_t table_arena;
  // Function bytecode objects emitted before archive wrapping.
  loom_ireevm_function_bytecode_t* bytecodes;
  // Number of initialized entries in |bytecodes|.
  uint32_t bytecode_count;
  // Aggregated compile-report summary for all emitted local functions.
  loom_ireevm_archive_emit_report_totals_t report_totals;
} loom_ireevm_archive_emit_state_t;

static void loom_ireevm_archive_emit_state_deinitialize(
    loom_ireevm_archive_emit_state_t* state) {
  for (uint32_t i = 0; i < state->bytecode_count; ++i) {
    loom_ireevm_function_bytecode_deinitialize(&state->bytecodes[i],
                                               state->allocator);
  }
  iree_arena_deinitialize(&state->table_arena);
  iree_arena_block_pool_deinitialize(&state->block_pool);
  loom_target_environment_deinitialize(&state->target_environment);
  state->initialized = false;
}

static iree_status_t loom_ireevm_archive_emit_state_initialize(
    loom_module_t* module, const loom_ireevm_archive_emit_options_t* options,
    iree_allocator_t allocator, loom_ireevm_archive_emit_state_t* out_state) {
  *out_state = (loom_ireevm_archive_emit_state_t){
      .module = module,
      .options = options,
      .allocator = allocator,
      .report = options ? options->report : NULL,
      .target_options =
          {
              .diagnostic_sink = options ? options->diagnostic_sink
                                         : (loom_diagnostic_sink_t){0},
              .source_resolver = options ? options->source_resolver
                                         : (loom_source_resolver_t){0},
              .max_errors = options ? options->max_errors : 0,
          },
  };
  out_state->max_errors = loom_target_entry_max_errors(
      &out_state->target_options, LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS);
  if (out_state->report != NULL) {
    loom_target_compile_report_initialize_if_empty(out_state->report,
                                                   allocator);
    out_state->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
    out_state->report->module_name =
        loom_ireevm_archive_emit_module_name(options);
  }

  iree_arena_block_pool_initialize(32 * 1024, allocator,
                                   &out_state->block_pool);
  iree_arena_initialize(&out_state->block_pool, &out_state->table_arena);
  out_state->initialized = true;

  iree_status_t status = loom_target_environment_initialize(
      &loom_ireevm_target_provider_set, &out_state->target_environment);
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        &out_state->target_environment, &out_state->low_registry);
  }
  if (iree_status_is_ok(status)) {
    loom_target_entry_diagnostic_emitter_initialize(
        module, &out_state->target_options, LOOM_EMITTER_VERIFIER,
        &out_state->diagnostic_emitter);
  } else {
    loom_ireevm_archive_emit_state_deinitialize(out_state);
  }
  return status;
}

static iree_status_t loom_ireevm_archive_emit_build_plan(
    loom_ireevm_archive_emit_state_t* state, bool* out_valid,
    loom_ireevm_module_plan_t* out_plan) {
  *out_valid = false;
  *out_plan = (loom_ireevm_module_plan_t){0};

  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(
      state->module, &state->target_options,
      LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(state->module);
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      state->module, &state->low_registry, &state->diagnostic_emitter,
      loom_target_selection_empty(), state->max_errors,
      loom_low_verify_provider_list_empty(), &low_verify_scratch,
      &low_verify_result));
  if (low_verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, &state->table_arena);
  bool plan_valid = false;
  IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_build(
      state->module, &fact_table,
      loom_target_entry_emitter(&state->diagnostic_emitter),
      &state->table_arena, &plan_valid, out_plan));
  if (!plan_valid) {
    return iree_ok_status();
  }
  if (out_plan->function_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IREE VM archive emission requires at least one VM low.func.def");
  }

  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_allocate_bytecodes(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan) {
  state->bytecodes = NULL;
  state->bytecode_count = 0;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->table_arena, plan->function_count, sizeof(*state->bytecodes),
      (void**)&state->bytecodes));
  memset(state->bytecodes, 0, plan->function_count * sizeof(*state->bytecodes));
  return iree_ok_status();
}

static void loom_ireevm_archive_emit_record_first_function(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_function_t* function) {
  if (state->report == NULL || state->bytecode_count != 0) {
    return;
  }
  loom_target_compile_report_record_target_bundle(
      state->report, &function->bundle_storage.bundle);
  state->report->lowered_symbol = function->symbol_name;
}

static void loom_ireevm_archive_emit_accumulate_frame_report(
    loom_ireevm_archive_emit_state_t* state,
    const loom_low_emission_frame_t* frame) {
  loom_ireevm_archive_emit_report_totals_t* totals = &state->report_totals;
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  uint64_t peak_live_units = 0;
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    peak_live_units = iree_max(peak_live_units,
                               liveness->pressure_summaries[i].peak_live_units);
  }
  totals->schedule_node_count += frame->schedule.node_count;
  totals->scheduled_node_count += frame->schedule.scheduled_node_count;
  totals->schedule_dependency_count += frame->schedule.dependency_count;
  totals->schedule_resource_use_count += frame->schedule.resource_use_count;
  totals->schedule_hazard_gap_count += frame->schedule.hazard_gap_count;
  totals->schedule_model_summary_count += frame->schedule.model_summary_count;
  totals->register_pressure_summary_count += liveness->pressure_summary_count;
  totals->register_pressure_peak_live_units =
      iree_max(totals->register_pressure_peak_live_units, peak_live_units);
  totals->allocation_assignment_count += frame->allocation.assignment_count;
  totals->allocation_spill_count += frame->allocation.spill_count;
  totals->allocation_spill_plan_count += frame->allocation.spill_plan_count;
  totals->allocation_coalesced_copy_count +=
      frame->allocation.coalesced_copy_count;
  totals->allocation_materialized_copy_count +=
      frame->allocation.materialized_copy_count;
}

static void loom_ireevm_archive_emit_accumulate_bytecode_report(
    loom_ireevm_archive_emit_state_t* state,
    const loom_low_emission_frame_t* frame,
    const loom_ireevm_function_bytecode_t* bytecode) {
  loom_ireevm_archive_emit_report_totals_t* totals = &state->report_totals;
  totals->emitted_instruction_count +=
      frame->schedule.scheduled_node_count + frame->schedule.block_count;
  totals->emitted_code_byte_count += bytecode->bytecode_length;
  totals->emitted_code_storage_byte_count += bytecode->data_length;
}

static void loom_ireevm_archive_emit_record_report_totals(
    loom_ireevm_archive_emit_state_t* state) {
  if (state->report == NULL) {
    return;
  }
  const loom_ireevm_archive_emit_report_totals_t* totals =
      &state->report_totals;
  loom_target_compile_report_record_schedule(
      state->report, totals->schedule_node_count, totals->scheduled_node_count,
      totals->schedule_dependency_count, totals->schedule_resource_use_count,
      totals->schedule_hazard_gap_count, totals->schedule_model_summary_count,
      totals->register_pressure_summary_count,
      totals->register_pressure_peak_live_units);
  loom_target_compile_report_record_allocation(
      state->report, totals->allocation_assignment_count,
      totals->allocation_spill_count, totals->allocation_spill_plan_count,
      totals->allocation_coalesced_copy_count,
      totals->allocation_materialized_copy_count);
  loom_target_compile_report_record_emission(
      state->report, totals->emitted_instruction_count,
      totals->emitted_code_byte_count, totals->emitted_code_storage_byte_count);
}

static iree_status_t loom_ireevm_archive_emit_function_argument_fixed_value(
    const loom_ireevm_archive_emit_state_t* state, loom_value_id_t argument_id,
    uint32_t* i32_register, uint32_t* ref_register,
    loom_low_allocation_fixed_value_t* out_fixed_value) {
  const loom_type_t type = loom_module_value_type(state->module, argument_id);
  if (!loom_low_type_is_register(type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "IREE VM ABI argument value %u is not a target-low register",
        (unsigned)argument_id);
  }

  const uint16_t register_class_id = loom_low_register_type_class_id(type);
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  loom_ireevm_register_class_layout_t layout = {0};
  iree_status_t status =
      loom_ireevm_register_class_layout(register_class_id, &layout);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status,
                                  "while mapping IREE VM ABI argument value %u",
                                  (unsigned)argument_id);
  }
  if (unit_count != layout.unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "IREE VM ABI argument value %u has unsupported register class "
        "'%.*s' with %u unit(s)",
        (unsigned)argument_id, (int)layout.class_name.size,
        layout.class_name.data, unit_count);
  }

  uint32_t* bank_register = layout.bank == LOOM_IREEVM_REGISTER_BANK_REF
                                ? ref_register
                                : i32_register;
  iree_host_size_t aligned_register = *bank_register;
  if (!iree_host_size_checked_align(aligned_register, layout.alignment,
                                    &aligned_register) ||
      aligned_register > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IREE VM ABI argument register base overflow");
  }
  *bank_register = (uint32_t)aligned_register;
  if (*bank_register > UINT32_MAX - layout.unit_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IREE VM ABI argument register span overflow");
  }
  const uint32_t location_base = *bank_register;
  *bank_register += layout.unit_count;
  *out_fixed_value = (loom_low_allocation_fixed_value_t){
      .value_id = argument_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      .location_base = location_base,
      .location_count = unit_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_function_fixed_values(
    loom_ireevm_archive_emit_state_t* state, loom_op_t* function_op,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count) {
  *out_fixed_values = NULL;
  *out_fixed_value_count = 0;

  loom_func_like_t function = loom_func_like_cast(state->module, function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "IREE VM archive emission expected a func-like low function");
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  if (argument_count == 0) {
    return iree_ok_status();
  }
  if (argument_ids == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "IREE VM archive emission found function arguments without IDs");
  }

  loom_low_allocation_fixed_value_t* fixed_values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->table_arena, argument_count,
                                sizeof(*fixed_values), (void**)&fixed_values));
  uint32_t i32_register = 0;
  uint32_t ref_register = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_function_argument_fixed_value(
        state, argument_ids[i], &i32_register, &ref_register,
        &fixed_values[i]));
  }

  *out_fixed_values = fixed_values;
  *out_fixed_value_count = argument_count;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_function(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan,
    const loom_ireevm_module_plan_function_t* function) {
  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_function_fixed_values(
      state, function->op, &fixed_values, &fixed_value_count));

  loom_low_emission_frame_t frame = {0};
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &state->low_registry.registry,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .emitter = loom_target_entry_emitter(&state->diagnostic_emitter),
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_NONE,
              .emitter = frame_options.emitter,
          },
  };
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      state->module, function->op, &frame_options, &spill_free_options,
      &state->table_arena, &frame));
  if (state->diagnostic_emitter.error_count != 0) {
    return iree_ok_status();
  }
  if (state->report != NULL) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_record_low_emission_frame(
        state->report, &frame));
    loom_ireevm_archive_emit_accumulate_frame_report(state, &frame);
  }

  loom_ireevm_function_bytecode_t* bytecode =
      &state->bytecodes[state->bytecode_count];
  IREE_RETURN_IF_ERROR(loom_ireevm_emit_function_bytecode(
      &frame.schedule, &frame.allocation, plan, state->allocator, bytecode));
  if (state->report != NULL) {
    loom_ireevm_archive_emit_accumulate_bytecode_report(state, &frame,
                                                        bytecode);
  }
  ++state->bytecode_count;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_functions(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_allocate_bytecodes(state, plan));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < plan->function_count && iree_status_is_ok(status);
       ++i) {
    const loom_ireevm_module_plan_function_t* function = &plan->functions[i];
    loom_ireevm_archive_emit_record_first_function(state, function);
    status = loom_ireevm_archive_emit_function(state, plan, function);
    if (state->diagnostic_emitter.error_count != 0) {
      break;
    }
  }
  return status;
}

static iree_status_t loom_ireevm_archive_emit_build_archive_imports(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan,
    const loom_ireevm_module_archive_import_t** out_imports) {
  *out_imports = NULL;
  if (plan->import_count == 0) {
    return iree_ok_status();
  }
  loom_ireevm_module_archive_import_t* imports = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->table_arena, plan->import_count,
                                sizeof(*imports), (void**)&imports));
  for (uint32_t i = 0; i < plan->import_count; ++i) {
    imports[i] = (loom_ireevm_module_archive_import_t){
        .full_name = plan->imports[i].full_name,
        .calling_convention = plan->imports[i].calling_convention,
    };
  }
  *out_imports = imports;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_build_archive_functions(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan,
    const loom_ireevm_module_archive_function_t** out_functions) {
  *out_functions = NULL;
  loom_ireevm_module_archive_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->table_arena, plan->function_count,
                                sizeof(*functions), (void**)&functions));
  for (uint32_t i = 0; i < plan->function_count; ++i) {
    functions[i] = (loom_ireevm_module_archive_function_t){
        .function_name = plan->functions[i].symbol_name,
        .calling_convention = plan->functions[i].calling_convention,
        .bytecode = &state->bytecodes[i],
    };
  }
  *out_functions = functions;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_build_archive_exports(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan,
    const loom_ireevm_module_archive_export_t** out_exports) {
  *out_exports = NULL;
  if (plan->export_count == 0) {
    return iree_ok_status();
  }
  loom_ireevm_module_archive_export_t* exports = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->table_arena, plan->export_count,
                                sizeof(*exports), (void**)&exports));
  for (uint32_t i = 0; i < plan->export_count; ++i) {
    exports[i] = (loom_ireevm_module_archive_export_t){
        .local_name = plan->exports[i].local_name,
        .internal_ordinal = plan->exports[i].internal_ordinal,
    };
  }
  *out_exports = exports;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_wrap_archive(
    loom_ireevm_archive_emit_state_t* state,
    const loom_ireevm_module_plan_t* plan,
    loom_ireevm_module_archive_t* out_archive) {
  const loom_ireevm_module_archive_import_t* imports = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_build_archive_imports(state, plan, &imports));
  const loom_ireevm_module_archive_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_build_archive_functions(
      state, plan, &functions));
  const loom_ireevm_module_archive_export_t* exports = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_build_archive_exports(state, plan, &exports));
  return loom_ireevm_emit_module_archive(
      loom_ireevm_archive_emit_module_name(state->options), imports,
      plan->import_count, functions, plan->function_count, exports,
      plan->export_count, state->allocator, out_archive);
}

static iree_status_t loom_ireevm_archive_emit(
    loom_ireevm_archive_emit_state_t* state,
    loom_ireevm_module_archive_t* out_archive, bool* out_emitted) {
  loom_ireevm_module_plan_t plan = {0};
  bool plan_valid = false;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_build_plan(state, &plan_valid, &plan));
  if (!plan_valid) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_functions(state, &plan));
  if (state->diagnostic_emitter.error_count != 0) {
    return iree_ok_status();
  }
  loom_ireevm_archive_emit_record_report_totals(state);
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_wrap_archive(state, &plan, out_archive));
  if (state->report != NULL) {
    loom_target_compile_report_record_artifact_size(state->report,
                                                    out_archive->data_length);
  }
  *out_emitted = true;
  return iree_ok_status();
}

iree_status_t loom_ireevm_emit_module_archive_from_ir(
    loom_module_t* module, const loom_ireevm_archive_emit_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_ireevm_module_archive_t* out_archive) {
  *out_emitted = false;
  *out_archive = (loom_ireevm_module_archive_t){0};

  loom_ireevm_archive_emit_state_t state = {0};
  iree_status_t status = loom_ireevm_archive_emit_state_initialize(
      module, options, allocator, &state);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_archive_emit(&state, out_archive, out_emitted);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  if (state.report != NULL) {
    loom_target_compile_report_record_status(state.report,
                                             iree_status_code(status));
  }
  if (state.initialized) {
    loom_ireevm_archive_emit_state_deinitialize(&state);
  }
  return status;
}
