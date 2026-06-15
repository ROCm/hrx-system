// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/interpreter.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/predicate.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"

typedef struct loom_pass_interpreter_epoch_t {
  // Number of module symbols at the observation point.
  iree_host_size_t symbol_count;
  // Number of module-level operations at the observation point.
  iree_host_size_t module_op_count;
  // Number of SSA values at the observation point.
  iree_host_size_t value_count;
  // Number of interned types at the observation point.
  iree_host_size_t type_count;
  // Number of encoding instances at the observation point.
  iree_host_size_t encoding_count;
} loom_pass_interpreter_epoch_t;

typedef struct loom_pass_interpreter_frame_t {
  // Active anchor kind.
  loom_pass_kind_t kind;
  // Current symbol when kind is LOOM_PASS_FUNCTION.
  loom_symbol_t* symbol;
  // Current function when kind is LOOM_PASS_FUNCTION.
  loom_func_like_t function;
} loom_pass_interpreter_frame_t;

typedef struct loom_pass_interpreter_state_t {
  // Program being executed.
  const loom_pass_program_t* program;
  // Mutable module receiving pass effects.
  loom_module_t* module;
  // Caller-supplied execution options.
  const loom_pass_interpreter_options_t* options;
  // Aggregate diagnostic result for the current program execution.
  loom_pass_run_result_t* result;
  // Next runtime pass invocation ordinal.
  iree_host_size_t next_invocation_ordinal;
  // Scoped value-fact workspace shared across pass invocations.
  loom_pass_value_fact_owner_t value_facts;
} loom_pass_interpreter_state_t;

typedef struct loom_pass_interpreter_diagnostic_counter_t {
  // Pass invocation receiving callback-local diagnostic counts.
  loom_pass_t* pass;
  // Caller-provided emitter that receives pass diagnostics.
  iree_diagnostic_emitter_t target;
  // Number of diagnostics emitted by the current pass invocation.
  iree_host_size_t emission_count;
  // Number of error diagnostics emitted by the current pass invocation.
  uint32_t error_count;
  // Number of warning diagnostics emitted by the current pass invocation.
  uint32_t warning_count;
  // Number of remark diagnostics emitted by the current pass invocation.
  uint32_t remark_count;
} loom_pass_interpreter_diagnostic_counter_t;

typedef struct loom_pass_interpreter_symbol_snapshot_entry_t {
  // Symbol entry captured before entering a pass.for body.
  loom_symbol_t* symbol;
  // Function-like view of symbol->defining_op.
  loom_func_like_t function;
} loom_pass_interpreter_symbol_snapshot_entry_t;

static const char* loom_pass_interpreter_anchor_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "func";
    default:
      return "unknown";
  }
}

static loom_pass_interpreter_epoch_t loom_pass_interpreter_epoch(
    const loom_module_t* module) {
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  return (loom_pass_interpreter_epoch_t){
      .symbol_count = module->symbols.count,
      .module_op_count = module_block ? module_block->op_count : 0,
      .value_count = module->values.count,
      .type_count = module->types.count,
      .encoding_count = module->encodings.count,
  };
}

static bool loom_pass_interpreter_epoch_changed(
    loom_pass_interpreter_epoch_t lhs, loom_pass_interpreter_epoch_t rhs) {
  return lhs.symbol_count != rhs.symbol_count ||
         lhs.module_op_count != rhs.module_op_count ||
         lhs.value_count != rhs.value_count ||
         lhs.type_count != rhs.type_count ||
         lhs.encoding_count != rhs.encoding_count;
}

static iree_string_view_t loom_pass_interpreter_symbol_name(
    const loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame) {
  if (!frame->symbol ||
      frame->symbol->name_id >= state->module->strings.count) {
    return IREE_SV("<none>");
  }
  return state->module->strings.entries[frame->symbol->name_id];
}

static iree_string_view_t loom_pass_interpreter_source_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t fallback) {
  if (!module || !loom_symbol_ref_is_valid(symbol_ref) ||
      symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return fallback;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return fallback;
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_pass_interpreter_count_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_pass_interpreter_diagnostic_counter_t* counter =
      (loom_pass_interpreter_diagnostic_counter_t*)user_data;
  ++counter->emission_count;
  switch (emission->error->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++counter->error_count;
      ++counter->pass->error_diagnostic_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++counter->warning_count;
      ++counter->pass->warning_diagnostic_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++counter->remark_count;
      ++counter->pass->remark_diagnostic_count;
      break;
    case LOOM_DIAGNOSTIC_COUNT_:
      break;
  }
  return iree_diagnostic_emit(counter->target, emission);
}

static bool loom_pass_interpreter_has_errors(
    const loom_pass_interpreter_state_t* state) {
  return state->result->error_count != 0;
}

static void loom_pass_interpreter_accumulate_diagnostics(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_diagnostic_counter_t* counter) {
  state->result->error_count += counter->error_count;
  state->result->warning_count += counter->warning_count;
  state->result->remark_count += counter->remark_count;
}

static iree_status_code_t loom_pass_interpreter_invocation_status_code(
    iree_status_code_t status_code,
    const loom_pass_interpreter_diagnostic_counter_t* counter) {
  if (status_code == IREE_STATUS_OK && counter->error_count != 0) {
    return IREE_STATUS_FAILED_PRECONDITION;
  }
  return status_code;
}

static iree_string_view_t loom_pass_interpreter_pipeline_name(
    const loom_pass_interpreter_state_t* state,
    const loom_pass_program_instruction_t* instruction) {
  const loom_op_t* pipeline_op = instruction->source.pipeline_op;
  if (!pipeline_op || !loom_pass_pipeline_isa(pipeline_op)) {
    return IREE_SV("<unknown>");
  }
  return loom_pass_interpreter_source_symbol_name(
      state->program->source_module, loom_pass_pipeline_symbol(pipeline_op),
      IREE_SV("<unknown>"));
}

static iree_status_t loom_pass_interpreter_make_pass(
    loom_pass_interpreter_state_t* state,
    const loom_pass_program_instruction_t* instruction,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* instance_arena, loom_pass_t* out_pass) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  *out_pass = (loom_pass_t){
      .info = invoke->info,
      .module_run = invoke->module_run,
      .instance_arena = instance_arena,
      .arena = instance_arena,
      .decoded_options = invoke->decoded_options,
      .diagnostic_emitter = diagnostic_emitter,
      .report = state->options->report,
      .environment = &state->options->environment,
      .value_facts = &state->value_facts,
  };
  const loom_pass_statistic_layout_t* statistic_layout =
      invoke->info->statistic_layout;
  if (!statistic_layout || statistic_layout->storage_size == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(instance_arena,
                                           statistic_layout->storage_size,
                                           &out_pass->statistic_storage));
  memset(out_pass->statistic_storage, 0, statistic_layout->storage_size);
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_emit_failure_diagnostic(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction) {
  if (!state->options->diagnostic_emitter.fn) {
    return iree_ok_status();
  }
  const loom_error_def_t* error = LOOM_ERR_STRUCTURE_028;
  if (!error) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "missing structured pass interpreter failure diagnostic");
  }
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(invoke->descriptor->key),
      loom_param_string(iree_make_cstring_view(
          loom_pass_interpreter_anchor_name(frame->kind))),
      loom_param_string(loom_pass_interpreter_symbol_name(state, frame)),
      loom_param_string(
          loom_pass_interpreter_pipeline_name(state, instruction)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = instruction->source.op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->diagnostic_emitter, &emission);
}

static iree_status_t loom_pass_interpreter_invoke_module(
    loom_pass_interpreter_state_t* state,
    const loom_pass_program_instruction_t* instruction, loom_pass_t* pass,
    bool* out_invocation_changed) {
  *out_invocation_changed = false;
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  loom_pass_interpreter_epoch_t before =
      loom_pass_interpreter_epoch(state->module);
  iree_status_t status = invoke->module_run(pass, state->module);
  loom_pass_interpreter_epoch_t after =
      loom_pass_interpreter_epoch(state->module);
  *out_invocation_changed =
      pass->changed || loom_pass_interpreter_epoch_changed(before, after);
  if (iree_status_is_ok(status)) {
    return iree_ok_status();
  }
  return status;
}

static iree_status_t loom_pass_interpreter_invoke_function(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, loom_pass_t* pass,
    bool* out_invocation_changed) {
  *out_invocation_changed = false;
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  if (!loom_func_like_isa(frame->function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function pass requires current function");
  }

  iree_arena_allocator_t function_arena;
  iree_arena_initialize(state->options->block_pool, &function_arena);
  pass->function_run = invoke->function_run;
  pass->arena = &function_arena;
  iree_arena_reset(&function_arena);

  loom_pass_interpreter_epoch_t before =
      loom_pass_interpreter_epoch(state->module);
  iree_status_t status =
      invoke->function_run(pass, state->module, frame->function);
  loom_pass_interpreter_epoch_t after =
      loom_pass_interpreter_epoch(state->module);

  pass->arena = pass->instance_arena;
  iree_arena_deinitialize(&function_arena);
  *out_invocation_changed =
      pass->changed || loom_pass_interpreter_epoch_changed(before, after);
  if (iree_status_is_ok(status)) {
    return iree_ok_status();
  }
  return status;
}

static iree_status_t loom_pass_interpreter_execute_range(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame, iree_host_size_t start,
    iree_host_size_t end, bool* out_changed);

static iree_status_t loom_pass_interpreter_invoke(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction,
    iree_host_size_t instruction_index, bool* out_changed) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  const iree_host_size_t invocation_ordinal = state->next_invocation_ordinal++;
  const iree_string_view_t pipeline_symbol =
      loom_pass_interpreter_pipeline_name(state, instruction);
  const iree_string_view_t symbol_name =
      loom_pass_interpreter_symbol_name(state, frame);
  iree_time_t start_time = 0;
  if (state->options->report) {
    start_time = iree_time_now();
  }
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(state->options->block_pool, &instance_arena);

  loom_pass_interpreter_diagnostic_counter_t diagnostic_counter = {
      .target = state->options->diagnostic_emitter,
  };
  iree_diagnostic_emitter_t pass_diagnostic_emitter = {
      .fn = loom_pass_interpreter_count_diagnostic,
      .user_data = &diagnostic_counter,
  };

  loom_pass_t pass = {0};
  iree_status_t trace_status = loom_pass_trace_emit(
      state->options->trace, &(loom_pass_trace_event_t){
                                 .module = state->module,
                                 .instruction = instruction,
                                 .instruction_index = instruction_index,
                                 .invocation_ordinal = invocation_ordinal,
                                 .pipeline_symbol = pipeline_symbol,
                                 .symbol_name = symbol_name,
                                 .anchor_kind = frame->kind,
                                 .point = LOOM_PASS_TRACE_POINT_BEFORE,
                                 .changed = false,
                                 .status_code = IREE_STATUS_OK,
                             });
  iree_status_t status = iree_ok_status();
  bool pass_execution_allowed = iree_status_is_ok(trace_status);
  if (pass_execution_allowed) {
    status = loom_pass_interpreter_make_pass(
        state, instruction, pass_diagnostic_emitter, &instance_arena, &pass);
  }
  diagnostic_counter.pass = &pass;

  bool created = false;
  if (iree_status_is_ok(status) && invoke->descriptor->create) {
    status = invoke->descriptor->create(&pass, iree_string_view_empty());
    created = iree_status_is_ok(status);
  }

  bool invocation_changed = false;
  if (iree_status_is_ok(status)) {
    if (invoke->info->kind == LOOM_PASS_MODULE) {
      status = loom_pass_interpreter_invoke_module(state, instruction, &pass,
                                                   &invocation_changed);
    } else if (invoke->info->kind == LOOM_PASS_FUNCTION) {
      status = loom_pass_interpreter_invoke_function(
          state, frame, instruction, &pass, &invocation_changed);
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported pass kind");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_changed |= invocation_changed;
  }
  if (invocation_changed) {
    loom_pass_value_fact_owner_invalidate(&state->value_facts);
  }

  if (created && invoke->descriptor->destroy) {
    invoke->descriptor->destroy(&pass);
  }

  iree_status_code_t status_code = loom_pass_interpreter_invocation_status_code(
      iree_status_code(status), &diagnostic_counter);
  if (iree_status_is_ok(trace_status)) {
    trace_status = loom_pass_trace_emit(
        state->options->trace,
        &(loom_pass_trace_event_t){
            .module = state->module,
            .instruction = instruction,
            .instruction_index = instruction_index,
            .invocation_ordinal = invocation_ordinal,
            .pipeline_symbol = pipeline_symbol,
            .symbol_name = symbol_name,
            .anchor_kind = frame->kind,
            .point = LOOM_PASS_TRACE_POINT_AFTER,
            .changed = invocation_changed,
            .status_code = status_code,
            .error_count = diagnostic_counter.error_count,
            .warning_count = diagnostic_counter.warning_count,
            .remark_count = diagnostic_counter.remark_count,
        });
  }

  iree_status_t report_status = iree_ok_status();
  loom_pass_interpreter_accumulate_diagnostics(state, &diagnostic_counter);

  if (state->options->report && pass_execution_allowed) {
    iree_duration_t duration_nanoseconds = iree_time_now() - start_time;
    report_status = loom_pass_report_append_invocation(
        state->options->report,
        &(loom_pass_report_invocation_options_t){
            .instruction = instruction,
            .instruction_index = instruction_index,
            .anchor_kind = frame->kind,
            .pipeline_symbol = pipeline_symbol,
            .symbol_name = symbol_name,
            .duration_nanoseconds = duration_nanoseconds,
            .changed = invocation_changed,
            .status_code = status_code,
            .statistic_storage = pass.statistic_storage,
            .detail_head = pass.report_detail_head,
            .detail_count = pass.report_detail_count,
        });
    if (iree_status_is_ok(report_status)) {
      pass.report_detail_head = NULL;
      pass.report_detail_tail = NULL;
      pass.report_detail_count = 0;
    }
  }

  const bool pass_failed = !iree_status_is_ok(status);
  loom_pass_report_discard_pass_details(&pass);
  iree_arena_deinitialize(&instance_arena);
  iree_status_t terminal_status = iree_status_join(status, trace_status);
  terminal_status = iree_status_join(terminal_status, report_status);
  if (pass_failed) {
    if (diagnostic_counter.emission_count == 0) {
      terminal_status = iree_status_join(
          terminal_status, loom_pass_interpreter_emit_failure_diagnostic(
                               state, frame, instruction));
    }
    return iree_status_annotate_f(
        terminal_status, "while executing pass '%.*s' at %s anchor on @%.*s",
        (int)invoke->descriptor->key.size, invoke->descriptor->key.data,
        loom_pass_interpreter_anchor_name(frame->kind), (int)symbol_name.size,
        symbol_name.data);
  }
  if (!iree_status_is_ok(terminal_status)) {
    return iree_status_annotate_f(
        terminal_status, "while tracing pass '%.*s' at %s anchor on @%.*s",
        (int)invoke->descriptor->key.size, invoke->descriptor->key.data,
        loom_pass_interpreter_anchor_name(frame->kind), (int)symbol_name.size,
        symbol_name.data);
  }
  return terminal_status;
}

static iree_status_t loom_pass_interpreter_build_function_snapshot(
    loom_pass_interpreter_state_t* state, iree_arena_allocator_t* arena,
    loom_pass_interpreter_symbol_snapshot_entry_t** out_entries,
    iree_host_size_t* out_count) {
  *out_entries = NULL;
  *out_count = 0;
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(state->module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }
    ++count;
  }
  if (count == 0) {
    return iree_ok_status();
  }

  loom_pass_interpreter_symbol_snapshot_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*entries),
                                                 (void**)&entries));
  iree_host_size_t entry_index = 0;
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(state->module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }
    entries[entry_index++] = (loom_pass_interpreter_symbol_snapshot_entry_t){
        .symbol = symbol,
        .function = function,
    };
  }
  *out_entries = entries;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_execute_for(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  if (frame->kind != LOOM_PASS_MODULE ||
      instruction->for_each_symbol.symbol_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.for execution anchor");
  }
  if (instruction->for_each_symbol.snapshot_kind !=
      LOOM_PASS_PROGRAM_SYMBOL_SNAPSHOT_FUNCTIONS_BY_SYMBOL_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.for snapshot kind");
  }

  iree_arena_allocator_t snapshot_arena;
  iree_arena_initialize(state->options->block_pool, &snapshot_arena);
  loom_pass_interpreter_symbol_snapshot_entry_t* entries = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = loom_pass_interpreter_build_function_snapshot(
      state, &snapshot_arena, &entries, &count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status) &&
                               !loom_pass_interpreter_has_errors(state);
       ++i) {
    loom_pass_interpreter_frame_t body_frame = {
        .kind = LOOM_PASS_FUNCTION,
        .symbol = entries[i].symbol,
        .function = entries[i].function,
    };
    bool body_changed = false;
    status = loom_pass_interpreter_execute_range(
        state, &body_frame, instruction->for_each_symbol.body_start,
        instruction->for_each_symbol.body_end, &body_changed);
    *out_changed |= body_changed;
  }
  iree_arena_deinitialize(&snapshot_arena);
  return status;
}

static iree_status_t loom_pass_interpreter_find_attr(
    loom_pass_program_attr_list_t attrs, iree_string_view_t name,
    const loom_pass_program_attr_t** out_attr) {
  *out_attr = NULL;
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    if (iree_string_view_equal(attrs.attrs[i].name, name)) {
      *out_attr = &attrs.attrs[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "missing pass predicate attr '%.*s'", (int)name.size,
                          name.data);
}

static const loom_pass_program_attr_t* loom_pass_interpreter_find_optional_attr(
    loom_pass_program_attr_list_t attrs, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    if (iree_string_view_equal(attrs.attrs[i].name, name)) {
      return &attrs.attrs[i];
    }
  }
  return NULL;
}

static const loom_op_t* loom_pass_interpreter_frame_op(
    const loom_pass_interpreter_frame_t* frame) {
  if (frame->kind == LOOM_PASS_FUNCTION &&
      loom_func_like_isa(frame->function)) {
    return frame->function.op;
  }
  return NULL;
}

static bool loom_pass_interpreter_lookup_attr_descriptor(
    const loom_module_t* module, const loom_op_t* op, iree_string_view_t name,
    const loom_attr_descriptor_t** out_descriptor, uint8_t* out_attr_index) {
  *out_descriptor = NULL;
  *out_attr_index = LOOM_ATTR_INDEX_NONE;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->attr_descriptors) {
    return false;
  }
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (iree_string_view_equal(loom_attr_descriptor_name(descriptor), name)) {
      *out_descriptor = descriptor;
      *out_attr_index = i;
      return true;
    }
  }
  return false;
}

static bool loom_pass_interpreter_attr_string_value_equal(
    const loom_pass_interpreter_state_t* state,
    const loom_attr_descriptor_t* descriptor, loom_attribute_t actual_attr,
    iree_string_view_t expected_string) {
  if (actual_attr.kind == LOOM_ATTR_STRING) {
    loom_string_id_t actual_string_id = loom_attr_as_string_id(actual_attr);
    if (actual_string_id >= state->module->strings.count) {
      return false;
    }
    return iree_string_view_equal(
        state->module->strings.entries[actual_string_id], expected_string);
  }
  if (actual_attr.kind == LOOM_ATTR_ENUM && descriptor &&
      descriptor->enum_case_names) {
    uint8_t enum_value = loom_attr_as_enum(actual_attr);
    if (enum_value >= descriptor->enum_case_count) {
      return false;
    }
    return iree_string_view_equal(
        loom_bstring_view(descriptor->enum_case_names[enum_value]),
        expected_string);
  }
  if (actual_attr.kind == LOOM_ATTR_SYMBOL) {
    iree_string_view_t symbol_name = loom_pass_interpreter_source_symbol_name(
        state->module, loom_attr_as_symbol(actual_attr), IREE_SV("<invalid>"));
    return iree_string_view_equal(symbol_name, expected_string);
  }
  return false;
}

static bool loom_pass_interpreter_attr_value_equal(
    const loom_pass_interpreter_state_t* state,
    const loom_attr_descriptor_t* descriptor, loom_attribute_t actual_attr,
    const loom_pass_program_attr_value_t* expected_value) {
  if (loom_attr_is_absent(actual_attr)) {
    return false;
  }
  switch (expected_value->kind) {
    case LOOM_ATTR_STRING:
      return loom_pass_interpreter_attr_string_value_equal(
          state, descriptor, actual_attr, expected_value->string_value);
    case LOOM_ATTR_I64:
      return actual_attr.kind == LOOM_ATTR_I64 &&
             loom_attr_as_i64(actual_attr) == expected_value->i64_value;
    case LOOM_ATTR_F64:
      return actual_attr.kind == LOOM_ATTR_F64 &&
             loom_attr_as_f64(actual_attr) == expected_value->f64_value;
    case LOOM_ATTR_BOOL:
      return actual_attr.kind == LOOM_ATTR_BOOL &&
             loom_attr_as_bool(actual_attr) == expected_value->bool_value;
    case LOOM_ATTR_ENUM:
      return actual_attr.kind == LOOM_ATTR_ENUM &&
             loom_attr_as_enum(actual_attr) == expected_value->enum_value;
    default:
      return false;
  }
}

static iree_status_t loom_pass_interpreter_evaluate_name_predicate(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_where_t* where, bool* out_match) {
  *out_match = false;
  const loom_pass_program_attr_t* value_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_find_attr(
      where->attrs, IREE_SV("value"), &value_attr));
  if (value_attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where name predicate requires string 'value' attr");
  }
  iree_string_view_t symbol_name =
      loom_pass_interpreter_symbol_name(state, frame);
  *out_match =
      iree_string_view_equal(symbol_name, value_attr->value.string_value);
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_evaluate_attr_predicate(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_where_t* where, bool* out_match) {
  *out_match = false;
  const loom_op_t* current_op = loom_pass_interpreter_frame_op(frame);
  if (!current_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where attr predicate requires current function");
  }
  const loom_pass_program_attr_t* name_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_find_attr(
      where->attrs, IREE_SV("name"), &name_attr));
  if (name_attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where attr predicate requires string 'name' attr");
  }

  const loom_attr_descriptor_t* descriptor = NULL;
  uint8_t attr_index = LOOM_ATTR_INDEX_NONE;
  if (!loom_pass_interpreter_lookup_attr_descriptor(
          state->module, current_op, name_attr->value.string_value, &descriptor,
          &attr_index)) {
    return iree_ok_status();
  }
  loom_attribute_t actual_attr = loom_op_attrs(current_op)[attr_index];
  if (loom_attr_is_absent(actual_attr)) {
    return iree_ok_status();
  }

  const loom_pass_program_attr_t* value_attr =
      loom_pass_interpreter_find_optional_attr(where->attrs, IREE_SV("value"));
  if (!value_attr) {
    *out_match = true;
    return iree_ok_status();
  }
  *out_match = loom_pass_interpreter_attr_value_equal(
      state, descriptor, actual_attr, &value_attr->value);
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_evaluate_trait_predicate(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_where_t* where, bool* out_match) {
  *out_match = false;
  const loom_op_t* current_op = loom_pass_interpreter_frame_op(frame);
  if (!current_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where trait predicate requires current function");
  }
  const loom_pass_program_attr_t* name_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_find_attr(
      where->attrs, IREE_SV("name"), &name_attr));
  if (name_attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where trait predicate requires string 'name' attr");
  }
  loom_trait_flags_t trait_flag = 0;
  if (!loom_pass_predicate_lookup_trait(name_attr->value.string_value,
                                        &trait_flag)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown pass.where trait '%.*s'",
                            (int)name_attr->value.string_value.size,
                            name_attr->value.string_value.data);
  }
  *out_match =
      (loom_op_effective_traits(state->module, current_op) & trait_flag) != 0;
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_evaluate_provider_predicate(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_match) {
  *out_match = false;
  if (!state->options->predicate_provider.evaluate) {
    const loom_pass_program_where_t* where = &instruction->where;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported pass.where predicate '%.*s'",
                            (int)where->predicate.size, where->predicate.data);
  }
  return state->options->predicate_provider.evaluate(
      state->options->predicate_provider.user_data,
      &(loom_pass_predicate_evaluate_context_t){
          .pipeline_module = state->program->source_module,
          .where_op = instruction->source.op,
          .anchor_kind = frame->kind,
          .predicate = instruction->where.predicate,
          .environment = &state->options->environment,
          .target_module = state->module,
          .symbol = frame->symbol,
          .function = frame->function,
      },
      out_match);
}

static iree_status_t loom_pass_interpreter_execute_where(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  const loom_pass_program_where_t* where = &instruction->where;
  bool match = false;
  if (iree_string_view_equal(where->predicate, IREE_SV("name"))) {
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_evaluate_name_predicate(
        state, frame, where, &match));
  } else if (iree_string_view_equal(where->predicate, IREE_SV("attr"))) {
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_evaluate_attr_predicate(
        state, frame, where, &match));
  } else if (iree_string_view_equal(where->predicate, IREE_SV("trait"))) {
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_evaluate_trait_predicate(
        state, frame, where, &match));
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_evaluate_provider_predicate(
        state, frame, instruction, &match));
  }
  if (!match) {
    return iree_ok_status();
  }
  return loom_pass_interpreter_execute_range(state, frame, where->body_start,
                                             where->body_end, out_changed);
}

static iree_status_t loom_pass_interpreter_execute_repeat(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  const loom_pass_program_repeat_t* repeat = &instruction->repeat;
  if (repeat->mode == LOOM_PASS_REPEAT_MODE_FIXED) {
    for (int64_t i = 0;
         i < repeat->count && !loom_pass_interpreter_has_errors(state); ++i) {
      bool body_changed = false;
      IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_range(
          state, frame, repeat->body_start, repeat->body_end, &body_changed));
      *out_changed |= body_changed;
    }
    return iree_ok_status();
  }
  if (repeat->mode != LOOM_PASS_REPEAT_MODE_UNTIL_CONVERGED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.repeat mode");
  }

  bool ever_changed = false;
  for (int64_t i = 0;
       i < repeat->max_iterations && !loom_pass_interpreter_has_errors(state);
       ++i) {
    bool body_changed = false;
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_range(
        state, frame, repeat->body_start, repeat->body_end, &body_changed));
    ever_changed |= body_changed;
    if (!body_changed) {
      *out_changed |= ever_changed;
      return iree_ok_status();
    }
  }
  *out_changed |= ever_changed;
  if (loom_pass_interpreter_has_errors(state)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      "pass.repeat<until_converged> exceeded max_iterations=%" PRId64,
      repeat->max_iterations);
}

static iree_status_t loom_pass_interpreter_execute_range(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame, iree_host_size_t start,
    iree_host_size_t end, bool* out_changed) {
  for (iree_host_size_t pc = start;
       pc < end && !loom_pass_interpreter_has_errors(state); ++pc) {
    const loom_pass_program_instruction_t* instruction =
        &state->program->instructions[pc];
    if (instruction->anchor_kind != frame->kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass program anchor mismatch at instruction %" PRIhsz, pc);
    }
    switch (instruction->kind) {
      case LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_invoke(
            state, frame, instruction, pc, out_changed));
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_for(
            state, frame, instruction, out_changed));
        pc = instruction->for_each_symbol.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_WHERE: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_where(
            state, frame, instruction, out_changed));
        pc = instruction->where.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_repeat(
            state, frame, instruction, out_changed));
        pc = instruction->repeat.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_FAIL:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s",
                                (int)instruction->message.message.size,
                                instruction->message.message.data);
      case LOOM_PASS_PROGRAM_INSTRUCTION_HALT:
        return iree_make_status(IREE_STATUS_ABORTED, "%.*s",
                                (int)instruction->message.message.size,
                                instruction->message.message.data);
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported pass program instruction");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_options_verify(
    const loom_pass_interpreter_options_t* options) {
  return loom_pass_environment_verify(&options->environment);
}

iree_status_t loom_pass_interpreter_run_module(
    const loom_pass_program_t* program, loom_module_t* module,
    const loom_pass_interpreter_options_t* options,
    loom_pass_run_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_options_verify(options));
  if (program->root_kind != LOOM_PASS_MODULE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected module-root pass program");
  }
  *out_result = (loom_pass_run_result_t){0};
  loom_pass_interpreter_state_t state = {
      .program = program,
      .module = module,
      .options = options,
      .result = out_result,
  };
  loom_pass_value_fact_owner_initialize(options->block_pool,
                                        &state.value_facts);
  loom_pass_interpreter_frame_t frame = {
      .kind = LOOM_PASS_MODULE,
  };
  bool changed = false;
  iree_status_t status = loom_pass_interpreter_execute_range(
      &state, &frame, 0, program->instruction_count, &changed);
  loom_pass_value_fact_owner_deinitialize(&state.value_facts);
  return status;
}

iree_status_t loom_pass_interpreter_run_function(
    const loom_pass_program_t* program, loom_module_t* module,
    loom_func_like_t function, const loom_pass_interpreter_options_t* options,
    loom_pass_run_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_options_verify(options));
  if (program->root_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected function-root pass program");
  }
  *out_result = (loom_pass_run_result_t){0};
  loom_pass_interpreter_state_t state = {
      .program = program,
      .module = module,
      .options = options,
      .result = out_result,
  };
  loom_pass_value_fact_owner_initialize(options->block_pool,
                                        &state.value_facts);
  loom_symbol_ref_t callee = loom_func_like_callee(function);
  loom_symbol_t* symbol = NULL;
  if (loom_symbol_ref_is_valid(callee) && callee.module_id == 0 &&
      callee.symbol_id < module->symbols.count) {
    symbol = &module->symbols.entries[callee.symbol_id];
  }
  loom_pass_interpreter_frame_t frame = {
      .kind = LOOM_PASS_FUNCTION,
      .symbol = symbol,
      .function = function,
  };
  bool changed = false;
  iree_status_t status = loom_pass_interpreter_execute_range(
      &state, &frame, 0, program->instruction_count, &changed);
  loom_pass_value_fact_owner_deinitialize(&state.value_facts);
  return status;
}
