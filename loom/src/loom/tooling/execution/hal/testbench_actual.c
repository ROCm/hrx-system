// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include <string.h>

#include "iree/tooling/device_util.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/compile/report_capture.h"
#include "loom/tooling/config/config.h"
#include "loom/util/fact_table.h"

typedef struct loom_run_hal_testbench_actual_sequence_span_t
    loom_run_hal_testbench_actual_sequence_span_t;
typedef struct loom_run_hal_testbench_actual_sequence_invocation_t
    loom_run_hal_testbench_actual_sequence_invocation_t;

struct loom_run_hal_testbench_actual_sequence_invocation_t {
  // HAL provider for this kernel launch.
  loom_run_hal_testbench_actual_provider_t* provider;
  // Contiguous kernel-launch span containing this invocation.
  loom_run_hal_testbench_actual_sequence_span_t* span;
};

struct loom_run_hal_testbench_actual_sequence_span_t {
  // Shared HAL context used by every provider in this span.
  loom_run_hal_testbench_context_t* context;
  // Prepared invocation entries indexed by span-local invocation ordinal.
  loom_run_hal_testbench_actual_sequence_invocation_t* invocations;
  // First source invocation ordinal represented by this span.
  iree_host_size_t first_invocation_index;
  // Number of contiguous kernel launches in this span.
  iree_host_size_t invocation_count;
  // Stable value-table payload addresses in flattened workload order.
  const loom_testbench_value_t** workload_values;
  // Number of entries in |workload_values|.
  iree_host_size_t workload_count;
  // Stable value-table payload addresses in flattened invocation input order.
  const loom_testbench_value_t** input_values;
  // Number of entries in |input_values|.
  iree_host_size_t input_count;
  // Buffer-valued entries from |input_values| in HAL binding order.
  const loom_testbench_value_t** binding_values;
  // Submission table populated from |binding_values| before each execution.
  iree_hal_buffer_binding_t* binding_table;
  // Binding byte lengths captured while recording a sample sequence.
  iree_device_size_t* binding_lengths;
  // Number of entries in each binding array.
  iree_host_size_t binding_count;
  // Reusable recording descriptors in invocation order.
  loom_run_hal_dispatch_sequence_step_t* steps;
  // Reusable command sequence indexed by case sample ordinal.
  loom_run_hal_dispatch_sequence_t* sample_sequences;
  // Number of entries in |sample_sequences|.
  iree_host_size_t sample_count;
  // Value-table slots currently addressed by |input_values|.
  const loom_testbench_value_slot_t* value_slots;
  // True once every provider has completed its compile attempt.
  bool providers_prepared;
  // True when one of the prepared providers rejected compilation.
  bool compile_rejected;
};

struct loom_run_hal_testbench_actual_sequence_execution_t {
  // Host allocator used for sequence execution storage.
  iree_allocator_t host_allocator;
  // Case plan whose kernel launches are mapped by this execution.
  const loom_testbench_case_plan_t* case_plan;
  // Prepared execution entries indexed by source invocation ordinal.
  loom_run_hal_testbench_actual_sequence_invocation_t* invocations;
  // Prepared contiguous kernel-launch spans.
  loom_run_hal_testbench_actual_sequence_span_t* spans;
  // Number of entries in |spans|.
  iree_host_size_t span_count;
};

void loom_run_hal_testbench_context_initialize(
    const loom_device_provider_registry_t* device_provider_registry,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_context_t* out_context) {
  *out_context = (loom_run_hal_testbench_context_t){
      .device_provider_registry = device_provider_registry,
      .host_allocator = iree_allocator_is_null(host_allocator)
                            ? iree_allocator_system()
                            : host_allocator,
      .device_event_sink = iree_hal_device_event_sink_stderr(),
  };
}

void loom_run_hal_testbench_context_set_device_event_sink(
    loom_run_hal_testbench_context_t* context,
    iree_hal_device_event_sink_t device_event_sink) {
  IREE_ASSERT(!context->runtime_initialized);
  context->device_event_sink = device_event_sink;
}

void loom_run_hal_testbench_context_set_runtime_sanitizer_options(
    loom_run_hal_testbench_context_t* context,
    const loom_sanitizer_options_t* sanitizer_options) {
  IREE_ASSERT(!context->runtime_initialized);
  context->runtime_sanitizer_options = *sanitizer_options;
  context->has_runtime_sanitizer_options = true;
}

void loom_run_hal_testbench_context_deinitialize(
    loom_run_hal_testbench_context_t* context) {
  if (context == NULL) {
    return;
  }
  if (context->runtime_initialized) {
    loom_run_hal_runtime_deinitialize(&context->runtime);
  }
  *context = (loom_run_hal_testbench_context_t){0};
}

static iree_status_t loom_run_hal_testbench_context_select_device_provider(
    loom_run_hal_testbench_context_t* context) {
  if (context->device_provider != NULL) {
    return iree_ok_status();
  }
  return loom_device_provider_registry_select(
      context->device_provider_registry, iree_hal_device_flag_list(),
      context->host_allocator, &context->device_provider);
}

iree_status_t loom_run_hal_testbench_context_validate_explicit_device(
    loom_run_hal_testbench_context_t* context) {
  if (iree_hal_device_flag_list().count == 0) {
    return iree_ok_status();
  }
  return loom_run_hal_testbench_context_select_device_provider(context);
}

iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context) {
  if (context->runtime_initialized) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_select_device_provider(context));
  loom_run_hal_runtime_options_t runtime_options;
  loom_run_hal_runtime_options_initialize(context->device_provider->driver_name,
                                          &runtime_options);
  runtime_options.event_sink = context->device_event_sink;
  loom_target_pipeline_options_t runtime_target_pipeline_options =
      context->device_provider->artifact_provider->default_pipeline_options;
  if (context->has_runtime_sanitizer_options) {
    runtime_target_pipeline_options.sanitizer =
        context->runtime_sanitizer_options;
  }
  runtime_options.runtime_features |=
      loom_run_hal_runtime_features_from_sanitizer_options(
          &runtime_target_pipeline_options.sanitizer);
  iree_status_t status = loom_run_hal_runtime_initialize(
      &runtime_options, context->host_allocator, &context->runtime);
  IREE_RETURN_IF_ERROR(status);
  context->runtime_initialized = true;
  return iree_ok_status();
}

iree_hal_buffer_params_t loom_run_hal_testbench_host_visible_buffer_params(
    void) {
  return (iree_hal_buffer_params_t){
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
}

static iree_status_t loom_run_hal_testbench_validate_kernel_launch(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation) {
  if (invocation->launch_schedule_depth > 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL execution does not yet support nested kernel launch schedules; "
        "launch in `%.*s` has schedule depth %" PRIhsz,
        (int)case_plan->name.size, case_plan->name.data,
        invocation->launch_schedule_depth);
  }
  IREE_ASSERT(invocation->kind == LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH);
  IREE_ASSERT(invocation->result_count == 0);
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_select_kernel_launch(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t** out_kernel_launch) {
  *out_kernel_launch = case_plan->first_kernel_launch;
  if (case_plan->kernel_launch_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL execution requires exactly one kernel launch in check.case "
        "`%.*s`; found %" PRIhsz,
        (int)case_plan->name.size, case_plan->name.data,
        case_plan->kernel_launch_count);
  }
  return loom_run_hal_testbench_validate_kernel_launch(case_plan,
                                                       *out_kernel_launch);
}

iree_status_t loom_run_hal_testbench_count_kernel_launches(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t* out_kernel_launch_count) {
  *out_kernel_launch_count = case_plan->kernel_launch_count;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_run_hal_testbench_validate_kernel_launch(case_plan, invocation));
  }
  return iree_ok_status();
}

void loom_run_hal_testbench_actual_provider_initialize(
    const loom_run_hal_testbench_actual_provider_options_t* options,
    loom_run_hal_testbench_actual_provider_t* out_provider) {
  *out_provider = (loom_run_hal_testbench_actual_provider_t){
      .context = options->context,
      .session = options->session,
      .target_environment = options->target_environment,
      .run_module = options->run_module,
      .pipeline = options->pipeline,
      .sanitizer = options->sanitizer,
      .config_set = options->config_set,
      .kernel_launch = options->kernel_launch,
      .diagnostic_sink = options->diagnostic_sink,
      .max_errors = options->max_errors,
      .report = options->report,
      .artifact_flags = options->artifact_flags,
      .artifact_manifest = options->artifact_manifest,
  };
  loom_run_hal_invocation_options_initialize(&out_provider->invocation_options);
}

void loom_run_hal_testbench_actual_provider_deinitialize(
    loom_run_hal_testbench_actual_provider_t* provider) {
  if (provider == NULL) {
    return;
  }
  if (provider->prepared_candidate_initialized) {
    loom_run_hal_prepared_candidate_deinitialize(&provider->prepared_candidate);
  }
  if (provider->candidate_initialized) {
    loom_run_hal_candidate_deinitialize(&provider->candidate);
  }
  loom_compile_pipeline_result_deinitialize(&provider->pipeline_result);
  loom_compile_pipeline_result_deinitialize(
      &provider->launch_config_pipeline_result);
  loom_module_free(provider->launch_config_module);
  if (provider->compile_module_initialized) {
    loom_run_module_deinitialize(&provider->compile_module);
  }
  if (provider->context != NULL) {
    iree_allocator_free(provider->context->host_allocator,
                        provider->workload_arguments);
    iree_allocator_free(provider->context->host_allocator,
                        provider->function_parameters);
  }
  *provider = (loom_run_hal_testbench_actual_provider_t){0};
}

static iree_status_t loom_run_hal_testbench_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u is outside the module symbol table",
                            (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u has an invalid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_resolve_func(
    loom_module_t* module, iree_string_view_t entry_symbol,
    loom_func_like_t* out_func) {
  *out_func = (loom_func_like_t){0};
  const loom_string_id_t entry_name_id =
      loom_module_lookup_string(module, entry_symbol);
  if (entry_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in selected "
                            "module string table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, entry_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in selected "
                            "module symbol table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry symbol '@%.*s' does not define a function",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  *out_func = func;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_materialize_config_set(
    loom_run_hal_testbench_actual_provider_t* provider) {
  loom_tooling_config_materialize_options_t options = {0};
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = provider->config_set;
  IREE_RETURN_IF_ERROR(loom_tooling_config_materialize_module(
      provider->compile_module.module, &options,
      loom_run_session_block_pool(provider->session), NULL));
  return loom_compile_report_record_materialized_config(
      provider->report, provider->compile_module.module, provider->config_set);
}

static iree_status_t loom_run_hal_testbench_clone_compile_module(
    loom_run_hal_testbench_actual_provider_t* provider) {
  const loom_module_t* source_module = provider->run_module->module;
  const loom_module_t* const source_modules[] = {source_module};
  iree_string_view_t module_name = iree_string_view_empty();
  if (source_module->name_id < source_module->strings.count) {
    module_name = source_module->strings.entries[source_module->name_id];
  }
  IREE_RETURN_IF_ERROR(loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = module_name,
      },
      loom_run_session_block_pool(provider->session),
      provider->context->host_allocator, &provider->compile_module.module));
  provider->compile_module_initialized = true;
  return iree_ok_status();
}

static void loom_run_hal_testbench_bind_compile_source(
    loom_run_hal_testbench_actual_provider_t* provider) {
  loom_run_module_t* compile_module = &provider->compile_module;
  const loom_run_module_t* run_module = provider->run_module;
  compile_module->filename = run_module->filename;
  compile_module->source = run_module->source;
  if (!run_module->has_source_entry) {
    return;
  }

  const iree_string_view_t source_filename = run_module->source_entry.filename;
  for (iree_host_size_t i = 0; i < compile_module->module->sources.count; ++i) {
    if (!iree_string_view_equal(compile_module->module->sources.entries[i],
                                source_filename)) {
      continue;
    }
    compile_module->source_entry = run_module->source_entry;
    compile_module->source_entry.source_id = (loom_source_id_t)i;
    compile_module->source_table_resolver = (loom_source_table_resolver_t){
        .entries = &compile_module->source_entry,
        .count = 1,
    };
    compile_module->has_source_entry = true;
    return;
  }
}

static iree_status_t loom_run_hal_testbench_link_selected_root(
    loom_run_hal_testbench_actual_provider_t* provider,
    const loom_module_t* source_module, iree_string_view_t entry_symbol,
    loom_module_t** out_module) {
  *out_module = NULL;
  const loom_module_t* const source_modules[] = {source_module};
  iree_string_view_t module_name = iree_string_view_empty();
  if (source_module->name_id < source_module->strings.count) {
    module_name = source_module->strings.entries[source_module->name_id];
  }
  const iree_string_view_t root_symbols[] = {entry_symbol};
  return loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = module_name,
          .root_symbols =
              {
                  .count = IREE_ARRAYSIZE(root_symbols),
                  .values = root_symbols,
              },
      },
      loom_run_session_block_pool(provider->session),
      provider->context->host_allocator, out_module);
}

static iree_status_t loom_run_hal_testbench_select_compile_root(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t entry_symbol) {
  const loom_module_t* source_module = provider->compile_module.module;
  loom_module_t* launch_config_module = NULL;
  iree_status_t status = loom_run_hal_testbench_link_selected_root(
      provider, source_module, entry_symbol, &launch_config_module);
  loom_module_t* compile_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_link_selected_root(
        provider, source_module, entry_symbol, &compile_module);
  }
  if (!iree_status_is_ok(status)) {
    loom_module_free(launch_config_module);
    return status;
  }

  loom_module_free(provider->compile_module.module);
  provider->compile_module.module = compile_module;
  provider->launch_config_module = launch_config_module;
  return iree_ok_status();
}

static void loom_run_hal_testbench_record_compile_rejection(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t stage, iree_string_view_t kind,
    iree_string_view_t message) {
  provider->compile_rejected = true;
  provider->compile_failure_stage = stage;
  provider->compile_failure_kind = kind;
  provider->compile_failure_message = message;
}

static iree_status_t loom_run_hal_testbench_forward_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_run_hal_testbench_actual_provider_t* provider =
      (loom_run_hal_testbench_actual_provider_t*)user_data;
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++provider->diagnostic_error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++provider->diagnostic_warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++provider->diagnostic_remark_count;
      break;
    default:
      break;
  }
  const loom_diagnostic_sink_t sink =
      provider->diagnostic_sink.fn != NULL
          ? provider->diagnostic_sink
          : (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink};
  return loom_diagnostic_emit(&sink, diagnostic);
}

static loom_diagnostic_sink_t loom_run_hal_testbench_counting_diagnostic_sink(
    loom_run_hal_testbench_actual_provider_t* provider) {
  return (loom_diagnostic_sink_t){
      .fn = loom_run_hal_testbench_forward_diagnostic,
      .user_data = provider,
  };
}

static uint32_t loom_run_hal_testbench_max_errors(
    const loom_run_hal_testbench_actual_provider_t* provider) {
  return provider->max_errors == 0 ? 20 : provider->max_errors;
}

static iree_status_t loom_run_hal_testbench_run_compile_pipeline(
    loom_run_hal_testbench_actual_provider_t* provider, loom_module_t* module,
    const loom_compile_pipeline_options_t* options, iree_string_view_t stage,
    loom_compile_pipeline_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_compile_run_pipeline(
      module, options, loom_run_session_block_pool(provider->session),
      out_result));
  if (out_result->pass.error_count != 0) {
    loom_run_hal_testbench_record_compile_rejection(
        provider, stage, IREE_SV("pass_diagnostics"), iree_string_view_empty());
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_resolve_target_requirement(
    loom_module_t* module, loom_func_like_t func,
    const loom_target_facts_t** out_target_requirement) {
  *out_target_requirement = NULL;

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, &module->arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module, loom_func_like_callee(func), &base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (function_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL testbench target selection requires a function symbol");
  }
  if (!loom_symbol_ref_is_valid(function_facts->target_symbol)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module, function_facts->target_symbol, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL testbench function target does not resolve to target facts");
  }
  *out_target_requirement = target_facts->projection;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_reflect_function_parameters(
    loom_run_hal_testbench_actual_provider_t* provider) {
  iree_hal_executable_t* executable = provider->prepared_candidate.executable;
  iree_string_view_t function_name =
      iree_string_view_trim(provider->invocation_options.function_name);
  if (iree_string_view_starts_with_char(function_name, '@')) {
    function_name = iree_string_view_remove_prefix(function_name, 1);
  }

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      executable, function_name, &function));
  iree_hal_executable_function_info_t function_info = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_function_info(executable, function, &function_info));

  // Some backends only reflect aggregate constant and binding counts and
  // therefore require source-type packing. A nonzero count is a complete ABI
  // contract and must match the source.
  if (function_info.parameter_count == 0) {
    return iree_ok_status();
  }
  if (function_info.parameter_count != provider->kernel_launch->input_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loaded HAL function '%.*s' reflects %u parameters for %" PRIhsz
        " source inputs",
        (int)function_name.size, function_name.data,
        (unsigned)function_info.parameter_count,
        provider->kernel_launch->input_count);
  }

  iree_hal_executable_function_parameter_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      provider->context->host_allocator, function_info.parameter_count,
      sizeof(*parameters), (void**)&parameters));
  iree_status_t status = iree_hal_executable_function_parameters(
      executable, function, function_info.parameter_count, parameters);
  if (iree_status_is_ok(status)) {
    provider->function_parameters = parameters;
    provider->function_parameter_count = function_info.parameter_count;
  } else {
    iree_allocator_free(provider->context->host_allocator, parameters);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_actual_provider_compile(
    loom_run_hal_testbench_actual_provider_t* provider) {
  if (provider->prepared_candidate_initialized || provider->compile_rejected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(provider->context));

  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_module_symbol_name_from_ref(
      provider->run_module->module, provider->kernel_launch->callee_ref,
      &entry_symbol));

  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_clone_compile_module(provider));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_materialize_config_set(provider));
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_select_compile_root(provider, entry_symbol));
  loom_run_hal_testbench_bind_compile_source(provider);

  if (provider->kernel_launch->workload_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(provider->context->host_allocator,
                                    provider->kernel_launch->workload_count,
                                    sizeof(*provider->workload_arguments),
                                    (void**)&provider->workload_arguments));
  }

  loom_func_like_t entry_func = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_func(
      provider->compile_module.module, entry_symbol, &entry_func));
  provider->invocation_options.function_name = entry_symbol;

  if (provider->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL kernel launches require a target environment");
  }

  if (!provider->compile_device_target_initialized) {
    const loom_device_provider_t* device_provider =
        provider->context->device_provider;
    const loom_target_facts_t* target_requirement = NULL;
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_target_requirement(
        provider->compile_module.module, entry_func, &target_requirement));
    IREE_RETURN_IF_ERROR(loom_device_provider_select_compatible_target(
        device_provider, &provider->context->runtime, target_requirement,
        &provider->compile_device_target));
    provider->compile_device_target_initialized = true;
  }

  loom_device_target_profile_t compile_target_profile = {0};
  IREE_RETURN_IF_ERROR(loom_device_target_profile_initialize(
      provider->context->device_provider, &provider->context->runtime,
      &provider->compile_device_target, &compile_target_profile));

  const loom_diagnostic_sink_t diagnostic_sink =
      loom_run_hal_testbench_counting_diagnostic_sink(provider);
  const uint32_t max_errors = loom_run_hal_testbench_max_errors(provider);

  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = provider->pipeline;
  pipeline_options.target_pipeline_options =
      provider->context->device_provider->artifact_provider
          ->default_pipeline_options;
  pipeline_options.target_pipeline_options.sanitizer = provider->sanitizer;
  pipeline_options.target_environment = provider->target_environment;
  const loom_target_specialization_request_t specialization_request = {
      .function_name = entry_symbol,
      .target_profile = &compile_target_profile.base,
  };
  pipeline_options.target_specializations =
      (loom_target_specialization_request_list_t){
          .values = &specialization_request,
          .count = 1,
      };
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(provider->session);
  pipeline_options.diagnostic_sink = diagnostic_sink;
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  pipeline_options.report = provider->report;

  // TODO(benvanik): Replace this independently prepared launch source with the
  // launch configuration artifact emitted by the device compilation.
  loom_compile_pipeline_options_t launch_config_pipeline_options =
      pipeline_options;
  launch_config_pipeline_options.pipeline = IREE_SV("default");
  launch_config_pipeline_options.default_pipeline =
      LOOM_COMPILE_DEFAULT_PIPELINE_EXPANDED_SOURCE;
  launch_config_pipeline_options.report = NULL;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_run_compile_pipeline(
      provider, provider->launch_config_module, &launch_config_pipeline_options,
      IREE_SV("launch_config"), &provider->launch_config_pipeline_result));
  if (provider->compile_rejected) {
    return iree_ok_status();
  }

  loom_func_like_t launch_config_func = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_func(
      provider->launch_config_module, entry_symbol, &launch_config_func));
  const loom_target_function_version_t* launch_config_function_version =
      loom_target_function_version_list_find(
          &provider->launch_config_pipeline_result.function_versions.list,
          launch_config_func);
  provider->launch_config_target_facts =
      launch_config_function_version->function_target_facts;

  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_run_compile_pipeline(
      provider, provider->compile_module.module, &pipeline_options,
      IREE_SV("compile"), &provider->pipeline_result));
  if (provider->compile_rejected) {
    return iree_ok_status();
  }

  loom_compile_options_t compile_options = {0};
  loom_compile_options_initialize(&compile_options);
  compile_options.function_versions =
      &provider->pipeline_result.function_versions.list;
  compile_options.target_pipeline_options =
      pipeline_options.target_pipeline_options;
  compile_options.diagnostic_sink = diagnostic_sink;
  compile_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  compile_options.max_errors = max_errors;
  compile_options.report = provider->report;
  compile_options.artifact_flags = provider->artifact_flags;
  compile_options.artifact_manifest = provider->artifact_manifest;

  provider->candidate_initialized = true;
  const iree_host_size_t emit_error_count = provider->diagnostic_error_count;
  iree_status_t status = loom_run_hal_candidate_emit_target(
      provider->context->device_provider, &provider->compile_device_target,
      &provider->compile_module, &compile_options,
      provider->context->host_allocator, &provider->candidate);
  provider->compile_report_available = true;
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (!provider->candidate.artifact_candidate.compiled) {
    if (provider->diagnostic_error_count != emit_error_count) {
      loom_run_hal_testbench_record_compile_rejection(
          provider, IREE_SV("emit"), IREE_SV("emit_diagnostics"),
          iree_string_view_empty());
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device provider '%.*s' did not emit an artifact or diagnostics",
        (int)provider->context->device_provider->artifact_provider->name.size,
        provider->context->device_provider->artifact_provider->name.data);
  }

  status = loom_run_hal_prepared_candidate_prepare(
      &provider->context->runtime, &provider->candidate.device_artifact,
      provider->context->host_allocator, &provider->prepared_candidate);
  if (iree_status_is_ok(status)) {
    provider->prepared_candidate_initialized = true;
    status = loom_run_hal_testbench_reflect_function_parameters(provider);
  }
  if (!iree_status_is_ok(status) && provider->prepared_candidate_initialized) {
    loom_run_hal_prepared_candidate_deinitialize(&provider->prepared_candidate);
    provider->prepared_candidate_initialized = false;
  }
  return status;
}

static iree_status_t loom_run_hal_testbench_invocation_options_push_constant(
    const loom_testbench_value_t* value, loom_type_t source_type,
    const iree_hal_executable_function_parameter_t* parameter,
    loom_run_hal_invocation_options_t* options) {
  if (!loom_testbench_value_is_scalar(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch constant input must be a scalar "
                            "value");
  }
  if (!loom_type_is_scalar(source_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch constant input must have a scalar "
                            "source type");
  }
  const loom_scalar_type_t source_scalar_type =
      loom_type_element_type(source_type);
  iree_host_size_t abi_word_count = 0;
  if (parameter != NULL) {
    if (parameter->type !=
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL executable parameter for scalar input is not a constant");
    }
    if (parameter->size == 0 ||
        (parameter->size % sizeof(options->constants[0])) != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL executable constant parameter has invalid byte length %u",
          (unsigned)parameter->size);
    }
    const iree_host_size_t expected_offset =
        options->constant_count * sizeof(options->constants[0]);
    if (parameter->offset != expected_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL executable constant parameter offset %u does not match next "
          "packed offset %" PRIhsz,
          (unsigned)parameter->offset, expected_offset);
    }
    abi_word_count = parameter->size / sizeof(options->constants[0]);
  }
  if (source_scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      source_scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    int64_t integer_value = 0;
    IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &integer_value));
    if (abi_word_count == 0) {
      abi_word_count = source_scalar_type == LOOM_SCALAR_TYPE_INDEX ? 1 : 2;
    }
    if (abi_word_count != 1 && abi_word_count != 2) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL dispatch %s constant has unsupported ABI word count %" PRIhsz,
          loom_scalar_type_name(source_scalar_type), abi_word_count);
    }
    if (abi_word_count == 1 &&
        (integer_value < INT32_MIN || integer_value > UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL dispatch %s constant value %" PRId64
                              " does not fit the 32-bit direct-constant ABI",
                              loom_scalar_type_name(source_scalar_type),
                              integer_value);
    }
    if (options->constant_count + abi_word_count >
        LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch constant count exceeds capacity "
          "%" PRIhsz,
          (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
    }
    const uint64_t raw_value = (uint64_t)integer_value;
    options->constants[options->constant_count++] = (uint32_t)raw_value;
    if (abi_word_count == 2) {
      options->constants[options->constant_count++] =
          (uint32_t)(raw_value >> 32);
    }
    return iree_ok_status();
  }

  uint32_t words[2] = {0};
  iree_host_size_t word_count = 0;
  IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
      &value->scalar, IREE_ARRAYSIZE(words), words, &word_count));
  if (abi_word_count != 0 && word_count != abi_word_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL dispatch %s constant requires %" PRIhsz
                            " reflected ABI words but materializes %" PRIhsz,
                            loom_scalar_type_name(source_scalar_type),
                            abi_word_count, word_count);
  }
  if (options->constant_count + word_count > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch constant count exceeds capacity "
                            "%" PRIhsz,
                            (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    options->constants[options->constant_count++] = words[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_validate_buffer_parameter(
    const iree_hal_executable_function_parameter_t* parameter,
    iree_host_size_t binding_ordinal) {
  if (parameter == NULL) {
    return iree_ok_status();
  }
  if (parameter->type != IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL executable parameter for buffer input is not a binding");
  }
  if (parameter->offset != binding_ordinal) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL executable binding ordinal %u does not match "
                            "next binding %" PRIhsz,
                            (unsigned)parameter->offset, binding_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_append_buffer_binding(
    const loom_testbench_value_t* input,
    const iree_hal_executable_function_parameter_t* parameter,
    loom_run_hal_binding_list_t* bindings) {
  if (!loom_testbench_value_is_buffer(input) ||
      input->buffer.kind == IREE_TOOLING_BUFFER_BINDING_KIND_NONE ||
      input->buffer.buffer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input buffer binding is invalid");
  }
  if (bindings->count >= bindings->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL invocation binding count exceeds capacity "
                            "%" PRIhsz,
                            bindings->capacity);
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_validate_buffer_parameter(
      parameter, bindings->count));
  iree_tooling_buffer_binding_t* binding = &bindings->values[bindings->count];
  *binding = input->buffer;
  iree_hal_buffer_retain(binding->buffer);
  iree_hal_buffer_view_retain(binding->buffer_view);
  ++bindings->count;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_input_append(
    loom_run_hal_binding_list_t* bindings, const loom_testbench_value_t* input,
    loom_type_t input_type,
    const iree_hal_executable_function_parameter_t* parameter,
    loom_run_hal_invocation_options_t* options) {
  if (loom_testbench_value_is_buffer(input)) {
    return loom_run_hal_testbench_append_buffer_binding(input, parameter,
                                                        bindings);
  }
  if (loom_testbench_value_is_scalar(input)) {
    return loom_run_hal_testbench_invocation_options_push_constant(
        input, input_type, parameter, options);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "HAL invocation input must be a buffer binding or "
                          "a scalar value");
}

iree_status_t loom_run_hal_testbench_invocation_inputs_from_values(
    const loom_testbench_value_t* inputs, const loom_type_t* input_types,
    const iree_hal_executable_function_parameter_t* input_parameters,
    iree_host_size_t input_count, loom_run_hal_invocation_options_t* options,
    iree_allocator_t allocator, loom_run_hal_binding_list_t* out_bindings) {
  if (input_count != 0 && (inputs == NULL || input_types == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input values and types are "
                            "required when input count is non-zero");
  }
  loom_run_hal_binding_list_initialize(out_bindings);
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      input_count, allocator, out_bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    const iree_hal_executable_function_parameter_t* parameter =
        input_parameters != NULL ? &input_parameters[i] : NULL;
    status = loom_run_hal_testbench_input_append(
        out_bindings, &inputs[i], input_types[i], parameter, options);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_bindings);
  }
  return status;
}

static iree_status_t loom_run_hal_testbench_evaluate_launch_config(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t workload_count,
    loom_run_hal_invocation_options_t* out_options) {
  if (workload_count != provider->kernel_launch->workload_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel launch workload count mismatch");
  }
  IREE_ASSERT(provider->launch_config_module != NULL);
  IREE_ASSERT(provider->launch_config_target_facts != NULL);

  const loom_kernel_launch_config_options_t options = {
      .function_symbol = provider->invocation_options.function_name,
      .workload_arguments = provider->workload_arguments,
      .workload_argument_count = workload_count,
      .required_fields = LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT,
      .function_target_facts = provider->launch_config_target_facts,
  };
  loom_kernel_launch_config_t config = {0};
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_evaluate(
      provider->launch_config_module,
      loom_run_session_block_pool(provider->session), &options, &config));
  if (loom_kernel_launch_config_has_failure(config.failure)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL launch configuration evaluation for '@%.*s' failed with code %u",
        (int)options.function_symbol.size, options.function_symbol.data,
        (unsigned)config.failure);
  }
  provider->resolved_launch_config = config;
  out_options->workgroup_count[0] = config.workgroup_count.x;
  out_options->workgroup_count[1] = config.workgroup_count.y;
  out_options->workgroup_count[2] = config.workgroup_count.z;
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  (void)out_results;
  loom_run_hal_testbench_actual_provider_t* provider =
      (loom_run_hal_testbench_actual_provider_t*)user_data;
  if (invocation != provider->kernel_launch) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual provider received an unexpected invocation");
  }
  IREE_ASSERT(result_count == 0);
  if (input_count != invocation->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel launch input count mismatch");
  }
  if (workload_count != invocation->workload_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel launch workload count mismatch");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_actual_provider_compile(provider));
  if (provider->compile_rejected) {
    return iree_ok_status();
  }

  loom_run_hal_invocation_options_t invocation_options =
      provider->invocation_options;
  for (iree_host_size_t i = 0; i < workload_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(
        &workloads[i], &provider->workload_arguments[i]));
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_evaluate_launch_config(
      provider, workload_count, &invocation_options));
  loom_run_hal_binding_list_t bindings = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      input_count, provider->context->host_allocator, &bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    const loom_value_id_t input_value_id =
        provider->kernel_launch->input_value_ids[i];
    const loom_type_t input_type =
        loom_module_value_type(provider->run_module->module, input_value_id);
    const iree_hal_executable_function_parameter_t* parameter =
        provider->function_parameter_count != 0
            ? &provider->function_parameters[i]
            : NULL;
    status = loom_run_hal_testbench_input_append(
        &bindings, &inputs[i], input_type, parameter, &invocation_options);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "preparing HAL actual input %" PRIhsz " for value ID %u", i,
          (unsigned)input_value_id);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(&bindings);
    return status;
  }

  loom_run_hal_invocation_plan_t plan = {0};
  loom_run_hal_iteration_t iteration = {0};
  status = loom_run_hal_invocation_plan_prepare_from_lists(
      &invocation_options, &bindings, /*expected_bindings=*/NULL,
      /*max_output_element_count=*/0, provider->context->host_allocator, &plan);
  loom_run_hal_binding_list_deinitialize(&bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_dispatch_plan(
        &provider->context->runtime, &provider->prepared_candidate, &plan,
        provider->context->host_allocator, &iteration);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
  return status;
}

static void loom_run_hal_testbench_actual_sequence_span_deinitialize(
    loom_run_hal_testbench_actual_sequence_span_t* span,
    iree_allocator_t host_allocator) {
  if (span == NULL) {
    return;
  }
  if (span->sample_sequences != NULL) {
    for (iree_host_size_t sample_index = 0; sample_index < span->sample_count;
         ++sample_index) {
      loom_run_hal_dispatch_sequence_deinitialize(
          &span->sample_sequences[sample_index]);
    }
  }
  iree_allocator_free(host_allocator, span->sample_sequences);
  iree_allocator_free(host_allocator, span->steps);
  iree_allocator_free(host_allocator, span->binding_lengths);
  iree_allocator_free(host_allocator, span->binding_table);
  iree_allocator_free(host_allocator, span->binding_values);
  iree_allocator_free(host_allocator, span->input_values);
  iree_allocator_free(host_allocator, span->workload_values);
  *span = (loom_run_hal_testbench_actual_sequence_span_t){0};
}

static iree_status_t loom_run_hal_testbench_actual_sequence_span_initialize(
    const loom_testbench_case_plan_t* case_plan,
    loom_run_hal_testbench_actual_sequence_invocation_t* invocations,
    iree_host_size_t first_invocation_index, iree_host_size_t invocation_count,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_actual_sequence_span_t* out_span) {
  *out_span = (loom_run_hal_testbench_actual_sequence_span_t){
      .context = invocations[0].provider->context,
      .invocations = invocations,
      .first_invocation_index = first_invocation_index,
      .invocation_count = invocation_count,
      .sample_count = case_plan->sample_count,
  };

  iree_host_size_t workload_count = 0;
  iree_host_size_t input_count = 0;
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t invocation_offset = 0;
       invocation_offset < invocation_count; ++invocation_offset) {
    const loom_run_hal_testbench_actual_provider_t* provider =
        invocations[invocation_offset].provider;
    IREE_ASSERT(provider != NULL);
    IREE_ASSERT(provider->context == out_span->context);
    const loom_testbench_invocation_plan_t* invocation =
        provider->kernel_launch;
    IREE_ASSERT(
        invocation ==
        &case_plan->invocations[first_invocation_index + invocation_offset]);
    if (!iree_host_size_checked_add(workload_count, invocation->workload_count,
                                    &workload_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL actual sequence workload count overflowed");
    }
    if (!iree_host_size_checked_add(input_count, invocation->input_count,
                                    &input_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL actual sequence input count overflowed");
    }
    iree_host_size_t step_binding_count = 0;
    for (iree_host_size_t input_index = 0;
         input_index < invocation->input_count; ++input_index) {
      const loom_type_t input_type = loom_module_value_type(
          invocation->module, invocation->input_value_ids[input_index]);
      if (loom_type_is_shaped(input_type)) {
        ++step_binding_count;
      } else if (!loom_type_is_scalar(input_type)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HAL kernel launch input %zu has unsupported type kind %u",
            input_index, (unsigned)loom_type_kind(input_type));
      }
    }
    if (step_binding_count > LOOM_RUN_HAL_MAX_BINDING_COUNT ||
        !iree_host_size_checked_add(binding_count, step_binding_count,
                                    &binding_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL actual sequence binding count exceeds supported limits");
    }
  }
  out_span->workload_count = workload_count;
  out_span->input_count = input_count;
  out_span->binding_count = binding_count;

  iree_status_t status = iree_ok_status();
  if (workload_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, workload_count,
                                         sizeof(*out_span->workload_values),
                                         (void**)&out_span->workload_values);
  }
  if (iree_status_is_ok(status) && input_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, input_count,
                                         sizeof(*out_span->input_values),
                                         (void**)&out_span->input_values);
  }
  if (iree_status_is_ok(status) && binding_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, binding_count,
                                         sizeof(*out_span->binding_values),
                                         (void**)&out_span->binding_values);
  }
  if (iree_status_is_ok(status) && binding_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, binding_count,
                                         sizeof(*out_span->binding_table),
                                         (void**)&out_span->binding_table);
  }
  if (iree_status_is_ok(status) && binding_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, binding_count,
                                         sizeof(*out_span->binding_lengths),
                                         (void**)&out_span->binding_lengths);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, invocation_count,
                                         sizeof(*out_span->steps),
                                         (void**)&out_span->steps);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, case_plan->sample_count,
                                    sizeof(*out_span->sample_sequences),
                                    (void**)&out_span->sample_sequences);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_actual_sequence_span_deinitialize(out_span,
                                                             host_allocator);
    return status;
  }
  if (workload_count != 0) {
    memset(out_span->workload_values, 0,
           workload_count * sizeof(*out_span->workload_values));
  }
  if (input_count != 0) {
    memset(out_span->input_values, 0,
           input_count * sizeof(*out_span->input_values));
  }
  if (binding_count != 0) {
    memset(out_span->binding_values, 0,
           binding_count * sizeof(*out_span->binding_values));
    memset(out_span->binding_table, 0,
           binding_count * sizeof(*out_span->binding_table));
    memset(out_span->binding_lengths, 0,
           binding_count * sizeof(*out_span->binding_lengths));
  }
  memset(out_span->steps, 0, invocation_count * sizeof(*out_span->steps));
  if (case_plan->sample_count != 0) {
    memset(out_span->sample_sequences, 0,
           case_plan->sample_count * sizeof(*out_span->sample_sequences));
  }

  iree_host_size_t binding_offset = 0;
  for (iree_host_size_t invocation_offset = 0;
       invocation_offset < invocation_count; ++invocation_offset) {
    loom_run_hal_testbench_actual_provider_t* provider =
        invocations[invocation_offset].provider;
    const loom_testbench_invocation_plan_t* invocation =
        provider->kernel_launch;
    iree_host_size_t step_binding_count = 0;
    for (iree_host_size_t input_index = 0;
         input_index < invocation->input_count; ++input_index) {
      const loom_type_t input_type = loom_module_value_type(
          invocation->module, invocation->input_value_ids[input_index]);
      step_binding_count += loom_type_is_shaped(input_type) ? 1 : 0;
    }
    out_span->steps[invocation_offset] =
        (loom_run_hal_dispatch_sequence_step_t){
            .candidate = &provider->prepared_candidate,
            .execution_epoch = invocation->execution_epoch,
            .binding_lengths = step_binding_count == 0
                                   ? NULL
                                   : &out_span->binding_lengths[binding_offset],
            .binding_count = step_binding_count,
        };
    binding_offset += step_binding_count;
  }
  return iree_ok_status();
}

void loom_run_hal_testbench_actual_sequence_execution_destroy(
    loom_run_hal_testbench_actual_sequence_execution_t* execution) {
  if (execution == NULL) {
    return;
  }
  const iree_allocator_t host_allocator = execution->host_allocator;
  for (iree_host_size_t span_index = 0; span_index < execution->span_count;
       ++span_index) {
    loom_run_hal_testbench_actual_sequence_span_deinitialize(
        &execution->spans[span_index], host_allocator);
  }
  iree_allocator_free(host_allocator, execution->spans);
  iree_allocator_free(host_allocator, execution->invocations);
  iree_allocator_free(host_allocator, execution);
}

iree_status_t loom_run_hal_testbench_actual_sequence_execution_create(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t provider_count,
    loom_run_hal_testbench_actual_provider_t* const* providers,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_actual_sequence_execution_t** out_execution) {
  *out_execution = NULL;
  loom_run_hal_testbench_actual_sequence_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  *execution = (loom_run_hal_testbench_actual_sequence_execution_t){
      .host_allocator = host_allocator,
      .case_plan = case_plan,
  };

  iree_status_t status = iree_ok_status();
  if (case_plan->invocation_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, case_plan->invocation_count,
        sizeof(*execution->invocations), (void**)&execution->invocations);
  }
  if (iree_status_is_ok(status) && case_plan->invocation_count != 0) {
    memset(execution->invocations, 0,
           case_plan->invocation_count * sizeof(*execution->invocations));
  }
  if (iree_status_is_ok(status) && provider_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, provider_count,
                                         sizeof(*execution->spans),
                                         (void**)&execution->spans);
  }
  if (iree_status_is_ok(status) && provider_count != 0) {
    memset(execution->spans, 0, provider_count * sizeof(*execution->spans));
  }

  iree_host_size_t provider_index = 0;
  for (iree_host_size_t invocation_index = 0;
       iree_status_is_ok(status) &&
       invocation_index < case_plan->invocation_count;) {
    if (case_plan->invocations[invocation_index].kind !=
        LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
      ++invocation_index;
      continue;
    }
    const iree_host_size_t first_invocation_index = invocation_index;
    do {
      IREE_ASSERT(provider_index < provider_count);
      loom_run_hal_testbench_actual_provider_t* provider =
          providers[provider_index++];
      IREE_ASSERT(provider != NULL);
      IREE_ASSERT(provider->kernel_launch ==
                  &case_plan->invocations[invocation_index]);
      execution->invocations[invocation_index].provider = provider;
      ++invocation_index;
    } while (invocation_index < case_plan->invocation_count &&
             case_plan->invocations[invocation_index].kind ==
                 LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH);
    loom_run_hal_testbench_actual_sequence_span_t* span =
        &execution->spans[execution->span_count++];
    status = loom_run_hal_testbench_actual_sequence_span_initialize(
        case_plan, &execution->invocations[first_invocation_index],
        first_invocation_index, invocation_index - first_invocation_index,
        host_allocator, span);
    for (iree_host_size_t i = first_invocation_index;
         iree_status_is_ok(status) && i < invocation_index; ++i) {
      execution->invocations[i].span = span;
    }
  }
  IREE_ASSERT(provider_index == provider_count);
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    loom_run_hal_testbench_actual_sequence_execution_destroy(execution);
  }
  return status;
}

static iree_host_size_t loom_run_hal_testbench_actual_sequence_invocation_index(
    const loom_run_hal_testbench_actual_sequence_execution_t* execution,
    const loom_testbench_invocation_plan_t* invocation) {
  IREE_ASSERT(invocation >= execution->case_plan->invocations);
  IREE_ASSERT(invocation < execution->case_plan->invocations +
                               execution->case_plan->invocation_count);
  return (iree_host_size_t)(invocation - execution->case_plan->invocations);
}

static iree_status_t loom_run_hal_testbench_actual_sequence_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  loom_run_hal_testbench_actual_sequence_execution_t* execution =
      (loom_run_hal_testbench_actual_sequence_execution_t*)user_data;
  const iree_host_size_t invocation_index =
      loom_run_hal_testbench_actual_sequence_invocation_index(execution,
                                                              invocation);
  loom_run_hal_testbench_actual_provider_t* provider =
      execution->invocations[invocation_index].provider;
  IREE_ASSERT(provider != NULL);
  return loom_run_hal_testbench_actual_invoke(
      provider, invocation, workload_count, workloads, input_count, inputs,
      result_count, out_results);
}

static iree_status_t loom_run_hal_testbench_actual_sequence_resolve_values(
    loom_run_hal_testbench_actual_sequence_span_t* span,
    const loom_testbench_value_table_t* table) {
  iree_host_size_t workload_offset = 0;
  iree_host_size_t input_offset = 0;
  iree_host_size_t binding_offset = 0;
  for (iree_host_size_t invocation_offset = 0;
       invocation_offset < span->invocation_count; ++invocation_offset) {
    const loom_testbench_invocation_plan_t* invocation =
        span->invocations[invocation_offset].provider->kernel_launch;
    for (iree_host_size_t workload_index = 0;
         workload_index < invocation->workload_count; ++workload_index) {
      const loom_testbench_value_t* workload = NULL;
      IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
          table, invocation->workload_value_ids[workload_index], &workload));
      span->workload_values[workload_offset++] = workload;
    }
    for (iree_host_size_t input_index = 0;
         input_index < invocation->input_count; ++input_index) {
      const loom_testbench_value_t* input = NULL;
      IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
          table, invocation->input_value_ids[input_index], &input));
      span->input_values[input_offset++] = input;
      const loom_type_t input_type = loom_module_value_type(
          invocation->module, invocation->input_value_ids[input_index]);
      if (loom_type_is_shaped(input_type)) {
        span->binding_values[binding_offset++] = input;
      }
    }
  }
  IREE_ASSERT(workload_offset == span->workload_count);
  IREE_ASSERT(input_offset == span->input_count);
  IREE_ASSERT(binding_offset == span->binding_count);
  span->value_slots = table->slots;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_actual_sequence_prepare_sample(
    loom_run_hal_testbench_actual_sequence_span_t* span,
    iree_host_size_t sample_ordinal) {
  iree_host_size_t workload_offset = 0;
  iree_host_size_t input_offset = 0;
  iree_host_size_t binding_offset = 0;
  for (iree_host_size_t invocation_offset = 0;
       invocation_offset < span->invocation_count; ++invocation_offset) {
    loom_run_hal_testbench_actual_provider_t* provider =
        span->invocations[invocation_offset].provider;
    loom_run_hal_dispatch_sequence_step_t* step =
        &span->steps[invocation_offset];
    step->options = provider->invocation_options;
    const loom_testbench_invocation_plan_t* invocation =
        provider->kernel_launch;
    for (iree_host_size_t workload_index = 0;
         workload_index < invocation->workload_count; ++workload_index) {
      IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(
          span->workload_values[workload_offset++],
          &provider->workload_arguments[workload_index]));
    }
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_evaluate_launch_config(
        provider, invocation->workload_count, &step->options));
    iree_host_size_t invocation_binding_ordinal = 0;
    for (iree_host_size_t input_index = 0;
         input_index < invocation->input_count; ++input_index) {
      const loom_testbench_value_t* input = span->input_values[input_offset++];
      const loom_type_t input_type = loom_module_value_type(
          invocation->module, invocation->input_value_ids[input_index]);
      const iree_hal_executable_function_parameter_t* parameter =
          provider->function_parameter_count != 0
              ? &provider->function_parameters[input_index]
              : NULL;
      if (loom_type_is_shaped(input_type)) {
        if (!loom_testbench_value_is_buffer(input) ||
            input->buffer.buffer == NULL) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "HAL kernel launch shaped input is not a buffer");
        }
        IREE_RETURN_IF_ERROR(loom_run_hal_testbench_validate_buffer_parameter(
            parameter, invocation_binding_ordinal));
        span->binding_lengths[binding_offset++] = input->buffer.byte_length;
        ++invocation_binding_ordinal;
      } else {
        IREE_RETURN_IF_ERROR(
            loom_run_hal_testbench_invocation_options_push_constant(
                input, input_type, parameter, &step->options));
      }
    }
  }
  IREE_ASSERT(workload_offset == span->workload_count);
  IREE_ASSERT(input_offset == span->input_count);
  IREE_ASSERT(binding_offset == span->binding_count);
  return loom_run_hal_dispatch_sequence_prepare(
      &span->context->runtime, span->invocation_count, span->steps,
      &span->sample_sequences[sample_ordinal]);
}

static void loom_run_hal_testbench_actual_sequence_populate_binding_table(
    loom_run_hal_testbench_actual_sequence_span_t* span) {
  for (iree_host_size_t binding_index = 0; binding_index < span->binding_count;
       ++binding_index) {
    const loom_testbench_value_t* input = span->binding_values[binding_index];
    span->binding_table[binding_index] = (iree_hal_buffer_binding_t){
        .buffer = input->buffer.buffer,
        .offset = input->buffer.byte_offset,
        .length = input->buffer.byte_length,
    };
  }
}

static iree_status_t loom_run_hal_testbench_actual_sequence_invoke_span(
    void* user_data, iree_host_size_t sample_ordinal,
    iree_host_size_t invocation_count,
    const loom_testbench_prepared_invocation_t* invocations,
    loom_testbench_value_table_t* table) {
  loom_run_hal_testbench_actual_sequence_execution_t* execution =
      (loom_run_hal_testbench_actual_sequence_execution_t*)user_data;
  const iree_host_size_t first_invocation_index =
      loom_run_hal_testbench_actual_sequence_invocation_index(
          execution, invocations[0].plan);
  loom_run_hal_testbench_actual_sequence_span_t* span =
      execution->invocations[first_invocation_index].span;
  IREE_ASSERT(span != NULL);
  IREE_ASSERT(span->first_invocation_index == first_invocation_index);
  IREE_ASSERT(span->invocation_count == invocation_count);
  IREE_ASSERT(sample_ordinal < span->sample_count);

  if (span->value_slots != table->slots) {
    IREE_RETURN_IF_ERROR(
        loom_run_hal_testbench_actual_sequence_resolve_values(span, table));
  }
  if (!span->providers_prepared) {
    for (iree_host_size_t provider_index = 0;
         provider_index < span->invocation_count; ++provider_index) {
      loom_run_hal_testbench_actual_provider_t* provider =
          span->invocations[provider_index].provider;
      IREE_RETURN_IF_ERROR(
          loom_run_hal_testbench_actual_provider_compile(provider));
      span->compile_rejected |= provider->compile_rejected;
    }
    span->providers_prepared = true;
  }
  if (span->compile_rejected) {
    return iree_ok_status();
  }

  loom_run_hal_dispatch_sequence_t* sample_sequence =
      &span->sample_sequences[sample_ordinal];
  if (sample_sequence->command_buffer == NULL) {
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_sequence_prepare_sample(
        span, sample_ordinal));
  }
  loom_run_hal_testbench_actual_sequence_populate_binding_table(span);
  return loom_run_hal_dispatch_sequence_execute(
      &span->context->runtime, sample_sequence,
      (iree_hal_buffer_binding_table_t){
          .count = span->binding_count,
          .bindings = span->binding_table,
      });
}

static iree_status_t loom_run_hal_testbench_actual_sequence_query_issue(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_sample_issue_t* out_issue) {
  *out_issue = (loom_testbench_sample_issue_t){0};
  loom_run_hal_testbench_actual_sequence_execution_t* execution =
      (loom_run_hal_testbench_actual_sequence_execution_t*)user_data;
  const iree_host_size_t invocation_index =
      loom_run_hal_testbench_actual_sequence_invocation_index(execution,
                                                              invocation);
  const loom_run_hal_testbench_actual_provider_t* provider =
      execution->invocations[invocation_index].provider;
  IREE_ASSERT(provider != NULL);
  if (provider->compile_rejected) {
    *out_issue = (loom_testbench_sample_issue_t){
        .category = LOOM_TESTBENCH_SAMPLE_ISSUE_COMPILE_REJECTED,
        .provider = IREE_SV("actual"),
        .stage = provider->compile_failure_stage,
        .kind = provider->compile_failure_kind,
        .message = provider->compile_failure_message,
    };
  }
  return iree_ok_status();
}

loom_testbench_invocation_provider_t
loom_run_hal_testbench_actual_sequence_execution_provider(
    loom_run_hal_testbench_actual_sequence_execution_t* execution) {
  return (loom_testbench_invocation_provider_t){
      .invoke = loom_run_hal_testbench_actual_sequence_invoke,
      .invoke_sequence = loom_run_hal_testbench_actual_sequence_invoke_span,
      .query_issue = loom_run_hal_testbench_actual_sequence_query_issue,
      .user_data = execution,
  };
}

iree_status_t loom_run_hal_testbench_actual_sequence_initialize(
    const loom_run_hal_testbench_actual_sequence_options_t* options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_sequence);
  *out_sequence = (loom_run_hal_testbench_actual_sequence_t){
      .host_allocator = options->context->host_allocator,
  };

  iree_host_size_t kernel_launch_count = 0;
  iree_status_t status = loom_run_hal_testbench_count_kernel_launches(
      options->case_plan, &kernel_launch_count);
  if (iree_status_is_ok(status) && kernel_launch_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL actual sequence requires at least one kernel launch in check.case "
        "`%.*s`",
        (int)options->case_plan->name.size, options->case_plan->name.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        out_sequence->host_allocator, kernel_launch_count,
        sizeof(*out_sequence->providers), (void**)&out_sequence->providers);
  }
  if (iree_status_is_ok(status)) {
    memset(out_sequence->providers, 0,
           kernel_launch_count * sizeof(*out_sequence->providers));
    out_sequence->provider_count = kernel_launch_count;
  }

  iree_host_size_t provider_index = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < options->case_plan->invocation_count;
       ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &options->case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
      continue;
    }
    const loom_run_hal_testbench_actual_provider_options_t provider_options = {
        .context = options->context,
        .session = options->session,
        .target_environment = options->target_environment,
        .run_module = options->run_module,
        .pipeline = options->pipeline,
        .sanitizer = options->sanitizer,
        .config_set = options->config_set,
        .kernel_launch = invocation,
        .diagnostic_sink = options->diagnostic_sink,
        .max_errors = options->max_errors,
        .artifact_flags = options->artifact_flags,
        .artifact_manifest = options->artifact_manifest,
    };
    loom_run_hal_testbench_actual_provider_initialize(
        &provider_options, &out_sequence->providers[provider_index++]);
  }
  loom_run_hal_testbench_actual_provider_t** providers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        out_sequence->host_allocator, out_sequence->provider_count,
        sizeof(*providers), (void**)&providers);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_sequence->provider_count; ++i) {
    providers[i] = &out_sequence->providers[i];
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_actual_sequence_execution_create(
        options->case_plan, out_sequence->provider_count, providers,
        out_sequence->host_allocator, &out_sequence->execution);
  }
  iree_allocator_free(out_sequence->host_allocator, providers);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_actual_sequence_deinitialize(out_sequence);
  }
  return status;
}

void loom_run_hal_testbench_actual_sequence_deinitialize(
    loom_run_hal_testbench_actual_sequence_t* sequence) {
  if (sequence == NULL) {
    return;
  }
  loom_run_hal_testbench_actual_sequence_execution_destroy(sequence->execution);
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    loom_run_hal_testbench_actual_provider_deinitialize(
        &sequence->providers[i]);
  }
  iree_allocator_free(sequence->host_allocator, sequence->providers);
  *sequence = (loom_run_hal_testbench_actual_sequence_t){0};
}

loom_testbench_invocation_provider_t
loom_run_hal_testbench_actual_sequence_provider(
    loom_run_hal_testbench_actual_sequence_t* sequence) {
  IREE_ASSERT(sequence->execution != NULL);
  return loom_run_hal_testbench_actual_sequence_execution_provider(
      sequence->execution);
}

iree_status_t loom_run_hal_testbench_materialize_invocation_from_table(
    const loom_testbench_value_table_t* table,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings) {
  const loom_testbench_invocation_plan_t* invocation = provider->kernel_launch;
  *out_options = provider->invocation_options;
  loom_run_hal_binding_list_initialize(out_bindings);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < invocation->workload_count; ++i) {
    const loom_testbench_value_t* workload = NULL;
    status = loom_testbench_value_table_lookup_borrow(
        table, invocation->workload_value_ids[i], &workload);
    if (iree_status_is_ok(status)) {
      status = loom_testbench_value_as_i64(workload,
                                           &provider->workload_arguments[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_evaluate_launch_config(
        provider, invocation->workload_count, out_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_binding_list_initialize_capacity(
        invocation->input_count, allocator, out_bindings);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < invocation->input_count; ++i) {
    loom_testbench_value_t value = {0};
    status = loom_testbench_value_table_lookup_retain(
        table, invocation->input_value_ids[i], &value);
    if (iree_status_is_ok(status)) {
      const loom_type_t input_type =
          loom_module_value_type(table->module, invocation->input_value_ids[i]);
      const iree_hal_executable_function_parameter_t* parameter =
          provider->function_parameter_count != 0
              ? &provider->function_parameters[i]
              : NULL;
      status = loom_run_hal_testbench_input_append(
          out_bindings, &value, input_type, parameter, out_options);
    }
    loom_testbench_value_deinitialize(&value);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_bindings);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_materialize_invocation_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings) {
  loom_testbench_value_table_t table = {0};
  iree_status_t status = loom_testbench_value_table_initialize(
      module, case_plan, allocator, &table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_case_sample(
        materializer_options, case_plan, sample_ordinal, &table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_materialize_invocation_from_table(
        &table, provider, allocator, out_options, out_bindings);
  }
  loom_testbench_value_table_deinitialize(&table);
  return status;
}

iree_status_t loom_run_hal_testbench_prepare_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_materializer_options_t* materializer_options,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_options_t invocation_options = {0};
  loom_run_hal_binding_list_t bindings = {0};
  iree_status_t status =
      loom_run_hal_testbench_materialize_invocation_for_sample(
          module_plan->module, materializer_options, case_plan, provider,
          sample_ordinal, allocator, &invocation_options, &bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &invocation_options, &bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, allocator, out_plan);
  }
  loom_run_hal_binding_list_deinitialize(&bindings);
  return status;
}
