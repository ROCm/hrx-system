// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include <string.h>

#include "iree/tooling/device_util.h"
#include "loom/ir/module.h"
#include "loom/target/profile.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/testbench_staging.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

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
  // Reusable dispatch sequence indexed by case sample ordinal.
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
    loomc_target_environment_t* target_environment,
    const loom_run_hal_target_route_t* target_routes,
    iree_host_size_t target_route_count, iree_allocator_t host_allocator,
    loom_run_hal_testbench_context_t* out_context) {
  *out_context = (loom_run_hal_testbench_context_t){
      .target_environment = target_environment,
      .target_routes = target_routes,
      .target_route_count = target_route_count,
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

iree_status_t loom_run_hal_testbench_context_add_module_runtime_requirements(
    loom_run_hal_testbench_context_t* context, const loomc_module_t* module,
    const loomc_sanitizer_options_t* sanitizer_options) {
  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_iree_hal_module_query_runtime_features(
          module, sanitizer_options,
          loomc_allocator_from_iree(context->host_allocator),
          &runtime_features)));
  const iree_hal_device_runtime_feature_flags_t missing_features =
      runtime_features & ~context->runtime_features;
  if (context->runtime_initialized && missing_features != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL runtime does not provision required module sanitizer features "
        "0x%016" PRIx64,
        (uint64_t)missing_features);
  }
  context->runtime_features |= runtime_features;
  return iree_ok_status();
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

static iree_status_t loom_run_hal_testbench_context_parse_device(
    loom_run_hal_testbench_context_t* context) {
  if (context->selected_target_route != NULL) {
    return iree_ok_status();
  }

  const iree_string_view_list_t device_uris = iree_hal_device_flag_list();
  if (device_uris.count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Loom HAL execution requires exactly one --device= URI; got %" PRIhsz,
        device_uris.count);
  }

  const iree_string_view_t device_uri = device_uris.values[0];
  iree_string_view_split(device_uri, ':', &context->driver_name, NULL);
  if (iree_string_view_is_empty(context->driver_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device=%.*s has no HAL driver name",
                            (int)device_uri.size, device_uri.data);
  }
  for (iree_host_size_t i = 0; i < context->target_route_count; ++i) {
    const loom_run_hal_target_route_t* route = &context->target_routes[i];
    if (iree_string_view_equal(route->driver_name, context->driver_name)) {
      context->selected_target_route = route;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--device=%.*s is not available in this Loom installation; no linked "
      "LoomC target adapter handles HAL driver '%.*s'",
      (int)device_uri.size, device_uri.data, (int)context->driver_name.size,
      context->driver_name.data);
}

iree_status_t loom_run_hal_testbench_context_validate_explicit_device(
    loom_run_hal_testbench_context_t* context) {
  if (iree_hal_device_flag_list().count == 0) {
    return iree_ok_status();
  }
  return loom_run_hal_testbench_context_parse_device(context);
}

iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context) {
  if (context->runtime_initialized) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_parse_device(context));
  loom_run_hal_runtime_options_t runtime_options;
  loom_run_hal_runtime_options_initialize(context->driver_name,
                                          &runtime_options);
  runtime_options.event_sink = context->device_event_sink;
  runtime_options.runtime_features = context->runtime_features;
  iree_status_t status = loom_run_hal_runtime_initialize(
      &runtime_options, context->host_allocator, &context->runtime);
  IREE_RETURN_IF_ERROR(status);
  context->runtime_initialized = true;
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_context_select_target(
    loom_run_hal_testbench_context_t* context, loomc_string_view_t identifier,
    loomc_target_profile_t* target_profile,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(context));
  IREE_ASSERT(context->selected_target_route != NULL);
  IREE_ASSERT(context->selected_target_route->provider != NULL);
  const loomc_iree_hal_target_provider_t* target_providers[] = {
      context->selected_target_route->provider,
  };
  const loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .identifier = identifier,
      .device = context->runtime.device,
      .physical_device_affinity =
          loom_run_hal_runtime_dispatch_physical_device_affinity(
              &context->runtime),
      .target_profile = target_profile,
      .providers = target_providers,
      .provider_count = IREE_ARRAYSIZE(target_providers),
  };
  return iree_status_from_loomc(loomc_target_select_iree_hal(
      context->target_environment, &options,
      loomc_allocator_from_iree(context->host_allocator), out_selection,
      out_result));
}

iree_status_t loom_run_hal_testbench_require_successful_result(
    const loomc_result_t* result, iree_string_view_t fallback_message) {
  if (result == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler operation returned no result");
  }
  if (loomc_result_succeeded(result)) {
    return iree_ok_status();
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic != NULL &&
        diagnostic->severity == LOOMC_DIAGNOSTIC_SEVERITY_ERROR &&
        !loomc_string_view_is_empty(diagnostic->message)) {
      const iree_string_view_t message =
          iree_string_view_from_loomc(diagnostic->message);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s",
                              (int)message.size, message.data);
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s",
                          (int)fallback_message.size, fallback_message.data);
}

iree_hal_buffer_params_t loom_run_hal_testbench_host_visible_buffer_params(
    void) {
  return (iree_hal_buffer_params_t){
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
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
      .compilation = options->compilation,
      .module = options->module != NULL        ? options->module
                : options->compilation != NULL ? options->compilation->module
                                               : NULL,
      .native_module = options->native_module,
      .pass_program = options->pass_program,
      .requested_target_profile = options->requested_target_profile,
      .sanitizer = options->sanitizer,
      .invocation = options->invocation,
      .result_callback = options->result_callback,
      .compile_report = options->compile_report,
      .artifact_manifest = options->artifact_manifest,
      .emit_artifact_flags = options->emit_artifact_flags,
      .launch_config_function = loomc_launch_config_function_invalid(),
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
  loomc_launch_config_program_release(provider->launch_config_program);
  loomc_result_release(provider->compiler_result);
  loomc_target_profile_release(provider->selected_target_profile);
  if (provider->context != NULL) {
    iree_allocator_free(provider->context->host_allocator,
                        provider->workload_argument_bits);
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
  *out_name = loom_string_table_get(&module->strings, symbol->name_id);
  return iree_ok_status();
}

static bool loom_run_hal_testbench_invocation_is_pipeline(
    const loom_testbench_invocation_plan_t* invocation) {
  if (invocation->kind == LOOM_TESTBENCH_INVOCATION_PIPELINE) {
    return true;
  }
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH ||
      invocation->callee_ref.symbol_id >= invocation->module->symbols.count) {
    return false;
  }
  const loom_symbol_t* symbol =
      &invocation->module->symbols.entries[invocation->callee_ref.symbol_id];
  return loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE);
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

static iree_status_t loom_run_hal_testbench_observe_result(
    loom_run_hal_testbench_actual_provider_t* provider,
    const loomc_result_t* result) {
  IREE_RETURN_IF_ERROR(
      loom_testbench_observe_compile_result(provider->result_callback, result));
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    switch (diagnostic->severity) {
      case LOOMC_DIAGNOSTIC_SEVERITY_ERROR:
        ++provider->diagnostic_error_count;
        break;
      case LOOMC_DIAGNOSTIC_SEVERITY_WARNING:
        ++provider->diagnostic_warning_count;
        break;
      case LOOMC_DIAGNOSTIC_SEVERITY_NOTE:
        ++provider->diagnostic_remark_count;
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

static void loom_run_hal_testbench_record_result_rejection(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t stage, iree_string_view_t default_message,
    const loomc_result_t* result) {
  const loomc_diagnostic_t* diagnostic = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* candidate = loomc_result_diagnostic_at(result, i);
    if (candidate->severity == LOOMC_DIAGNOSTIC_SEVERITY_ERROR) {
      diagnostic = candidate;
      break;
    }
  }
  iree_string_view_t kind = IREE_SV("rejected");
  iree_string_view_t message = default_message;
  if (diagnostic != NULL) {
    const iree_string_view_t diagnostic_kind =
        iree_string_view_from_loomc(diagnostic->code);
    const iree_string_view_t diagnostic_message =
        iree_string_view_from_loomc(diagnostic->message);
    if (!iree_string_view_is_empty(diagnostic_kind)) {
      kind = diagnostic_kind;
    }
    if (!iree_string_view_is_empty(diagnostic_message)) {
      message = diagnostic_message;
    }
  }
  loom_run_hal_testbench_record_compile_rejection(provider, stage, kind,
                                                  message);
}

static iree_status_t loom_run_hal_testbench_resolve_export_name(
    const loomc_module_t* module, iree_string_view_t entry_symbol,
    iree_string_view_t* out_export_name) {
  loomc_module_function_t function = {0};
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_lookup_function(
      module, loomc_string_view_from_iree(entry_symbol), &function)));
  *out_export_name = iree_string_view_from_loomc(function.symbol_name);
  loomc_module_function_export_info_t export_info = {0};
  if (loomc_module_function_try_get_export_info(module, &function,
                                                &export_info) &&
      (export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL) != 0 &&
      !loomc_string_view_is_empty(export_info.export_symbol)) {
    *out_export_name = iree_string_view_from_loomc(export_info.export_symbol);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_select_compile_artifacts(
    const loomc_result_t* result,
    loom_run_hal_testbench_compile_artifacts_t* out_artifacts) {
  *out_artifacts = (loom_run_hal_testbench_compile_artifacts_t){0};
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    const loomc_artifact_t** slot = NULL;
    if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      slot = &out_artifacts->executable;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
      slot = &out_artifacts->launch_config;
    } else if (loomc_string_view_equal(
                   artifact->format,
                   loomc_make_cstring_view(
                       LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON)) ||
               loomc_string_view_equal(
                   artifact->format,
                   loomc_make_cstring_view(
                       LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT))) {
      slot = &out_artifacts->compile_report;
    } else if (loomc_string_view_equal(
                   artifact->format,
                   loomc_make_cstring_view(
                       LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON))) {
      slot = &out_artifacts->artifact_manifest;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_TEXT) {
      slot = &out_artifacts->target_listing;
    }
    if (slot != NULL && *slot != NULL) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "compiler returned duplicate HAL artifacts");
    }
    if (slot != NULL) {
      *slot = artifact;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_reflect_function_parameters(
    loom_run_hal_testbench_actual_provider_t* provider) {
  iree_hal_executable_t* executable = provider->prepared_candidate.executable;
  const iree_string_view_t function_name =
      provider->invocation_options.function_name;

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
  if (function_info.parameter_count != provider->invocation->input_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loaded HAL function '%.*s' reflects %u parameters for %" PRIhsz
        " source inputs",
        (int)function_name.size, function_name.data,
        (unsigned)function_info.parameter_count,
        provider->invocation->input_count);
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
  if (provider->compilation == NULL || provider->module == NULL ||
      provider->native_module == NULL || provider->invocation == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL actual provider requires compilation, module, "
                            "and invocation state");
  }
  const bool is_pipeline =
      loom_run_hal_testbench_invocation_is_pipeline(provider->invocation);
  if (!is_pipeline &&
      provider->invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL actual provider requires a kernel or finite pipeline invocation");
  }
  if (is_pipeline && provider->invocation->workload_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL pipeline specialization arguments are not implemented");
  }
  provider->sequence_representation =
      is_pipeline
          ? LOOM_RUN_HAL_DISPATCH_SEQUENCE_REPRESENTATION_DIRECT
          : LOOM_RUN_HAL_DISPATCH_SEQUENCE_REPRESENTATION_COMMAND_BUFFER;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_add_module_runtime_requirements(
          provider->context, provider->module, provider->sanitizer));

  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_module_symbol_name_from_ref(
      provider->native_module, provider->invocation->callee_ref,
      &entry_symbol));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_export_name(
      provider->module, entry_symbol,
      &provider->invocation_options.function_name));

  loomc_iree_hal_target_selection_t selection = {0};
  loomc_result_t* result = NULL;
  iree_status_t status = loom_run_hal_testbench_context_select_target(
      provider->context, loomc_make_cstring_view("Loom testbench live device"),
      provider->requested_target_profile, &selection, &result);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_observe_result(provider, result);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
    provider->compiler_result = result;
    result = NULL;
    loom_run_hal_testbench_record_result_rejection(
        provider, IREE_SV("target"),
        IREE_SV("HAL target selection rejected the live device"),
        provider->compiler_result);
  }
  if (iree_status_is_ok(status) && !provider->compile_rejected) {
    IREE_ASSERT(provider->context->selected_target_route != NULL);
    provider->selected_target_profile = selection.target_profile;
    selection.target_profile = NULL;
    provider->executable_target = selection.executable_target;
    const loom_target_profile_t* native_profile =
        loomc_target_profile_get_interop_view(
            provider->selected_target_profile);
    const loom_target_bundle_t* target_bundle =
        loom_target_profile_bundle(native_profile);
    if (target_bundle == NULL || target_bundle->snapshot == NULL) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "selected HAL target profile has no native target snapshot");
    } else {
      provider->target_snapshot = target_bundle->snapshot;
    }
  }
  loomc_result_release(result);
  result = NULL;
  loomc_target_profile_release(selection.target_profile);
  if (!iree_status_is_ok(status) || provider->compile_rejected) {
    return status;
  }

  loomc_module_t* compile_module = NULL;
  status = iree_status_from_loomc(loomc_module_clone(
      provider->module, provider->compilation->workspace,
      loomc_allocator_from_iree(provider->context->host_allocator),
      &compile_module));

  loomc_artifact_manifest_options_t artifact_manifest =
      provider->artifact_manifest;
  const void* emit_option_chain = NULL;
  if (artifact_manifest.mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE) {
    artifact_manifest.type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS;
    artifact_manifest.structure_size = sizeof(artifact_manifest);
    artifact_manifest.next = emit_option_chain;
    emit_option_chain = &artifact_manifest;
  }
  loomc_compile_report_options_t compile_report = provider->compile_report;
  if (compile_report.mode != LOOMC_COMPILE_REPORT_MODE_NONE) {
    compile_report.type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS;
    compile_report.structure_size = sizeof(compile_report);
    compile_report.next = emit_option_chain;
    emit_option_chain = &compile_report;
  }
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = emit_option_chain,
      .artifact_flags =
          provider->emit_artifact_flags | LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_string_view_t root = loomc_string_view_from_iree(entry_symbol);
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = provider->sanitizer,
      .roots = &root,
      .root_count = 1,
      .target_profile = provider->selected_target_profile,
      .config = provider->compilation->config,
      .emit_options = &emit_options,
      .artifact_flags =
          is_pipeline ? 0 : LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compile_artifact(
        provider->compilation->compiler, provider->compilation->workspace,
        provider->pass_program, compile_module, &compile_options,
        loomc_allocator_from_iree(provider->context->host_allocator), &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_observe_result(provider, result);
  }
  if (iree_status_is_ok(status)) {
    provider->compiler_result = result;
    result = NULL;
    status = loom_run_hal_testbench_select_compile_artifacts(
        provider->compiler_result, &provider->artifacts);
  }
  if (iree_status_is_ok(status) &&
      !loomc_result_succeeded(provider->compiler_result)) {
    loom_run_hal_testbench_record_result_rejection(
        provider, IREE_SV("compile"),
        IREE_SV("HAL compilation rejected the source module"),
        provider->compiler_result);
  }
  loomc_result_release(result);
  loomc_module_release(compile_module);
  if (!iree_status_is_ok(status) || provider->compile_rejected) {
    return status;
  }
  if (provider->artifacts.executable == NULL ||
      (!is_pipeline && provider->artifacts.launch_config == NULL)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        is_pipeline
            ? "compiler did not return a pipeline executable artifact"
            : "compiler did not return executable and launch-config artifacts");
  }

  if (!is_pipeline) {
    status = iree_status_from_loomc(loomc_launch_config_program_load(
        provider->artifacts.launch_config,
        loomc_allocator_from_iree(provider->context->host_allocator),
        &provider->launch_config_program));
    if (iree_status_is_ok(status)) {
      status =
          iree_status_from_loomc(loomc_launch_config_program_lookup_function(
              provider->launch_config_program,
              loomc_string_view_from_iree(
                  provider->invocation_options.function_name),
              &provider->launch_config_function));
    }
  }
  if (iree_status_is_ok(status) && provider->invocation->workload_count != 0) {
    status = iree_allocator_malloc_array(
        provider->context->host_allocator, provider->invocation->workload_count,
        sizeof(*provider->workload_argument_bits),
        (void**)&provider->workload_argument_bits);
  }

  const loom_device_artifact_t device_artifact = {
      .executable_target = provider->executable_target,
      .contents = iree_byte_sequence_from_loomc(
          provider->artifacts.executable->contents),
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_prepared_candidate_prepare(
        &provider->context->runtime, &device_artifact,
        provider->context->host_allocator, &provider->prepared_candidate);
  }
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
    const loom_target_snapshot_t* target_snapshot,
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
  if (parameter &&
      parameter->type != IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL executable parameter for scalar input is not a constant");
  }
  iree_host_size_t byte_length = parameter ? parameter->size : 0;
  uint64_t raw_value = 0;
  if (source_scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      source_scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    int64_t integer_value = 0;
    IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &integer_value));
    if (!parameter) {
      if (!target_snapshot) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "HAL dispatch %s constant requires a selected target carrier",
            loom_scalar_type_name(source_scalar_type));
      }
      const uint32_t target_bitwidth =
          source_scalar_type == LOOM_SCALAR_TYPE_INDEX
              ? target_snapshot->index_bitwidth
              : target_snapshot->offset_bitwidth;
      byte_length = target_bitwidth / 8;
    }
    if (byte_length != 4 && byte_length != 8) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL dispatch %s constant has unsupported ABI byte length %" PRIhsz,
          loom_scalar_type_name(source_scalar_type), byte_length);
    }
    if (source_scalar_type == LOOM_SCALAR_TYPE_OFFSET && integer_value < 0) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL dispatch offset constant value %" PRId64
                              " is negative",
                              integer_value);
    }
    const bool fits_32_bit_carrier =
        source_scalar_type == LOOM_SCALAR_TYPE_INDEX
            ? integer_value >= INT32_MIN && integer_value <= INT32_MAX
            : (uint64_t)integer_value <= UINT32_MAX;
    if (byte_length == 4 && !fits_32_bit_carrier) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL dispatch %s constant value %" PRId64
                              " does not fit its 32-bit target carrier",
                              loom_scalar_type_name(source_scalar_type),
                              integer_value);
    }
    raw_value = (uint64_t)integer_value;
  } else {
    uint32_t words[2] = {0};
    iree_host_size_t word_count = 0;
    IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
        &value->scalar, IREE_ARRAYSIZE(words), words, &word_count));
    const iree_host_size_t materialized_byte_length =
        word_count * sizeof(words[0]);
    if (!parameter) {
      byte_length = materialized_byte_length;
    }
    const iree_host_size_t source_byte_length =
        iree_max(1, loom_scalar_type_bitwidth(source_scalar_type) / 8);
    if ((byte_length != 1 && byte_length != 2 && byte_length != 4 &&
         byte_length != 8) ||
        byte_length < source_byte_length ||
        byte_length > materialized_byte_length) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL dispatch %s constant cannot use reflected byte length %" PRIhsz,
          loom_scalar_type_name(source_scalar_type), byte_length);
    }
    if (word_count == 1) {
      raw_value = words[0];
    } else {
      memcpy(&raw_value, words, sizeof(raw_value));
    }
  }

  // Reflection owns the byte layout: adjacent source arguments may have ABI
  // padding, and narrow scalars need not consume an entire 32-bit word.
  const iree_host_size_t byte_offset =
      parameter ? parameter->offset : options->constant_byte_length;
  if (byte_offset > sizeof(options->constants) ||
      byte_length > sizeof(options->constants) - byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch constants exceed capacity %" PRIhsz,
                            sizeof(options->constants));
  }
  if (byte_offset > options->constant_byte_length) {
    memset(options->constants + options->constant_byte_length, 0,
           byte_offset - options->constant_byte_length);
  }
  uint8_t bytes[sizeof(raw_value)];
  iree_unaligned_store_le_u64(bytes, raw_value);
  memcpy(options->constants + byte_offset, bytes, byte_length);
  options->constant_byte_length =
      iree_max(options->constant_byte_length, byte_offset + byte_length);
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
    loom_type_t input_type, const loom_target_snapshot_t* target_snapshot,
    const iree_hal_executable_function_parameter_t* parameter,
    loom_run_hal_invocation_options_t* options) {
  if (loom_testbench_value_is_buffer(input)) {
    return loom_run_hal_testbench_append_buffer_binding(input, parameter,
                                                        bindings);
  }
  if (loom_testbench_value_is_scalar(input)) {
    return loom_run_hal_testbench_invocation_options_push_constant(
        input, input_type, target_snapshot, parameter, options);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "HAL invocation input must be a buffer binding or "
                          "a scalar value");
}

iree_status_t loom_run_hal_testbench_invocation_inputs_from_values(
    const loom_testbench_value_t* inputs, const loom_type_t* input_types,
    const loom_target_snapshot_t* target_snapshot,
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
        out_bindings, &inputs[i], input_types[i], target_snapshot, parameter,
        options);
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
  if (workload_count != provider->invocation->workload_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation workload count mismatch");
  }
  if (loom_run_hal_testbench_invocation_is_pipeline(provider->invocation)) {
    // Array-program distribution is part of the compiled executable. One HAL
    // dispatch invokes the complete array program.
    out_options->workgroup_count[0] = 1;
    out_options->workgroup_count[1] = 1;
    out_options->workgroup_count[2] = 1;
    return iree_ok_status();
  }
  IREE_ASSERT(provider->launch_config_program != NULL);
  loomc_launch_config_t config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(config),
  };
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_launch_config_program_invoke(
          provider->launch_config_program, provider->launch_config_function,
          provider->workload_argument_bits, workload_count, &config)));
  provider->resolved_launch_config = config;
  out_options->workgroup_count[0] = config.workgroup_count.x;
  out_options->workgroup_count[1] = config.workgroup_count.y;
  out_options->workgroup_count[2] = config.workgroup_count.z;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_workload_argument_bits(
    const loom_testbench_value_t* value, uint64_t* out_bits) {
  int64_t signed_value = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &signed_value));
  *out_bits = (uint64_t)signed_value;
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_actual_provider_materialize_invocation(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings) {
  *out_options = (loom_run_hal_invocation_options_t){0};
  loom_run_hal_binding_list_initialize(out_bindings);
  if (!provider->prepared_candidate_initialized) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual provider must be prepared before materializing values");
  }
  const loom_testbench_invocation_plan_t* invocation = provider->invocation;
  if (input_count != invocation->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input count mismatch");
  }
  if (workload_count != invocation->workload_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation workload count mismatch");
  }

  *out_options = provider->invocation_options;
  for (iree_host_size_t i = 0; i < workload_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_workload_argument_bits(
        &workloads[i], &provider->workload_argument_bits[i]));
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_evaluate_launch_config(
      provider, workload_count, out_options));
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      input_count, provider->context->host_allocator, out_bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    const loom_value_id_t input_value_id = invocation->input_value_ids[i];
    const loom_type_t input_type =
        loom_module_value_type(provider->native_module, input_value_id);
    const iree_hal_executable_function_parameter_t* parameter =
        provider->function_parameter_count != 0
            ? &provider->function_parameters[i]
            : NULL;
    status = loom_run_hal_testbench_input_append(
        out_bindings, &inputs[i], input_type, provider->target_snapshot,
        parameter, out_options);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "preparing HAL actual input %" PRIhsz " for value ID %u", i,
          (unsigned)input_value_id);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_bindings);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  (void)out_results;
  loom_run_hal_testbench_actual_provider_t* provider =
      (loom_run_hal_testbench_actual_provider_t*)user_data;
  if (invocation != provider->invocation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual provider received an unexpected invocation");
  }
  IREE_ASSERT(result_count == 0);
  if (input_count != invocation->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input count mismatch");
  }
  if (workload_count != invocation->workload_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation workload count mismatch");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_actual_provider_compile(provider));
  if (provider->compile_rejected) {
    return iree_ok_status();
  }

  loom_run_hal_invocation_options_t invocation_options = {0};
  loom_run_hal_binding_list_t bindings = {0};
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_actual_provider_materialize_invocation(
          provider, workload_count, workloads, input_count, inputs,
          &invocation_options, &bindings));

  loom_run_hal_invocation_plan_t plan = {0};
  loom_run_hal_iteration_t iteration = {0};
  loom_run_hal_testbench_staging_t staging = {0};
  iree_status_t status = loom_run_hal_invocation_plan_prepare_from_lists(
      &invocation_options, &bindings, /*expected_bindings=*/NULL,
      /*max_output_element_count=*/0, provider->context->host_allocator, &plan);
  loom_run_hal_binding_list_deinitialize(&bindings);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_t device_bindings[LOOM_RUN_HAL_MAX_BINDING_COUNT];
    for (iree_host_size_t i = 0; i < plan.bindings.count; ++i) {
      const iree_tooling_buffer_binding_t* binding = &plan.bindings.values[i];
      device_bindings[i] = (iree_hal_buffer_binding_t){
          .buffer = binding->buffer,
          .offset = binding->byte_offset,
          .length = binding->byte_length,
      };
    }
    status = loom_run_hal_testbench_staging_initialize(
        &provider->context->runtime, plan.bindings.count, device_bindings,
        provider->context->host_allocator, &staging);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < plan.bindings.count; ++i) {
        iree_tooling_buffer_binding_t* binding = &plan.bindings.values[i];
        iree_hal_buffer_retain(device_bindings[i].buffer);
        iree_tooling_buffer_binding_deinitialize(binding);
        *binding = (iree_tooling_buffer_binding_t){
            .kind = IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER,
            .buffer = device_bindings[i].buffer,
            .byte_offset = device_bindings[i].offset,
            .byte_length = device_bindings[i].length,
        };
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_dispatch_plan(
        &provider->context->runtime, &provider->prepared_candidate, &plan,
        provider->context->host_allocator, &iteration);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_readback(
        &provider->context->runtime, &staging);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
  status = iree_status_join(
      status, loom_run_hal_testbench_staging_deinitialize(&staging));
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
    const loom_testbench_invocation_plan_t* invocation = provider->invocation;
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
    const loom_testbench_invocation_plan_t* invocation = provider->invocation;
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
            .representation = provider->sequence_representation,
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
      IREE_ASSERT(provider->invocation ==
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
        span->invocations[invocation_offset].provider->invocation;
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
    const loom_testbench_invocation_plan_t* invocation = provider->invocation;
    for (iree_host_size_t workload_index = 0;
         workload_index < invocation->workload_count; ++workload_index) {
      IREE_RETURN_IF_ERROR(loom_run_hal_testbench_workload_argument_bits(
          span->workload_values[workload_offset++],
          &provider->workload_argument_bits[workload_index]));
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
                input, input_type, provider->target_snapshot, parameter,
                &step->options));
      }
    }
  }
  IREE_ASSERT(workload_offset == span->workload_count);
  IREE_ASSERT(input_offset == span->input_count);
  IREE_ASSERT(binding_offset == span->binding_count);
  return loom_run_hal_dispatch_sequence_prepare(
      &span->context->runtime, span->invocation_count, span->steps,
      span->context->host_allocator, &span->sample_sequences[sample_ordinal]);
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
  if (!loom_run_hal_dispatch_sequence_is_prepared(sample_sequence)) {
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_sequence_prepare_sample(
        span, sample_ordinal));
  }
  loom_run_hal_testbench_actual_sequence_populate_binding_table(span);
  loom_run_hal_testbench_staging_t staging = {0};
  iree_status_t status = loom_run_hal_testbench_staging_initialize(
      &span->context->runtime, span->binding_count, span->binding_table,
      execution->host_allocator, &staging);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch_sequence_execute(
        &span->context->runtime, sample_sequence,
        (iree_hal_buffer_binding_table_t){
            .count = span->binding_count,
            .bindings = span->binding_table,
        });
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_readback(&span->context->runtime,
                                                     &staging);
  }
  status = iree_status_join(
      status, loom_run_hal_testbench_staging_deinitialize(&staging));
  return status;
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
        .compilation = options->compilation,
        .native_module = options->native_module,
        .pass_program = options->pass_program,
        .requested_target_profile = options->requested_target_profile,
        .sanitizer = options->sanitizer,
        .invocation = invocation,
        .result_callback = options->result_callback,
        .compile_report = options->compile_report,
        .artifact_manifest = options->artifact_manifest,
        .emit_artifact_flags = options->emit_artifact_flags,
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
  const loom_testbench_invocation_plan_t* invocation = provider->invocation;
  *out_options = provider->invocation_options;
  loom_run_hal_binding_list_initialize(out_bindings);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < invocation->workload_count; ++i) {
    const loom_testbench_value_t* workload = NULL;
    status = loom_testbench_value_table_lookup_borrow(
        table, invocation->workload_value_ids[i], &workload);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_testbench_workload_argument_bits(
          workload, &provider->workload_argument_bits[i]);
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
          out_bindings, &value, input_type, provider->target_snapshot,
          parameter, out_options);
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
  iree_status_t status = loom_testbench_value_table_initialize_case(
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
