// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/one_shot.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/internal/arena.h"
#include "loom/link/linker.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/target/entry_selection.h"
#include "loom/target/selection.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/compile/report_capture.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/hal/compile_request.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/io/file.h"

void loom_run_hal_one_shot_options_initialize(
    loom_run_hal_one_shot_options_t* out_options) {
  *out_options = (loom_run_hal_one_shot_options_t){0};
  out_options->workgroup_count[0] = 1;
  out_options->workgroup_count[1] = 1;
  out_options->workgroup_count[2] = 1;
}

static const loom_op_t* loom_run_hal_one_shot_lookup_entry_kernel(
    const loom_module_t* module, iree_string_view_t function_name) {
  // Symbols are stored without their textual sigil.
  (void)iree_string_view_consume_prefix_char(&function_name, '@');
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, function_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  return loom_kernel_def_isa(symbol->defining_op) ? symbol->defining_op : NULL;
}

static const loom_op_t* loom_run_hal_one_shot_find_single_kernel(
    const loom_module_t* module) {
  const loom_op_t* selected_kernel = NULL;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_kernel_def_isa(symbol->defining_op)) {
      continue;
    }
    if (selected_kernel != NULL) {
      return NULL;
    }
    selected_kernel = symbol->defining_op;
  }
  return selected_kernel;
}

static iree_status_t loom_run_hal_one_shot_select_kernel_root(
    const loom_module_t* module, iree_string_view_t function_name,
    iree_string_view_t* out_root) {
  *out_root = iree_string_view_empty();
  if (!iree_string_view_is_empty(function_name)) {
    const loom_op_t* kernel =
        loom_run_hal_one_shot_lookup_entry_kernel(module, function_name);
    if (kernel == NULL) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "kernel function '%.*s' was not found",
                              (int)function_name.size, function_name.data);
    }
    (void)iree_string_view_consume_prefix_char(&function_name, '@');
    *out_root = function_name;
    return iree_ok_status();
  }

  const loom_op_t* selected_kernel =
      loom_run_hal_one_shot_find_single_kernel(module);
  if (selected_kernel == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL execution requires exactly one kernel or --function=<symbol>");
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->defining_op == selected_kernel) {
      *out_root = module->strings.entries[symbol->name_id];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "selected kernel has no module symbol");
}

static iree_status_t loom_run_hal_one_shot_materialize_kernel_root(
    loom_run_session_t* session, iree_allocator_t host_allocator,
    loom_run_module_t* run_module, iree_string_view_t* inout_root_name) {
  const loom_module_t* const source_modules[] = {run_module->module};
  iree_string_view_t module_name = iree_string_view_empty();
  if (run_module->module->name_id < run_module->module->strings.count) {
    module_name =
        run_module->module->strings.entries[run_module->module->name_id];
  }
  const iree_string_view_t roots[] = {*inout_root_name};
  loom_module_t* linked_module = NULL;
  IREE_RETURN_IF_ERROR(loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = module_name,
          .root_symbols =
              {
                  .count = IREE_ARRAYSIZE(roots),
                  .values = roots,
              },
      },
      loom_run_session_block_pool(session), host_allocator, &linked_module));

  const loom_string_id_t root_name_id =
      loom_module_lookup_string(linked_module, *inout_root_name);
  if (root_name_id == LOOM_STRING_ID_INVALID) {
    loom_module_free(linked_module);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "materialized kernel root has no module string");
  }
  *inout_root_name = linked_module->strings.entries[root_name_id];
  loom_module_free(run_module->module);
  run_module->module = linked_module;
  return iree_ok_status();
}

bool loom_run_hal_one_shot_options_apply_static_workgroup_count(
    const loom_module_t* module, iree_string_view_t function_name,
    loom_run_hal_one_shot_options_t* options) {
  if (module == NULL || options == NULL) {
    return false;
  }
  const loom_op_t* kernel =
      iree_string_view_is_empty(function_name)
          ? loom_run_hal_one_shot_find_single_kernel(module)
          : loom_run_hal_one_shot_lookup_entry_kernel(module, function_name);
  if (kernel == NULL) {
    return false;
  }
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  if (!loom_kernel_def_static_workgroup_count(module, kernel,
                                              &workgroup_count)) {
    return false;
  }
  options->workgroup_count[0] = workgroup_count.x;
  options->workgroup_count[1] = workgroup_count.y;
  options->workgroup_count[2] = workgroup_count.z;
  return true;
}

void loom_run_hal_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_hal_one_shot_result_t* out_result) {
  *out_result = (loom_run_hal_one_shot_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_hal_one_shot_result_deinitialize(
    loom_run_hal_one_shot_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_hal_one_shot_result_t){0};
}

static bool loom_run_hal_one_shot_accept_entry(
    void* user_data, const loom_target_entry_t* entry) {
  (void)user_data;
  (void)entry;
  return true;
}

static iree_status_t loom_run_hal_one_shot_select_entry(
    const loom_run_hal_one_shot_request_t* request,
    iree_arena_allocator_t* arena, bool* out_selected,
    loom_target_entry_t* out_entry) {
  *out_selected = false;
  *out_entry = (loom_target_entry_t){0};

  const loom_target_entry_options_t options = {
      .entry_symbol = request->options->function_name,
      .diagnostic_sink = request->compile_options->diagnostic_sink,
      .source_resolver = request->compile_options->source_resolver,
      .max_errors = request->compile_options->max_errors,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      request->run_module->module, &options, LOOM_EMITTER_VERIFIER,
      &diagnostic_emitter);
  const loom_target_entry_predicate_t predicate = {
      .fn = loom_run_hal_one_shot_accept_entry,
  };
  return loom_target_entry_select_entry(
      request->run_module->module, &options, predicate, &diagnostic_emitter,
      IREE_SV("HAL execution"), arena, out_selected, out_entry);
}

static iree_status_t loom_run_hal_one_shot_write_artifact(
    iree_string_view_t path, const iree_byte_sequence_t* contents,
    iree_string_view_t artifact_name, iree_allocator_t allocator) {
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (contents == NULL || iree_byte_sequence_length(contents) == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL %.*s artifact is empty",
                            (int)artifact_name.size, artifact_name.data);
  }
  return loom_tooling_write_output_byte_sequence(path, contents, allocator);
}

static iree_status_t loom_run_hal_one_shot_write_candidate_artifacts(
    const loom_run_hal_one_shot_request_t* request,
    const loom_run_hal_candidate_t* candidate) {
  IREE_RETURN_IF_ERROR(loom_run_hal_one_shot_write_artifact(
      request->options->target_artifact_output_path,
      candidate->artifact_candidate.artifact.target_artifact_data,
      IREE_SV("target-native"), request->host_allocator));
  return loom_run_hal_one_shot_write_artifact(
      request->options->executable_output_path,
      candidate->artifact_candidate.artifact.executable_data,
      IREE_SV("executable"), request->host_allocator);
}

iree_status_t loom_run_hal_one_shot_probe(
    const loom_device_provider_t* device_provider,
    const loom_run_hal_one_shot_probe_request_t* request) {
  loom_run_hal_runtime_t runtime = {0};
  loom_device_target_t target = {0};

  loom_run_hal_runtime_options_t runtime_options;
  loom_run_hal_runtime_options_initialize(device_provider->driver_name,
                                          &runtime_options);
  iree_status_t status = loom_run_hal_runtime_initialize(
      &runtime_options, request->host_allocator, &runtime);
  if (iree_status_is_ok(status)) {
    const iree_string_view_t target_value =
        iree_string_view_trim(request->target);
    if (iree_string_view_is_empty(target_value)) {
      status = loom_device_provider_select_target(device_provider, &runtime,
                                                  &target);
    } else {
      loom_target_specification_t specification = {0};
      status = loom_target_specification_parse(target_value, &specification);
      const loom_target_profile_t* profile = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_target_environment_select_profile(
            request->target_environment, &specification, &profile);
      }
      if (iree_status_is_ok(status)) {
        status = loom_device_provider_select_profile_target(
            device_provider, &runtime, profile, specification.selector,
            &target);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &request->result->output, "device provider: %.*s\nhal driver: %.*s\n",
        (int)device_provider->artifact_provider->name.size,
        device_provider->artifact_provider->name.data,
        (int)device_provider->driver_name.size,
        device_provider->driver_name.data);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t target_key = loom_device_target_key(&target);
    status = iree_string_builder_append_format(
        &request->result->output, "device target key: %.*s\n",
        (int)target_key.size, target_key.data);
  }
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

iree_status_t loom_run_hal_one_shot_run(
    const loom_device_provider_t* device_provider,
    const loom_run_hal_one_shot_request_t* request) {
  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_candidate_t candidate = {0};
  loom_compile_pipeline_result_t pipeline_result = {0};
  loom_run_hal_invocation_result_t invocation_result = {0};
  loom_run_hal_invocation_result_initialize(request->host_allocator,
                                            &invocation_result);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, request->host_allocator,
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_string_view_t root_name = iree_string_view_empty();
  iree_status_t status = loom_run_hal_one_shot_select_kernel_root(
      request->run_module->module, request->options->function_name, &root_name);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_one_shot_materialize_kernel_root(
        request->session, request->host_allocator, request->run_module,
        &root_name);
  }

  loom_run_hal_runtime_options_t runtime_options;
  loom_run_hal_runtime_options_initialize(device_provider->driver_name,
                                          &runtime_options);
  runtime_options.runtime_features |=
      loom_run_hal_runtime_features_from_sanitizer_options(
          &request->compile_options->target_pipeline_options.sanitizer);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_runtime_initialize(&runtime_options,
                                             request->host_allocator, &runtime);
  }

  const loom_target_facts_t* target_requirement = NULL;
  if (iree_status_is_ok(status) &&
      iree_string_view_is_empty(
          iree_string_view_trim(request->compile_request_options.target))) {
    bool entry_selected = false;
    loom_target_entry_t entry = {0};
    status = loom_run_hal_one_shot_select_entry(request, &arena,
                                                &entry_selected, &entry);
    if (iree_status_is_ok(status) && !entry_selected) {
      request->result->exit_code = 1;
    } else if (iree_status_is_ok(status)) {
      target_requirement = entry.target_facts;
    }
  }

  loom_run_hal_compile_request_t compile_request = {0};
  if (iree_status_is_ok(status) && request->result->exit_code == 0) {
    const iree_string_view_t roots[] = {root_name};
    loom_compile_request_options_t compile_request_options =
        request->compile_request_options;
    compile_request_options.roots = (iree_string_view_list_t){
        .count = IREE_ARRAYSIZE(roots),
        .values = roots,
    };
    const loom_run_hal_compile_resolve_options_t resolve_options = {
        .module = request->run_module->module,
        .compile = compile_request_options,
        .target_requirement = target_requirement,
        .target_environment = request->target_environment,
        .device_provider = device_provider,
        .runtime = &runtime,
    };
    status = loom_run_hal_compile_request_resolve(&resolve_options,
                                                  &compile_request);
  }

  loom_compile_options_t compile_options = *request->compile_options;
  compile_options.target_pipeline_options =
      device_provider->artifact_provider->default_pipeline_options;
  compile_options.target_pipeline_options.sanitizer =
      request->compile_options->target_pipeline_options.sanitizer;
  loom_device_target_profile_t device_target_profile = {0};
  const loom_target_profile_t* specialization_profile = NULL;
  if (iree_status_is_ok(status) && request->result->exit_code == 0) {
    status = loom_run_hal_compile_request_target_profile(
        &compile_request, device_provider, &runtime, &device_target_profile,
        &specialization_profile);
  }
  if (iree_status_is_ok(status) && request->result->exit_code == 0) {
    const loom_target_specialization_request_t specialization_request = {
        .function_name = root_name,
        .target_profile = specialization_profile,
    };
    loom_compile_pipeline_options_t pipeline_options = {0};
    loom_compile_pipeline_options_initialize(&pipeline_options);
    pipeline_options.pipeline = request->pipeline;
    pipeline_options.target_pipeline_options =
        compile_options.target_pipeline_options;
    pipeline_options.target_environment = request->target_environment;
    pipeline_options.target_specializations =
        (loom_target_specialization_request_list_t){
            .values = &specialization_request,
            .count = 1,
        };
    pipeline_options.low_descriptor_registry =
        loom_run_session_low_descriptor_registry(request->session);
    pipeline_options.diagnostic_sink = compile_options.diagnostic_sink;
    pipeline_options.source_resolver = compile_options.source_resolver;
    pipeline_options.max_errors = compile_options.max_errors;
    pipeline_options.report = compile_options.report;
    status = loom_compile_run_pipeline(
        request->run_module->module, &pipeline_options,
        loom_run_session_block_pool(request->session), &pipeline_result);
    if (iree_status_is_ok(status) && pipeline_result.pass.error_count != 0) {
      request->result->exit_code = 1;
    }
    compile_options.function_versions = &pipeline_result.function_versions.list;
  }
  if (iree_status_is_ok(status) && request->result->exit_code == 0) {
    status = loom_run_hal_candidate_emit_target(
        device_provider, &compile_request.device_target, request->run_module,
        &compile_options, request->host_allocator, &candidate);
  }
  if (iree_status_is_ok(status) && !candidate.artifact_candidate.compiled) {
    request->result->exit_code = 1;
  }
  if (iree_status_is_ok(status) && candidate.artifact_candidate.compiled) {
    status =
        loom_run_hal_one_shot_write_candidate_artifacts(request, &candidate);
  }
  if (iree_status_is_ok(status) && candidate.artifact_candidate.compiled &&
      !request->options->emit_only) {
    loom_run_hal_invocation_request_t invocation_request = {0};
    loom_run_hal_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.artifact = &candidate.device_artifact;
    invocation_request.options.function_name = root_name;
    invocation_request.options.workgroup_count[0] =
        request->options->workgroup_count[0];
    invocation_request.options.workgroup_count[1] =
        request->options->workgroup_count[1];
    invocation_request.options.workgroup_count[2] =
        request->options->workgroup_count[2];
    if (request->options->constant_count >
        IREE_ARRAYSIZE(invocation_request.options.constants)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch constant count %" PRIhsz " exceeds maximum %" PRIhsz,
          request->options->constant_count,
          IREE_ARRAYSIZE(invocation_request.options.constants));
    }
    if (iree_status_is_ok(status)) {
      invocation_request.options.constant_count =
          request->options->constant_count;
      memcpy(invocation_request.options.constants, request->options->constants,
             request->options->constant_count *
                 sizeof(request->options->constants[0]));
      invocation_request.bindings = (loom_run_hal_binding_specs_t){
          .values = request->options->bindings.values,
          .count = request->options->bindings.count,
      };
      invocation_request.expected_bindings = (loom_run_hal_binding_specs_t){
          .values = request->options->expected_bindings.values,
          .count = request->options->expected_bindings.count,
      };
      invocation_request.max_output_element_count =
          request->options->max_output_element_count;
      status = loom_run_hal_invocation_run(
          &invocation_request, request->host_allocator, &invocation_result);
    }
  }
  if (iree_status_is_ok(status) && candidate.artifact_candidate.compiled &&
      !request->options->emit_only) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->output,
        iree_string_builder_view(&invocation_result.output));
  }
  if (iree_status_is_ok(status) && request->compile_report_capture != NULL) {
    status = loom_compile_report_capture_append_output(
        request->compile_report_capture, &request->result->output);
  }

  loom_run_hal_invocation_result_deinitialize(&invocation_result);
  loom_run_hal_candidate_deinitialize(&candidate);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  loom_run_hal_runtime_deinitialize(&runtime);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}
