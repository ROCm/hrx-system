// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/compile.h"

#include "iree/base/byte_sequence.h"
#include "loom/codegen/low/repr.h"
#include "loom/link/linker.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/compile/preparation.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/input.h"

static iree_status_t loom_check_compile_verify_input(
    loom_module_t* module, const loom_compile_pipeline_options_t* options) {
  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
  };
  loom_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(
      module, &entry_options, options->max_errors, &result));
  if (result.error_count != 0) {
    return iree_ok_status();
  }
  loom_target_entry_diagnostic_emitter_t emitter;
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_VERIFIER, &emitter);
  loom_low_verify_scratch_t scratch =
      loom_low_verify_scratch_for_module(module);
  loom_low_verify_result_t low_result = {0};
  return loom_target_entry_verify_low_module(
      module, options->low_descriptor_registry, &entry_options, &emitter,
      options->max_errors, loom_low_verify_provider_list_empty(), &scratch,
      &low_result);
}

static iree_status_t loom_check_compile_emit(
    const loom_compile_request_t* request, loom_module_t* module,
    const loom_compile_pipeline_options_t* pipeline_options,
    const loom_compile_pipeline_result_t* pipeline_result,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    bool* out_compiled) {
  *out_compiled = false;
  loom_compile_options_t compile_options;
  loom_compile_options_initialize(&compile_options);
  compile_options.target_pipeline_options =
      pipeline_options->target_pipeline_options;
  compile_options.diagnostic_sink = pipeline_options->diagnostic_sink;
  compile_options.source_resolver = pipeline_options->source_resolver;
  compile_options.function_versions = &pipeline_result->function_versions.list;
  if (request->producer.kind == LOOM_COMPILE_PRODUCER_ARTIFACT) {
    loom_artifact_candidate_t candidate = {0};
    iree_status_t status = loom_artifact_candidate_emit_target(
        request->producer.value.artifact_provider, &request->explicit_target,
        module, &compile_options, allocator, &candidate);
    if (iree_status_is_ok(status)) {
      *out_compiled =
          candidate.compiled &&
          candidate.artifact.target_artifact_data != NULL &&
          iree_byte_sequence_length(candidate.artifact.target_artifact_data) >
              0 &&
          candidate.artifact.executable_data != NULL &&
          iree_byte_sequence_length(candidate.artifact.executable_data) > 0;
    }
    loom_artifact_candidate_deinitialize(&candidate);
    return status;
  }

  const loom_target_emitter_t* emitter = request->producer.value.target_emitter;
  const loom_target_entry_options_t entry_options = {
      .function_versions = compile_options.function_versions,
      .diagnostic_sink = compile_options.diagnostic_sink,
      .source_resolver = compile_options.source_resolver,
      .max_errors = compile_options.max_errors,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
  // Emitted diagnostics live in the collector arena. Target emitters may
  // rewind their scratch arena before returning, so the two lifetimes must not
  // alias.
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  const loom_target_emit_request_t emit_request = {
      .target_environment = pipeline_options->target_environment,
      .low_descriptor_registry =
          &pipeline_options->low_descriptor_registry->registry,
      .module = module,
      .function_versions = compile_options.function_versions,
      .identifier = emitter->default_identifier,
      .diagnostic_emitter = loom_target_entry_emitter(&diagnostic_emitter),
      .scratch_arena = &scratch_arena,
      .allocator = allocator,
  };
  loom_target_emit_artifact_t artifact = {0};
  bool emitted = false;
  iree_status_t status = emitter->emit(&emit_request, &emitted, &artifact);
  if (iree_status_is_ok(status)) {
    *out_compiled = emitted;
  }
  loom_target_emit_artifact_release(&artifact);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_check_compile_request(
    const loom_compile_request_t* request, const loom_module_t* source_module,
    const loom_source_table_resolver_t* source_table,
    const loom_compile_pipeline_options_t* base_pipeline_options,
    loom_check_diagnostic_collector_t* collector,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator) {
  const iree_host_size_t initial_error_count = collector->error_count;
  loom_source_table_projection_t source_projection = {
      .table = *source_table, .arena = collector->arena};
  loom_compile_pipeline_options_t projected_options = *base_pipeline_options;
  projected_options.source_resolver = (loom_source_resolver_t){
      .fn = loom_source_table_resolve, .user_data = &source_projection.table};
  const loom_compile_pipeline_options_t* pipeline_options = &projected_options;
  const loom_module_t* const sources[] = {source_module};
  const loom_link_options_t link_options = {
      .module_name = source_module->name_id < source_module->strings.count
                         ? loom_string_table_get(&source_module->strings,
                                                 source_module->name_id)
                         : iree_string_view_empty(),
      .source_callback = {.fn = loom_source_table_project,
                          .user_data = &source_projection},
  };
  loom_module_t* module = NULL;
  iree_status_t status = loom_link_materialized_modules(
      sources, IREE_ARRAYSIZE(sources), &link_options, block_pool, allocator,
      &module);
  collector->module = module;
  uint32_t error_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_compile_materialize_request(request, pipeline_options,
                                              &source_projection, block_pool,
                                              allocator, &module, &error_count);
    collector->module = module;
  }
  loom_compile_pipeline_result_t pipeline_result = {0};
  if (iree_status_is_ok(status) &&
      collector->error_count == initial_error_count) {
    status = loom_compile_run_request_pipeline(
        request, module, pipeline_options, block_pool, &pipeline_result);
  }
  if (iree_status_is_ok(status) &&
      collector->error_count == initial_error_count) {
    status = loom_tooling_config_require_resolved_module(module, NULL);
  }
  if (iree_status_is_ok(status) &&
      collector->error_count == initial_error_count) {
    bool compiled = false;
    status = loom_check_compile_emit(request, module, pipeline_options,
                                     &pipeline_result, block_pool, allocator,
                                     &compiled);
    if (iree_status_is_ok(status) && !compiled &&
        collector->error_count == initial_error_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiler produced neither an artifact nor an error");
    }
  }
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  loom_module_free(module);
  collector->module = source_module;
  return status;
}

iree_status_t loom_check_execute_compile(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    const loom_check_compile_options_t* options,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  iree_status_t status =
      loom_target_environment_initialize_low_descriptor_registry(
          options->environment->target_environment, &low_registry);
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &collector},
      .max_errors = 20,
  };
  loom_input_request_t load_request = *input_request;
  loom_input_module_t input = {0};
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_print_context_initialize(
        &low_registry.registry, &collector.type_print_context);
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry.registry, &parse_options.low_asm_environment);
    loom_low_repr_environment_initialize(&low_registry.registry,
                                         &load_request.low_repr_environment);
    status =
        loom_check_load_input(test_case, &load_request, environment, context,
                              block_pool, &parse_options, allocator, &input);
  }
  collector.module = input.module;
  loom_compile_pipeline_options_t pipeline_options;
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.target_environment =
      options->environment->target_environment;
  pipeline_options.low_descriptor_registry = &low_registry;
  pipeline_options.cleanup_pattern_provider_set =
      options->environment->cleanup_pattern_provider_set;
  pipeline_options.diagnostic_sink = parse_options.diagnostic_sink;
  pipeline_options.source_resolver = loom_input_module_source_resolver(&input);
  pipeline_options.max_errors = parse_options.max_errors;
  if (iree_status_is_ok(status) && input.module != NULL &&
      collector.error_count == 0) {
    status = loom_check_compile_verify_input(input.module, &pipeline_options);
  }
  if (iree_status_is_ok(status) && input.module != NULL &&
      collector.error_count == 0) {
    loom_tooling_config_materialize_options_t config_options;
    loom_tooling_config_materialize_options_initialize(&config_options);
    config_options.config_set = options->config_set;
    status = loom_tooling_config_materialize_module(
        input.module, &config_options, block_pool, NULL);
  }
  loom_compile_request_t request = {0};
  if (iree_status_is_ok(status) && input.module != NULL &&
      collector.error_count == 0) {
    const loom_compile_request_options_t request_options = {
        .target = options->target,
    };
    status = loom_compile_request_resolve(
        input.module, &request_options,
        options->environment->artifact_provider_registry,
        options->environment->target_environment, &request);
    if (iree_status_is_ok(status) &&
        loom_compile_request_is_command(&request)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "compiler qualification requires a kernel or module product; "
          "command artifact sets require the loom-compile publication "
          "workflow");
    }
    if (iree_status_is_ok(status)) {
      pipeline_options.target_pipeline_options =
          request.producer.kind == LOOM_COMPILE_PRODUCER_ARTIFACT
              ? request.producer.value.artifact_provider
                    ->default_pipeline_options
              : request.producer.value.target_emitter->default_pipeline_options;
      pipeline_options.target_pipeline_options.sanitizer = options->sanitizer;
    }
  }
  if (iree_status_is_ok(status) && input.module != NULL &&
      collector.error_count == 0) {
    if (request.product == LOOM_COMPILE_PRODUCT_KERNEL) {
      // Testbench launches are independent deployment units. Compiling their
      // roots separately also preserves targets with per-artifact dispatch
      // ABIs.
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < input.module->symbols.count; ++i) {
        const loom_symbol_t* symbol = &input.module->symbols.entries[i];
        if (!loom_compile_request_symbol_is_implicit_root(
                input.module, request.product, symbol)) {
          continue;
        }
        const iree_string_view_t root =
            loom_string_table_get(&input.module->strings, symbol->name_id);
        request.roots = (iree_string_view_list_t){.count = 1, .values = &root};
        status = loom_check_compile_request(
            &request, input.module, &input.sources.table, &pipeline_options,
            &collector, block_pool, allocator);
      }
    } else {
      status = loom_check_compile_request(
          &request, input.module, &input.sources.table, &pipeline_options,
          &collector, block_pool, allocator);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_diagnostic_collector_finish(
        &collector, test_case, case_index, report, allocator, result);
    result->final_outcome = result->raw_outcome;
  }
  loom_input_module_deinitialize(&input);
  iree_arena_deinitialize(&arena);
  return status;
}
