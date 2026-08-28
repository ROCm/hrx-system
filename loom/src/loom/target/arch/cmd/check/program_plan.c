// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/check/program_plan.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/ops/command/ops.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tools/loom-check/diagnostics.h"

typedef struct loom_cmd_program_plan_check_options_t {
  // Command program symbol names selected in caller order.
  iree_string_view_t* root_names;
  // Number of entries in |root_names|.
  iree_host_size_t root_count;
} loom_cmd_program_plan_check_options_t;

static bool loom_cmd_program_plan_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("command-program"));
}

static iree_status_t loom_cmd_program_plan_check_parse_roots(
    iree_string_view_t roots_text, iree_arena_allocator_t* arena,
    loom_cmd_program_plan_check_options_t* options) {
  if (iree_string_view_is_empty(roots_text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program emit requires a root symbol");
  }

  iree_host_size_t root_count = 1;
  for (iree_host_size_t i = 0; i < roots_text.size; ++i) {
    if (roots_text.data[i] == ',') ++root_count;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, root_count,
                                                 sizeof(*options->root_names),
                                                 (void**)&options->root_names));

  iree_string_view_t remaining = roots_text;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    iree_string_view_t root = iree_string_view_empty();
    iree_string_view_t tail = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &root, &tail);
    if (!iree_string_view_starts_with(root, IREE_SV("@")) || root.size == 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program root expected a symbol name, got '%.*s'",
          (int)root.size, root.data);
    }
    root = iree_string_view_substr(root, 1, IREE_HOST_SIZE_MAX);
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(root, options->root_names[j])) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate command program root '@%.*s'",
                                (int)root.size, root.data);
      }
    }
    options->root_names[i] = root;
    remaining = tail;
  }
  options->root_count = root_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_parse_options(
    const loom_check_emit_provider_request_t* request,
    loom_cmd_program_plan_check_options_t* out_options) {
  *out_options = (loom_cmd_program_plan_check_options_t){0};
  IREE_ASSERT(
      iree_string_view_equal(request->target_name, IREE_SV("command-program")));

  iree_string_view_t roots_text = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &roots_text,
                         &option_text);
  roots_text = iree_string_view_trim(roots_text);
  option_text = iree_string_view_trim(option_text);
  IREE_RETURN_IF_ERROR(loom_cmd_program_plan_check_parse_roots(
      roots_text, request->case_arena, out_options));

  if (!iree_string_view_is_empty(option_text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected command program emit option '%.*s'",
                            (int)option_text.size, option_text.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_resolve_roots(
    loom_module_t* module, const loom_cmd_program_plan_check_options_t* options,
    iree_arena_allocator_t* arena, loom_symbol_ref_t** out_root_refs) {
  *out_root_refs = NULL;
  loom_symbol_ref_t* root_refs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, options->root_count, sizeof(*root_refs), (void**)&root_refs));
  for (iree_host_size_t i = 0; i < options->root_count; ++i) {
    const iree_string_view_t name = options->root_names[i];
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    if (name_id == LOOM_STRING_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "command program root '@%.*s' was not found",
                              (int)name.size, name.data);
    }
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "command program root '@%.*s' was not found",
                              (int)name.size, name.data);
    }
    loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
    if (!defining_op || !loom_command_program_def_isa(defining_op)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "symbol '@%.*s' is not a command.program.def",
                              (int)name.size, name.data);
    }
    root_refs[i] = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }
  *out_root_refs = root_refs;
  return iree_ok_status();
}

// Indexes the check module and prepares only the command implementation and
// exact kernel facets requested by its roots. Kernel bodies remain unopened.
static iree_status_t loom_cmd_program_plan_check_prepare_roots(
    loom_module_t* source_module, const loom_symbol_ref_t* source_root_refs,
    iree_host_size_t root_count, iree_arena_allocator_t* arena,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_cmd_program_plan_t* out_plan) {
  *out_valid = false;
  *out_plan = (loom_cmd_program_plan_t){0};

  loom_link_module_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(loom_link_module_index_allocate(
      source_module->context, block_pool, host_allocator, &index));
  iree_host_size_t provider_ordinal = 0;
  iree_status_t status = loom_link_module_index_add_materialized(
      index, source_module,
      &(loom_link_module_index_add_options_t){
          .provider_name = IREE_SV("command_program_check"),
      },
      &provider_ordinal);

  const loom_link_module_index_module_t* indexed_module = NULL;
  if (iree_status_is_ok(status)) {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, provider_ordinal);
    IREE_ASSERT(provider != NULL);
    IREE_ASSERT_EQ(provider->module_count, 1u);
    indexed_module =
        loom_link_module_index_module_at(index, provider->module_start_ordinal);
    IREE_ASSERT(indexed_module != NULL);
  }

  iree_host_size_t* root_symbol_ordinals = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, root_count,
                                       sizeof(*root_symbol_ordinals),
                                       (void**)&root_symbol_ordinals);
  }
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    IREE_ASSERT_EQ(source_root_refs[i].module_id, 0u);
    root_symbol_ordinals[i] =
        indexed_module->symbol_start_ordinal + source_root_refs[i].symbol_id;
  }
  if (iree_status_is_ok(status)) {
    const loom_link_plan_materialization_environment_t environment = {
        .context = source_module->context,
        .block_pool = block_pool,
        .allocator = host_allocator,
    };
    status = loom_cmd_program_plan_prepare_index(
        index, root_symbol_ordinals, root_count, /*options=*/NULL,
        loom_pass_builtin_registry(), diagnostic_emitter, &environment, arena,
        out_valid, out_plan);
  }

  loom_link_module_index_free(index);
  return status;
}

static iree_status_t loom_cmd_program_plan_check_print_roots(
    const loom_check_emit_provider_request_t* request,
    const loom_cmd_program_plan_t* plan) {
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_initialize(
      &request->low_registry->registry, &low_asm_environment);
  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM,
      .low_asm_environment = low_asm_environment,
  };
  for (iree_host_size_t i = 0; i < plan->root_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          &request->result->actual_output, "\n"));
    }
    IREE_RETURN_IF_ERROR(loom_text_print_operation_to_builder_with_options(
        plan->root_module, plan->roots[i].function_op,
        &request->result->actual_output, &print_options));
  }
  return iree_ok_status();
}

// Exercises the complete portable artifact boundary for every prepared root.
// Textual expectations continue to describe the source and Low semantics while
// this closure proves that the same production plan serializes into an artifact
// accepted by the artifact's untrusted-byte parser.
static iree_status_t loom_cmd_program_plan_check_roundtrip_artifacts(
    const loom_cmd_program_plan_t* plan, iree_allocator_t host_allocator) {
  loom_cmd_program_artifact_set_t artifact_set = {0};
  IREE_RETURN_IF_ERROR(
      loom_cmd_program_artifact_set_build(plan, &artifact_set, host_allocator));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < artifact_set.programs.count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_artifact_t* artifact =
        &artifact_set.programs.values[i];
    if (iree_status_is_ok(status)) {
      loom_cmd_program_t program = {0};
      status = loom_cmd_program_parse(
          iree_make_const_byte_span(artifact->data.data,
                                    artifact->data.data_length),
          &program);
    }
  }
  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  return status;
}

static iree_status_t loom_cmd_program_plan_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_cmd_program_plan_check_options_t options = {0};
  IREE_RETURN_IF_ERROR(
      loom_cmd_program_plan_check_parse_options(request, &options));

  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.default_pipeline =
      LOOM_COMPILE_DEFAULT_PIPELINE_EXPANDED_SOURCE;
  pipeline_options.target_environment =
      request->environment->target_environment;
  pipeline_options.low_descriptor_registry = request->low_registry;
  pipeline_options.diagnostic_sink =
      (loom_diagnostic_sink_t){.fn = loom_check_diagnostic_collector_sink,
                               .user_data = request->diagnostic_collector};
  pipeline_options.source_resolver = request->source_resolver;
  pipeline_options.max_errors = 20;

  loom_compile_pipeline_result_t pipeline_result = {0};
  iree_status_t status =
      loom_compile_run_pipeline(request->module, &pipeline_options,
                                request->block_pool, &pipeline_result);
  loom_cmd_program_plan_t plan = {0};
  bool plan_valid = false;
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0) {
    loom_symbol_ref_t* source_root_refs = NULL;
    status = loom_cmd_program_plan_check_resolve_roots(
        request->module, &options, request->case_arena, &source_root_refs);
    if (iree_status_is_ok(status)) {
      loom_check_diagnostic_emitter_capture_t capture = {
          .diagnostic_collector = request->diagnostic_collector,
          .module = request->module,
          .source_resolver = request->source_resolver,
          .emitter = LOOM_EMITTER_PASS,
      };
      status = loom_cmd_program_plan_check_prepare_roots(
          request->module, source_root_refs, options.root_count,
          request->case_arena, request->block_pool, request->host_allocator,
          (iree_diagnostic_emitter_t){
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &capture,
          },
          &plan_valid, &plan);
      if (iree_status_is_ok(status) && !plan_valid &&
          capture.emission_count == 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "command program preparation failed without a diagnostic");
      }
    }
  }
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0 &&
      plan_valid) {
    status = loom_cmd_program_plan_check_roundtrip_artifacts(
        &plan, request->host_allocator);
  }
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0 &&
      plan_valid) {
    status = loom_cmd_program_plan_check_print_roots(request, &plan);
  }
  loom_cmd_program_plan_deinitialize(&plan);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  return status;
}

static iree_status_t loom_cmd_program_plan_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "command-program");
}

const loom_check_emit_provider_t loom_cmd_program_plan_check_emit_provider = {
    .name = IREE_SVL("command program plan"),
    .match = loom_cmd_program_plan_check_emit_provider_matches,
    .execute = loom_cmd_program_plan_check_emit_provider_execute,
    .append_names = loom_cmd_program_plan_check_emit_provider_append_names,
};
