// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/check/program_plan.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tools/loom-check/diagnostics.h"

typedef enum loom_cmd_program_plan_check_output_e {
  LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_PROGRAM = 0,
  LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_LAUNCH_CONFIG = 1,
  LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_KERNEL = 2,
} loom_cmd_program_plan_check_output_t;

typedef struct loom_cmd_program_plan_check_options_t {
  // Prepared product selected by the emit target name.
  loom_cmd_program_plan_check_output_t output;
  // Command program symbol names selected in caller order.
  iree_string_view_t* root_names;
  // Number of entries in |root_names|.
  iree_host_size_t root_count;
  // Root-local kernel dependency selected by command-kernel.
  uint32_t dependency_index;
} loom_cmd_program_plan_check_options_t;

static bool loom_cmd_program_plan_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("command-program")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("command-launch-config")) ||
         iree_string_view_equal(target_name, IREE_SV("command-kernel"));
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
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("command-program"))) {
    out_options->output = LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_PROGRAM;
  } else if (iree_string_view_equal(request->target_name,
                                    IREE_SV("command-launch-config"))) {
    out_options->output = LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_LAUNCH_CONFIG;
  } else {
    out_options->output = LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_KERNEL;
  }

  iree_string_view_t roots_text = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &roots_text,
                         &option_text);
  roots_text = iree_string_view_trim(roots_text);
  option_text = iree_string_view_trim(option_text);
  IREE_RETURN_IF_ERROR(loom_cmd_program_plan_check_parse_roots(
      roots_text, request->case_arena, out_options));

  if (out_options->output != LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_KERNEL) {
    if (!iree_string_view_is_empty(option_text)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unexpected command program emit option '%.*s'",
                              (int)option_text.size, option_text.data);
    }
    return iree_ok_status();
  }
  if (out_options->root_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-kernel requires exactly one command program root");
  }

  iree_string_view_t dependency_name = iree_string_view_empty();
  iree_string_view_t dependency_value = iree_string_view_empty();
  iree_string_view_split(option_text, '=', &dependency_name, &dependency_value);
  if (!iree_string_view_equal(dependency_name, IREE_SV("dependency")) ||
      dependency_value.size == 0 ||
      !iree_string_view_atoi_uint32(dependency_value,
                                    &out_options->dependency_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-kernel requires a dependency=N option");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_resolve_roots(
    loom_module_t* module, const loom_cmd_program_plan_check_options_t* options,
    iree_arena_allocator_t* arena, const loom_op_t*** out_root_ops) {
  const loom_op_t** root_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, options->root_count, sizeof(*root_ops), (void**)&root_ops));
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
    root_ops[i] = defining_op;
  }
  *out_root_ops = root_ops;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_print_module(
    const loom_check_emit_provider_request_t* request,
    const loom_module_t* module) {
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_initialize(
      &request->low_registry->registry, &low_asm_environment);
  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM,
      .low_asm_environment = low_asm_environment,
  };
  return loom_text_print_module_to_builder_with_options(
      module, &request->result->actual_output, &print_options);
}

// Exercises the complete portable artifact boundary for every prepared root.
// Textual expectations continue to describe the source and Low semantics while
// this closure proves that the same production plan serializes into an artifact
// accepted by the artifact's untrusted-byte parser.
static bool loom_cmd_program_plan_check_tables_equal(
    loom_cmd_program_table_t lhs, loom_cmd_program_table_t rhs) {
  return lhs.data == rhs.data && lhs.count == rhs.count;
}

static bool loom_cmd_program_plan_check_views_equal(
    const loom_cmd_program_t* lhs, const loom_cmd_program_t* rhs) {
  return lhs->storage.data == rhs->storage.data &&
         lhs->storage.data_length == rhs->storage.data_length &&
         lhs->requirements.fixed_buffer_count ==
             rhs->requirements.fixed_buffer_count &&
         lhs->requirements.rebindable_binding_count ==
             rhs->requirements.rebindable_binding_count &&
         lhs->requirements.executable_count ==
             rhs->requirements.executable_count &&
         lhs->requirements.entry_count == rhs->requirements.entry_count &&
         lhs->requirements.transient.binding_index ==
             rhs->requirements.transient.binding_index &&
         lhs->requirements.transient.required_byte_length ==
             rhs->requirements.transient.required_byte_length &&
         lhs->requirements.transient.minimum_alignment ==
             rhs->requirements.transient.minimum_alignment &&
         lhs->requirements.launch_counts.binding_index ==
             rhs->requirements.launch_counts.binding_index &&
         lhs->requirements.launch_counts.required_byte_length ==
             rhs->requirements.launch_counts.required_byte_length &&
         lhs->requirements.launch_counts.minimum_alignment ==
             rhs->requirements.launch_counts.minimum_alignment &&
         loom_cmd_program_plan_check_tables_equal(lhs->buffer_refs,
                                                  rhs->buffer_refs) &&
         loom_cmd_program_plan_check_tables_equal(lhs->entry_schemas,
                                                  rhs->entry_schemas) &&
         loom_cmd_program_plan_check_tables_equal(lhs->entry_schema_kinds,
                                                  rhs->entry_schema_kinds) &&
         lhs->argument_data.data == rhs->argument_data.data &&
         lhs->argument_data.data_length == rhs->argument_data.data_length &&
         loom_cmd_program_plan_check_tables_equal(lhs->commands,
                                                  rhs->commands) &&
         loom_cmd_program_plan_check_tables_equal(lhs->parameter_roots,
                                                  rhs->parameter_roots) &&
         loom_cmd_program_plan_check_tables_equal(lhs->parameters,
                                                  rhs->parameters) &&
         lhs->parameter_keys.data == rhs->parameter_keys.data &&
         lhs->parameter_keys.data_length == rhs->parameter_keys.data_length;
}

static iree_status_t loom_cmd_program_plan_check_roundtrip_artifacts(
    const loom_cmd_program_plan_t* plan, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < plan->root_count && iree_status_is_ok(status); ++i) {
    iree_byte_span_t data = iree_byte_span_empty();
    loom_cmd_program_t direct_program = {0};
    status = loom_cmd_program_plan_serialize_root(
        plan, i, block_pool, &data, &direct_program, host_allocator);
    if (iree_status_is_ok(status)) {
      loom_cmd_program_t parsed_program = {0};
      status = loom_cmd_program_parse(
          iree_make_const_byte_span(data.data, data.data_length),
          &parsed_program);
      if (iree_status_is_ok(status) && !loom_cmd_program_plan_check_views_equal(
                                           &direct_program, &parsed_program)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "compiler-bound command view disagrees with artifact parser");
      }
    }
    iree_allocator_free(host_allocator, data.data);
  }
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
    const loom_op_t** root_ops = NULL;
    status = loom_cmd_program_plan_check_resolve_roots(
        request->module, &options, request->case_arena, &root_ops);
    if (iree_status_is_ok(status)) {
      loom_check_diagnostic_emitter_capture_t capture = {
          .diagnostic_collector = request->diagnostic_collector,
          .module = request->module,
          .source_resolver = request->source_resolver,
          .emitter = LOOM_EMITTER_PASS,
      };
      status = loom_cmd_program_plan_prepare(
          request->module, root_ops, options.root_count,
          loom_pass_builtin_registry(),
          (iree_diagnostic_emitter_t){
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &capture,
          },
          request->block_pool, &plan_valid, &plan, request->host_allocator);
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
        &plan, request->block_pool, request->host_allocator);
  }
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0 &&
      plan_valid) {
    const loom_module_t* output_module = NULL;
    switch (options.output) {
      case LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_PROGRAM:
        output_module = plan.root_module;
        break;
      case LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_LAUNCH_CONFIG:
        output_module = plan.launch_module;
        break;
      case LOOM_CMD_PROGRAM_PLAN_CHECK_OUTPUT_KERNEL: {
        const loom_cmd_program_root_t* root = &plan.roots[0];
        if (options.dependency_index >= root->executable_count) {
          status = iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "command root '@%.*s' has %u kernel executables; dependency "
              "%u is out of range",
              (int)options.root_names[0].size, options.root_names[0].data,
              root->executable_count, options.dependency_index);
          break;
        }
        const uint32_t unit_index =
            root->executable_unit_indices[options.dependency_index];
        output_module = plan.dependency_units[unit_index].module;
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_cmd_program_plan_check_print_module(request, output_module);
    }
  }
  loom_cmd_program_plan_deinitialize(&plan);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  return status;
}

static iree_status_t loom_cmd_program_plan_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "command-program, command-launch-config, command-kernel");
}

const loom_check_emit_provider_t loom_cmd_program_plan_check_emit_provider = {
    .name = IREE_SVL("command program plan"),
    .match = loom_cmd_program_plan_check_emit_provider_matches,
    .execute = loom_cmd_program_plan_check_emit_provider_execute,
    .append_names = loom_cmd_program_plan_check_emit_provider_append_names,
};
