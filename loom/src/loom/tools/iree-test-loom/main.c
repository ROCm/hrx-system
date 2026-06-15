// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom: executes check.case records from ordinary Loom modules.

#include "loom/tools/iree-test-loom/main.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tooling/testbench/reference.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG(string, case, "",
          "Optional check.case symbol to execute, such as '@smoke'. Empty "
          "executes all cases in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for the selected case "
          "or cases. Negative executes all planned samples.");
IREE_FLAG_NAMED(int32_t, max_samples_per_case, "max-samples-per-case",
                LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
                "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used for HAL actual invocations. Use 'default', "
          "'none', '@symbol', or a comma-separated pass list.");

enum {
  // Generic HAL requirements plus target-linked requirement providers.
  IREE_TEST_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
};

static iree_status_t iree_test_loom_register_context(void* user_data,
                                                     loom_context_t* context) {
  const iree_test_loom_configuration_t* configuration =
      (const iree_test_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_tooling_context_register_tool_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

static iree_string_view_t iree_test_loom_normalize_case_name(
    iree_string_view_t case_name) {
  case_name = iree_string_view_trim(case_name);
  if (iree_string_view_starts_with(case_name, IREE_SV("@"))) {
    return iree_string_view_substr(case_name, 1, IREE_HOST_SIZE_MAX);
  }
  return case_name;
}

static bool iree_test_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(case_plan->name, selected_case_name);
}

static iree_status_t iree_test_loom_validate_sample_flag(
    iree_host_size_t sample_count, iree_host_size_t* out_sample_ordinal,
    bool* out_has_sample) {
  *out_sample_ordinal = 0;
  *out_has_sample = false;
  if (FLAG_sample < 0) {
    return iree_ok_status();
  }
  *out_sample_ordinal = (iree_host_size_t)FLAG_sample;
  *out_has_sample = true;
  if (*out_sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected case sample count %" PRIhsz,
                            *out_sample_ordinal, sample_count);
  }
  return iree_ok_status();
}

static bool iree_test_loom_case_has_actual_invocation(
    const loom_testbench_case_plan_t* case_plan) {
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    if (case_plan->invocations[i].kind == LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_test_loom_append_skipped_case(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_result_t* requirement_result,
    iree_string_builder_t* skipped_output, bool* inout_first_skipped_case) {
  if (!*inout_first_skipped_case) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(skipped_output, ","));
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(skipped_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"provider\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, requirement_result->provider));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"op\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream,
      loom_testbench_requirement_op_kind_name(requirement_result->op_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"code\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream,
      loom_testbench_requirement_skip_code_name(requirement_result->code)));
  if (!iree_string_view_is_empty(requirement_result->provider_code)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"provider_code\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, requirement_result->provider_code));
  }
  if (!iree_string_view_is_empty(requirement_result->display_message)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"display_message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, requirement_result->display_message));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  *inout_first_skipped_case = false;
  return iree_ok_status();
}

static iree_status_t iree_test_loom_configure_hal_actual_sequence(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_run_hal_testbench_context_t* hal_context,
    loom_testbench_case_execution_options_t* execution_options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(hal_context));

  execution_options->materializer.device = hal_context->runtime.device;
  execution_options->materializer.device_allocator =
      iree_hal_device_allocator(hal_context->runtime.device);
  execution_options->materializer.buffer_params =
      loom_run_hal_testbench_host_visible_buffer_params();

  const loom_run_hal_testbench_actual_sequence_options_t sequence_options = {
      .context = hal_context,
      .session = session,
      .target_environment = configuration->target_environment,
      .filename = run_module->filename,
      .source = run_module->source,
      .pipeline = iree_make_cstring_view(FLAG_pipeline),
      .test_module = module_plan->module,
      .case_plan = case_plan,
  };
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_sequence_initialize(
      &sequence_options, out_sequence));
  execution_options->invocation.invoke_actual =
      (loom_testbench_invocation_callback_t){
          .fn = loom_run_hal_testbench_actual_sequence_invoke,
          .user_data = out_sequence,
      };
  return iree_ok_status();
}

static iree_status_t iree_test_loom_run_case_samples(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* base_execution_options,
    loom_run_hal_testbench_context_t* hal_context,
    iree_arena_allocator_t* arena, iree_string_builder_t* sample_output,
    iree_string_builder_t* skipped_output, bool* inout_first_sample,
    bool* inout_first_skipped_case, iree_host_size_t* inout_sample_count,
    iree_host_size_t* inout_failed_sample_count,
    iree_host_size_t* inout_skipped_case_count) {
  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  loom_testbench_requirement_provider_t
      requirement_providers[IREE_TEST_LOOM_MAX_REQUIREMENT_PROVIDERS] = {0};
  iree_host_size_t requirement_provider_count = 0;
  loom_run_hal_testbench_requirement_provider_initialize(
      hal_context, &requirement_providers[requirement_provider_count++]);
  if (configuration->populate_requirement_providers.fn != NULL) {
    IREE_RETURN_IF_ERROR(configuration->populate_requirement_providers.fn(
        configuration->populate_requirement_providers.user_data, hal_context,
        IREE_ARRAYSIZE(requirement_providers), requirement_providers,
        &requirement_provider_count));
  }
  if (requirement_provider_count > IREE_ARRAYSIZE(requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "iree-test-loom requirement provider capacity exceeded");
  }
  loom_testbench_requirement_provider_registry_t requirement_registry = {0};
  loom_testbench_requirement_provider_registry_initialize(
      requirement_providers, requirement_provider_count, &requirement_registry);
  loom_testbench_requirement_result_t requirement_result = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_evaluate_case_requirements(
      module_plan->module, case_plan, &requirement_registry,
      &requirement_result));
  if (requirement_result.skipped) {
    IREE_RETURN_IF_ERROR(iree_test_loom_append_skipped_case(
        case_plan, &requirement_result, skipped_output,
        inout_first_skipped_case));
    ++*inout_skipped_case_count;
    return iree_ok_status();
  }

  iree_host_size_t selected_sample_ordinal = 0;
  bool has_selected_sample = false;
  IREE_RETURN_IF_ERROR(iree_test_loom_validate_sample_flag(
      case_plan->sample_count, &selected_sample_ordinal, &has_selected_sample));

  iree_status_t status = iree_ok_status();
  loom_testbench_case_execution_options_t execution_options =
      *base_execution_options;
  loom_run_hal_testbench_actual_sequence_t hal_actual_sequence = {0};
  loom_testbench_reference_matmul_oracle_options_t matmul_oracle_options = {0};
  loom_testbench_oracle_provider_t oracle_providers[2] = {0};
  bool hal_actual_sequence_initialized = false;
  if (iree_test_loom_case_has_actual_invocation(case_plan)) {
    status = iree_test_loom_configure_hal_actual_sequence(
        configuration, session, run_module, module_plan, case_plan, hal_context,
        &execution_options, &hal_actual_sequence);
    hal_actual_sequence_initialized = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      matmul_oracle_options =
          (loom_testbench_reference_matmul_oracle_options_t){
              .device = hal_context->runtime.device,
              .device_allocator =
                  iree_hal_device_allocator(hal_context->runtime.device),
              .result_buffer_params =
                  loom_run_hal_testbench_host_visible_buffer_params(),
              .host_allocator = execution_options.materializer.host_allocator,
          };
      loom_testbench_reference_matmul_oracle_provider_initialize(
          &matmul_oracle_options, &oracle_providers[0]);
      loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
          &matmul_oracle_options, &oracle_providers[1]);
      execution_options.invocation.oracle_providers =
          loom_make_testbench_oracle_provider_list(
              oracle_providers, IREE_ARRAYSIZE(oracle_providers));
    }
  }

  loom_testbench_prepared_case_t prepared_case = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_prepare_case_execution(
        &execution_options, module_plan, case_index, arena, &prepared_case);
  }

  bool executor_initialized = false;
  loom_testbench_case_executor_t executor = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_case_executor_initialize(
        &prepared_case, &execution_options, &executor);
    executor_initialized = iree_status_is_ok(status);
  }

  const iree_host_size_t begin_sample =
      has_selected_sample ? selected_sample_ordinal : 0;
  const iree_host_size_t end_sample = has_selected_sample
                                          ? selected_sample_ordinal + 1
                                          : case_plan->sample_count;
  for (iree_host_size_t sample_ordinal = begin_sample;
       iree_status_is_ok(status) && sample_ordinal < end_sample;
       ++sample_ordinal) {
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      if (!*inout_first_sample) {
        status = iree_string_builder_append_cstring(sample_output, ",");
      }
      if (iree_status_is_ok(status)) {
        loom_output_stream_t stream;
        loom_output_stream_for_builder(sample_output, &stream);
        status = loom_testbench_case_sample_result_write_json(&sample_result,
                                                              &stream);
      }
    }
    if (iree_status_is_ok(status)) {
      *inout_first_sample = false;
      ++*inout_sample_count;
      if (!sample_result.passed) {
        ++*inout_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  if (hal_actual_sequence_initialized) {
    loom_run_hal_testbench_actual_sequence_deinitialize(&hal_actual_sequence);
  }
  iree_arena_reset(arena);
  return status;
}

static iree_status_t iree_test_loom_write_report(
    iree_host_size_t case_count, iree_host_size_t sample_count,
    iree_host_size_t failed_sample_count, iree_host_size_t skipped_case_count,
    iree_string_view_t samples, iree_string_view_t skipped_cases,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "{\"format\":\"loom.test.v0\""));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"case_count\":%" PRIhsz, case_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"sample_count\":%" PRIhsz, sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failed_sample_count\":%" PRIhsz, failed_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"skipped_case_count\":%" PRIhsz, skipped_case_count));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"samples\":["));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, samples));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "],\"skipped_cases\":["));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(output, skipped_cases));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "]}\n"));
  return iree_ok_status();
}

static void iree_test_loom_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## iree-test-loom\n"
      "\n"
      "`iree-test-loom` executes `check.case` records from ordinary Loom\n"
      "modules and writes a structured `loom.test.v0` JSON report. Use it for\n"
      "correctness before promoting the same cases to `check.benchmark` rows.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "iree-test-loom module.loom\n"
      "iree-test-loom module.loom --case=@case_q8_block_unroll_wg64\n"
      "iree-test-loom module.loom --case=@sampled_choice --sample=1\n"
      "iree-test-loom module.loom --max-samples-per-case=16\n"
      "iree-test-loom module.loom --pipeline=@hal_actual_pipeline\n"
      "```\n"
      "\n"
      "`--case=@name` selects one checked case; empty selection runs cases in\n"
      "source order. `--sample=N` selects one planned sample for sampled "
      "cases.\n"
      "`--max-samples-per-case=N` bounds planning for generator-heavy cases.\n"
      "\n"
      "### Actual invocations\n"
      "\n"
      "A case can mix reference/oracle checks with HAL actual invocations. "
      "Actual\n"
      "invocations use the selected HAL artifact provider and execution "
      "provider\n"
      "linked into this binary. `--pipeline=default|none|@symbol|pass,list`\n"
      "controls the HAL actual compile pipeline.\n"
      "\n"
      "### Report shape\n"
      "\n"
      "```shell\n"
      "iree-test-loom module.loom --case=@smoke | jq '.failed_sample_count'\n"
      "iree-test-loom module.loom | jq '.samples[] | {case, sample_ordinal, "
      "passed}'\n"
      "iree-test-loom module.loom | jq '.skipped_cases[]? | {case, provider, "
      "op, code, provider_code}'\n"
      "```\n"
      "\n"
      "The report carries `case_count`, `sample_count`, "
      "`failed_sample_count`,\n"
      "`skipped_case_count`, `samples`, and `skipped_cases`. Skipped cases "
      "use\n"
      "stable `op` and `code` fields; `provider_code` is provider-defined and\n"
      "`display_message` is human-facing only. A nonzero failed\n"
      "sample count makes the process fail after the JSON report is "
      "written.\n");
}

int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Executes check.case records from a normal Loom module.\n"
      "\n"
      "Usage:\n"
      "  iree-test-loom file.loom --case=@smoke\n"
      "  cat module.loom | iree-test-loom -\n"
      "  iree-test-loom --agents_md\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      iree_test_loom_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_hal_testbench_context_t hal_context = {0};
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_string_builder_t sample_output;
  iree_string_builder_initialize(allocator, &sample_output);
  iree_string_builder_t skipped_output;
  iree_string_builder_initialize(allocator, &skipped_output);
  iree_string_builder_t report_output;
  iree_string_builder_initialize(allocator, &report_output);
  int exit_code = 0;

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-test-loom accepts at most one input file or '-' for stdin; got "
        "%d inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && FLAG_max_samples_per_case <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max-samples-per-case must be positive; got "
                              "%d",
                              (int)FLAG_max_samples_per_case);
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_test_loom_register_context,
        .user_data = (void*)configuration,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_testbench_context_initialize(
        configuration->hal_artifact_provider_registry, allocator, &hal_context);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }

  if (iree_status_is_ok(status)) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case =
        (iree_host_size_t)FLAG_max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    const iree_string_view_t selected_case_name =
        iree_test_loom_normalize_case_name(iree_make_cstring_view(FLAG_case));
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;

    iree_host_size_t selected_case_count = 0;
    iree_host_size_t sample_count = 0;
    iree_host_size_t failed_sample_count = 0;
    iree_host_size_t skipped_case_count = 0;
    bool first_sample = true;
    bool first_skipped_case = true;
    for (iree_host_size_t case_index = 0;
         iree_status_is_ok(status) && case_index < module_plan.case_count;
         ++case_index) {
      const loom_testbench_case_plan_t* case_plan =
          &module_plan.cases[case_index];
      if (!iree_test_loom_case_matches_selection(case_plan,
                                                 selected_case_name)) {
        continue;
      }
      ++selected_case_count;
      status = iree_test_loom_run_case_samples(
          configuration, &session, &run_module, &module_plan, case_index,
          &execution_options, &hal_context, &execution_arena, &sample_output,
          &skipped_output, &first_sample, &first_skipped_case, &sample_count,
          &failed_sample_count, &skipped_case_count);
    }
    if (iree_status_is_ok(status) && selected_case_count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND, "no check.case matched '%.*s'",
          (int)selected_case_name.size, selected_case_name.data);
    }
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_write_report(
          selected_case_count, sample_count, failed_sample_count,
          skipped_case_count, iree_string_builder_view(&sample_output),
          iree_string_builder_view(&skipped_output), &report_output);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_tooling_write_stdout(iree_string_builder_view(&report_output));
    }
    if (iree_status_is_ok(status) && failed_sample_count != 0) {
      exit_code = 1;
    }
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_string_builder_deinitialize(&report_output);
  iree_string_builder_deinitialize(&skipped_output);
  iree_string_builder_deinitialize(&sample_output);
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_hal_testbench_context_deinitialize(&hal_context);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
