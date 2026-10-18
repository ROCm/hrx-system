// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-format: converts Loom modules between text and bytecode formats.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "loom/testing/target_environment.h"
#include "loom/testing/test_file_format.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/io/module_format.h"
#include "loomc/iree.h"
#include "loomc/target/configured_ir.h"

IREE_FLAG(string, from, "auto", "Input format: auto, text, bc, or bytecode.");
IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG(bool, check, false,
          "Checks that .loom modules or .loom-test input sections are already "
          "in canonical form without writing output.");
IREE_FLAG(bool, in_place, false,
          "Formats one or more .loom modules or .loom-test input sections in "
          "place.");

typedef enum loom_format_action_e {
  LOOM_FORMAT_ACTION_CONVERT = 0,
  LOOM_FORMAT_ACTION_CHECK,
  LOOM_FORMAT_ACTION_IN_PLACE,
} loom_format_action_t;

static const char* loom_format_action_flag(loom_format_action_t action) {
  switch (action) {
    case LOOM_FORMAT_ACTION_CONVERT:
      return "conversion";
    case LOOM_FORMAT_ACTION_CHECK:
      return "--check";
    case LOOM_FORMAT_ACTION_IN_PLACE:
      return "--in-place";
  }
  return "unknown action";
}

static loomc_source_format_t loom_format_source_format(
    loom_module_format_t format) {
  switch (format) {
    case LOOM_MODULE_FORMAT_TEXT:
      return LOOMC_SOURCE_FORMAT_TEXT;
    case LOOM_MODULE_FORMAT_BYTECODE:
      return LOOMC_SOURCE_FORMAT_BYTECODE;
    default:
      return LOOMC_SOURCE_FORMAT_UNKNOWN;
  }
}

static iree_status_t loom_format_check_result(const loomc_result_t* result) {
  bool succeeded = false;
  IREE_RETURN_IF_ERROR(
      loom_tooling_cli_print_loomc_result(stderr, result, &succeeded));
  return succeeded ? iree_ok_status()
                   : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "invalid input module");
}

static iree_status_t loom_format_convert_source(
    iree_const_byte_span_t input, iree_string_view_t filename,
    loom_module_format_t input_format, loom_module_format_t output_format,
    loomc_context_t* context, loomc_workspace_t* workspace,
    iree_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = loom_format_source_format(input_format),
      .identifier = loomc_string_view_from_iree(filename),
      .contents = loomc_byte_span_from_iree(input),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_source_create(&source_options, compiler_allocator, &source)));
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_deserialize_from_source(
          context, workspace, source, NULL, compiler_allocator, &module,
          &result));
  loomc_source_release(source);
  if (iree_status_is_ok(status)) {
    status = loom_format_check_result(result);
  }
  loomc_result_release(result);
  result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_module_verify_options_t verify_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
        .structure_size = sizeof(verify_options),
        .max_errors = 100,
        .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
    };
    status = iree_status_from_loomc(loomc_module_verify(
        module, &verify_options, compiler_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_format_check_result(result);
  }
  if (iree_status_is_ok(status)) {
    const loomc_module_serialize_options_t serialize_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        .structure_size = sizeof(serialize_options),
        .format = loom_format_source_format(output_format),
    };
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &serialize_options, compiler_allocator, out_source));
  }
  loomc_result_release(result);
  loomc_module_release(module);
  return status;
}

static iree_status_t loom_format_create_context(iree_allocator_t allocator,
                                                loomc_context_t** out_context) {
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_target_environment_t* inputs[2] = {NULL, NULL};
  loomc_target_environment_t* environment = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured_ir(
          compiler_allocator, &inputs[0]));
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_test_ir(
        compiler_allocator, &inputs[1]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create(
        IREE_ARRAYSIZE(inputs), inputs, compiler_allocator, &environment));
  }
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, compiler_allocator, out_context));
  }
  loomc_target_environment_release(environment);
  loomc_target_environment_release(inputs[1]);
  loomc_target_environment_release(inputs[0]);
  return status;
}

static iree_status_t loom_format_process_input(
    iree_string_view_t input_path, iree_string_view_t output_path,
    loom_format_action_t action, loom_module_format_t input_format,
    loom_module_format_t output_format, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, bool* out_changed) {
  *out_changed = false;

  const iree_string_view_t filename =
      loom_tooling_file_path_is_stdio(input_path)
          ? iree_make_cstring_view("<stdin>")
          : input_path;
  iree_io_file_contents_t* contents = NULL;
  loomc_source_t* output_source = NULL;
  iree_string_builder_t formatted_source;
  iree_string_builder_initialize(allocator, &formatted_source);
  iree_string_view_t output = iree_string_view_empty();

  iree_status_t status =
      loom_tooling_read_input_file(input_path, allocator, &contents);
  loom_module_format_t resolved_input_format = input_format;
  if (iree_status_is_ok(status) &&
      resolved_input_format == LOOM_MODULE_FORMAT_AUTO) {
    resolved_input_format =
        loom_module_format_detect_input(contents->const_buffer);
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT &&
      resolved_input_format != LOOM_MODULE_FORMAT_TEXT) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s requires text input, but '%.*s' contains %s",
                              loom_format_action_flag(action),
                              (int)filename.size, filename.data,
                              loom_module_format_name(resolved_input_format));
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT &&
      iree_string_view_ends_with(filename, IREE_SV(".loom-test"))) {
    status = loom_test_file_format(
        loom_tooling_file_contents_string_view(contents), filename, context,
        workspace, block_pool, allocator, &formatted_source);
    output = iree_string_builder_view(&formatted_source);
  } else if (iree_status_is_ok(status)) {
    status = loom_format_convert_source(
        contents->const_buffer, filename, resolved_input_format, output_format,
        context, workspace, allocator, &output_source);
    if (iree_status_is_ok(status)) {
      const loomc_byte_span_t bytes = loomc_source_contents(output_source);
      output =
          iree_make_string_view((const char*)bytes.data, bytes.data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    switch (action) {
      case LOOM_FORMAT_ACTION_CONVERT:
        status = loom_tooling_write_output_file(output_path, output, allocator);
        break;
      case LOOM_FORMAT_ACTION_CHECK:
        if (!iree_string_view_equal(
                loom_tooling_file_contents_string_view(contents), output)) {
          status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                    "'%.*s' is not canonically formatted; run "
                                    "loom-format --in-place <input>",
                                    (int)filename.size, filename.data);
        }
        break;
      case LOOM_FORMAT_ACTION_IN_PLACE: {
        const iree_string_view_t input_text =
            loom_tooling_file_contents_string_view(contents);
        if (!iree_string_view_equal(input_text, output)) {
          // Release the input mapping before opening the same path for
          // overwrite. Windows mappings can keep the file locked after the
          // original read handle closes.
          iree_io_file_contents_free(contents);
          contents = NULL;
          status =
              loom_tooling_write_output_file(input_path, output, allocator);
          if (iree_status_is_ok(status)) {
            *out_changed = true;
          }
        }
        break;
      }
    }
  }

  loomc_source_release(output_source);
  iree_string_builder_deinitialize(&formatted_source);
  iree_io_file_contents_free(contents);
  return status;
}

static void loom_format_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-format\n"
      "\n"
      "`loom-format` converts Loom modules between text `.loom` and bytecode\n"
      "`.loombc` encodings. Use it to prepackage provider libraries or to\n"
      "round-trip a generated module before linking or compiling.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-format source.loom --from=text --to=bc --output=source.loombc\n"
      "loom-format source.loombc --from=bc --to=text --output=source.loom\n"
      "loom-format source.loom --check\n"
      "loom-format source.loom-test --check\n"
      "loom-format --check first.loom second.loom\n"
      "loom-format --in-place first.loom second.loom\n"
      "cat source.loom | loom-format --from=text --to=bc "
      "--output=source.loombc\n"
      "loom-format source.loom --from=auto --to=text\n"
      "```\n"
      "\n"
      "`--from=auto` detects bytecode by the LOOM file magic and treats every\n"
      "other input as text. `--to=text` prints canonical text IR. `--to=bc`\n"
      "writes bytecode suitable for `loom-link --library=...` and\n"
      "`loom-compile` input. The complete module is verified before either\n"
      "output is written; external calls require a matching `func.decl` or\n"
      "definition. Text conversion resolves target-low syntax using the\n"
      "configured target descriptors. For `.loom-test` files, `--check` and\n"
      "`--in-place` canonicalize module input sections while preserving\n"
      "expected output and explicitly authored source-conversion fixtures.\n"
      "Precisely annotated invalid input is also preserved. `--check` writes\n"
      "no output; `--in-place` rewrites only noncanonical files after their\n"
      "input modules have parsed. Verify-mode semantic failures require a\n"
      "matching diagnostic annotation.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-format",
      "Converts Loom IR modules between text and bytecode formats.\n"
      "\n"
      "Usage:\n"
      "  loom-format [--from=auto|text|bc] [--to=text|bc] [--output=file] "
      "[file]\n"
      "  loom-format [--from=auto|text] --check [file ...]\n"
      "  loom-format [--from=auto|text] --in-place file ...\n"
      "  cat module.loom | loom-format --from=text --to=bc "
      "--output=module.loombc\n"
      "  loom-format --agents_md\n"
      "\n"
      "Input defaults to stdin when no file is provided. Output defaults to "
      "stdout.\n"
      "Conversion accepts one input. Check and in-place modes accept multiple "
      "text inputs, including .loom-test containers.\n"
      "The auto input format detects bytecode by the LOOM file magic and "
      "treats\n"
      "all other input as text. The complete module is verified before output "
      "is written.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_format_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;

  loom_module_format_t input_format = LOOM_MODULE_FORMAT_AUTO;
  loom_module_format_t output_format = LOOM_MODULE_FORMAT_TEXT;
  loom_format_action_t action = LOOM_FORMAT_ACTION_CONVERT;
  iree_status_t status = loom_module_format_parse(
      iree_make_cstring_view(FLAG_from), /*allow_auto=*/true, &input_format);
  if (iree_status_is_ok(status)) {
    status = loom_module_format_parse(iree_make_cstring_view(FLAG_to),
                                      /*allow_auto=*/false, &output_format);
  }
  if (iree_status_is_ok(status)) {
    if (FLAG_check && FLAG_in_place) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "--check and --in-place are mutually exclusive");
    } else if (FLAG_check) {
      action = LOOM_FORMAT_ACTION_CHECK;
    } else if (FLAG_in_place) {
      action = LOOM_FORMAT_ACTION_IN_PLACE;
    }
  }
  if (iree_status_is_ok(status) && action == LOOM_FORMAT_ACTION_CONVERT &&
      argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-format accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT &&
      output_format != LOOM_MODULE_FORMAT_TEXT) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s requires --to=text",
                         loom_format_action_flag(action));
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT &&
      input_format == LOOM_MODULE_FORMAT_BYTECODE) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s requires text input",
                         loom_format_action_flag(action));
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT &&
      !loom_tooling_file_path_is_stdio(iree_make_cstring_view(FLAG_output))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s does not accept --output",
                              loom_format_action_flag(action));
  }
  if (iree_status_is_ok(status) && action == LOOM_FORMAT_ACTION_IN_PLACE &&
      argc < 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--in-place requires at least one input file");
  }
  if (iree_status_is_ok(status) && action != LOOM_FORMAT_ACTION_CONVERT) {
    for (int i = 1; i < argc; ++i) {
      const iree_string_view_t input_path = iree_make_cstring_view(argv[i]);
      if (loom_tooling_file_path_is_stdio(input_path) &&
          (action == LOOM_FORMAT_ACTION_IN_PLACE || argc > 2)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "%s batch inputs must be file paths; stdin is supported only by "
            "single-input --check",
            loom_format_action_flag(action));
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_format_create_context(allocator, &context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_workspace_create(
        NULL, loomc_allocator_from_iree(allocator), &workspace));
  }

  int exit_code = 0;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  } else {
    const int input_count = argc < 2 ? 1 : argc - 1;
    int failure_count = 0;
    int changed_count = 0;
    for (int i = 0; i < input_count; ++i) {
      const iree_string_view_t input_path =
          argc < 2 ? iree_string_view_empty()
                   : iree_make_cstring_view(argv[i + 1]);
      bool changed = false;
      iree_status_t input_status = loom_format_process_input(
          input_path, iree_make_cstring_view(FLAG_output), action, input_format,
          output_format, context, workspace, &block_pool, allocator, &changed);
      if (!iree_status_is_ok(input_status)) {
        iree_status_fprint(stderr, input_status);
        iree_status_free(input_status);
        ++failure_count;
        continue;
      }
      if (changed) {
        ++changed_count;
        printf("loom-format: formatted %.*s\n", (int)input_path.size,
               input_path.data);
      }
    }
    if (action == LOOM_FORMAT_ACTION_IN_PLACE) {
      FILE* summary_stream = failure_count == 0 ? stdout : stderr;
      fprintf(summary_stream,
              "loom-format: %d formatted, %d unchanged, %d failed\n",
              changed_count, input_count - changed_count - failure_count,
              failure_count);
    } else if (action == LOOM_FORMAT_ACTION_CHECK && input_count > 1) {
      FILE* summary_stream = failure_count == 0 ? stdout : stderr;
      fprintf(summary_stream, "loom-format: %s (%d checked, %d failed)\n",
              failure_count == 0 ? "PASS" : "FAIL", input_count, failure_count);
    }
    if (failure_count != 0) {
      exit_code = 1;
    }
  }

  loomc_workspace_release(workspace);
  loomc_context_release(context);
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
