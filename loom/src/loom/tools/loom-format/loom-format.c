// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-format: converts Loom modules between text and bytecode formats.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/target/arch/cmd/provider.h"
#include "loom/target/configured/provider.h"
#include "loom/target/provider.h"
#include "loom/target/test/provider.h"
#include "loom/testing/test_file_format.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-format/convert.h"
#include "loom/util/stream.h"

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

static iree_status_t loom_format_stderr_diagnostic_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  FILE* file = (FILE*)user_data;
  loom_output_stream_t stream;
  loom_output_stream_for_file(file, &stream);
  return loom_diagnostic_format(diagnostic, &stream);
}

static iree_status_t loom_format_write_output(
    iree_string_view_t path, const loom_format_output_t* output,
    iree_allocator_t allocator) {
  return loom_tooling_write_output_file(
      path, iree_make_string_view((const char*)output->data, output->length),
      allocator);
}

static iree_status_t loom_format_check_canonical(
    iree_string_view_t filename, const iree_io_file_contents_t* contents,
    const loom_format_output_t* output) {
  const iree_string_view_t input_text =
      loom_tooling_file_contents_string_view(contents);
  const iree_string_view_t canonical_text =
      iree_make_string_view((const char*)output->data, output->length);
  if (iree_string_view_equal(input_text, canonical_text)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "'%.*s' is not canonically formatted; run loom-format --in-place "
      "<input>",
      (int)filename.size, filename.data);
}

static iree_status_t loom_format_process_input(
    iree_string_view_t input_path, iree_string_view_t output_path,
    loom_format_action_t action, loom_module_format_t input_format,
    loom_module_format_t output_format, loom_context_t* context,
    iree_arena_block_pool_t* block_pool,
    loom_text_low_asm_environment_t low_asm_environment,
    iree_allocator_t allocator, bool* out_changed) {
  *out_changed = false;

  const iree_string_view_t filename =
      loom_tooling_file_path_is_stdio(input_path)
          ? iree_make_cstring_view("<stdin>")
          : input_path;
  iree_io_file_contents_t* contents = NULL;
  loom_format_output_t output = {0};

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
    iree_string_builder_t formatted_source;
    iree_string_builder_initialize(allocator, &formatted_source);
    status = loom_test_file_format(
        loom_tooling_file_contents_string_view(contents), filename, context,
        block_pool, low_asm_environment, allocator, &formatted_source);
    if (iree_status_is_ok(status)) {
      output.length = iree_string_builder_size(&formatted_source);
      output.data =
          (uint8_t*)iree_string_builder_take_storage(&formatted_source);
    }
    iree_string_builder_deinitialize(&formatted_source);
  } else if (iree_status_is_ok(status)) {
    loom_format_convert_options_t convert_options = {
        .input_format = resolved_input_format,
        .output_format = output_format,
        .diagnostic_sink =
            {
                .fn = loom_format_stderr_diagnostic_sink,
                .user_data = stderr,
            },
        .low_asm_environment = low_asm_environment,
        .text_print_flags =
            LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_PREFER_LOW_ASM,
    };
    status =
        loom_format_convert(contents->const_buffer, filename, context,
                            block_pool, &convert_options, &output, allocator);
  }
  if (iree_status_is_ok(status)) {
    switch (action) {
      case LOOM_FORMAT_ACTION_CONVERT:
        status = loom_format_write_output(output_path, &output, allocator);
        break;
      case LOOM_FORMAT_ACTION_CHECK:
        status = loom_format_check_canonical(filename, contents, &output);
        break;
      case LOOM_FORMAT_ACTION_IN_PLACE: {
        const iree_string_view_t input_text =
            loom_tooling_file_contents_string_view(contents);
        const iree_string_view_t canonical_text =
            iree_make_string_view((const char*)output.data, output.length);
        if (!iree_string_view_equal(input_text, canonical_text)) {
          // Release the input mapping before opening the same path for
          // overwrite. Windows mappings can keep the file locked after the
          // original read handle closes.
          iree_io_file_contents_free(contents);
          contents = NULL;
          status = loom_format_write_output(input_path, &output, allocator);
          if (iree_status_is_ok(status)) {
            *out_changed = true;
          }
        }
        break;
      }
    }
  }

  loom_format_output_deinitialize(&output, allocator);
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

  loom_context_t context = {0};
  bool context_initialized = false;
  loom_target_environment_t target_environment = {0};
  bool target_environment_initialized = false;
  const loom_target_provider_t** target_providers = NULL;
  loom_target_provider_set_t target_provider_set = {0};
  loom_target_low_descriptor_registry_t low_descriptor_registry = {0};
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  loom_text_low_asm_environment_t low_asm_environment = {0};

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
    const loom_target_provider_set_t* configured_provider_set =
        loom_configured_target_provider_set();
    const iree_host_size_t provider_count =
        configured_provider_set->provider_count + 2;
    status = iree_allocator_malloc(allocator,
                                   provider_count * sizeof(*target_providers),
                                   (void**)&target_providers);
    if (iree_status_is_ok(status)) {
      if (configured_provider_set->provider_count > 0) {
        memcpy(target_providers, configured_provider_set->providers,
               configured_provider_set->provider_count *
                   sizeof(*target_providers));
      }
      target_providers[configured_provider_set->provider_count] =
          &loom_cmd_target_provider;
      target_providers[configured_provider_set->provider_count + 1] =
          &loom_test_target_provider;
      target_provider_set =
          loom_target_provider_set_make(target_providers, provider_count);
      status = loom_target_environment_initialize(&target_provider_set,
                                                  &target_environment);
      target_environment_initialized = iree_status_is_ok(status);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_context_initialize(allocator, &context);
    context_initialized = true;
    status =
        loom_tooling_context_register_tool_dialects_with_target_environment(
            &target_environment, &context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        &target_environment, &low_descriptor_registry);
  }
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
        &low_descriptor_registry.registry,
        loom_target_environment_low_asm_diagnostic_provider_list(
            &target_environment),
        &low_asm_storage, &low_asm_environment);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context);
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
          output_format, &context, &block_pool, low_asm_environment, allocator,
          &changed);
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

  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  if (target_environment_initialized) {
    loom_target_environment_deinitialize(&target_environment);
  }
  iree_allocator_free(allocator, target_providers);
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
