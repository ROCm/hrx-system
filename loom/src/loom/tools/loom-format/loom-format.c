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
#include "loom/error/diagnostic.h"
#include "loom/target/configured/provider.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-format/convert.h"
#include "loom/util/stream.h"

IREE_FLAG(string, from, "auto", "Input format: auto, text, bc, or bytecode.");
IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");

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
      "cat source.loom | loom-format --from=text --to=bc "
      "--output=source.loombc\n"
      "loom-format source.loom --from=auto --to=text\n"
      "```\n"
      "\n"
      "`--from=auto` detects bytecode by the LOOM file magic and treats every\n"
      "other input as text. `--to=text` prints canonical text IR. `--to=bc`\n"
      "writes bytecode suitable for `loom-link --library=...` and\n"
      "`loom-compile` input.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-format",
      "Converts Loom IR modules between text and bytecode formats.\n"
      "\n"
      "Usage:\n"
      "  loom-format [--from=auto|text|bc] [--to=text|bc] [--output=file] "
      "[file]\n"
      "  cat module.loom | loom-format --from=text --to=bc "
      "--output=module.loombc\n"
      "  loom-format --agents_md\n"
      "\n"
      "Input defaults to stdin when no file is provided. Output defaults to "
      "stdout.\n"
      "The auto input format detects bytecode by the LOOM file magic and "
      "treats\n"
      "all other input as text.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_format_print_agents_markdown(stdout);
      return 0;
    }
  }
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  iree_io_file_contents_t* contents = NULL;
  loom_format_output_t output = {0};

  loom_module_format_t input_format = LOOM_MODULE_FORMAT_AUTO;
  loom_module_format_t output_format = LOOM_MODULE_FORMAT_TEXT;
  iree_status_t status = loom_module_format_parse(
      iree_make_cstring_view(FLAG_from), /*allow_auto=*/true, &input_format);
  if (iree_status_is_ok(status)) {
    status = loom_module_format_parse(iree_make_cstring_view(FLAG_to),
                                      /*allow_auto=*/false, &output_format);
  }
  if (iree_status_is_ok(status) && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-format accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status)) {
    loom_context_initialize(allocator, &context);
    context_initialized = true;
    status =
        loom_tooling_context_register_tool_dialects_with_target_environment(
            loom_configured_target_environment(), &context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context);
  }

  iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  iree_string_view_t filename =
      (argc < 2 || loom_tooling_file_path_is_stdio(input_path))
          ? iree_make_cstring_view("<stdin>")
          : input_path;

  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
  }
  if (iree_status_is_ok(status)) {
    loom_format_convert_options_t convert_options = {
        .input_format = input_format,
        .output_format = output_format,
        .diagnostic_sink =
            {
                .fn = loom_format_stderr_diagnostic_sink,
                .user_data = stderr,
            },
    };
    status =
        loom_format_convert(contents->const_buffer, filename, &context,
                            &block_pool, &convert_options, &output, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_format_write_output(iree_make_cstring_view(FLAG_output),
                                      &output, allocator);
  }

  int exit_code = 0;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_format_output_deinitialize(&output, allocator);
  iree_io_file_contents_free(contents);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return exit_code;
}
