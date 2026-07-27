// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-benchmark-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-benchmark-loom/main.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tools/iree-benchmark-loom/help.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/run.h"

int iree_benchmark_loom_main(
    int argc, char** argv,
    const iree_benchmark_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = iree_allocator_system();
  iree_string_builder_t command_line_json;
  iree_string_builder_initialize(allocator, &command_line_json);
  iree_status_t status = iree_benchmark_loom_append_command_line_json(
      argc, argv, &command_line_json);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(1);
    return 1;
  }

  iree_benchmark_loom_set_usage(configuration->tool_name);
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (iree_benchmark_loom_cli_flags_request_agents_md()) {
    iree_benchmark_loom_print_agents_md(stdout);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(0);
    return 0;
  }

  int exit_code = 0;
  iree_benchmark_loom_options_t options = {0};
  status = iree_benchmark_loom_options_from_flags(&options);
  if (argc > 2) {
    status = iree_status_join(
        status, iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "iree-benchmark-loom accepts at most one input file or '-' "
                    "for stdin; got %d inputs",
                    argc - 1));
  }
  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  if (iree_status_is_ok(status)) {
    iree_benchmark_loom_file_run_options_t run_options = {
        .configuration = configuration,
        .benchmark_options = &options,
        .input_path = input_path,
        .command_line_json = iree_string_builder_view(&command_line_json),
        .event_sink = NULL,
        .host_allocator = allocator,
    };
    iree_benchmark_loom_run_result_t run_result = {0};
    status = iree_benchmark_loom_run_file(&run_options, &run_result);
    exit_code = run_result.exit_code;
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_string_builder_deinitialize(&command_line_json);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
