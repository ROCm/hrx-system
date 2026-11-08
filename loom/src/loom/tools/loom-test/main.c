// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-test command-line adapter.

#include "loom/tools/loom-test/main.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tools/loom-test/options.h"
#include "loom/tools/loom-test/run.h"
#include "loom/util/stream.h"

int loom_test_main(int argc, char** argv,
                   const loom_test_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Executes check.case and check.scenario records from a normal Loom "
      "module.\n"
      "\n"
      "Usage:\n"
      "  loom-test file.loom --case=@smoke\n"
      "  cat module.loom | loom-test -\n"
      "  loom-test --agents_md\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_test_print_agents_markdown(stdout);
      return 0;
    }
  }

  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  loom_test_options_t test_options = {0};
  iree_status_t status = loom_test_options_from_flags(&test_options);
  if (iree_status_is_ok(status) && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-test accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  int exit_code = 1;
  if (iree_status_is_ok(status)) {
    loom_output_stream_t report_output;
    loom_output_stream_for_file(stdout, &report_output);
    const loom_test_file_run_options_t run_options = {
        .configuration = configuration,
        .test_options = &test_options,
        .input_path = argc < 2 ? iree_string_view_empty()
                               : iree_make_cstring_view(argv[1]),
        .report_output = &report_output,
        .host_allocator = iree_allocator_system(),
    };
    loom_test_run_result_t run_result = {0};
    status = loom_test_run_file(&run_options, &run_result);
    exit_code = run_result.exit_code;
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
