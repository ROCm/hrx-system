// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-check command-line implementation.

#include "loom/tools/loom-check/main.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tools/loom-check/file.h"
#include "loom/tools/loom-check/json_output.h"
#include "loom/tools/loom-check/output.h"

IREE_FLAG(bool, update, false,
          "Rewrite test files with actual output in the expected\n"
          "section (after // ----). Inserts the separator for non-empty\n"
          "output when absent.\n"
          "Cannot be used with stdin or verify mode.");
IREE_FLAG(bool, verbose, false,
          "Print PASS/FAIL/SKIP for every case, not just failures.");

typedef struct loom_check_json_flag_t {
  bool enabled;
  loom_check_json_output_mode_t output_mode;
} loom_check_json_flag_t;

static const char* loom_check_json_output_mode_name(
    loom_check_json_output_mode_t output_mode) {
  switch (output_mode) {
    case LOOM_CHECK_JSON_OUTPUT_FAILURES:
      return "failures";
    case LOOM_CHECK_JSON_OUTPUT_SUMMARY:
      return "summary";
    case LOOM_CHECK_JSON_OUTPUT_ALL:
      return "all";
  }
  return "unknown";
}

static iree_status_t loom_check_parse_json_flag(iree_string_view_t flag_name,
                                                void* storage,
                                                iree_string_view_t value) {
  (void)flag_name;
  loom_check_json_flag_t* flag = (loom_check_json_flag_t*)storage;

  flag->enabled = true;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, iree_make_cstring_view("failures"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_FAILURES;
  } else if (iree_string_view_equal(value, iree_make_cstring_view("summary"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_SUMMARY;
  } else if (iree_string_view_equal(value, iree_make_cstring_view("all"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_ALL;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid --json mode '%.*s'; expected failures, summary, or all",
        (int)value.size, value.data);
  }

  return iree_ok_status();
}

static void loom_check_print_json_flag(iree_string_view_t flag_name,
                                       void* storage, FILE* file) {
  const loom_check_json_flag_t* flag = (const loom_check_json_flag_t*)storage;
  if (!flag->enabled) {
    fprintf(file, "# --%.*s[=failures|summary|all]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  fprintf(file, "--%.*s=%s\n", (int)flag_name.size, flag_name.data,
          loom_check_json_output_mode_name(flag->output_mode));
}

static loom_check_json_flag_t FLAG_json = {
    .enabled = false,
    .output_mode = LOOM_CHECK_JSON_OUTPUT_FAILURES,
};
IREE_FLAG_CALLBACK(loom_check_parse_json_flag, loom_check_print_json_flag,
                   &FLAG_json, json,
                   "Structured JSON output to stdout. Bare --json is the same\n"
                   "as --json=failures. Modes: failures, summary, all.");

static void loom_check_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-check\n"
      "\n"
      "`loom-check` is the Loom IR golden-test runner for `.loom-test` files.\n"
      "Checked-in Bazel test targets are the stable unit because they "
      "preserve\n"
      "the CI test environment and carry the fixture path they own.\n"
      "\n"
      "### Verify tests\n"
      "\n"
      "```shell\n"
      "iree-bazel-test --config=asan //loom/src/loom/...\n"
      "iree-bazel-test --config=asan "
      "//loom/src/loom/tools/loom-check/test:test\n"
      "```\n"
      "\n"
      "### Update expected output\n"
      "\n"
      "Pass the update flag through Bazel with `--test_arg=--update`:\n"
      "\n"
      "```shell\n"
      "iree-bazel-test --config=asan <loom-check-test-target> "
      "--test_arg=--update\n"
      "```\n"
      "\n"
      "`iree-bazel-test` detects this flag and uses Bazel's standalone\n"
      "TestRunner strategy so update-capable tests can rewrite checked-in\n"
      "fixture files. After building a target, its generated executable can\n"
      "also be run directly and will append the fixture path automatically:\n"
      "\n"
      "```shell\n"
      "iree-bazel-build <loom-check-test-target>\n"
      "bazel-bin/path/to/generated-test --update\n"
      "```\n"
      "\n"
      "### Direct use\n"
      "\n"
      "Direct tool runs are useful when working with an explicit file path:\n"
      "\n"
      "```shell\n"
      "iree-bazel-run //loom/src/loom/tools/loom-check -- "
      "path/to/file.loom-test\n"
      "iree-bazel-run //loom/src/loom/tools/loom-check -- --update "
      "path/to/file.loom-test\n"
      "```\n"
      "\n"
      "`--update` cannot be used with stdin or verify-mode cases.\n"
      "\n"
      "### Emit output discipline\n"
      "\n"
      "Prefer IR output checks for compiler behavior. Use large JSON emit "
      "goldens\n"
      "only when the JSON structure is the unit under test. Prefer concise\n"
      "text emit targets, such as `low-allocation`, when the test only needs\n"
      "to prove compiler facts.\n");
}

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

int loom_check_main(int argc, char** argv,
                    const loom_check_environment_t* base_environment) {
  if (!base_environment) {
    fprintf(stderr, "loom-check environment is required\n");
    return 1;
  }

  iree_flags_set_usage(
      "loom-check",
      "Test runner for .loom-test check files.\n"
      "\n"
      "Parses .loom-test files into cases, executes each case according to "
      "its\n"
      "mode directive, and reports pass/fail/skip results with diffs or\n"
      "diagnostic details on failure. Use .loom for ordinary Loom IR files.\n"
      "\n"
      "Usage:\n"
      "  loom-check [flags] [file]\n"
      "  cat test.loom-test | loom-check\n"
      "  loom-check --agents_md\n"
      "\n"
      "Update workflow:\n"
      "  Checked-in .loom-test expectations are updated through Bazel test\n"
      "  targets so the test environment and fixture path stay attached:\n"
      "    iree-bazel-test --config=asan <loom-check-test-target> "
      "--test_arg=--update\n"
      "  The iree-bazel-test wrapper automatically uses Bazel's standalone\n"
      "  TestRunner strategy for --test_arg=--update so fixture files are\n"
      "  writable. After building a target, its generated executable can also\n"
      "  be run directly and will append the fixture path automatically:\n"
      "    bazel-bin/path/to/generated-test --update\n"
      "\n"
      "Modes (set via // RUN: directive, default is roundtrip):\n"
      "  roundtrip   Parse, print, compare against expected output.\n"
      "  verify      Parse, verify, match diagnostics against annotations.\n"
      "  pass <p>    Parse, run pass pipeline <p>, print, compare.\n"
      "  pass-report <p>\n"
      "              Parse, run pass pipeline <p>, print compile report,\n"
      "              compare.\n"
      "  format <f>  Parse, convert to format <f>, convert back, compare.\n"
      "  emit <t>    Parse, emit analysis or target-structured output <t>,\n"
      "              print, compare.\n"
      "              Core targets include liveness-json, low-schedule-json,\n"
      "              low-allocation, low-allocation-json, low-packet-json,\n"
      "              target-low-registry-manifest, and source-low.\n"
      "              source-low emits target-lowering\n"
      "              artifacts or pipeline text and accepts\n"
      "              output=module|low|pipeline|prepared-pipeline,\n"
      "              control-flow=cfg|structured-low, and\n"
      "              diagnostics=none|memory|all.\n"
      "              low-allocation, low-allocation-json, and low-packet-json\n"
      "              accept "
      "fixed=%value:<physical_register|target_id>:<base>:<count>\n"
      "              allocation anchors.\n"
      "              low-schedule-json, low-allocation-json, and\n"
      "              low-packet-json accept output=json|none.\n"
      "              low-allocation and low-allocation-json accept\n"
      "              diagnostics=none|predicted-spills|copy-decisions|\n"
      "              placement-decisions|all.\n"
      "              low-schedule-json and low-packet-json accept\n"
      "              strategy=source|pressure|latency-hiding|resource-stall "
      "and\n"
      "              diagnostics=none|packets|all. low-schedule-json also\n"
      "              accepts "
      "cliff=<reg-class>:<units>:<tier-before>:<tier-after>.\n"
      "              Linked providers may add\n"
      "              more.\n"
      "File format:\n"
      "  A .loom-test file contains one or more cases separated by // ====.\n"
      "  Each case has directives at the top, then input IR, and\n"
      "  optionally a // ---- separator followed by expected output.\n"
      "  When // ---- is absent, the expected output equals the input\n"
      "  (round-trip identity test).\n"
      "\n"
      "  The first // ==== separator must appear after the first case body.\n"
      "  A // RUN: directive in the first case sets the file-level default\n"
      "  mode. Cases without their own // RUN: inherit from it.\n"
      "\n"
      "  Directives:\n"
      "    // RUN: <mode> [args]    Set the test mode (one per case).\n"
      "    // REQUIRES: <name>[, ...] Skip when requirements are unavailable.\n"
      "    // XFAIL: <reason>       Mark as expected failure.\n"
      "    // TEMPLATE: <path>      File-level corpus template metadata.\n"
      "    Known REQUIRES names: loom-check-test-unavailable and names from "
      "providers linked\n"
      "    into this runner.\n"
      "    TEMPLATE is only accepted in the file preamble before the first "
      "// ====.\n"
      "    CASE directives are intentionally unsupported; function symbols are "
      "case names.\n"
      "\n"
      "  Annotations (verify mode):\n"
      "    // ERROR: DOMAIN/CODE \"substring\"\n"
      "    // ERROR@+1: PARSE/006\n"
      "    // WARNING@-2: \"some message\"\n"
      "    // REMARK: TYPE\n"
      "    Domain and code are optional (omit to match any).\n"
      "    @+N/@-N targets a line relative to the annotation.\n"
      "\n"
      "Examples:\n"
      "  # Round-trip: print output must match input exactly.\n"
      "  echo '// RUN: roundtrip\n"
      "  func.def @f() {\n"
      "  }' | loom-check\n"
      "\n"
      "  # Verify: parse error must match the annotation.\n"
      "  echo '// RUN: verify\n"
      "  // ERROR@+1: PARSE/006\n"
      "  bogus.nonexistent' | loom-check\n"
      "\n"
      "Exit code is 0 when all cases pass, 1 if any fail.\n");

  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_check_print_agents_markdown(stdout);
      return 0;
    }
  }
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, host_allocator, &block_pool);

  // Initialize context with the dialects selected by this loom-check binary.
  loom_context_t context;
  loom_context_initialize(host_allocator, &context);
  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-check accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_check_context_register_and_finalize(base_environment, &context);
  }

  iree_host_size_t pass_count = 0;
  iree_host_size_t fail_count = 0;
  iree_host_size_t skip_count = 0;
  const loom_check_process_options_t process_options = {
      .update = FLAG_update,
      .verbose = FLAG_verbose,
      .json_enabled = FLAG_json.enabled,
      .json_output_mode = FLAG_json.output_mode,
  };

  if (iree_status_is_ok(status)) {
    loom_check_environment_t environment = *base_environment;
    if (argc < 2) {
      // No positional args: read from stdin.
      status = loom_check_read_and_process(
          iree_string_view_empty(), &process_options, &environment, &context,
          &block_pool, host_allocator, &pass_count, &fail_count, &skip_count);
    } else {
      status = loom_check_read_and_process(
          iree_make_cstring_view(argv[1]), &process_options, &environment,
          &context, &block_pool, host_allocator, &pass_count, &fail_count,
          &skip_count);
    }
  }

  if (iree_status_is_ok(status)) {
    loom_check_print_summary(pass_count, fail_count, skip_count);
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);

  if (had_error || fail_count > 0) {
    return 1;
  }
  return 0;
}
