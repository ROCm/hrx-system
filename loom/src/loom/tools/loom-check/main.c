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
#include "loom/sanitizer/options.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/compile/configured.h"
#include "loom/tooling/context/context.h"
#include "loom/tools/loom-check/file.h"
#include "loom/tools/loom-check/json_output.h"
#include "loom/tools/loom-check/output.h"

IREE_FLAG(bool, update, false,
          "Rewrite test files with actual output in the expected\n"
          "section (after // ----) and synchronize TEMPLATE cases. Inserts\n"
          "the separator for non-empty output when absent.\n"
          "Cannot be used with stdin or verify mode.");
IREE_FLAG_NAMED(
    bool, check_templates, "check-templates", false,
    "Check TEMPLATE freshness without executing RUN directives or\n"
    "writing files. Accepts multiple input files. Used by precommit;\n"
    "ordinary test execution does not read template sources.");
IREE_FLAG(bool, verbose, false,
          "Print PASS/FAIL/SKIP for every case, not just failures.");
IREE_FLAG_NAMED(
    string, input_format, "input-format", "",
    "Source format for stdin or nonstandard filenames.\n"
    "Defaults to the filename suffix; INPUT directives override it.");
IREE_FLAG_NAMED(bool, list_input_formats, "list-input-formats", false,
                "List input formats linked into this binary and exit.");
IREE_FLAG(string, template_root, "",
          "Filesystem root for TEMPLATE paths during --check-templates or\n"
          "--update. Unused during ordinary test execution.\n"
          "Defaults to the current working directory.");
IREE_FLAG(string, target, "",
          "Qualify every input case for this family:selector through the "
          "offline compiler. Matches diagnostic annotations or requires "
          "a nonempty artifact. Does not execute RUN directives, compare "
          "their goldens, apply XFAIL, or require execution hardware.");
IREE_FLAG_LIST(string, config, "Compile-time key=value binding for --target.");
IREE_FLAG_LIST_NAMED(string, config_file, "config-file",
                     "JSON/JSONC compile-time bindings for --target.");
IREE_FLAG(string, sanitizer, "none",
          "Compiler sanitizer checks for --target, such as access.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Compiler sanitizer reporting: default, trap, or report-only.");

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
IREE_FLAG_LIST_NAMED(
    string, source_prefix_map, "source-prefix-map",
    "Remap source paths in diagnostics and loc() output. Repeat as\n"
    "--source-prefix-map=old=new; entries are applied in reverse order so the\n"
    "last matching map wins. Use old= to strip a prefix.");

static void loom_check_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-check\n"
      "\n"
      "`loom-check` checks compiler output and diagnostics from test files.\n"
      "`.loom-test` selects Loom text. `--list-input-formats` shows linked\n"
      "source importers and their filename suffixes. Each case is "
      "independent;\n"
      "`// INPUT: <format> [options]` selects source options, while `// RUN:`\n"
      "selects the operation to check.\n"
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
      "Checked-in template-backed fixtures are self-contained. Ordinary\n"
      "runs and `--target` compilation do not read their template sources.\n"
      "Precommit checks freshness with `--check-templates`, which accepts\n"
      "multiple files without executing RUN directives or writing files.\n"
      "\n"
      "### Compile for a profile\n"
      "\n"
      "`--target=family:selector` compiles every source case or each\n"
      "kernel in a linked `.loombc` test module through final artifact "
      "emission.\n"
      "A pass means a nonempty artifact or exactly matched source diagnostic\n"
      "annotations. No device is opened and numerical checks are not "
      "executed.\n"
      "RUN goldens, XFAIL, and execution requirements belong to the separate\n"
      "ordinary check outcome.\n"
      "\n"
      "```shell\n"
      "loom-check --target=amdgpu:gfx942 offsets.loom-test\n"
      "loom-check --target=spirv:vulkan1.3+bda test-module.loombc\n"
      "```\n"
      "\n"
      "`compile_targets` on `loom_test` uses the same root-owned test module\n"
      "as `execution_profile`; it adds independent host-only tests. On\n"
      "`loom_check_test_suite`, it maps existing source paths to typed target\n"
      "profile labels. Running an owning test includes its compiler children.\n"
      "`--config` and `--config-file` bind compile-time configuration.\n"
      "`--sanitizer` and `--sanitizer-reporting` select compiler "
      "instrumentation.\n"
      "`--update` is rejected in compilation mode.\n"
      "\n"
      "### Update expected output\n"
      "\n"
      "Pass the update flag through Bazel with `--test_arg=--update`:\n"
      "For a target with compiler children, select its `_run` child.\n"
      "\n"
      "```shell\n"
      "iree-bazel-test --config=asan <loom-check-test-target> "
      "--test_arg=--update\n"
      "```\n"
      "\n"
      "`iree-bazel-test` detects this flag and uses Bazel's standalone\n"
      "TestRunner strategy so update-capable tests can rewrite checked-in\n"
      "fixture files. It supplies the checkout root for template updates.\n"
      "After building a target, its generated executable can\n"
      "also be run directly and will append the fixture path automatically:\n"
      "\n"
      "```shell\n"
      "iree-bazel-build <loom-check-test-target>\n"
      "bazel-bin/path/to/generated-test --template-root=\"$PWD\" --update\n"
      "```\n"
      "\n"
      "### Direct use\n"
      "\n"
      "Direct tool runs are useful when working with an explicit file path:\n"
      "\n"
      "```shell\n"
      "iree-bazel-run //loom/src/loom/tools/loom-check -- "
      "path/to/file.loom-test\n"
      "iree-bazel-run //loom/src/loom/tools/loom-check -- "
      "--template-root=. --update path/to/file.loom-test\n"
      "```\n"
      "\n"
      "Explicit checks and updates resolve TEMPLATE paths from the current\n"
      "directory unless `--template-root` is explicit. Both use the same\n"
      "synchronizer. `--update` cannot be used with stdin or\n"
      "verify-mode cases.\n"
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
      "  loom-check --check-templates [flags] file...\n"
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
      "  writable, and supplies the checkout root for template updates.\n"
      "  After building a target, its generated executable can also\n"
      "  be run directly and will append the fixture path automatically:\n"
      "    bazel-bin/path/to/generated-test --template-root=\"$PWD\" --update\n"
      "\n"
      "Modes (set via // RUN: directive, default is roundtrip):\n"
      "  with-checks <mode> ...\n"
      "              Match CHECK/CHECK-NOT whole-line globs instead of a "
      "golden.\n"
      "              Applies to textual output; --update preserves the "
      "checks.\n"
      "  roundtrip   Parse, print, compare against expected output.\n"
      "  verify      Parse, verify, match diagnostics against annotations.\n"
      "  pass <p>    Parse, run flat pass pipeline <p>, print, compare.\n"
      "  pass @p     Parse, run named pass.pipeline @p, print, compare.\n"
      "  with-low-asm pass <p>\n"
      "              Preserve authored Low assembly in pass output.\n"
      "  pass-report <p>\n"
      "              Parse, run pass pipeline <p>, print pass report,\n"
      "              compare.\n"
      "  compile-report <p>\n"
      "              Parse, run pass pipeline <p>, print compile report,\n"
      "              compare.\n"
      "  format <f>  Parse, convert to format <f>, convert back, compare.\n"
      "  emit <t>    Parse, emit analysis or target-structured output <t>,\n"
      "              print, compare.\n"
      "              Core targets include liveness-json, low-schedule-json,\n"
      "              low-allocation, low-allocation-json, low-packet-json,\n"
      "              low-compile-report @function,\n"
      "              target-low-registry-manifest, and source-low.\n"
      "              pipeline-plan @pipeline max-instances=<count> checks\n"
      "              concrete pipeline planning diagnostics without a target.\n"
      "              source-low emits target-lowering\n"
      "              artifacts or pipeline text and accepts\n"
      "              @function target=family:selector for specialization,\n"
      "              output=module|low|pipeline|prepared-pipeline|none,\n"
      "              control-flow=cfg|structured-low,\n"
      "              sanitizer=none|access|value|operation|race|all,\n"
      "              sanitizer-reporting=default|trap|report-only, and\n"
      "              diagnostics=none|memory|operand-forms|all.\n"
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
      "              low-schedule-json accepts diagnostics=none|pressure|\n"
      "              resources|hazards|candidates|model|all,\n"
      "              cliff=<reg-class>:<units>:<tier-before>:<tier-after>,\n"
      "              and <reg-class>=<units> pressure budgets.\n"
      "              low-packet-json accepts diagnostics=none|packets|all.\n"
      "              Linked providers may add\n"
      "              more.\n"
      "File format:\n"
      "  A .<format>-test file contains cases separated by // ====.\n"
      "  .loom-test selects Loom text; --list-input-formats lists importers.\n"
      "  Each case has directives at the top, then input source, and\n"
      "  optionally a // ---- separator followed by expected output.\n"
      "  When // ---- is absent, the expected output equals the input\n"
      "  (round-trip identity test).\n"
      "\n"
      "  The first // ==== separator must appear after the first case body.\n"
      "  A // RUN: directive in the first case sets the file-level default\n"
      "  mode. Cases without their own // RUN: inherit from it.\n"
      "\n"
      "  Directives:\n"
      "    // INPUT: <format> [options]\n"
      "                            Select source input independently of RUN.\n"
      "                            The first case supplies defaults; each\n"
      "                            later INPUT replaces them for that case.\n"
      "    // RUN: [with-locations] <mode> [args]\n"
      "                            Set the test mode (one per case). The\n"
      "                            with-locations modifier prints loc()\n"
      "                            annotations for roundtrip and pass output.\n"
      "    // REQUIRES: <name>[, ...] Skip when requirements are unavailable.\n"
      "    // XFAIL: <reason>       Mark as expected failure.\n"
      "    // TEMPLATE: <path>      Root-relative authoritative corpus "
      "source.\n"
      "    // TEMPLATE-EXCLUDE: @<case> <reason>\n"
      "                            Omit one architecturally inapplicable "
      "case.\n"
      "    Known REQUIRES names come from providers linked into this runner.\n"
      "    TEMPLATE is only accepted in the file preamble before the first "
      "// ====. Precommit rejects stale files via --check-templates;\n"
      "    ordinary test execution does not read template sources.\n"
      "    TEMPLATE-EXCLUDE belongs in that preamble, requires an exact case\n"
      "    name and a reason, and rejects duplicate or unknown names. "
      "Entirely\n"
      "    inapplicable corpora need no fixture; excluding every case fails.\n"
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
      "    Lines are relative to the case input. Annotations target the main\n"
      "    source; diagnostics from included files retain their own identity.\n"
      "    Expected CHECK patterns may use CHECK: or // CHECK: spelling.\n"
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

  if (FLAG_list_input_formats) {
    const loom_input_provider_t* const builtin_providers[] = {
        &loom_input_text_provider, &loom_input_bytecode_provider};
    const iree_host_size_t builtin_count = IREE_ARRAYSIZE(builtin_providers);
    for (iree_host_size_t i = 0;
         i < builtin_count + base_environment->input_providers.count; ++i) {
      const loom_input_provider_t* provider =
          i < builtin_count
              ? builtin_providers[i]
              : base_environment->input_providers.values[i - builtin_count];
      printf("%.*s:", (int)provider->name.size, provider->name.data);
      for (iree_host_size_t j = 0; j < provider->suffixes.count; ++j) {
        iree_string_view_t suffix = provider->suffixes.values[j];
        printf(" %.*s", (int)suffix.size, suffix.data);
      }
      printf("\n");
    }
    return 0;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, host_allocator, &block_pool);

  // Initialize context with the dialects selected by this loom-check binary.
  loom_context_t context;
  loom_context_initialize(host_allocator, &context);
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(host_allocator, &config_set);
  const iree_string_view_t target = iree_make_cstring_view(FLAG_target);
  const loom_tooling_compile_environment_t* compile_environment =
      iree_string_view_is_empty(target)
          ? NULL
          : loom_tooling_configured_compile_environment();
  iree_status_t status = iree_ok_status();
  if (FLAG_check_templates &&
      (FLAG_update || !iree_string_view_is_empty(target) ||
       FLAG_json.enabled)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--check-templates cannot be combined with "
                              "--update, --target, or --json");
  } else if (FLAG_check_templates && argc < 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--check-templates requires input files");
  } else if (!FLAG_check_templates && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-check accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status)) {
    if (compile_environment != NULL) {
      status =
          loom_tooling_context_register_tool_dialects_with_target_environment(
              compile_environment->target_environment, &context);
      if (iree_status_is_ok(status)) {
        status = loom_context_finalize(&context);
      }
    } else {
      status =
          loom_check_context_register_and_finalize(base_environment, &context);
    }
  }
  const iree_flag_string_list_t config_files = FLAG_config_file_list();
  const iree_flag_string_list_t configs = FLAG_config_list();
  loom_sanitizer_options_t sanitizer = {0};
  if (iree_status_is_ok(status)) {
    status = loom_sanitizer_options_parse_checks(
        iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
        &sanitizer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_sanitizer_reporting_mode_parse(
        iree_make_cstring_view(FLAG_sanitizer_reporting),
        IREE_SV("--sanitizer-reporting"), &sanitizer.reporting_mode);
  }
  if (iree_status_is_ok(status) && compile_environment == NULL &&
      (config_files.count > 0 || configs.count > 0 || sanitizer.checks != 0 ||
       sanitizer.reporting_mode != LOOM_SANITIZER_REPORTING_MODE_DEFAULT)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "compiler options require --target");
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < config_files.count; ++i) {
    status = loom_tooling_config_set_append_json_file(
        &config_set, config_files.values[i], host_allocator);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < configs.count;
       ++i) {
    status = loom_tooling_config_set_append_assignment(&config_set,
                                                       configs.values[i]);
  }

  iree_host_size_t pass_count = 0;
  iree_host_size_t fail_count = 0;
  iree_host_size_t skip_count = 0;
  const loom_check_process_options_t process_options = {
      .compile = {.target = target,
                  .environment = compile_environment,
                  .config_set = &config_set,
                  .sanitizer = sanitizer},
      .input_format = iree_make_cstring_view(FLAG_input_format),
      .mode = FLAG_check_templates ? LOOM_CHECK_PROCESS_CHECK_TEMPLATES
              : FLAG_update        ? LOOM_CHECK_PROCESS_UPDATE
                                   : LOOM_CHECK_PROCESS_EXECUTE,
      .verbose = FLAG_verbose,
      .json_enabled = FLAG_json.enabled,
      .json_output_mode = FLAG_json.output_mode,
      .source_path_options =
          {
              .prefix_maps = FLAG_source_prefix_map_list(),
          },
      .template_root = iree_make_cstring_view(FLAG_template_root),
  };

  if (iree_status_is_ok(status)) {
    loom_check_environment_t environment = *base_environment;
    if (argc < 2) {
      // No positional args: read from stdin.
      status = loom_check_read_and_process(
          iree_string_view_empty(), &process_options, &environment, &context,
          &block_pool, host_allocator, &pass_count, &fail_count, &skip_count);
    } else {
      for (int i = 1; iree_status_is_ok(status) && i < argc; ++i) {
        status = loom_check_read_and_process(
            iree_make_cstring_view(argv[i]), &process_options, &environment,
            &context, &block_pool, host_allocator, &pass_count, &fail_count,
            &skip_count);
      }
    }
  }

  if (iree_status_is_ok(status) && !FLAG_check_templates) {
    loom_check_print_summary(pass_count, fail_count, skip_count);
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  loom_context_deinitialize(&context);
  loom_tooling_config_set_deinitialize(&config_set);
  iree_arena_block_pool_deinitialize(&block_pool);

  if (had_error || fail_count > 0) {
    return 1;
  }
  return 0;
}
