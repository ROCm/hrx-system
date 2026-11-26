// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-analyze: static analysis for Loom sources. Reports operation sequences
// that are written more than once so they can be factored into motifs.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-analyze/clone_analysis.h"
#include "loom/tools/loom-analyze/inputs.h"
#include "loom/tools/loom-analyze/report.h"
#include "loom/tools/loom-analyze/session.h"

#define LOOM_ANALYZE_VERSION "0.1.0"

// Exit codes follow the linter convention: findings and tool failures are
// distinguishable so CI can gate on findings without masking broken input.
#define LOOM_ANALYZE_EXIT_CLEAN 0
#define LOOM_ANALYZE_EXIT_FINDINGS 1
#define LOOM_ANALYZE_EXIT_ERROR 2

IREE_FLAG(int32_t, min_ops, 6,
          "Minimum clone size in operations, counting nested operations.");
IREE_FLAG(int32_t, min_occurrences, 2,
          "Minimum number of copies reported as a clone.");
IREE_FLAG(string, parameterize, "none",
          "Comparison relaxations, comma separated: none; constants to match "
          "sequences whose *.constant values differ (candidates for motif "
          "parameters); types to match sequences differing only in element "
          "types (reported, not applicable: motifs cannot be type-generic).");
IREE_FLAG(string, format, "text", "Report format: text, json, or sarif.");
IREE_FLAG(string, output, "-",
          "Report path. Use '-' or the empty string for stdout.");
IREE_FLAG_LIST(string, exclude,
               "Path pattern to skip, using '*' and '?' wildcards against the "
               "whole '/'-separated path. Repeatable.");

typedef struct loom_analyze_loaded_input_t {
  iree_io_file_contents_t* contents;
  loomc_module_t* module;
} loom_analyze_loaded_input_t;

// Reads, parses and verifies one input, then indexes it. Returns OK with
// |*out_accepted| false when the input has diagnostics; those are printed.
static iree_status_t loom_analyze_load_input(
    const char* path, loom_analyze_session_t* session,
    loom_analyze_analyzer_t* analyzer, iree_allocator_t allocator,
    loom_analyze_loaded_input_t* out_input, bool* out_accepted) {
  *out_accepted = false;
  const iree_string_view_t filename = iree_make_cstring_view(path);
  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(filename, allocator, &out_input->contents));
  const loom_module_t* native_module = NULL;
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_analyze_session_load(
      session, filename, out_input->contents->const_buffer, stderr,
      &out_input->module, &native_module, &accepted));
  if (accepted) {
    IREE_RETURN_IF_ERROR(
        loom_analyze_analyzer_add_module(analyzer, native_module));
  }
  *out_accepted = accepted;
  return iree_ok_status();
}

static void loom_analyze_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-analyze\n"
      "\n"
      "`loom-analyze` is a static analyzer for Loom sources. It reports "
      "operation\n"
      "sequences that are written more than once so they can be factored into\n"
      "a motif: a `func.def inline` reached through `func.call`, or a\n"
      "`template.def` provider reached through `template.apply`.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-analyze kernels/\n"
      "loom-analyze --min-ops=10 kernels/ lib/motifs.loom\n"
      "loom-analyze --parameterize=constants kernels/\n"
      "loom-analyze --format=sarif --output=loom-analyze.sarif kernels/\n"
      "loom-analyze --exclude='*/testdata/*' src/\n"
      "```\n"
      "\n"
      "Directories are searched recursively for `.loom` files. Sequences "
      "match\n"
      "when they are equal up to SSA renaming: operations, attributes,\n"
      "symbols, constants and types compare by content; operands compare by\n"
      "dataflow within the sequence; operands of commutative operations\n"
      "compare as a multiset. Nested regions match recursively. Module-level\n"
      "declarations are not candidates.\n"
      "\n"
      "### Rules\n"
      "\n"
      "- `loom-analyze/duplicate-sequence`: a sequence of at least "
      "`--min-ops`\n"
      "  operations appears at least `--min-occurrences` times.\n"
      "- `loom-analyze/reimplemented-motif`: a sequence repeats the complete "
      "body of\n"
      "  an existing `func.def` or `template.def`; call it instead.\n"
      "\n"
      "Copies inside a `kernel.def` launch-config region are noted, because\n"
      "launch configuration cannot call functions today.\n"
      "\n"
      "### Output and exit status\n"
      "\n"
      "`--format=text` prints `file:line:col: warning: ... [rule]` "
      "diagnostics\n"
      "with `note:` lines for each copy and a summary.\n"
      "`--format=sarif` writes SARIF 2.1.0 with stable\n"
      "`partialFingerprints`.\n"
      "\n"
      "`--format=json` writes a refactor plan. Findings with\n"
      "`\"selected\": true` cover disjoint ops and can be applied together.\n"
      "Each copy lists its exact source range, its `inputs` (motif\n"
      "parameters) and `outputs` (call results) by name and type, and\n"
      "whether it is `callable`. loom-analyze never edits source.\n"
      "\n"
      "Exit status is 0 without findings, 1 with findings, and 2 when flags\n"
      "are invalid or an input cannot be read, parsed or verified.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-analyze",
      "Finds operation sequences written more than once in Loom sources.\n"
      "\n"
      "Usage:\n"
      "  loom-analyze [--min-ops=N] [--min-occurrences=K] "
      "[--parameterize=none|constants,types]\n"
      "          [--format=text|json|sarif] [--output=file] "
      "[--exclude=pattern] path...\n"
      "  loom-analyze --agents_md\n"
      "\n"
      "Paths name .loom files or directories searched recursively for .loom\n"
      "files. Exit status: 0 no findings, 1 findings, 2 usage or input "
      "error.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_analyze_print_agents_markdown(stdout);
      return LOOM_ANALYZE_EXIT_CLEAN;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  loom_analyze_session_t* session = NULL;
  loom_analyze_analyzer_t* analyzer = NULL;
  loom_analyze_input_list_t inputs;
  loom_analyze_input_list_initialize(allocator, &inputs);
  loom_analyze_loaded_input_t* loaded = NULL;
  iree_host_size_t loaded_count = 0;
  int exit_code = LOOM_ANALYZE_EXIT_CLEAN;

  loom_analyze_options_t options;
  loom_analyze_options_initialize(&options);
  loom_analyze_report_format_t format = LOOM_ANALYZE_REPORT_FORMAT_TEXT;
  iree_status_t status = loom_analyze_report_parse_format(
      iree_make_cstring_view(FLAG_format), &format);
  if (iree_status_is_ok(status) && FLAG_min_ops < 1) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "--min-ops must be at least 1; got %d", FLAG_min_ops);
  }
  if (iree_status_is_ok(status) && FLAG_min_occurrences < 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--min-occurrences must be at least 2; got %d",
                              FLAG_min_occurrences);
  }
  if (iree_status_is_ok(status)) {
    options.min_op_count = (uint32_t)FLAG_min_ops;
    options.min_occurrence_count = (uint32_t)FLAG_min_occurrences;
    status = loom_analyze_parse_parameterization(
        iree_make_cstring_view(FLAG_parameterize), &options.parameterization);
  }
  if (iree_status_is_ok(status) && argc < 2) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "loom-analyze requires at least one input file or "
                         "directory");
  }
  if (iree_status_is_ok(status)) {
    const iree_flag_string_list_t excludes = FLAG_exclude_list();
    for (int i = 1; i < argc && iree_status_is_ok(status); ++i) {
      status = loom_analyze_input_list_add_path(
          &inputs, iree_make_cstring_view(argv[i]), excludes.values,
          excludes.count);
    }
    loom_analyze_input_list_finalize(&inputs);
  }
  if (iree_status_is_ok(status) && inputs.count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no .loom inputs remain after expansion and "
                              "--exclude filtering");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, inputs.count,
                                         sizeof(*loaded), (void**)&loaded);
  }
  if (iree_status_is_ok(status)) {
    status = loom_analyze_session_create(allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    status = loom_analyze_analyzer_create(&options, allocator, &analyzer);
  }

  // Inputs that fail to load are reported and skipped; the remaining inputs
  // are still analyzed and the run exits with the error status.
  iree_host_size_t failed_count = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < inputs.count;
       ++i) {
    bool accepted = false;
    iree_status_t input_status =
        loom_analyze_load_input(inputs.paths[i], session, analyzer, allocator,
                                &loaded[loaded_count++], &accepted);
    if (!iree_status_is_ok(input_status)) {
      iree_status_fprint(stderr, input_status);
      iree_status_free(input_status);
      ++failed_count;
    } else if (!accepted) {
      ++failed_count;
    }
  }

  const loom_analyze_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_analyze_analyzer_run(analyzer, &result);
  }
  if (iree_status_is_ok(status)) {
    loom_tooling_output_stream_t output;
    status = loom_tooling_output_stream_open(
        iree_make_cstring_view(FLAG_output), allocator, &output);
    if (iree_status_is_ok(status)) {
      status = loom_analyze_report_write(result, &options, format,
                                         IREE_SV(LOOM_ANALYZE_VERSION),
                                         &output.stream);
      iree_status_t close_status = loom_tooling_output_stream_close(&output);
      status = iree_status_join(status, close_status);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = LOOM_ANALYZE_EXIT_ERROR;
  } else if (failed_count > 0) {
    fprintf(stderr,
            "loom-analyze: %" PRIhsz " input(s) could not be analyzed\n",
            failed_count);
    exit_code = LOOM_ANALYZE_EXIT_ERROR;
  } else if (loom_analyze_report_result_count(result) > 0) {
    exit_code = LOOM_ANALYZE_EXIT_FINDINGS;
  }

  // Modules must outlive the analyzer, whose results borrow their sources.
  loom_analyze_analyzer_free(analyzer);
  for (iree_host_size_t i = 0; i < loaded_count; ++i) {
    loomc_module_release(loaded[i].module);
    iree_io_file_contents_free(loaded[i].contents);
  }
  iree_allocator_free(allocator, loaded);
  loom_analyze_input_list_deinitialize(&inputs);
  loom_analyze_session_free(session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
