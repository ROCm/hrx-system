// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-reduce: rewrites Loom sources to remove duplicated operation sequences
// found by the loom-analyze clone analysis.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-analyze/clone_analysis.h"
#include "loom/tools/loom-analyze/inputs.h"
#include "loom/tools/loom-analyze/session.h"
#include "loom/tools/loom-reduce/reduce.h"

#define LOOM_REDUCE_EXIT_OK 0
#define LOOM_REDUCE_EXIT_ERROR 2

IREE_FLAG(string, output_dir, "",
          "Empty directory receiving a rewritten copy of the source root.");
IREE_FLAG(string, scope, "file",
          "file: define each motif beside its copies (no link changes). "
          "corpus: share motifs across files within each top-level directory "
          "of the root.");
IREE_FLAG(string, motif_file, "loom_reduce_{group}_motifs.loom",
          "Motif file name created in each top-level directory (corpus "
          "scope); {group} becomes the directory name.");
IREE_FLAG(int32_t, min_ops, 4,
          "Minimum duplicated sequence size in operations. Sequences whose "
          "calls would not remove operations are skipped.");
IREE_FLAG(string, parameterize, "none",
          "none: share motifs only between identical copies. constants: also "
          "between copies differing in top-level constants, which stay at "
          "each call site and are passed to the motif.");
IREE_FLAG(int32_t, min_occurrences, 2,
          "Minimum number of copies of a duplicated sequence.");
IREE_FLAG(string, report, "",
          "Write a JSON report of applied and skipped work.");
IREE_FLAG(string, links, "",
          "Write the link inputs each rewritten file now needs (corpus "
          "scope).");
IREE_FLAG(bool, no_verify, false,
          "Skip parsing and verifying rewritten files.");
IREE_FLAG_LIST(string, exclude,
               "Path pattern to skip, using '*' and '?' wildcards against the "
               "whole '/'-separated path. Repeatable.");

typedef struct loom_reduce_loaded_t {
  iree_io_file_contents_t* contents;
  loomc_module_t* module;
  char* relative_path;
} loom_reduce_loaded_t;

typedef struct loom_reduce_verify_state_t {
  loom_analyze_session_t* session;
} loom_reduce_verify_state_t;

// Parses and verifies rewritten text in memory.
static iree_status_t loom_reduce_verify_text(void* user_data,
                                             iree_string_view_t relative_path,
                                             iree_string_view_t text,
                                             bool* out_accepted) {
  loom_reduce_verify_state_t* state = (loom_reduce_verify_state_t*)user_data;
  loomc_module_t* module = NULL;
  const loom_module_t* native_module = NULL;
  iree_status_t status = loom_analyze_session_load(
      state->session, relative_path,
      iree_make_const_byte_span(text.data, text.size), /*diagnostics=*/NULL,
      &module, &native_module, out_accepted);
  loomc_module_release(module);
  return status;
}

// Returns |path| relative to |root| with '/' separators.
static iree_status_t loom_reduce_relative_path(iree_string_view_t root,
                                               const char* path,
                                               iree_allocator_t allocator,
                                               char** out_relative) {
  iree_string_view_t view = iree_make_cstring_view(path);
  if (iree_string_view_starts_with(view, root)) {
    view = iree_string_view_substr(view, root.size, IREE_HOST_SIZE_MAX);
  }
  while (view.size && (view.data[0] == '/' || view.data[0] == '\\')) {
    view = iree_string_view_substr(view, 1, IREE_HOST_SIZE_MAX);
  }
  while (iree_string_view_starts_with(view, IREE_SV("./"))) {
    view = iree_string_view_substr(view, 2, IREE_HOST_SIZE_MAX);
  }
  char* relative = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, view.size + 1, (void**)&relative));
  for (iree_host_size_t i = 0; i < view.size; ++i) {
    relative[i] = view.data[i] == '\\' ? '/' : view.data[i];
  }
  relative[view.size] = 0;
  *out_relative = relative;
  return iree_ok_status();
}

static iree_status_t loom_reduce_join(iree_string_view_t directory,
                                      iree_string_view_t relative,
                                      iree_string_builder_t* out_path) {
  iree_string_builder_reset(out_path);
  return iree_string_builder_append_format(out_path, "%.*s/%.*s",
                                           (int)directory.size, directory.data,
                                           (int)relative.size, relative.data);
}

// Copies every file under |root| into |output_dir|.
static iree_status_t loom_reduce_copy_tree(iree_string_view_t root,
                                           iree_string_view_t output_dir,
                                           iree_allocator_t allocator) {
  loom_analyze_input_list_t files;
  loom_analyze_input_list_initialize(allocator, &files);
  iree_status_t status = loom_analyze_input_list_add_tree(&files, root);
  iree_string_builder_t path;
  iree_string_builder_initialize(allocator, &path);
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    char* relative = NULL;
    status =
        loom_reduce_relative_path(root, files.paths[i], allocator, &relative);
    iree_io_file_contents_t* contents = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_tooling_read_input_file(
          iree_make_cstring_view(files.paths[i]), allocator, &contents);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_reduce_join(output_dir, iree_make_cstring_view(relative), &path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_tooling_create_parent_directory(
          iree_string_builder_view(&path), allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_tooling_write_output_file(
          iree_string_builder_view(&path),
          loom_tooling_file_contents_string_view(contents), allocator);
    }
    iree_io_file_contents_free(contents);
    iree_allocator_free(allocator, relative);
  }
  iree_string_builder_deinitialize(&path);
  loom_analyze_input_list_deinitialize(&files);
  return status;
}

static iree_status_t loom_reduce_write_stream(
    const char* path, iree_allocator_t allocator,
    iree_status_t (*write)(const loom_reduce_plan_t*, loom_output_stream_t*,
                           uint64_t),
    const loom_reduce_plan_t* plan, uint64_t op_count) {
  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(loom_tooling_output_stream_open(
      iree_make_cstring_view(path), allocator, &output));
  iree_status_t status = write(plan, &output.stream, op_count);
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

static iree_status_t loom_reduce_write_report(const loom_reduce_plan_t* plan,
                                              loom_output_stream_t* stream,
                                              uint64_t op_count) {
  return loom_reduce_plan_write_report(plan, op_count, stream);
}

static iree_status_t loom_reduce_write_links(const loom_reduce_plan_t* plan,
                                             loom_output_stream_t* stream,
                                             uint64_t op_count) {
  (void)op_count;
  return loom_reduce_plan_write_links(plan, stream);
}

static void loom_reduce_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-reduce\n"
      "\n"
      "`loom-reduce` rewrites Loom sources to remove duplicated operation\n"
      "sequences. It runs the loom-analyze clone analysis on a source root,\n"
      "turns each selected exact duplicate into one `func.def inline` motif,\n"
      "and replaces every callable copy with a call. It writes a rewritten\n"
      "copy of the root and never edits sources in place.\n"
      "\n"
      "```shell\n"
      "loom-reduce --output-dir=reduced/ kernels/\n"
      "loom-reduce --scope=corpus --links=links.json --report=reduce.json \\\n"
      "    --output-dir=reduced/ kernels/\n"
      "```\n"
      "\n"
      "`--scope=file` defines each motif at the end of the file holding its\n"
      "copies; no declarations or link inputs change. `--scope=corpus` groups\n"
      "copies by the first directory under the root, writes each group's\n"
      "motifs to `<dir>/loom_reduce_<dir>_motifs.loom` with the declarations\n"
      "they reference, adds\n"
      "`func.decl` lines to calling files, and records in `--links` which\n"
      "files each rewritten source now needs linked.\n"
      "\n"
      "Copies of an existing `func.def` call it rather than getting a new\n"
      "motif; in corpus scope its definition moves unchanged into the group's\n"
      "motif file when copies live in other files. `reduce.json` lists under\n"
      "`reuse_fallbacks` why reuse was impossible where a new motif was made.\n"
      "Run repeatedly on the output until no motifs are reported, then once\n"
      "more with `--parameterize=constants` to share motifs between copies\n"
      "that differ in constants interleaved with the shared operations.\n"
      "\n"
      "Every rewritten file is parsed and verified in memory; motifs touching "
      "a\n"
      "failing file are re-admitted one at a time and those that still fail\n"
      "are reported as reverted. Rewrites that cannot be expressed (block\n"
      "terminators, nested buffer.alloca, dependent parameter types, call\n"
      "cycles) are skipped with a reason. A top-level buffer.alloca stays at\n"
      "each call site with the operations it depends on, and its buffer is\n"
      "passed to the motif. Copies inside `pure` functions get a `pure` motif\n"
      "and `func.call pure`. Inline motifs usually compile to identical code;\n"
      "compare compiled kernels and measure any that change.\n"
      "\n"
      "Exit status is 0 when the rewrite completes and 2 for usage, input, or\n"
      "output errors.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-reduce",
      "Rewrites Loom sources to remove duplicated operation sequences.\n"
      "\n"
      "Usage:\n"
      "  loom-reduce --output-dir=DIR [--scope=file|corpus] [--report=FILE]\n"
      "              [--links=FILE] [--min-ops=N] [--exclude=pattern] ROOT\n"
      "  loom-reduce --agents_md\n"
      "\n"
      "ROOT is searched recursively for .loom files. The whole tree is copied\n"
      "to DIR and only the copy is rewritten.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_reduce_print_agents_markdown(stdout);
      return LOOM_REDUCE_EXIT_OK;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  loom_analyze_session_t* session = NULL;
  loom_analyze_analyzer_t* analyzer = NULL;
  loom_reduce_plan_t* plan = NULL;
  loom_analyze_input_list_t inputs;
  loom_analyze_input_list_initialize(allocator, &inputs);
  loom_reduce_loaded_t* loaded = NULL;
  loom_reduce_input_t* plan_inputs = NULL;
  iree_host_size_t loaded_count = 0;
  iree_string_builder_t path;
  iree_string_builder_initialize(allocator, &path);
  const iree_string_view_t output_dir = iree_make_cstring_view(FLAG_output_dir);
  iree_string_view_t root = iree_string_view_empty();

  loom_reduce_options_t options = {
      .scope = LOOM_REDUCE_SCOPE_FILE,
      .motif_file_name = iree_make_cstring_view(FLAG_motif_file),
  };
  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "loom-reduce requires exactly one source root "
                              "directory");
  } else {
    root = iree_make_cstring_view(argv[1]);
    while (root.size > 1 && (root.data[root.size - 1] == '/' ||
                             root.data[root.size - 1] == '\\')) {
      --root.size;
    }
  }
  if (iree_status_is_ok(status) && iree_string_view_is_empty(output_dir)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--output-dir is required");
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t scope = iree_make_cstring_view(FLAG_scope);
    if (iree_string_view_equal(scope, IREE_SV("corpus"))) {
      options.scope = LOOM_REDUCE_SCOPE_CORPUS;
    } else if (!iree_string_view_equal(scope, IREE_SV("file"))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown --scope '%s'; expected file or corpus",
                                FLAG_scope);
    }
  }
  if (iree_status_is_ok(status) &&
      (FLAG_min_ops < 1 || FLAG_min_occurrences < 2)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--min-ops must be at least 1 and "
                              "--min-occurrences at least 2");
  }
  if (iree_status_is_ok(status)) {
    // An existing non-empty output directory could mix old and new files.
    loom_analyze_input_list_t existing;
    loom_analyze_input_list_initialize(allocator, &existing);
    iree_status_t probe =
        loom_analyze_input_list_add_tree(&existing, output_dir);
    if (iree_status_is_ok(probe) && existing.count > 0) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "output directory '%s' is not empty",
                                FLAG_output_dir);
    }
    iree_status_free(probe);
    loom_analyze_input_list_deinitialize(&existing);
  }
  if (iree_status_is_ok(status)) {
    const iree_flag_string_list_t excludes = FLAG_exclude_list();
    status = loom_analyze_input_list_add_path(&inputs, root, excludes.values,
                                              excludes.count);
    loom_analyze_input_list_finalize(&inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, inputs.count + 1,
                                         sizeof(*loaded), (void**)&loaded);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(allocator, inputs.count + 1,
                                    sizeof(*plan_inputs), (void**)&plan_inputs);
  }
  if (iree_status_is_ok(status)) {
    status = loom_analyze_session_create(allocator, &session);
  }
  loom_analyze_options_t analyze_options;
  loom_analyze_options_initialize(&analyze_options);
  analyze_options.min_op_count = (uint32_t)FLAG_min_ops;
  analyze_options.min_occurrence_count = (uint32_t)FLAG_min_occurrences;
  if (iree_status_is_ok(status)) {
    status = loom_analyze_parse_parameterization(
        iree_make_cstring_view(FLAG_parameterize),
        &analyze_options.parameterization);
    if (iree_status_is_ok(status) &&
        (analyze_options.parameterization &
         ~(loom_analyze_parameterization_t)
             LOOM_ANALYZE_PARAMETERIZE_CONSTANTS)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--parameterize accepts none or constants; motifs cannot be generic "
          "over element types");
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_analyze_analyzer_create(&analyze_options, allocator, &analyzer);
  }

  // Every source must load: a rewrite cannot be checked against a file that
  // does not verify to begin with.
  for (iree_host_size_t i = 0; i < inputs.count && iree_status_is_ok(status);
       ++i) {
    loom_reduce_loaded_t* entry = &loaded[loaded_count++];
    const iree_string_view_t filename = iree_make_cstring_view(inputs.paths[i]);
    status =
        loom_tooling_read_input_file(filename, allocator, &entry->contents);
    const loom_module_t* native_module = NULL;
    bool accepted = false;
    if (iree_status_is_ok(status)) {
      status = loom_analyze_session_load(
          session, filename, entry->contents->const_buffer, stderr,
          &entry->module, &native_module, &accepted);
    }
    if (iree_status_is_ok(status) && !accepted) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "input '%s' does not parse and verify",
                                inputs.paths[i]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_analyze_analyzer_add_module(analyzer, native_module);
    }
    if (iree_status_is_ok(status)) {
      status = loom_reduce_relative_path(root, inputs.paths[i], allocator,
                                         &entry->relative_path);
    }
    if (iree_status_is_ok(status)) {
      plan_inputs[i] = (loom_reduce_input_t){
          .load_path = filename,
          .relative_path = iree_make_cstring_view(entry->relative_path),
          .text = loom_tooling_file_contents_string_view(entry->contents),
          .module = native_module,
      };
    }
  }

  const loom_analyze_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_analyze_analyzer_run(analyzer, &result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_reduce_plan_create(&options, plan_inputs, inputs.count,
                                     result, allocator, &plan);
  }
  loom_reduce_verify_state_t verify_state = {.session = session};
  if (iree_status_is_ok(status)) {
    loom_reduce_verifier_t verifier = {NULL, NULL};
    if (!FLAG_no_verify) {
      verifier.fn = loom_reduce_verify_text;
      verifier.user_data = &verify_state;
    }
    status = loom_reduce_plan_apply(plan, verifier);
  }
  if (iree_status_is_ok(status)) {
    status = loom_reduce_copy_tree(root, output_dir, allocator);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < loom_reduce_plan_output_count(plan);
       ++i) {
    iree_string_view_t relative, text;
    loom_reduce_plan_output(plan, i, &relative, &text);
    status = loom_reduce_join(output_dir, relative, &path);
    if (iree_status_is_ok(status)) {
      status = loom_tooling_create_parent_directory(
          iree_string_builder_view(&path), allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_tooling_write_output_file(iree_string_builder_view(&path),
                                              text, allocator);
    }
  }
  if (iree_status_is_ok(status) && FLAG_report[0]) {
    status = loom_reduce_write_stream(FLAG_report, allocator,
                                      loom_reduce_write_report, plan,
                                      result->summary.op_count);
  }
  if (iree_status_is_ok(status) && FLAG_links[0]) {
    status = loom_reduce_write_stream(FLAG_links, allocator,
                                      loom_reduce_write_links, plan, 0);
  }

  int exit_code = LOOM_REDUCE_EXIT_OK;
  if (iree_status_is_ok(status)) {
    loom_reduce_summary_t summary;
    loom_reduce_plan_summary(plan, &summary);
    const uint64_t total = result->summary.op_count;
    fprintf(
        stdout,
        "loom-reduce: %u motif(s) replace %u copies in %u file(s); "
        "removes %" PRIu64 " of %" PRIu64
        " ops (%.1f%%)\n"
        "loom-reduce: %u skipped, %u reverted after verification\n"
        "loom-reduce: %u existing motif(s) reused; %u new motif(s) made "
        "where reuse was impossible\n",
        summary.motif_count, summary.replaced_copy_count,
        summary.changed_file_count, summary.removed_op_count, total,
        total ? 100.0 * (double)summary.removed_op_count / (double)total : 0.0,
        summary.skipped_count, summary.reverted_count, summary.reused_count,
        summary.reuse_fallback_count);
    if (summary.linked_file_count) {
      fprintf(stdout, "loom-reduce: %u file(s) now need new link inputs%s%s\n",
              summary.linked_file_count, FLAG_links[0] ? "; see " : "",
              FLAG_links);
    }
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = LOOM_REDUCE_EXIT_ERROR;
  }

  // The plan borrows analyzer results, which borrow the modules.
  loom_reduce_plan_free(plan);
  loom_analyze_analyzer_free(analyzer);
  for (iree_host_size_t i = 0; i < loaded_count; ++i) {
    loomc_module_release(loaded[i].module);
    iree_io_file_contents_free(loaded[i].contents);
    iree_allocator_free(allocator, loaded[i].relative_path);
  }
  iree_allocator_free(allocator, loaded);
  iree_allocator_free(allocator, plan_inputs);
  iree_string_builder_deinitialize(&path);
  loom_analyze_input_list_deinitialize(&inputs);
  loom_analyze_session_free(session);
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
