// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-test/options.h"

#include <string.h>

#include "iree/base/tooling/flags.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/input/flags.h"
#include "loom/tooling/testbench/testbench.h"

IREE_FLAG(string, case, "",
          "Optional check.case or check.scenario symbol to execute, such as "
          "'@smoke'. Empty executes all records in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for the selected case "
          "or cases. Negative executes all planned samples.");
IREE_FLAG_LIST(
    string, library,
    "Source or bytecode library linked into the authored input. Repeat as "
    "--library=path.loombc. Libraries are linked whole in argument order.");
IREE_FLAG_NAMED(
    string, file_output_root, "file-output-root", "",
    "Root for relative check.file.write paths. Required only when a "
    "selected case writes a file. Parent directories are created as needed; "
    "'-' is not a directory.");
IREE_FLAG_NAMED(int32_t, max_samples_per_case, "max-samples-per-case",
                LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
                "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used for HAL kernel launches. 'default' runs the "
          "normal compiler pipeline. 'none' disables all compiler "
          "transformations and requires emission-ready input. Use '@symbol' "
          "or a comma-separated pass list for an explicit pipeline.");
IREE_FLAG(string, target, "",
          "Optional compiler target as `family:selector` for every HAL kernel "
          "launch. The selected device must be able to load the target. Empty "
          "selects a compatible target from the device and authored kernel.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding for kernel and function compilation. "
    "Repeat as --config=key=value. "
    "Bindings not referenced by the loaded module are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file for kernel and function compilation. "
    "Repeat for multiple files. "
    "Nested object keys are flattened with '.' separators.");
IREE_FLAG(string, sanitizer, "none",
          "Sanitizer checks inserted by the target pipeline. Use 'none', "
          "'access', 'value', 'operation', 'race', 'asan', 'ubsan', 'tsan', "
          "'all', or a '|' separated list.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Sanitizer reporting mode used by the target pipeline. Use "
                "'default', 'trap', or 'report-only'.");
IREE_FLAG_LIST_NAMED(
    string, xfail, "xfail",
    "Required record failure as '@record=DOMAIN/NNN[,DOMAIN/NNN...]'. "
    "Repeat once per record. XPASS and diagnostic drift fail the run.");
IREE_FLAG_LIST_NAMED(
    string, allow_failure, "allow-failure",
    "Permitted record failure as '@record=DOMAIN/NNN[,DOMAIN/NNN...]'. "
    "A pass is valid, while any other diagnostic fails the run.");

static iree_string_view_list_t loom_test_flag_list_view(
    iree_flag_string_list_t list) {
  return (iree_string_view_list_t){
      .count = list.count,
      .values = list.values,
  };
}

void loom_test_options_initialize(loom_test_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->sample_ordinal = -1;
  out_options->max_samples_per_case =
      LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE;
  out_options->pipeline = IREE_SV("default");
}

iree_status_t loom_test_options_from_flags(loom_test_options_t* out_options) {
  loom_test_options_initialize(out_options);
  out_options->input = loom_input_options_from_flags();
  out_options->selected_case = iree_make_cstring_view(FLAG_case);
  out_options->sample_ordinal = FLAG_sample;
  out_options->libraries = loom_test_flag_list_view(FLAG_library_list());
  out_options->file_output_root = iree_make_cstring_view(FLAG_file_output_root);
  if (FLAG_max_samples_per_case <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--max-samples-per-case must be positive; got %d",
                            (int)FLAG_max_samples_per_case);
  }
  out_options->max_samples_per_case =
      (iree_host_size_t)FLAG_max_samples_per_case;
  out_options->pipeline = iree_make_cstring_view(FLAG_pipeline);
  out_options->target = iree_make_cstring_view(FLAG_target);
  out_options->config_assignments =
      loom_test_flag_list_view(FLAG_config_list());
  out_options->config_files = loom_test_flag_list_view(FLAG_config_file_list());
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_parse_checks(
      iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
      &out_options->sanitizer));
  IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_parse(
      iree_make_cstring_view(FLAG_sanitizer_reporting),
      IREE_SV("--sanitizer-reporting"),
      &out_options->sanitizer.reporting_mode));
  out_options->expected_failures = loom_test_flag_list_view(FLAG_xfail_list());
  out_options->allowed_failures =
      loom_test_flag_list_view(FLAG_allow_failure_list());
  return loom_test_options_validate(out_options);
}

iree_status_t loom_test_options_validate(const loom_test_options_t* options) {
  if (options->max_samples_per_case == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_samples_per_case must be positive");
  }
  if (iree_string_view_equal(options->file_output_root, IREE_SV("-"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--file-output-root must name a directory, not '-'");
  }
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(&options->sanitizer));
  if (loom_sanitizer_options_is_enabled(&options->sanitizer) &&
      !loom_tooling_cli_pipeline_uses_default(
          iree_string_view_trim(options->pipeline))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--sanitizer and --sanitizer-reporting require --pipeline=default");
  }
  return iree_ok_status();
}
