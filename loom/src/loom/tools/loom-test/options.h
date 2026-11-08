// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured loom-test options and command-line flag adaptation.

#ifndef LOOM_TOOLS_LOOM_TEST_OPTIONS_H_
#define LOOM_TOOLS_LOOM_TEST_OPTIONS_H_

#include "iree/base/api.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/input/format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_test_options_t {
  // Source format, provider options, and retained diagnostic path remapping.
  loom_input_options_t input;
  // Optional check.case or check.scenario symbol selected for execution.
  iree_string_view_t selected_case;
  // Optional concrete sample ordinal, or negative to execute all samples.
  int32_t sample_ordinal;
  // Source or bytecode libraries linked whole in argument order.
  iree_string_view_list_t libraries;
  // Explicit root for relative check.file.write output paths.
  iree_string_view_t file_output_root;
  // Maximum planned concrete samples retained for each check.case.
  iree_host_size_t max_samples_per_case;
  // Pass pipeline used before target artifact emission.
  iree_string_view_t pipeline;
  // Optional explicit `family:selector` compiler target.
  iree_string_view_t target;
  // Direct config bindings parsed from repeated --config=key=value flags.
  iree_string_view_list_t config_assignments;
  // Config object file paths parsed from repeated --config-file=path flags.
  iree_string_view_list_t config_files;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Strict expected-failure qualifications in command-line order.
  iree_string_view_list_t expected_failures;
  // Pass-allowed failure qualifications in command-line order.
  iree_string_view_list_t allowed_failures;
} loom_test_options_t;

// Initializes default options for a direct structured run.
void loom_test_options_initialize(loom_test_options_t* out_options);

// Returns structured options borrowed from the parsed command-line flags.
iree_status_t loom_test_options_from_flags(loom_test_options_t* out_options);

// Validates options shared by direct and command-line callers.
iree_status_t loom_test_options_validate(const loom_test_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_TEST_OPTIONS_H_
