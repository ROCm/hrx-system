// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/compile_options.h"

#include "loom/error/diagnostic.h"

enum {
  LOOM_RUN_DEFAULT_MAX_COMPILE_ERRORS = 20,
};

void loom_run_candidate_compile_options_initialize(
    loom_run_candidate_compile_options_t* out_options) {
  *out_options = (loom_run_candidate_compile_options_t){
      .module_name = IREE_SVL("loom"),
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_RUN_DEFAULT_MAX_COMPILE_ERRORS,
  };
}
