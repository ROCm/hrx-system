// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/options.h"

#include "loom/error/diagnostic.h"

enum {
  LOOM_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

void loom_compile_options_initialize(loom_compile_options_t* out_options) {
  *out_options = (loom_compile_options_t){
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_COMPILE_DEFAULT_MAX_ERRORS,
  };
}
