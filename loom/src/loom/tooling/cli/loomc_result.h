// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line presentation of public LoomC operation results.

#ifndef LOOM_TOOLING_CLI_LOOMC_RESULT_H_
#define LOOM_TOOLING_CLI_LOOMC_RESULT_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "loomc/result.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints every diagnostic in the non-null |result| and returns its operation
// outcome through |out_succeeded|. The status reports output failure only.
// Source identifiers are expected to carry their final logical identity from
// admission through every later compiler operation.
iree_status_t loom_tooling_cli_print_loomc_result(FILE* file,
                                                  const loomc_result_t* result,
                                                  bool* out_succeeded);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CLI_LOOMC_RESULT_H_
