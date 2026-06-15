// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM one-shot execution backend for Loom run tools.

#ifndef LOOM_TOOLING_EXECUTION_IREEVM_BACKEND_H_
#define LOOM_TOOLING_EXECUTION_IREEVM_BACKEND_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/execution_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Execution backend named "vm" for the IREE VM archive path.
extern const loom_run_execution_backend_t loom_ireevm_execution_backend;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_IREEVM_BACKEND_H_
