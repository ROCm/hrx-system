// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test context helpers.
//
// These helpers register the same base dialect surface used by user-facing
// Loom tools without depending on any particular tool executable. Unit tests
// and generated synthetic fuzzers should still register the minimal dialect set
// they exercise so accidental dependencies stay visible in BUILD files.

#ifndef LOOM_TESTING_CONTEXT_H_
#define LOOM_TESTING_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers production dialects, the synthetic test dialect, and built-in
// encoding families.
//
// The context must have been initialized and must not have been finalized yet.
iree_status_t loom_testing_context_register_all_dialects(
    loom_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TESTING_CONTEXT_H_
