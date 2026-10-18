// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LoomC diagnostic conversion for annotation matching.

#ifndef LOOM_TESTING_TEST_DIAGNOSTIC_LOOMC_H_
#define LOOM_TESTING_TEST_DIAGNOSTIC_LOOMC_H_

#include "loom/testing/test_diagnostic.h"
#include "loomc/diagnostic.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copies one public diagnostic into arena-owned storage independent of its
// result. The checker and formatter use the same annotation matching records.
iree_status_t loom_test_diagnostic_materialize_loomc(
    const loomc_diagnostic_t* source, iree_arena_allocator_t* arena,
    loom_test_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TEST_DIAGNOSTIC_LOOMC_H_
