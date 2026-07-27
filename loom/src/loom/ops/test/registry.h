// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit registration hook for the synthetic test dialect.
//
// Production context initialization intentionally excludes the test dialect so
// embedders do not pull parser/verifier scaffolding into JIT builds by linking
// the normal op registry. Developer tools and tests that consume test.* IR use
// this helper as an explicit opt-in.

#ifndef LOOM_OPS_TEST_REGISTRY_H_
#define LOOM_OPS_TEST_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the synthetic test dialect vtables with |context|.
//
// The context must have been initialized and not yet finalized. The vtable
// storage is static and remains valid for the process lifetime.
iree_status_t loom_test_dialect_register(loom_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TEST_REGISTRY_H_
