// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only pass registry table audits.
//
// Pass registries are static descriptor tables linked into the compiler. Normal
// pass lookup, option decoding, and invocation paths trust them as executable
// construction invariants; this verifier is for tests that validate authored
// or generated descriptor tables before they are deployed.

#ifndef LOOM_PASS_TESTING_REGISTRY_VERIFY_H_
#define LOOM_PASS_TESTING_REGISTRY_VERIFY_H_

#include "iree/base/api.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies registry ordering, key uniqueness, and descriptor shape.
iree_status_t loom_pass_registry_verify(const loom_pass_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TESTING_REGISTRY_VERIFY_H_
