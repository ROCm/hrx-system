// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit core test-low descriptor registry package.
//
// This package intentionally links only synthetic test-low descriptor sets.
// Core compiler tests and developer tools use it when they need a tiny
// deterministic descriptor package instead of real backend tables.

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the core test-low descriptor-set registry.
void loom_target_core_test_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

// Looks up a descriptor set by key in the core test-low registry, or returns
// NULL when no descriptor set matches.
const loom_low_descriptor_set_t*
loom_target_core_test_low_descriptor_set_lookup(iree_string_view_t key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_
