// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only target-low registry package audits.
//
// Target registry packages are linked static data. Production target selection
// trusts their provider and bundle tables; this verifier is for package tests
// and target bring-up checks that validate the authored tables explicitly.

#ifndef LOOM_TARGET_TESTING_LOW_DESCRIPTOR_REGISTRY_VERIFY_H_
#define LOOM_TARGET_TESTING_LOW_DESCRIPTOR_REGISTRY_VERIFY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/requirements.h"
#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies that the linked target-low registry package is internally
// consistent.
iree_status_t loom_target_low_descriptor_registry_verify(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TESTING_LOW_DESCRIPTOR_REGISTRY_VERIFY_H_
