// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only low descriptor table audits.
//
// These routines verify generator-owned static descriptor tables and
// registries. They intentionally live outside the production descriptor
// library: after a target's rodata is linked into a compiler executable, normal
// lowering and emission paths trust it as part of the binary's construction
// contract.

#ifndef LOOM_CODEGEN_LOW_TESTING_DESCRIPTORS_VERIFY_H_
#define LOOM_CODEGEN_LOW_TESTING_DESCRIPTORS_VERIFY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies structural integrity of a descriptor set. This checks table spans,
// string offsets, descriptor key uniqueness, and cross-table references; it
// does not perform full target legality.
iree_status_t loom_low_descriptor_set_verify(
    const loom_low_descriptor_set_t* descriptor_set);

// Verifies descriptor registry integrity. Registry validation is an explicit
// static-table audit for tests and table-generation checks.
iree_status_t loom_low_descriptor_registry_verify(
    const loom_low_descriptor_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TESTING_DESCRIPTORS_VERIFY_H_
