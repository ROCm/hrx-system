// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical required physical-storage identities for a local value domain.

#ifndef LOOM_CODEGEN_LOW_STORAGE_IDENTITY_H_
#define LOOM_CODEGEN_LOW_STORAGE_IDENTITY_H_

#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds one canonical owner ordinal per value in the acquired domain's
// definition scope. Mandatory aliases and destructive ties share an owner;
// copies, slices, concats and control-flow payloads remain independent.
// Returns NULL when all values are independent. Otherwise the arena-owned
// array has value_count + additional_value_capacity entries, including an
// independent reserved tail for values introduced during preparation.
iree_status_t loom_low_storage_identity_build(
    const loom_local_value_domain_t* domain,
    iree_host_size_t additional_value_capacity, iree_arena_allocator_t* arena,
    loom_value_ordinal_t** out_origins);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_IDENTITY_H_
