// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Build-time verification for static source-representation provider tables.

#ifndef LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_VERIFY_H_
#define LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_VERIFY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/source_representation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies every static span, reference, identity, canonical domain, target
// data row, and descriptor recipe in |provider|. Target table tests call this
// once at build time; production planning trusts verified static tables.
iree_status_t loom_low_source_representation_provider_verify(
    const loom_low_source_representation_provider_t* provider,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_VERIFY_H_
