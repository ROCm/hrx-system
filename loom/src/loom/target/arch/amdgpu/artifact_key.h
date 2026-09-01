// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU compiler artifact-key parsing and formatting.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_KEY_H_
#define LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_KEY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a canonical AMDGPU artifact key into a compiler-owned identity.
//
// The key begins with an exact, generic, or overlay compiler target and may
// carry AMDHSA feature coordinates such as `:sramecc+:xnack-`.
iree_status_t loom_amdgpu_artifact_key_parse(
    iree_string_view_t value, loom_amdgpu_target_identity_t* out_identity);

// Appends the canonical AMDGPU artifact key for |identity| to |builder|.
//
// Explicit ON and OFF feature states are emitted in stable target-family
// order. ANY and UNSUPPORTED states do not contribute suffixes.
iree_status_t loom_amdgpu_artifact_key_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder);

// Formats the canonical AMDGPU artifact key for |identity| into caller-owned
// storage. |out_artifact_key| borrows |buffer|.
iree_status_t loom_amdgpu_artifact_key_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_artifact_key);

// Formats the canonical AMDGPU artifact key for |identity| into |arena|.
iree_status_t loom_amdgpu_artifact_key_format_arena(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_artifact_key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_KEY_H_
