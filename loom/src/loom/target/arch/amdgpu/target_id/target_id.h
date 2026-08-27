// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target identity formatting and resolved low-target queries.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_resolved_target_t loom_low_resolved_target_t;

// Returns the AMDGPU processor selected by a resolved low target, or NULL.
const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_low_resolved_target_t* target);

// Returns the compiler-semantic processor properties selected by a resolved
// low target, or NULL.
const loom_amdgpu_processor_properties_t*
loom_amdgpu_target_processor_properties_from_resolved_target(
    const loom_low_resolved_target_t* target);

// Appends the canonical bare AMDGPU artifact target ID for |identity| to
// |builder|.
//
// Explicit ON and OFF feature states are emitted in stable target-family
// order. ANY and UNSUPPORTED states do not contribute suffixes.
iree_status_t loom_amdgpu_artifact_target_key_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder);

// Formats the canonical bare AMDGPU artifact target ID for |identity| into
// caller-owned storage. |out_target_id| borrows |buffer|.
iree_status_t loom_amdgpu_artifact_target_key_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_target_id);

// Formats the canonical bare AMDGPU artifact target ID for |identity| into
// |arena|.
iree_status_t loom_amdgpu_artifact_target_key_format_arena(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id);

// Formats the AMDHSA code-object target ID projected from |identity| into
// |arena|.
//
// The canonical target is projected to its backend processor and only feature
// states represented by the AMDHSA ABI are emitted. Target-overlay semantics
// remain in the enclosing artifact identity.
iree_status_t loom_amdgpu_amdhsa_code_object_target_id_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_
