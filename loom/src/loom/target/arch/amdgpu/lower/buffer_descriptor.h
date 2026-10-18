// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained buffer descriptor bounds and their Low materialization.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_DESCRIPTOR_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_DESCRIPTOR_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_buffer_extent_plan_t
    loom_amdgpu_buffer_extent_plan_t;

// Selects a source-view extent using canonical facts and actual producer
// carriers. The retained recipe belongs to the function plan. NULL selects the
// full U32 descriptor range when the view has no tighter representable bound.
// Explicit resource extents take precedence over this view-derived recipe.
iree_status_t loom_amdgpu_plan_buffer_extent(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_buffer_extent_plan_t** out_plan);

// Selects an exactly representable root-relative source-view end and makes it
// the descriptor range even when the ABI resource carries a broader extent.
// NULL means the view end cannot be represented by the target descriptor.
iree_status_t loom_amdgpu_plan_buffer_view_extent(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_buffer_extent_plan_t** out_plan);

// Emits a buffer descriptor from a Low resource and its retained view bound.
// Resource extent/control attributes are the ABI contract; source analysis is
// never consulted. Only allocation can fail while executing the recipe.
iree_status_t loom_amdgpu_emit_hal_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding,
    const loom_low_source_memory_access_plan_t* source_access,
    const loom_amdgpu_buffer_extent_plan_t* plan,
    loom_value_id_t* out_low_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_DESCRIPTOR_H_
