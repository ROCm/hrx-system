// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU exact matrix-fragment publication recipe costs.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_COST_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_COST_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Costs the candidate-specific direct scalar/packed descriptor recipe. Returns
// false when any descriptor required by the recipe is unavailable or malformed.
bool loom_amdgpu_fragment_publication_cost_direct(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_amdgpu_fragment_memory_publication_packet_t* packets,
    uint16_t packet_count, loom_low_representation_cost_t* out_cost);

// Costs the candidate-specific cross-lane packed descriptor recipe. Returns
// false when any descriptor required by the recipe is unavailable or malformed.
bool loom_amdgpu_fragment_publication_cost_crosslane(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_matrix_fragment_packed_b16_publication_t* publication,
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags,
    loom_amdgpu_descriptor_ref_t store_descriptor_ref,
    loom_low_representation_cost_t* out_cost);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_COST_H_
