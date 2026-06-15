// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU collective combining-kind mapping support.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_COMBINE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_COMBINE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a combining kind and payload shape to the native AMDGPU VGPR packet used
// by reductions and scans.
bool loom_amdgpu_collective_combine_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref);

// Returns the 32-bit identity bit pattern for a native AMDGPU combining packet.
bool loom_amdgpu_collective_combine_identity_bits(loom_combining_kind_t kind,
                                                  uint32_t* out_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_COMBINE_H_
