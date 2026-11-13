// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU descriptor semantics used across target planning and emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns AMDGPU schedule state reads for structural low materializations.
loom_low_schedule_structural_state_read_list_t
loom_amdgpu_descriptor_structural_state_reads(void);

// Models storage-address materialization with the target's immediate VGPR move.
loom_low_schedule_structural_model_t
loom_amdgpu_descriptor_storage_address_model(
    const loom_low_descriptor_set_t* descriptor_set);

// Number of native instructions in a descriptor's indivisible expansion.
static inline uint32_t loom_amdgpu_descriptor_instruction_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return descriptor == loom_amdgpu_descriptor_ref_descriptor(
                           descriptor_set,
                           LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_SYMBOL)
             ? 3u
             : 1u;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
