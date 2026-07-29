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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
