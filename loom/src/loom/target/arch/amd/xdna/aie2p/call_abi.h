// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P module-internal direct-call ABI classification.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register-bank demand for one side of a direct-call boundary.
typedef struct loom_aie2p_call_abi_requirements_t {
  // Number of one-unit aie2p.er scalar carriers.
  uint16_t scalar_count;
  // Number of one-unit aie2p.ep pointer carriers.
  uint16_t pointer_count;
} loom_aie2p_call_abi_requirements_t;

// Adds the physical direct-call carrier required by one source type. Buffers
// and views pass their data address in one pointer register. Returns false when
// the type has no representation in the module-internal direct-call ABI.
bool loom_aie2p_call_abi_add_source_type(
    loom_type_t source_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements);

// Adds the physical direct-call carrier required by one lowered type. Returns
// false unless the type is one one-unit aie2p.er or aie2p.ep register value.
bool loom_aie2p_call_abi_add_low_type(
    loom_type_t low_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements);

// Returns true when one argument or result side fits the fixed register ABI.
bool loom_aie2p_call_abi_requirements_fit(
    loom_aie2p_call_abi_requirements_t requirements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_
