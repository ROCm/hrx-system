// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P module-internal direct-call ABI classification.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_

#include "loom/codegen/low/allocation/call.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // r0-r14 carry scalar arguments and results. r15 joins the preserved bank.
  LOOM_AIE2P_CALL_ABI_SCALAR_REGISTER_COUNT = 15,
  // p0 and p1 are the address members of the two load-FIFO tuples. Pointer
  // arguments begin at p2 so helpers retain both input streams.
  LOOM_AIE2P_CALL_ABI_POINTER_REGISTER_OFFSET = 2,
  // r15 and r18-r22 survive direct helper calls. r16 and r17 form the fixed l8
  // predicate carrier, while r23 is the link shadow and r24-r31 have other
  // architectural roles. Helpers reserve these general-purpose registers
  // instead of emitting save and restore traffic around every call.
  LOOM_AIE2P_CALL_ABI_PRESERVED_SCALAR_REGISTER_COUNT = 6,
};

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

// Returns the physical register for one caller-preserved scalar-bank ordinal.
// |ordinal| must be less than
// LOOM_AIE2P_CALL_ABI_PRESERVED_SCALAR_REGISTER_COUNT.
uint16_t loom_aie2p_call_abi_preserved_scalar_register(uint16_t ordinal);

// Adds one lowered carrier to |inout_requirements| and selects its fixed
// physical location. The requirements describe one argument or result side
// and must start empty for each side.
iree_status_t loom_aie2p_call_abi_select_location(
    loom_type_t low_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements,
    loom_low_allocation_abi_location_t* out_location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CALL_ABI_H_
