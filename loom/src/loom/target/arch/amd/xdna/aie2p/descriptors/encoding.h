// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final adaptation from allocated AIE2P Low descriptors to instruction slots.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_

#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encodes one compiler-verified descriptor after register allocation.
// Assignment pointers correspond positionally to descriptor operand rows,
// including results. Immediate values correspond positionally to descriptor
// immediates and symbolic values have already been resolved. Low verification,
// allocation, and generated target tables prove every join consumed here.
loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal,
    const loom_low_allocation_assignment_t* const* operand_assignments,
    const int64_t* immediate_values);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_
