// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural Low operations admitted by AIE2P core programs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CORE_STRUCTURE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CORE_STRUCTURE_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_aie2p_core_structure_kind_e {
  // Operation has no representation in an AIE2P core program.
  LOOM_AIE2P_CORE_STRUCTURE_UNSUPPORTED = 0,
  // Compiler-only operation omitted from native emission.
  LOOM_AIE2P_CORE_STRUCTURE_COMPILE_TIME = 1,
  // Structural register transfer realized after physical allocation.
  LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE = 2,
  // Runtime value or storage declaration with no emitted instruction.
  LOOM_AIE2P_CORE_STRUCTURE_DECLARATION = 3,
  // Function-local storage address materialized as a native instruction.
  LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS = 4,
  // Unconditional CFG branch materialized as native control flow.
  LOOM_AIE2P_CORE_STRUCTURE_BRANCH = 5,
  // Conditional CFG branch materialized as native control flow.
  LOOM_AIE2P_CORE_STRUCTURE_CONDITIONAL_BRANCH = 6,
  // Function return materialized as native control flow.
  LOOM_AIE2P_CORE_STRUCTURE_RETURN = 7,
  // Direct module-local function call materialized as native control flow.
  LOOM_AIE2P_CORE_STRUCTURE_CALL = 8,
} loom_aie2p_core_structure_kind_t;

// Classifies |op| against the closed structural representation accepted by
// AIE2P core verification and consumed by native bundle planning. Descriptor
// packets are outside this classification and use the generated descriptor
// contract instead.
loom_aie2p_core_structure_kind_t loom_aie2p_core_structure_classify(
    const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CORE_STRUCTURE_H_
