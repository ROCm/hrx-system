// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"

#include "loom/ops/low/ops.h"

loom_aie2p_core_structure_kind_t loom_aie2p_core_structure_classify(
    const loom_op_t* op) {
  if (loom_traits_are_compile_time_only(op->traits)) {
    return LOOM_AIE2P_CORE_STRUCTURE_COMPILE_TIME;
  }
  switch (op->kind) {
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_MOVE:
    case LOOM_OP_LOW_SLICE:
    case LOOM_OP_LOW_CONCAT:
      return LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE;
    case LOOM_OP_LOW_LIVE_IN:
    case LOOM_OP_LOW_RESOURCE:
    case LOOM_OP_LOW_STORAGE_RESERVE:
    case LOOM_OP_LOW_STORAGE_VIEW:
      return LOOM_AIE2P_CORE_STRUCTURE_DECLARATION;
    case LOOM_OP_LOW_STORAGE_ADDRESS:
      return LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS;
    case LOOM_OP_LOW_BR:
      return LOOM_AIE2P_CORE_STRUCTURE_BRANCH;
    case LOOM_OP_LOW_COND_BR:
      return LOOM_AIE2P_CORE_STRUCTURE_CONDITIONAL_BRANCH;
    case LOOM_OP_LOW_RETURN:
      return LOOM_AIE2P_CORE_STRUCTURE_RETURN;
    case LOOM_OP_LOW_FUNC_CALL:
      return LOOM_AIE2P_CORE_STRUCTURE_CALL;
    default:
      return LOOM_AIE2P_CORE_STRUCTURE_UNSUPPORTED;
  }
}
