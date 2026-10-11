// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/read_motion.h"

#include "loom/ops/op_defs.h"

bool loom_motion_read_can_cross_op(const loom_module_t* module,
                                   const loom_op_t* op) {
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return true;
  }
  if (op->region_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_may_write(traits)) {
    return false;
  }
  return !iree_any_bit_set(
      traits, LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_HINT |
                  LOOM_TRAIT_POISON_BOUNDARY | LOOM_TRAIT_CONVERGENT |
                  LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_UNIQUE_IDENTITY);
}
