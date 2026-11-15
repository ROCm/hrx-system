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
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_order_reads(traits)) {
    return false;
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_region_has_read_ordering_effects(regions[i])) {
      return false;
    }
  }
  return true;
}
