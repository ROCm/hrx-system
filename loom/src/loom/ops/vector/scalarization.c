// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/scalarization.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

#define LOOM_VECTOR_SCALARIZATION_OP_INDEX(op_kind) \
  ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_VECTOR_SCALARIZATION_ROW(vector_op, scalar_op, flags_, \
                                      seed_operand_index_)          \
  [LOOM_VECTOR_SCALARIZATION_OP_INDEX(vector_op)] = {               \
      .lane_op_kind = (scalar_op),                                  \
      .flags = (flags_),                                            \
      .seed_operand_index = (seed_operand_index_),                  \
  },

const loom_vector_scalarization_t
    loom_vector_scalarization_rows[LOOM_OP_VECTOR_COUNT_] = {
#include "loom/ops/vector/scalarization_rows.inl"
};

#undef LOOM_VECTOR_SCALARIZATION_ROW
#undef LOOM_VECTOR_SCALARIZATION_OP_INDEX
