// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semantic classification for Check expectation operations.

#ifndef LOOM_OPS_CHECK_EXPECTATION_H_
#define LOOM_OPS_CHECK_EXPECTATION_H_

#include <stdbool.h>

#include "loom/ops/check/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Comparison behavior requested by a check.expect operation.
typedef enum loom_check_expectation_kind_e {
  // Operation is not a Check expectation.
  LOOM_CHECK_EXPECTATION_NONE = 0,
  // Exact scalar or shaped-value equality.
  LOOM_CHECK_EXPECTATION_EQUAL = 1,
  // Bit-for-bit scalar or shaped-value equality.
  LOOM_CHECK_EXPECTATION_BITWISE = 2,
  // Floating-point comparison with authored tolerances.
  LOOM_CHECK_EXPECTATION_APPROXIMATE = 3,
  // Dynamic and static shape comparison.
  LOOM_CHECK_EXPECTATION_SHAPE = 4,
  // Structured runtime-event matching.
  LOOM_CHECK_EXPECTATION_EVENT = 5,
  // Signed integer-distance comparison.
  LOOM_CHECK_EXPECTATION_CLOSE_SIGNED = 6,
  // Unsigned integer-distance comparison.
  LOOM_CHECK_EXPECTATION_CLOSE_UNSIGNED = 7,
} loom_check_expectation_kind_t;

#define LOOM_CHECK_EXPECTATION_KIND_COUNT_ 8

static_assert(sizeof(loom_check_expectation_kind_t) == sizeof(int),
              "expectation kinds must retain their existing ABI width");

// Classifies |op| by its Check expectation semantics.
//
// This is the authoritative concrete-to-semantic mapping shared by Check
// verification and testbench planning. Consumers switch on the returned kind
// instead of maintaining concrete operation lists.
static inline loom_check_expectation_kind_t loom_check_expectation_classify(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_CHECK_EXPECT_EQUAL:
      return LOOM_CHECK_EXPECTATION_EQUAL;
    case LOOM_OP_CHECK_EXPECT_BITWISE:
      return LOOM_CHECK_EXPECTATION_BITWISE;
    case LOOM_OP_CHECK_EXPECT_APPROXIMATE:
      return LOOM_CHECK_EXPECTATION_APPROXIMATE;
    case LOOM_OP_CHECK_EXPECT_SHAPE:
      return LOOM_CHECK_EXPECTATION_SHAPE;
    case LOOM_OP_CHECK_EXPECT_EVENT:
      return LOOM_CHECK_EXPECTATION_EVENT;
    case LOOM_OP_CHECK_EXPECT_CLOSE_SIGNED:
      return LOOM_CHECK_EXPECTATION_CLOSE_SIGNED;
    case LOOM_OP_CHECK_EXPECT_CLOSE_UNSIGNED:
      return LOOM_CHECK_EXPECTATION_CLOSE_UNSIGNED;
    default:
      return LOOM_CHECK_EXPECTATION_NONE;
  }
}

// Returns true for expectations comparing paired actual and expected values.
static inline bool loom_check_expectation_kind_is_pair(
    loom_check_expectation_kind_t kind) {
  switch (kind) {
    case LOOM_CHECK_EXPECTATION_EQUAL:
    case LOOM_CHECK_EXPECTATION_BITWISE:
    case LOOM_CHECK_EXPECTATION_APPROXIMATE:
    case LOOM_CHECK_EXPECTATION_CLOSE_SIGNED:
    case LOOM_CHECK_EXPECTATION_CLOSE_UNSIGNED:
      return true;
    default:
      return false;
  }
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_CHECK_EXPECTATION_H_
