// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_VALUE_TREE_H_
#define LOOM_IR_VALUE_TREE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static structure of an ordered value tree. Inputs and results remain ordinary
// SSA fields, each in its own preorder leaf order. The shape contains no value
// identities, offsets, native ABI selectors, or compiler planning state.
//
// Each nonnegative i64 token stores its kind in the low two bits and immediate
// child count in the remaining bits. VALUE and ABSENT have zero children;
// RESULT is a VALUE with payload 1 and consumes the next ordinary SSA result.
// RECORD denotes a fixed positional tuple; RANGE denotes a variable-length
// list. Absence, an empty tuple, and an empty list are distinct semantic
// values. The whole shape describes exactly one root, including for an absent
// group.
enum loom_value_tree_kind_e {
  LOOM_VALUE_TREE_VALUE = 0,
  LOOM_VALUE_TREE_ABSENT = 1,
  LOOM_VALUE_TREE_RECORD = 2,
  LOOM_VALUE_TREE_RANGE = 3,
  LOOM_VALUE_TREE_RESULT = 4,
};

// Validates external shape data and exact coverage of input and result spans
// without allocation or recursion. Shape and value counts use the owning IR
// fields' u16 domain. A NULL shape or an empty shape is invalid, including for
// empty value spans. Consumers after verification use the shape directly.
bool loom_value_tree_validate(const int64_t* shape, uint16_t shape_count,
                              uint16_t value_count, uint16_t result_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_VALUE_TREE_H_
