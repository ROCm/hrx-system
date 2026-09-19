// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural ancestry queries over stable constructed IR.

#ifndef LOOM_IR_ANCESTRY_H_
#define LOOM_IR_ANCESTRY_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |ancestor_op| is |op| or transitively owns a region
// containing it. Queries walk the maintained parent_op chain in O(depth).
bool loom_op_is_ancestor_of(const loom_op_t* ancestor_op, const loom_op_t* op);

// Returns true when |ancestor_op| transitively owns |block| through one of its
// regions. Nonempty regions use maintained operation ancestry in O(depth).
// Op-less regions fall back to a structural walk because regions intentionally
// carry no parent pointer.
bool loom_op_contains_block(const loom_op_t* ancestor_op,
                            const loom_block_t* block);

// Returns true when |value_id|'s definition is |ancestor_op| or is nested in
// one of its regions. Invalid value IDs and declaration arguments return false.
bool loom_op_subtree_defines_value(const loom_module_t* module,
                                   const loom_op_t* ancestor_op,
                                   loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_ANCESTRY_H_
