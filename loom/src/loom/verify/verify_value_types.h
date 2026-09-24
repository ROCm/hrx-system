// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_VALUE_TYPES_H_
#define LOOM_VERIFY_VERIFY_VALUE_TYPES_H_

#include "loom/verify/verify_state.h"

// Checks a structurally verified callable exit against its function-like
// result tuple.
iree_status_t loom_verify_func_like_exit(loom_verify_state_t* state,
                                         const loom_op_t* func_op,
                                         const loom_op_t* exit_op);

// Checks a structurally verified loop's entry arguments against its recurring
// result type scheme, including counted induction variable type and arity.
void loom_verify_loop_entry_types(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_loop_like_vtable_t* loop);

// Checks references carried by operation value types in the current scope.
void loom_verify_value_type_refs(loom_verify_state_t* state,
                                 const loom_op_t* op,
                                 const loom_op_vtable_t* vtable);

// Checks argument type references after the block's arguments are defined.
// The owner anchors diagnostics because block arguments have no locations.
void loom_verify_block_arg_type_refs(loom_verify_state_t* state,
                                     const loom_block_t* block,
                                     const loom_op_t* owner);

// Checks retained type and predicate references carried by operation
// attributes. Symbol definitions may refer to their own declaration-local
// values.
void loom_verify_attribute_value_refs(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable);

#endif  // LOOM_VERIFY_VERIFY_VALUE_TYPES_H_
