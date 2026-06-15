// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_CONSTRAINTS_H_
#define LOOM_VERIFY_VERIFY_CONSTRAINTS_H_

#include "loom/verify/verify_state.h"

void loom_verify_semantic_constraints(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable);

#endif  // LOOM_VERIFY_VERIFY_CONSTRAINTS_H_
