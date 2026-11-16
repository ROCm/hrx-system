// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_ARGUMENT_ACCESS_H_
#define LOOM_VERIFY_VERIFY_ARGUMENT_ACCESS_H_

#include "loom/ops/op_defs.h"

typedef struct loom_verify_state_t loom_verify_state_t;

// Checks argument-bound shape/types and retains annotated bodies for the
// deferred interprocedural check. Operation structure must already be valid.
iree_status_t loom_verify_argument_access_record(loom_verify_state_t* state,
                                                 loom_func_like_t function);

// Checks promises against the shared storage-access graph, after structural
// verification has established valid bodies and call edges throughout the
// scope.
iree_status_t loom_verify_argument_access_check(loom_verify_state_t* state);

#endif  // LOOM_VERIFY_VERIFY_ARGUMENT_ACCESS_H_
