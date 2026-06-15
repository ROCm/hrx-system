// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar dialect adapters for shared combining semantics.

#ifndef LOOM_OPS_SCALAR_COMBINING_H_
#define LOOM_OPS_SCALAR_COMBINING_H_

#include "loom/ops/combining.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a combining kind to the scalar binary op that implements one pairwise
// combine step. Returns false if the kind has no scalar op representation.
bool loom_scalar_combining_kind_op(loom_combining_kind_t kind,
                                   loom_op_kind_t* out_op_kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCALAR_COMBINING_H_
