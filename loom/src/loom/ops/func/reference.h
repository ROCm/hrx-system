// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// First-class function reference type utilities.

#ifndef LOOM_OPS_FUNC_REFERENCE_H_
#define LOOM_OPS_FUNC_REFERENCE_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the structural function signature carried by |reference_type|.
// Returns none when the type is not a func.ref, its signature parameter is
// invalid, or the parameter does not identify a function type.
loom_type_t loom_func_ref_resolve_signature(const loom_module_t* module,
                                            loom_type_t reference_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNC_REFERENCE_H_
