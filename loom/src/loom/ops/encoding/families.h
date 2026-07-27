// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Built-in encoding family registry.
//
// The text format keeps encoding family syntax generic (`#family<attrs...>`),
// while contexts register known family contracts for role inference and
// family-specific verification. This header exposes the small process-static
// family list used by tools that want the standard Loom dialect vocabulary.

#ifndef LOOM_OPS_ENCODING_FAMILIES_H_
#define LOOM_OPS_ENCODING_FAMILIES_H_

#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers all built-in encoding family vtables with |context|.
iree_status_t loom_context_register_builtin_encoding_vtables(
    loom_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_FAMILIES_H_
