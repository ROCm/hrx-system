// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_OP_REGISTRY_H_
#define LOOM_OPS_OP_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers production dialect vtables and built-in encoding families.
//
// The context must have been initialized and must not have been
// finalized yet. The test dialect is intentionally not registered here;
// developer tools and tests that need it must opt in explicitly.
iree_status_t loom_op_registry_register_all_dialects(
    loom_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_OP_REGISTRY_H_
