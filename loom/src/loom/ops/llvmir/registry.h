// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_LLVMIR_REGISTRY_H_
#define LOOM_OPS_LLVMIR_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the LLVMIR dialect with |context|.
iree_status_t loom_llvmir_ops_register_dialect(loom_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_LLVMIR_REGISTRY_H_
