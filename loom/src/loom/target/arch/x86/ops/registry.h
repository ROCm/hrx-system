// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_X86_OPS_REGISTRY_H_
#define LOOM_TARGET_ARCH_X86_OPS_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the x86 target-family dialect with |context|.
iree_status_t loom_x86_ops_register_dialect(loom_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_OPS_REGISTRY_H_
