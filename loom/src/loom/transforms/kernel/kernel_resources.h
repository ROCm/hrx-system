// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel resource ABI normalization.
//
// Exported device functions use ordered opaque buffer arguments as ABI resource
// roots. Typed logical access must be made explicit inside the function body
// via buffer.view so alias analyses and later slab-packing passes can reason
// from a visible buffer root to every derived view.

#ifndef LOOM_TRANSFORMS_KERNEL_RESOURCES_H_
#define LOOM_TRANSFORMS_KERNEL_RESOURCES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_normalize_kernel_resources_pass_info(void);

iree_status_t loom_normalize_kernel_resources_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_RESOURCES_H_
