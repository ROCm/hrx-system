// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel resource ABI normalization.
//
// Realized kernel entries use ordered opaque buffer arguments as invocation
// ABI resource roots. Typed logical access is made
// explicit inside the body via buffer.view so alias analyses and later
// slab-packing passes can reason from a visible buffer root to every derived
// view. Staged arguments, pipeline constructions, and ordinary device functions
// retain their authored callable signatures. An execution realizer normalizes
// a pipeline only after establishing its independent invocation boundary.

#ifndef LOOM_TRANSFORMS_KERNEL_RESOURCES_H_
#define LOOM_TRANSFORMS_KERNEL_RESOURCES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_normalize_kernel_resources_pass_info(void);

// Rewrites an execution entry's runtime view arguments to buffer roots and
// inserts typed buffer.view operations in its body. The caller owns this
// invocation ABI: semantic callers have already been consumed or adapted.
// Specialization arguments retain their types. Reports the number of rewritten
// arguments; the caller invalidates its retained facts when nonzero.
iree_status_t loom_kernel_resources_normalize(loom_module_t* module,
                                              loom_func_like_t function,
                                              uint16_t* out_normalized_count);

iree_status_t loom_normalize_kernel_resources_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_RESOURCES_H_
