// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Promotes simple invocation-private fragment buffers into SSA values.
//
// The pass recognizes rank-1 private alloca views that are initialized by one
// full-domain scalar copy loop and replaces later scalar private loads with
// vector.extract operations from a single vector.load. The transformation keeps
// frontend fragment intent out of target lowering without inventing a target
// private memory fallback.

#ifndef LOOM_TRANSFORMS_PROMOTE_PRIVATE_FRAGMENTS_H_
#define LOOM_TRANSFORMS_PROMOTE_PRIVATE_FRAGMENTS_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_promote_private_fragments_pass_info(void);

iree_status_t loom_promote_private_fragments_run(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PROMOTE_PRIVATE_FRAGMENTS_H_
