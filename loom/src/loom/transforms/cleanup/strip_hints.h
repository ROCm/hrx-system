// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_STRIP_HINTS_H_
#define LOOM_TRANSFORMS_STRIP_HINTS_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the hint-stripping pass.
const loom_pass_info_t* loom_strip_hints_pass_info(void);

// Removes compiler hint ops. Hints carry no semantic memory effects and are
// preserved by ordinary DCE/canonicalization so that pass pipelines can decide
// explicitly when code-quality guidance should be discarded.
iree_status_t loom_strip_hints_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_STRIP_HINTS_H_
