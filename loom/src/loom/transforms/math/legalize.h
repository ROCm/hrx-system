// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-aware source-to-source math legalization.

#ifndef LOOM_TRANSFORMS_MATH_LEGALIZE_H_
#define LOOM_TRANSFORMS_MATH_LEGALIZE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the legalize-math pass.
const loom_pass_info_t* loom_math_legalize_pass_info(void);

// Creates legalize-math pass state from a textual option dictionary.
iree_status_t loom_math_legalize_create(loom_pass_t* pass,
                                        iree_string_view_t options);

// Rewrites semantic scalar/vector math ops to source IR supported by the
// current legalization policy. Unsupported ops are preserved unless a concrete
// recipe/policy declares that they must diagnose.
iree_status_t loom_math_legalize_run(loom_pass_t* pass, loom_module_t* module,
                                     loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_MATH_LEGALIZE_H_
