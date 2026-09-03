// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the callable inliner pass metadata.
const loom_pass_info_t* loom_inline_callables_pass_info(void);

// Creates an inline-callables pass instance.
iree_status_t loom_inline_callables_create(loom_pass_t* pass,
                                           iree_string_view_t options);

// Runs module-level required callable inlining.
iree_status_t loom_inline_callables_run(loom_pass_t* pass,
                                        loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_
