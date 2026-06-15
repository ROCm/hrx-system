// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the template selection pass metadata.
const loom_pass_info_t* loom_template_selection_pass_info(void);

// Creates a template selection pass invocation.
iree_status_t loom_template_selection_create(loom_pass_t* pass,
                                             iree_string_view_t options);

// Runs module-level template provider selection.
iree_status_t loom_template_selection_run(loom_pass_t* pass,
                                          loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_
