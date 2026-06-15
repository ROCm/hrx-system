// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_DCE_H_
#define LOOM_TRANSFORMS_SYMBOL_DCE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the symbol DCE pass metadata.
const loom_pass_info_t* loom_symbol_dce_pass_info(void);

// Runs module-level symbol reachability DCE.
iree_status_t loom_symbol_dce_run(loom_pass_t* pass, loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_DCE_H_
