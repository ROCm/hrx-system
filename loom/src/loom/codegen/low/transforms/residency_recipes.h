// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final target-low residency recipe sealing.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_RESIDENCY_RECIPES_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_RESIDENCY_RECIPES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static pass metadata for low-seal-residency-recipes.
const loom_pass_info_t* loom_low_seal_residency_recipes_pass_info(void);

// Seals the exact target-low producer membership of residency candidates.
iree_status_t loom_low_seal_residency_recipes_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_RESIDENCY_RECIPES_H_
