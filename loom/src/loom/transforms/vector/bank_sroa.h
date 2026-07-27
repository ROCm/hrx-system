// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural scalar replacement for loop-carried vector banks.

#ifndef LOOM_TRANSFORMS_VECTOR_BANK_SROA_H_
#define LOOM_TRANSFORMS_VECTOR_BANK_SROA_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns pass metadata for vector-bank scalar replacement.
const loom_pass_info_t* loom_vector_bank_sroa_pass_info(void);

// Replaces statically addressed loop-carried vector banks with one carried
// scalar or tail-vector value per bank slot.
iree_status_t loom_vector_bank_sroa_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_BANK_SROA_H_
