// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-Low verification pass adapter.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_VERIFICATION_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_VERIFICATION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static metadata for the target-Low verification pass.
const loom_pass_info_t* loom_low_verification_pass_info(void);

// Verifies every target-Low function against the linked target environment.
iree_status_t loom_low_verification_run(loom_pass_t* pass,
                                        loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_VERIFICATION_H_
