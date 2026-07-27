// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass descriptor entry points for target-aware source legalization.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_TARGET_LEGALIZE_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_TARGET_LEGALIZE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_low_target_legalize_pass_info(void);

iree_status_t loom_low_target_legalize_create(loom_pass_t* pass,
                                              iree_string_view_t options);

iree_status_t loom_low_target_legalize_run(loom_pass_t* pass,
                                           loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_TARGET_LEGALIZE_H_
