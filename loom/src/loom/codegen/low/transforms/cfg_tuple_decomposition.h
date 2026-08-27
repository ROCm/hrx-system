// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Decomposes low CFG register tuples whose uses are already lane-local.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_DECOMPOSITION_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_DECOMPOSITION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_low_decompose_cfg_tuples_pass_info(void);

iree_status_t loom_low_decompose_cfg_tuples_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_DECOMPOSITION_H_
