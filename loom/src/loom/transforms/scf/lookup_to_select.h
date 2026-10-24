// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projects keyed SSA tables into conditional whole-value selections.

#ifndef LOOM_TRANSFORMS_SCF_LOOKUP_TO_SELECT_H_
#define LOOM_TRANSFORMS_SCF_LOOKUP_TO_SELECT_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_scf_lookup_to_select_pass_info(void);

iree_status_t loom_scf_lookup_to_select_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SCF_LOOKUP_TO_SELECT_H_
