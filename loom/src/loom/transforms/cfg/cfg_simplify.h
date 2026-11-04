// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simplifies explicit CFG block structure and block-argument forwarding.

#ifndef LOOM_TRANSFORMS_CFG_SIMPLIFY_H_
#define LOOM_TRANSFORMS_CFG_SIMPLIFY_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_facts_t loom_target_facts_t;

const loom_pass_info_t* loom_cfg_simplify_pass_info(void);

// Simplifies one function through the same CFG and fact-maintenance lifecycle
// as the pass. The caller supplies independent scratch facts when enclosing
// construction facts must remain live. The function signature is preserved.
iree_status_t loom_cfg_simplify_function(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts, iree_arena_allocator_t* arena,
    loom_pass_value_fact_owner_t* value_facts, bool* out_changed);

iree_status_t loom_cfg_simplify_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CFG_SIMPLIFY_H_
