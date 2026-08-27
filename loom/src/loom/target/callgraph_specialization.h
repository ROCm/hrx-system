// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Invocation-target specialization across retained semantic call graphs.

#ifndef LOOM_TARGET_CALLGRAPH_SPECIALIZATION_H_
#define LOOM_TARGET_CALLGRAPH_SPECIALIZATION_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the target callgraph specialization pass metadata.
const loom_pass_info_t* loom_target_callgraph_specialization_pass_info(void);

// Extends invocation-local target versions through retained semantic callees.
iree_status_t loom_target_callgraph_specialization_run(loom_pass_t* pass,
                                                       loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_CALLGRAPH_SPECIALIZATION_H_
