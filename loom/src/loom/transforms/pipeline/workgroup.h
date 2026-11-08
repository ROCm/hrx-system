// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_WORKGROUP_H_
#define LOOM_TRANSFORMS_PIPELINE_WORKGROUP_H_

#include "loom/ops/builder.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Realizes one cooperative strand as an ordinary single-workgroup kernel.
// Construction and channel binding precede this boundary: strands are outlined,
// specialization arguments are bound, and communication is ordinary executable
// IR. The retained wave domain owns launch geometry; the body observes that
// same geometry through kernel queries. No work loop is split into dispatches.
const loom_pass_info_t* loom_pipeline_workgroup_pass_info(void);
iree_status_t loom_pipeline_workgroup_run(loom_pass_t* pass,
                                          loom_module_t* module,
                                          loom_func_like_t function);

// Contributes the shared realization to a provider's execution-lowering phase.
// Only source functions in this target family enter the function pass.
iree_status_t loom_pipeline_workgroup_build_pipeline(loom_builder_t* builder,
                                                     iree_string_view_t family);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_WORKGROUP_H_
