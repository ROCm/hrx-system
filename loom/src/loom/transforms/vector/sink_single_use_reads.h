// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_SINK_SINGLE_USE_READS_H_
#define LOOM_TRANSFORMS_VECTOR_SINK_SINGLE_USE_READS_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns pass metadata for source-level read sinking.
const loom_pass_info_t* loom_sink_single_use_reads_pass_info(void);

// Sinks single-use multi-lane vector read operations toward their sole user
// when doing so only crosses pure/read-only operations in the same block.
iree_status_t loom_sink_single_use_reads_run(loom_pass_t* pass,
                                             loom_module_t* module,
                                             loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_SINK_SINGLE_USE_READS_H_
