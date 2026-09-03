// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Configured options for transitional target emitters.

#ifndef LOOM_TOOLING_COMPILE_CONFIGURED_EMITTER_H_
#define LOOM_TOOLING_COMPILE_CONFIGURED_EMITTER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Invokes a configured target emitter with any configured target-projection
// options it requires. This preserves debug-only target emitters while native
// products migrate to product-format providers.
iree_status_t loom_configured_target_emitter_emit(
    const loom_target_emitter_t* emitter,
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_CONFIGURED_EMITTER_H_
