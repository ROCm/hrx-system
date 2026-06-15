// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check support for Wasm binary emission tests.

#ifndef LOOM_TARGET_EMIT_WASM_CHECK_LOOM_CHECK_H_
#define LOOM_TARGET_EMIT_WASM_CHECK_LOOM_CHECK_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits target-low Wasm modules and compares external-tool disassembly.
extern const loom_check_emit_provider_t loom_wasm_loom_check_emit_provider;

// Probes Wasm external-tool requirements.
extern const loom_check_requirement_provider_t
    loom_wasm_loom_check_requirement_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_CHECK_LOOM_CHECK_H_
