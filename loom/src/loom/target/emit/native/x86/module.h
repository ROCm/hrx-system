// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core x86 native module emission.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_MODULE_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_MODULE_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an object from prepared Low with the x86 architecture's fact identity.
// The composing architecture supplies its identity so explicitly requested
// formats enforce the same family boundary as canonical format selection.
// Ordinary scalar functions use SysV independently of the compiler host.
// The returned artifact owns its bytes independently of compiler storage.
iree_status_t loom_x86_module_emit(
    const loom_target_emit_request_t* request,
    const loom_target_fact_type_t* target_fact_type, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_MODULE_H_
