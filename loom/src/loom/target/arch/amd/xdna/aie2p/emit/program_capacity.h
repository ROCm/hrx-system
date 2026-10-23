// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Architectural program-memory admission for one AIE2P core.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CAPACITY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CAPACITY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Admits |required_byte_length| against the AIE2P core program-memory
// capacity. An oversized authored program emits a target diagnostic and
// returns OK with |out_admitted| false. Diagnostic-sink failures are returned
// as statuses.
iree_status_t loom_aie2p_program_capacity_admit(
    const loom_module_t* module, const loom_op_t* function_op,
    uint64_t required_byte_length, iree_diagnostic_emitter_t emitter,
    bool* out_admitted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CAPACITY_H_
