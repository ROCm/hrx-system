// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-level AIE2P load-FIFO state semantics.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_FIFO_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_FIFO_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |type| is the opaque aie2p.load_fifo state type.
bool loom_aie2p_load_fifo_type_isa(const loom_module_t* module,
                                   loom_type_t type);

// Verifies that both architectural entry-state results use aie2p.load_fifo.
iree_status_t loom_aie2p_load_fifo_state_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_FIFO_H_
