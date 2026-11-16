// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LOADING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LOADING_H_

#include "loom/target/arch/amd/xdna/aie2p/program_clone.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds an ordinary returning Low helper taking one DMA descriptor selector.
// It disables the previous hardware-loop endpoint, queues a code image through
// the admitted control route, consumes the header/status completion, and
// returns only after successful delivery. Exact register/status effects
// participate in normal callable compilation. The physical queue route and
// command are private to this compilation workspace.
iree_status_t loom_aie2p_program_loading_build(loom_module_t* module,
                                               uint32_t queue_header,
                                               uint32_t queue_command,
                                               loom_symbol_ref_t* out_helper);

// Inserts loading calls at producer-indexed call boundaries before allocation.
// |descriptors| has one entry per retained call: UINT32_MAX leaves it
// unchanged; other values select an admitted code-delivery descriptor. Updates
// the direct callee and call-site inventories together. Every allocation is
// module-owned.
iree_status_t loom_aie2p_program_loading_insert(
    loom_module_t* module, loom_symbol_ref_t helper,
    const uint32_t* descriptors, loom_aie2p_program_clone_function_t* function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LOADING_H_
