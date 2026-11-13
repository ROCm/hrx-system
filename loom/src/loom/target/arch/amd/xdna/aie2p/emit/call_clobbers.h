// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical effects visible to the caller of one compiled AIE2P function.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALL_CLOBBERS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALL_CLOBBERS_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#ifdef __cplusplus
extern "C" {
#endif

// Composes native writes, argument/result transport, and completed direct
// callee contracts. |restored_register| is INVALID or a register saved on entry
// and restored on every return. Its native writes remain in the code facts but
// are absent from the caller-visible clobbers. Native writes include every
// overlapping register view, including aliases across classes.
// The arena owns the sorted, duplicate-free result. Callee rows retain their
// original register views: widening them to every alias and back into atomic
// units would falsely clobber the untouched portion of an overlapping view.
// Only result allocation can fail.
iree_status_t loom_aie2p_call_clobbers_build(
    loom_aie2p_register_unit_set_t native_writes,
    loom_aie2p_physical_register_id_t restored_register,
    const loom_low_call_contract_t* carriers,
    loom_symbol_ref_array_t direct_callees,
    loom_low_call_contract_provider_t call_contracts,
    iree_arena_allocator_t* arena, loom_low_call_clobber_list_t* out_clobbers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALL_CLOBBERS_H_
