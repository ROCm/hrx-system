// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_STORAGE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_STORAGE_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves live buffer and view captures from retained reference facts and
// invocation-owned placement. Selects the consuming core's address aperture;
// storage identity, byte displacement, and layout do not require another walk.
iree_status_t loom_aie2p_native_select_storage_accesses(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* invocation,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker,
    bool* out_valid);

// Emits an ordinary Low helper returning one selected local address. Both
// channel endpoints and direct captures use the same physical pointer ABI.
iree_status_t loom_aie2p_native_address_helper(
    loom_aie2p_native_context_t* context, uint32_t address,
    loom_symbol_ref_t* out_symbol);

// Replaces captured arguments in the consuming clone. View types come from
// that instance so their shape/layout SSA dependencies remain in its domain.
// This emits address formation only, with no payload copy or new allocation.
iree_status_t loom_aie2p_native_materialize_storage_accesses(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_aie2p_native_worker_t* worker,
    const loom_pipeline_worker_materialization_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_STORAGE_H_
