// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Detached native-object contribution emission for one AIE2P Low leaf.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits one arena-owned native object contribution from |plan|.
//
// Section bytes and names are copied into |arena| and no longer borrow the
// source Loom module. Symbols and fixups remain contribution-relative so a
// later array-image linker can gather independently compiled leaves without
// reopening worker object files or retaining compiler IR.
iree_status_t loom_aie2p_leaf_object_emit(
    const loom_aie2p_bundle_plan_t* plan, iree_arena_allocator_t* arena,
    loom_native_object_contribution_t* out_object);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_
