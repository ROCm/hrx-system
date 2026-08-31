// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final-placement relocation for AIE2P native object contributions.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_RELOCATION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_RELOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/emit/native/contribution.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_aie2p_native_relocation_kind_e {
  LOOM_AIE2P_NATIVE_RELOCATION_KIND_NONE = 0,
  // Absolute core program-memory address in J, JZ, or JNZ cpmaddr.
  LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE = 1,
} loom_aie2p_native_relocation_kind_t;

// Applies all AIE2P fixups after contribution assembly and address placement.
//
// Section addresses and contribution layouts in |assembly| must be final. The
// function mutates the arena-owned assembled section payloads in place. Symbol
// and fixup layout scratch is allocated from |scratch_arena|.
iree_status_t loom_aie2p_native_object_apply_fixups(
    const loom_native_object_contribution_t* object,
    loom_native_section_contribution_assembly_t* assembly,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_RELOCATION_H_
