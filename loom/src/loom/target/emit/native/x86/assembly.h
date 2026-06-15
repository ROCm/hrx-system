// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 native assembly-fragment emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an Intel-syntax x86 assembly fragment for one scheduled and allocated
// x86 low.func.def. The fragment assumes exact physical-register inputs and
// outputs; it does not emit a platform ABI prologue/epilogue or object file.
// Values must be physically allocated and unspilled.
iree_status_t loom_x86_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_ASSEMBLY_H_
