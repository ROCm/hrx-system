// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Durable low-IR representation of source-derived memory access summaries.

#ifndef LOOM_CODEGEN_LOW_MEMORY_ACCESS_IR_H_
#define LOOM_CODEGEN_LOW_MEMORY_ACCESS_IR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attaches the serializable precision in |summary| to |low_op|. Summaries that
// carry only descriptor-equivalent memory-space precision require no attribute
// and leave the operation unchanged.
iree_status_t loom_low_memory_access_ir_attach(
    loom_module_t* module, loom_op_t* low_op,
    const loom_low_memory_access_summary_t* summary);

// Reconstructs source-derived memory access records attached to direct packet
// operations in |low_func_op|. The returned table and records borrow |arena|;
// record operations and encoded attribute payloads continue to borrow the IR.
iree_status_t loom_low_memory_access_table_build_from_ir(
    const loom_op_t* low_func_op, iree_arena_allocator_t* arena,
    loom_low_memory_access_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_MEMORY_ACCESS_IR_H_
