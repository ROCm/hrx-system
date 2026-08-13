// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low scalar address expression materialization.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ADDRESS_MATERIALIZATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_ADDRESS_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lowers one amdgpu.address.add_scaled_u32 |op| using an existing rewriter
// and interned one-unit SGPR |sgpr_type|. This lets enclosing target
// materializers fold address expansion into an operation traversal they
// already require.
iree_status_t loom_amdgpu_address_materialize_expression(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type);

// Lowers AMDGPU scalar address expressions in |function_op| into ordered
// native SALU packet sequences. The structural operations retain complete
// address semantics through source-to-low cleanup so ordinary SSA CSE can
// share them before the native packets expose SCC state dependencies.
iree_status_t loom_amdgpu_address_materialize_expressions(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_materialized_count,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ADDRESS_MATERIALIZATION_H_
