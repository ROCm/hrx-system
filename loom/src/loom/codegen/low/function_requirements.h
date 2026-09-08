// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Declared interfaces and storage of one immutable, verified Low function.

#ifndef LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_

#include "loom/codegen/low/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Requirements available before scheduling, allocation, or native emission.
//
// Resource operations borrow the source function. Storage records belong to
// the supplied arena and use source value IDs. Both describe one immutable
// function snapshot and must be rebuilt after its declarations change. Storage
// includes explicit reservations, not spills that allocation may introduce.
typedef struct loom_low_function_requirements_t {
  // Low resource declarations in function body order.
  const loom_op_t* const* resources;
  // Number of resource declarations.
  iree_host_size_t resource_count;
  // Packed function-local storage reservations.
  loom_low_storage_layout_t storage_layout;
  // Number of operations in the immediate function body.
  iree_host_size_t node_count;
  // Number of Low returns that transfer control back to a caller.
  iree_host_size_t return_count;
} loom_low_function_requirements_t;

// Inventories one verified Low function body without resolving a target or
// acquiring the module's value-ordinal scratch map. The function's immediate
// region owns its ABI imports and storage; nested regions are not entry scopes.
iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_
