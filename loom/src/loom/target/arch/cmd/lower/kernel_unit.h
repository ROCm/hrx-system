// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned kernel units derived from command-program launch sites.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_KERNEL_UNIT_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_KERNEL_UNIT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independently owned dependency unit derived from a source launch site.
//
// The unit owns a selectively linked module containing the launched kernel and
// its dependency closure. Its executable ABI may be narrower than the source
// kernel ABI because every consumer of the derived entry is owned by the
// command-program compilation. The source module and source kernel ABI remain
// unchanged.
typedef struct loom_cmd_kernel_unit_t {
  // Owned module containing the derived kernel and its dependency closure.
  loom_module_t* module;

  // Derived kernel definition in |module|.
  loom_op_t* kernel_op;

  // Number of workload arguments on the source kernel.
  uint16_t source_workload_count;

  // Number of workload arguments retained by the derived kernel.
  uint16_t workload_count;

  // Source workload ordinal for each retained derived workload argument.
  const uint16_t* source_workload_ordinals;

  // Number of device-ABI arguments on the source kernel.
  uint16_t source_argument_count;

  // Number of device-ABI arguments retained by the derived kernel.
  uint16_t argument_count;

  // Source argument ordinal for each retained derived argument.
  const uint16_t* source_argument_ordinals;
} loom_cmd_kernel_unit_t;

// Exact dependency-closed source projection for one kernel definition.
//
// The source module must remain immutable while this projection is used. All
// projection storage belongs to the arena passed to
// loom_cmd_kernel_unit_source_prepare.
typedef struct loom_cmd_kernel_unit_source_t {
  // Borrowed immutable source module.
  const loom_module_t* module;

  // Kernel definition selected from module.
  loom_op_t* kernel_op;

  // Exact module-local symbol projection.
  struct {
    // Strictly increasing source symbol ordinals.
    const iree_host_size_t* ordinals;

    // Number of entries in ordinals.
    iree_host_size_t count;

    // Entry in ordinals that identifies kernel_op.
    iree_host_size_t kernel_selection_ordinal;
  } symbols;
} loom_cmd_kernel_unit_source_t;

// Prepares the exact source projection for one kernel definition.
//
// |references| must describe the same immutable source module. Ordinary symbol
// dependencies and providers for reachable template-family demands are
// included in the projection. No source IR is cloned or retained by the
// result.
iree_status_t loom_cmd_kernel_unit_source_prepare(
    const loom_module_t* source_module, loom_op_t* source_kernel_op,
    const loom_symbol_reference_table_t* references,
    iree_arena_allocator_t* arena, loom_cmd_kernel_unit_source_t* out_source);

// Returns true when two launches of the same resolved kernel definition have
// equivalent kernel-unit boundaries.
//
// The caller owns resolved kernel-definition identity. Launch boundaries match
// when they carry equal boundary-projected scalar facts for every workload and
// device-ABI operand. Non-scalar operand identity does not affect the derived
// unit.
bool loom_cmd_kernel_unit_boundaries_equivalent(
    const loom_module_t* source_module, const loom_op_t* lhs_launch_op,
    const loom_op_t* rhs_launch_op,
    const loom_value_fact_table_t* source_facts);

// Returns a process-local hash of one resolved kernel-unit boundary.
//
// The resolved kernel definition participates in the identity. Scalar workload
// and device argument facts are projected through the same boundary semantics
// used by loom_cmd_kernel_unit_boundaries_equivalent. Hash collisions require
// an exact equivalence check.
uint32_t loom_cmd_kernel_unit_boundary_hash(
    const loom_module_t* source_module, const loom_op_t* source_kernel_op,
    const loom_op_t* source_launch_op,
    const loom_value_fact_table_t* source_facts);

// Materializes one compiler-owned kernel unit from |source| and
// |source_launch_op|.
//
// |source_facts| must describe the function containing the launch and remain
// valid for the duration of this call. Workload operands map to the kernel
// launch-configuration formals through the symbol kernel contract. Device-ABI
// operands map to the kernel body through CallLike. Exact and abstract scalar
// facts are consumed while specializing the derived unit; no source-module
// analysis state is retained.
//
// On success |out_unit| owns its module and must be deinitialized. On failure
// |out_unit| is empty and the source module is unchanged.
iree_status_t loom_cmd_kernel_unit_materialize(
    const loom_cmd_kernel_unit_source_t* source,
    const loom_op_t* source_launch_op,
    const loom_value_fact_table_t* source_facts,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_cmd_kernel_unit_t* out_unit);

// Releases all storage owned by |unit| and resets it to empty.
void loom_cmd_kernel_unit_deinitialize(loom_cmd_kernel_unit_t* unit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_KERNEL_UNIT_H_
