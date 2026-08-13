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
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independently owned kernel entry derived from a source launch site.
//
// The entry owns a selectively linked module containing the launched kernel
// and its dependency closure. Its executable ABI may be narrower than the
// source kernel ABI because every consumer of the derived entry is owned by
// the command-program compilation. The source module and source kernel ABI
// remain unchanged.
typedef struct loom_cmd_kernel_entry_t {
  // Owned module containing the derived kernel and its dependency closure.
  loom_module_t* module;

  // Derived kernel definition in |module|.
  loom_op_t* kernel_op;

  // Exact target symbol ordinal in the closed source module, or invalid when
  // the source kernel is targetless.
  loom_symbol_id_t source_target_symbol_id;

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
} loom_cmd_kernel_entry_t;

// One independently compilable kernel unit containing several named exports.
//
// All exports have the same exact source target. Their order is the unit-local
// export order referenced by command-root entry requirements.
typedef struct loom_cmd_kernel_unit_t {
  // Owned hermetic module containing every kernel export and dependency.
  loom_module_t* module;

  // Module-owned kernel definitions in unit-local export order.
  loom_op_t** kernel_ops;

  // Number of entries in |kernel_ops|.
  uint32_t export_count;
} loom_cmd_kernel_unit_t;

// Returns true when two source launches have the same kernel-unit identity.
//
// Launches match when they reference the same linked kernel symbol and carry
// equal boundary-projected scalar facts for every workload and device-ABI
// operand. Non-scalar operand identity does not affect the derived unit.
bool loom_cmd_kernel_entry_launches_equivalent(
    const loom_module_t* source_module, const loom_op_t* lhs_launch_op,
    const loom_op_t* rhs_launch_op,
    const loom_value_fact_table_t* source_facts);

// Materializes one compiler-owned kernel entry from |source_launch_op|.
//
// |source_facts| must describe the function containing the launch and remain
// valid for the duration of this call. Workload operands map to the kernel
// launch-configuration formals through the symbol kernel contract. Device-ABI
// operands map to the kernel body through CallLike. Exact and abstract scalar
// facts are consumed while specializing the derived unit; no source-module
// analysis state is retained.
//
// |entry_ordinal| provides a stable plan-local suffix so several specialized
// copies of one source kernel can coexist in a packed unit.
//
// On success |out_entry| owns its module and must be deinitialized. On failure
// |out_entry| is empty and the source module is unchanged.
iree_status_t loom_cmd_kernel_entry_materialize(
    const loom_module_t* source_module, const loom_op_t* source_launch_op,
    const loom_value_fact_table_t* source_facts, uint32_t entry_ordinal,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_cmd_kernel_entry_t* out_entry);

// Links |entry_indices| from |entries| into one hermetic compilation unit.
//
// Entries must share one exact source target and have unique derived export
// names. The output export order matches |entry_indices|.
iree_status_t loom_cmd_kernel_unit_pack(const loom_cmd_kernel_entry_t* entries,
                                        const uint32_t* entry_indices,
                                        uint32_t entry_count,
                                        iree_arena_block_pool_t* block_pool,
                                        iree_allocator_t allocator,
                                        loom_cmd_kernel_unit_t* out_unit);

// Releases all storage owned by |entry| and resets it to empty.
void loom_cmd_kernel_entry_deinitialize(loom_cmd_kernel_entry_t* entry);

// Releases all storage owned by |unit| and resets it to empty.
void loom_cmd_kernel_unit_deinitialize(loom_cmd_kernel_unit_t* unit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_KERNEL_UNIT_H_
