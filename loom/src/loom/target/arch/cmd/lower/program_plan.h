// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned command root and dependency-unit preparation.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/pass/registry.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/lower/kernel_unit.h"
#include "loom/target/arch/cmd/lower/launch_graph.h"
#include "loom/target/arch/cmd/lower/parameters.h"
#include "loom/target/arch/cmd/lower/transients.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source supplying one atomic executable-entry binding requirement.
typedef enum loom_cmd_entry_requirement_source_e {
  // Compiler-owned kernel unit that can produce an executable entry.
  LOOM_CMD_ENTRY_REQUIREMENT_SOURCE_KERNEL_UNIT = 0,
  // Bodyless configured entry declaration supplied by the host.
  LOOM_CMD_ENTRY_REQUIREMENT_SOURCE_DECLARATION = 1,
} loom_cmd_entry_requirement_source_t;

// One plan-wide atomic executable-entry binding requirement.
//
// A root-local slot resolves this entire row at once. The executable object
// and executable-local entry token may not be selected independently.
typedef struct loom_cmd_entry_requirement_t {
  // Active requirement source.
  loom_cmd_entry_requirement_source_t source;
  // Source-specific plan-owned contract.
  union {
    // Index into the containing plan's kernel-unit table.
    uint32_t kernel_unit_index;
    // Configured entry declaration owned by the plan's entry module.
    const loom_op_t* declaration_op;
  } contract;
} loom_cmd_entry_requirement_t;

// One prepared command root within a program plan.
//
// The lowered command function addresses root-local executable and entry slots.
// |entry_requirement_indices| maps each slot to one atomic plan-wide binding
// requirement. The launch function in the plan's shared host module evaluates
// this root's dynamic launch counts. All referenced artifacts remain valid
// after the source module is released.
typedef struct loom_cmd_program_root_t {
  // Lowered command root in the plan's shared root module.
  loom_op_t* function_op;

  // External resource-table shape carried through command lowering.
  loom_cmd_abi_layout_t abi_layout;

  // Host launch-count function in the plan's shared launch module.
  loom_op_t* launch_function_op;

  // Number of unique dynamic xyz tuples returned by |launch_function_op|.
  uint32_t launch_tuple_count;

  // Plan-wide body-backed kernel units used by this root in first-use order.
  uint32_t* dependency_unit_indices;

  // Number of entries in |dependency_unit_indices|.
  uint32_t dependency_count;

  // Plan-wide requirement for each root-local executable/entry slot.
  uint32_t* entry_requirement_indices;

  // Number of entries in |entry_requirement_indices|.
  uint32_t entry_requirement_count;

  // Concrete immutable parameter requirements and their fixed placement.
  loom_cmd_parameter_requirement_table_t parameters;

  // Aggregate issue-time storage required by command-program allocas.
  loom_cmd_transient_requirement_t transient;

  // Host-produced workgroup-count storage consumed by static dispatches.
  loom_cmd_program_launch_count_requirement_t launch_counts;
} loom_cmd_program_root_t;

// Immutable command roots and their union dependency graph.
//
// The root module contains every selected command symbol lowered to the
// portable cmd low ISA. The launch module contains one pure host function per
// selected root. Equivalent logical launch sites across all roots share one
// selectively linked and specialized kernel unit. Bodyless direct dispatches
// retain selectively linked configured entry declarations and never
// materialize a kernel body. No compilation or artifact emission occurs while
// preparing the plan.
typedef struct loom_cmd_program_plan_t {
  // Owned module containing all lowered command roots.
  loom_module_t* root_module;

  // Owned module containing all root launch-count functions.
  loom_module_t* launch_module;

  // Owned module containing bodyless configured entry declarations.
  loom_module_t* entry_module;

  // Selected roots in caller order.
  loom_cmd_program_root_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Unique independently owned body-backed kernel dependencies.
  loom_cmd_kernel_unit_t* dependency_units;

  // Number of entries in |dependency_units|.
  iree_host_size_t dependency_count;

  // Unique atomic executable-entry requirements in plan-wide order.
  loom_cmd_entry_requirement_t* entry_requirements;

  // Number of entries in |entry_requirements|.
  iree_host_size_t entry_requirement_count;

  // Host allocator used for all plan-owned host tables.
  iree_allocator_t host_allocator;
} loom_cmd_program_plan_t;

// Prepares command-program roots for independent compilation.
//
// |source_program_ops| must contain unique linked module-boundary
// command.program.def operations. Preparation selectively links their union
// dependency closure into one module, flattens command-program composition,
// resolves root-local control flow and explicit unroll policies, interns
// equivalent logical launch sites across roots into private dependency units,
// retains configured entry declarations for bodyless direct dispatches,
// materializes one launch-count program per root, assigns root-local atomic
// entry slots, and lowers every command root. Dependency kernel bodies retain
// their ordinary target-compilation path. The source module is unchanged and
// need not outlive the returned plan.
//
// |pass_registry| must provide the standard canonicalize and unroll-scf-for
// function passes used to resolve root-local source structure. It is a
// compiler-owned resource rather than part of the authored program contract.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, leave |out_plan| empty, and return OK. A valid plan sets |out_valid|
// to true and transfers all referenced modules to |out_plan|, which must be
// deinitialized by the caller.
iree_status_t loom_cmd_program_plan_prepare(
    const loom_module_t* source_module,
    const loom_op_t* const* source_program_ops,
    iree_host_size_t source_program_count,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid,
    loom_cmd_program_plan_t* out_plan, iree_allocator_t host_allocator);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
