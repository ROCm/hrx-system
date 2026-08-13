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
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One root-local executable entry requirement.
typedef struct loom_cmd_program_entry_t {
  // Root-local executable requirement providing this entry.
  uint32_t executable_index;

  // Unit-local named export ordinal in the executable's compilation unit.
  uint32_t unit_export_index;
} loom_cmd_program_entry_t;

// One prepared command root within a program plan.
//
// The lowered command function addresses the plan-wide dependency table. Its
// launch function in the plan's shared host module evaluates this root's
// dynamic launch counts. Both remain valid after the source module is released.
typedef struct loom_cmd_program_root_t {
  // Lowered command root in the plan's shared root module.
  loom_op_t* function_op;

  // External resource-table shape carried through command lowering.
  loom_cmd_abi_layout_t abi_layout;

  // Host launch-config function in the plan's shared launch module.
  loom_op_t* launch_function_op;

  // Number of unique dynamic xyz tuples stored by |launch_function_op|.
  uint32_t launch_tuple_count;

  // Plan-wide dependency unit index for each root-local executable slot.
  uint32_t* executable_unit_indices;

  // Number of entries in |executable_unit_indices|.
  uint32_t executable_count;

  // Root-local entry requirements in portable command entry-table order.
  loom_cmd_program_entry_t* entries;

  // Number of entries in |entries|.
  uint32_t entry_count;

  // Concrete immutable parameter requirements and their fixed placement.
  loom_cmd_parameter_requirement_table_t parameters;

  // Aggregate issue-time storage required by command-program allocas.
  loom_cmd_transient_requirement_t transient;
} loom_cmd_program_root_t;

// Immutable command roots and their union dependency graph.
//
// The root module contains every selected command symbol lowered to the
// portable cmd low ISA. The launch module contains one host config function per
// selected root. Equivalent dependency launch sites across all roots share one
// selectively linked and specialized kernel unit. No compilation or artifact
// emission occurs while preparing the plan.
typedef struct loom_cmd_program_plan_t {
  // Owned module containing all lowered command roots.
  loom_module_t* root_module;

  // Owned module containing all root launch-config functions.
  loom_module_t* launch_module;

  // Selected roots in caller order.
  loom_cmd_program_root_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Unique independently owned multi-export kernel compilation units.
  loom_cmd_kernel_unit_t* dependency_units;

  // Number of entries in |dependency_units|.
  iree_host_size_t dependency_count;

  // Arena owning immutable plan metadata from the caller's shared block pool.
  iree_arena_allocator_t arena;

  // Host allocator used for owned module storage.
  iree_allocator_t host_allocator;
} loom_cmd_program_plan_t;

// Prepares command-program roots for independent compilation.
//
// |source_program_ops| must contain unique linked module-boundary
// command.program.def operations. Preparation selectively links their union
// dependency closure into one module, flattens command-program composition,
// resolves root-local control flow and explicit unroll policies, interns
// equivalent launch sites across roots into private dependency units,
// materializes one launch-config function per root, assigns plan-wide dense
// dependency slots, and lowers every command root. Dependency kernel bodies
// retain their ordinary target-compilation path. The source module is unchanged
// and need not outlive the returned plan.
//
// |pass_registry| must provide the standard canonicalize and unroll-scf-for
// function passes used to resolve root-local source structure. It is a
// compiler-owned resource rather than part of the authored program contract.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, leave |out_plan| empty, and return OK. A valid plan sets |out_valid|
// to true and transfers all referenced modules to |out_plan|, which must be
// deinitialized by the caller. |block_pool| must outlive the returned plan.
iree_status_t loom_cmd_program_plan_prepare(
    const loom_module_t* source_module,
    const loom_op_t* const* source_program_ops,
    iree_host_size_t source_program_count,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid,
    loom_cmd_program_plan_t* out_plan, iree_allocator_t host_allocator);

// Returns the complete external resource requirements fixed for |root|.
//
// The result is the authoritative contract shared by serialization and unit
// loading. It contains no references to plan storage.
loom_cmd_program_requirements_t loom_cmd_program_root_requirements(
    const loom_cmd_program_root_t* root);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
