// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-program parameter enumeration and fixed-root placement.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PARAMETERS_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PARAMETERS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/runtime/command/program.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned parameter requirements and their canonical default placement.
//
// Entries retain no source IR identity. Keys and tables remain valid until
// deinitialization even after the source module has been released.
typedef struct loom_cmd_parameter_requirement_table_t {
  // Fixed roots in fixed-buffer table order.
  loom_cmd_program_parameter_root_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Unique root/key parameters in first source occurrence order.
  loom_cmd_program_parameter_t* entries;

  // Number of entries in |entries|.
  iree_host_size_t count;

  // Owned backing storage referenced by |entries[*].key|.
  char* key_storage;

  // Exact concatenated byte length of the canonical parameter keys.
  iree_host_size_t key_storage_length;
} loom_cmd_parameter_requirement_table_t;

// Scratch-owned placement facts consumed by command low conversion.
typedef struct loom_cmd_parameter_layout_t {
  // Derived parameter result values mapped to placed fixed-buffer ranges.
  const loom_cmd_buffer_range_t* buffer_ranges;

  // Number of entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;

  // Number of dense fixed-buffer ABI resources.
  uint32_t fixed_buffer_count;

  // Number of dense issue-time storage-binding ABI resources.
  uint32_t rebindable_binding_count;
} loom_cmd_parameter_layout_t;

// Enumerates concrete parameter references and assigns command buffer roles.
//
// Each reachable command.parameter result must have an exact byte footprint
// and substitutions evaluable to nonnegative indices. Parameter source roots
// become fixed resources; all other launch bindings remain rebindable.
// References to the same concrete key within one root share placement with
// the maximum requested extent and alignment. Parameters are packed
// independently within each source root in first source occurrence order
// using a canonical 256-byte minimum alignment.
//
// Exact view results rooted in storage bindings or parameter results are
// preserved in the same lower plan. Parameter-relative offsets compose with
// the parameter's placement; fixed/rebindable roles and subranges propagate
// without per-launch rediscovery. command.parameter.base borrows the suffix
// beginning at its named parameter; its open-ended range does not enlarge
// that parameter root's finite packed requirement. |bindings| is populated in
// source storage-binding order. |out_requirements| owns its tables through
// |host_allocator|. |out_layout| references only |scratch_arena| storage and
// remains valid until that arena is reset. The module's value-ordinal scratch
// is acquired temporarily; no source IR is changed. An unresolved parameter
// base emits a diagnostic and leaves |out_valid| false without a layout.
iree_status_t loom_cmd_parameter_layout_build(
    loom_module_t* module, loom_func_like_t program,
    const loom_value_fact_table_t* fact_table,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t host_allocator,
    loom_cmd_buffer_binding_t* bindings, iree_host_size_t binding_count,
    loom_cmd_parameter_requirement_table_t* out_requirements,
    loom_cmd_parameter_layout_t* out_layout, bool* out_valid);

// Releases all storage owned by |table| and resets it to empty.
void loom_cmd_parameter_requirement_table_deinitialize(
    loom_cmd_parameter_requirement_table_t* table,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PARAMETERS_H_
