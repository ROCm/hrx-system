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
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One concrete immutable parameter required by a command-program root.
typedef struct loom_cmd_parameter_requirement_t {
  // Fully substituted parameter key owned by the containing table.
  iree_string_view_t key;

  // Source command-program launch-binding ordinal selecting the root group.
  uint16_t source_binding_ordinal;

  // Dense fixed-buffer table index assigned to the source root.
  uint32_t fixed_buffer_index;

  // Byte offset of the parameter relative to the fixed-buffer range.
  uint64_t byte_offset;

  // Exact byte length of the parameter payload.
  uint64_t byte_length;

  // Minimum required alignment of the placed parameter payload.
  uint64_t minimum_alignment;
} loom_cmd_parameter_requirement_t;

// Aggregate requirement for one fixed parameter-buffer root.
typedef struct loom_cmd_parameter_root_requirement_t {
  // Source command-program launch-binding ordinal selecting this root.
  uint16_t source_binding_ordinal;

  // Dense fixed-buffer table index assigned to this root.
  uint32_t fixed_buffer_index;

  // Minimum byte length required to contain every assigned parameter.
  uint64_t required_byte_length;

  // Minimum required alignment of the supplied fixed-buffer range.
  uint64_t minimum_alignment;
} loom_cmd_parameter_root_requirement_t;

// Owned parameter requirements and their canonical default placement.
//
// Entries retain no source IR identity. Keys and tables remain valid until
// deinitialization even after the source module has been released.
typedef struct loom_cmd_parameter_requirement_table_t {
  // Fixed roots in fixed-buffer table order.
  loom_cmd_parameter_root_requirement_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Concrete parameters in source traversal order.
  loom_cmd_parameter_requirement_t* entries;

  // Number of entries in |entries|.
  iree_host_size_t count;

  // Owned backing storage referenced by |entries[*].key|.
  char* key_storage;
} loom_cmd_parameter_requirement_table_t;

// Scratch-owned placement facts consumed by command low conversion.
typedef struct loom_cmd_parameter_layout_t {
  // Derived parameter result values mapped to placed fixed-buffer ranges.
  const loom_cmd_buffer_range_t* buffer_ranges;

  // Number of entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;

  // Number of dense fixed-buffer ABI resources.
  uint32_t fixed_buffer_count;

  // Number of dense issue-time launch-binding ABI resources.
  uint32_t rebindable_binding_count;
} loom_cmd_parameter_layout_t;

// Enumerates concrete parameter references and assigns command buffer roles.
//
// Each reachable command.parameter result must have an exact byte footprint
// and substitutions evaluable to nonnegative indices. Parameter source roots
// become fixed resources; all other launch bindings remain rebindable.
// Parameters are packed independently within each source root in source
// traversal order using a canonical 256-byte minimum alignment.
//
// Exact ordinary view results rooted in launch bindings are preserved in the
// same lower plan, so fixed/rebindable roles and subranges propagate without
// per-launch rediscovery. |bindings| is populated in source launch-binding
// order. |out_requirements| references |storage_arena| and remains valid until
// that arena is reset. |out_layout| references only |scratch_arena| storage and
// remains valid until that arena is reset.
iree_status_t loom_cmd_parameter_layout_build(
    const loom_module_t* module, loom_func_like_t program,
    const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* storage_arena, loom_cmd_buffer_binding_t* bindings,
    iree_host_size_t binding_count,
    loom_cmd_parameter_requirement_table_t* out_requirements,
    loom_cmd_parameter_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PARAMETERS_H_
