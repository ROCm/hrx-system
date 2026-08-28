// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic Core VM constant, global, and read-only data layout planning.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_RESOURCES_H_
#define LOOM_TARGET_EMIT_VM_MODULE_RESOURCES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// One direct read-only data block in physical ordinal order.
typedef struct loom_vm_module_rodata_layout_t {
  // Borrowed immutable bytes owned by the source module.
  iree_const_byte_span_t contents;
  // Minimum power-of-two host-address alignment of |contents|.
  uint32_t minimum_alignment;
} loom_vm_module_rodata_layout_t;

// Complete physical resource layout for one emitted Core VM module.
typedef struct loom_vm_module_resource_layout_t {
  // Constant-pool cells indexed by physical pool ordinal.
  iree_vm_bytecode_v0_constant_cell_t* constant_cells;
  // Number of entries in |constant_cells|.
  uint32_t constant_count;
  // Total value-global count.
  uint32_t value_global_count;
  // Dense immutable value-global prefix length.
  uint32_t immutable_value_global_count;
  // Total ref-global count.
  uint32_t ref_global_count;
  // Dense immutable ref-global prefix length.
  uint32_t immutable_ref_global_count;
  // Logical ref types indexed by physical ref-global ordinal.
  loom_type_t* ref_global_types;
  // Wire descriptors indexed by physical ref-global ordinal.
  iree_vm_bytecode_v0_global_ref_descriptor_row_t* ref_global_descriptors;
  // Total function-global count.
  uint32_t function_global_count;
  // Dense immutable function-global prefix length.
  uint32_t immutable_function_global_count;
  // Logical function types indexed by physical function-global ordinal.
  loom_type_t* function_global_types;
  // Wire descriptors indexed by physical function-global ordinal.
  iree_vm_bytecode_v0_global_function_descriptor_row_t*
      function_global_descriptors;
  // Read-only data blocks indexed by physical rodata ordinal.
  loom_vm_module_rodata_layout_t* rodata;
  // Number of entries in |rodata|.
  uint32_t rodata_count;
  // Maximum rodata block alignment, including the section minimum.
  uint32_t rodata_section_alignment;
} loom_vm_module_resource_layout_t;

// Collects and validates physical VM resource records in |module|.
//
// Constant, global, and rodata ordinals must be dense and unique in their
// independent physical domains. Immutable globals must occupy the complete
// prefix of each global domain. Returned arrays are owned by |arena|;
// referenced contents continue to borrow source module storage.
iree_status_t loom_vm_module_resource_layout_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_vm_module_resource_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_RESOURCES_H_
