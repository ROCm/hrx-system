// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Task entry parameter packing and reflected logical interfaces.

#ifndef LOOM_TARGET_ABI_TASK_PARAMETER_LAYOUT_H_
#define LOOM_TARGET_ABI_TASK_PARAMETER_LAYOUT_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/task/executable/library/abi.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_task_parameter_layout_t {
  // Dispatch requirements represented by the versioned executable schema.
  iree_hal_executable_dispatch_attrs_v0_t attributes;
  // Module-owned logical parameter types in declaration order.
  const loom_type_t* parameter_types;
  // Arena-owned parameter records in logical declaration order.
  const iree_hal_executable_dispatch_parameter_v0_t* parameters;
} loom_task_parameter_layout_t;

// Parses the logical kernel interface independently of its physical signature.
// `signature` supplies source types and `offsets` supplies one binding ordinal
// or constant byte offset per parameter. Unused parameters retain their slots.
// Both entry materialization and native library emission consume this contract.
// Invalid authored layouts emit a source diagnostic and leave |out_accepted|
// false. Only allocation and diagnostic delivery failures return a status.
iree_status_t loom_task_parameter_layout_parse(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_named_attr_slice_t layout, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_task_parameter_layout_t* out_abi);

// Builds the canonical task parameter layout from a logical signature.
// Bindings are dense in declaration order; constants are aligned to their
// scalar element size, capped at eight bytes. Index/offset values use 64 bits;
// boolean lanes occupy one byte. This contract is independent of register ABI.
// The returned dictionary and its signature/offsets are module-owned.
// Unsupported parameter types or extents emit a diagnostic at |source_op|
// and leave |out_accepted| false and |out_layout| empty.
iree_status_t loom_task_parameter_layout_build(
    loom_module_t* module, const loom_op_t* source_op, const loom_type_t* types,
    uint16_t count, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* scratch_arena, bool* out_accepted,
    loom_named_attr_slice_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ABI_TASK_PARAMETER_LAYOUT_H_
