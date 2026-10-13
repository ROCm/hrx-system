// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_HAL_OPS_H_
#define LOOM_OPS_HAL_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_HAL_QUEUE_COPY = LOOM_OP_KIND(LOOM_DIALECT_HAL, 0),
  LOOM_OP_HAL_COUNT_ = 1,
};

// LOOM_OP_HAL_QUEUE_COPY: Submits a buffer copy with explicit timeline dependencies and visibility boundaries. The HAL captures descriptors during the call; buffer and backing lifetimes follow the selected HAL flags. Absent before/after barriers preserve conservative defaults; empty lists request no extra payload barriers. Transition recipes contain qualified HAL actions.
// hal.queue.copy (%queue, [(%ready, %value)], [(%done, %value)], (%source, %source_offset), (%target, %target_offset), %length, (none, []), %flags)
LOOM_DEFINE_ISA(loom_hal_queue_copy_isa, LOOM_OP_HAL_QUEUE_COPY)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_hal_queue_copy_arguments, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_hal_queue_copy_shape, 0)
iree_status_t loom_hal_queue_copy_build(
    loom_builder_t* builder,
    const int64_t* shape,
    iree_host_size_t shape_count,
    const loom_value_id_t* arguments,
    iree_host_size_t arguments_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_hal_method_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the hal dialect.
const loom_op_vtable_t* const* loom_hal_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the hal dialect.
const loom_op_semantics_t* loom_hal_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a hal op kind, or empty metadata.
loom_op_semantics_t loom_hal_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/hal/ops.inc"

#endif  // LOOM_OPS_HAL_OPS_H_
