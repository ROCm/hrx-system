// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_GROUP_OPS_H_
#define LOOM_OPS_GROUP_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_GROUP_CREATE = LOOM_OP_KIND(LOOM_DIALECT_GROUP, 0),
  LOOM_OP_GROUP_COUNT_ = 1,
};

// LOOM_OP_GROUP_CREATE: Create a distinct scheduling group with SSA-defined cardinality. The cardinality may be specialized from workload or target queries; consumers use ordinary value facts when an exact count is required.
// %workers = group.create %worker_count : index -> group
LOOM_DEFINE_ISA(loom_group_create_isa, LOOM_OP_GROUP_CREATE)
LOOM_DEFINE_OPERAND(loom_group_create_cardinality, 0)
LOOM_DEFINE_RESULT(loom_group_create_result, 0)
iree_status_t loom_group_create_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t cardinality,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_group_create_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the group dialect.
const loom_op_vtable_t* const* loom_group_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the group dialect.
const loom_op_semantics_t* loom_group_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a group op kind, or empty metadata.
loom_op_semantics_t loom_group_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_GROUP_OPS_H_
