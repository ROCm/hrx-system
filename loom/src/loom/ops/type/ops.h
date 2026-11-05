// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_TYPE_OPS_H_
#define LOOM_OPS_TYPE_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_TYPE_DECL = LOOM_OP_KIND(LOOM_DIALECT_TYPE, 0),
  LOOM_OP_TYPE_COUNT_ = 1,
};

// LOOM_OP_TYPE_DECL: Declare an external managed payload identity under a local name. The namespace and payload name identify the type independently of its local spelling. Providers supply payload behavior and ownership; the declaration exposes no layout, address, or implementation.
// type.decl import("input", "decoder") @Decoder
LOOM_DEFINE_ISA(loom_type_decl_isa, LOOM_OP_TYPE_DECL)
LOOM_DEFINE_ATTR_SYMBOL(loom_type_decl_name, 0)
LOOM_DEFINE_ATTR_STRING(loom_type_decl_namespace, 1)
LOOM_DEFINE_ATTR_STRING(loom_type_decl_type_name, 2)
iree_status_t loom_type_decl_build(
    loom_builder_t* builder,
    loom_string_id_t namespace_,
    loom_string_id_t type_name,
    loom_symbol_ref_t name,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_type_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the type dialect.
const loom_op_vtable_t* const* loom_type_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the type dialect.
const loom_op_semantics_t* loom_type_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a type op kind, or empty metadata.
loom_op_semantics_t loom_type_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/type/ops.inc"

#endif  // LOOM_OPS_TYPE_OPS_H_
