// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_METADATA_OPS_H_
#define LOOM_OPS_METADATA_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_METADATA_MODULE = LOOM_OP_KIND(LOOM_DIALECT_METADATA, 0),
  LOOM_OP_METADATA_COUNT_ = 1,
};

// LOOM_OP_METADATA_MODULE: Attach one stable typed metadata value to the module. Keys are exact strings and unique within the module metadata scope. Values are ordinary Loom attributes so targets and tooling can project the subset their output contracts support.
// metadata.module "model.revision" = u64(3)
LOOM_DEFINE_ISA(loom_metadata_module_isa, LOOM_OP_METADATA_MODULE)
LOOM_DEFINE_ATTR_STRING(loom_metadata_module_key, 0)
LOOM_DEFINE_ATTR_ANY(loom_metadata_module_value, 1)
iree_status_t loom_metadata_module_build(
    loom_builder_t* builder,
    loom_string_id_t key,
    loom_attribute_t value,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the metadata dialect.
const loom_op_vtable_t* const* loom_metadata_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the metadata dialect.
const loom_op_semantics_t* loom_metadata_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a metadata op kind, or empty metadata.
loom_op_semantics_t loom_metadata_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_METADATA_OPS_H_
