// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_MODULE_OPS_H_
#define LOOM_OPS_MODULE_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_MODULE_IMPORT = LOOM_OP_KIND(LOOM_DIALECT_MODULE, 0),
  LOOM_OP_MODULE_COUNT_ = 1,
};

// LOOM_OP_MODULE_IMPORT: Name a source provider that may define the listed module symbols. The provider is an opaque resolver-defined key, and the symbol list records availability without creating dependency or liveness edges. Provider identity and import order never participate in template provider matching. Template families are declared locally by every using module rather than sourced through imports. This metadata is consumed only by compile-time linking.
// module.import "motif/format/ggml.loom" [@decode_q4, @decode_q6]
LOOM_DEFINE_ISA(loom_module_import_isa, LOOM_OP_MODULE_IMPORT)
LOOM_DEFINE_ATTR_STRING(loom_module_import_provider, 0)
LOOM_DEFINE_ATTR_SYMBOL_SET(loom_module_import_symbols, 1)
iree_status_t loom_module_import_build(
    loom_builder_t* builder,
    loom_string_id_t provider,
    loom_symbol_ref_array_t symbols,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the module dialect.
const loom_op_vtable_t* const* loom_module_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the module dialect.
const loom_op_semantics_t* loom_module_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a module op kind, or empty metadata.
loom_op_semantics_t loom_module_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_MODULE_OPS_H_
