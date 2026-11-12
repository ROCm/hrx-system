// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_TILE_OPS_H_
#define LOOM_OPS_TILE_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_TILE_REFINE = LOOM_OP_KIND(LOOM_DIALECT_TILE, 0),
  LOOM_OP_TILE_DUPLICATE = LOOM_OP_KIND(LOOM_DIALECT_TILE, 1),
  LOOM_OP_TILE_COUNT_ = 2,
};

// LOOM_OP_TILE_REFINE: Assert more precise shape and encoding information for the same tile elements. Source and result describe the same value and ownership identity; this operation performs no conversion or copy.
// %refined = tile.refine %source : tile<[%count]xf32> -> tile<16xf32>
LOOM_DEFINE_ISA(loom_tile_refine_isa, LOOM_OP_TILE_REFINE)
LOOM_DEFINE_OPERAND(loom_tile_refine_source, 0)
LOOM_DEFINE_RESULT(loom_tile_refine_result, 0)
iree_status_t loom_tile_refine_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_value_refinement_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_value_refinement_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_value_refinement_type_transfer(
    loom_type_transfer_context_t* context,
    const loom_module_t* module, loom_op_t* op);
iree_status_t loom_value_refinement_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TILE_DUPLICATE: Create another local owner of a tile and its dependent dimensions or encoding. The duplicate carries the same elements through a distinct SSA ownership identity.
// %copy_m, %copy_layout, %copy = tile.duplicate %m, %layout, %source : index, encoding<layout>, tile<[%copy_m]xf32, %copy_layout>
LOOM_DEFINE_ISA(loom_tile_duplicate_isa, LOOM_OP_TILE_DUPLICATE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_tile_duplicate_sources, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_tile_duplicate_duplicates, 0)
iree_status_t loom_tile_duplicate_build(
    loom_builder_t* builder,
    const loom_value_id_t* sources,
    iree_host_size_t sources_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_value_duplication_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Returns the vtable array for the tile dialect.
const loom_op_vtable_t* const* loom_tile_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the tile dialect.
const loom_op_semantics_t* loom_tile_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a tile op kind, or empty metadata.
loom_op_semantics_t loom_tile_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/tile/ops.inc"

#endif  // LOOM_OPS_TILE_OPS_H_
