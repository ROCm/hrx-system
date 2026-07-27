// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_ENCODING_OPS_H_
#define LOOM_OPS_ENCODING_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_ENCODING_LAYOUT_DENSE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 0),
  LOOM_OP_ENCODING_LAYOUT_STRIDED = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 1),
  LOOM_OP_ENCODING_DEFINE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 2),
  LOOM_OP_ENCODING_ISA = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 3),
  LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 4),
  LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 5),
  LOOM_OP_ENCODING_ASSUME_SPEC = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 6),
  LOOM_OP_ENCODING_COUNT_ = 7,
};

// LOOM_OP_ENCODING_LAYOUT_DENSE: Construct a dense row-major address layout. The consuming view type provides the rank and logical extents.
// %layout = encoding.layout.dense : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_dense_isa, LOOM_OP_ENCODING_LAYOUT_DENSE)
LOOM_DEFINE_RESULT(loom_encoding_layout_dense_result, 0)
iree_status_t loom_encoding_layout_dense_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_dense_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_ENCODING_LAYOUT_STRIDED: Construct an address layout from per-dimension element strides. Static and dynamic stride values are interleaved in one bracket list.
// %layout = encoding.layout.strided [%row_stride, 1] : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_strided_isa, LOOM_OP_ENCODING_LAYOUT_STRIDED)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_encoding_layout_strided_strides, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_strided_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_encoding_layout_strided_static_strides, 0)
iree_status_t loom_encoding_layout_strided_build(
    loom_builder_t* builder,
    const loom_value_id_t* strides,
    iree_host_size_t strides_count,
    const int64_t* static_strides,
    iree_host_size_t static_strides_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_strided_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_layout_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_DEFINE: Create an encoding value from a static encoding specification.
// %enc = encoding.define #q8_0<block=32> : encoding<schema>
LOOM_DEFINE_ISA(loom_encoding_define_isa, LOOM_OP_ENCODING_DEFINE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_encoding_define_params, 0)
LOOM_DEFINE_RESULT(loom_encoding_define_result, 0)
LOOM_DEFINE_ATTR_ENCODING(loom_encoding_define_spec, 0)
LOOM_DEFINE_ATTR_DICT(loom_encoding_define_param_names, 1)
iree_status_t loom_encoding_define_build(
    loom_builder_t* builder,
    uint16_t spec,
    const loom_named_value_t* params,
    iree_host_size_t params_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_define_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_define_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_ISA: Test if an encoding belongs to a category.
// %is_quantized = encoding.isa %enc, "quantized" : i1
LOOM_DEFINE_ISA(loom_encoding_isa_isa, LOOM_OP_ENCODING_ISA)
LOOM_DEFINE_OPERAND(loom_encoding_isa_enc, 0)
LOOM_DEFINE_RESULT(loom_encoding_isa_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_encoding_isa_category, 0)
iree_status_t loom_encoding_isa_build(
    loom_builder_t* builder,
    loom_value_id_t enc,
    loom_string_id_t category,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE: Refine an existing address-layout encoding value with the fact that it is dense row-major. The result is the same encoding value in SSA form with stronger local facts.
// %dense = encoding.layout.assume.dense %layout : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_assume_dense_isa, LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE)
LOOM_DEFINE_OPERAND(loom_encoding_layout_assume_dense_layout, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_assume_dense_result, 0)
iree_status_t loom_encoding_layout_assume_dense_build(
    loom_builder_t* builder,
    loom_value_id_t layout,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_assume_dense_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED: Refine an existing address-layout encoding value with the fact that it is strided and has the given rank. Per-axis stride values remain unknown unless a concrete encoding.layout.strided value is available.
// %strided = encoding.layout.assume.strided %layout {rank = 2} : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_assume_strided_isa, LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED)
LOOM_DEFINE_OPERAND(loom_encoding_layout_assume_strided_layout, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_assume_strided_result, 0)
LOOM_DEFINE_ATTR_I64(loom_encoding_layout_assume_strided_rank, 0)
iree_status_t loom_encoding_layout_assume_strided_build(
    loom_builder_t* builder,
    loom_value_id_t layout,
    int64_t rank,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_assume_strided_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_layout_assume_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_ASSUME_SPEC: Refine an existing encoding value with an exact static encoding specification. Dynamic values remain ordinary SSA operands elsewhere; this op only states the selected static family and static parameters.
// %schema2 = encoding.assume.spec %schema, #ggml_q4_0<block_elems=32, storage_bytes=18> : encoding<schema>
LOOM_DEFINE_ISA(loom_encoding_assume_spec_isa, LOOM_OP_ENCODING_ASSUME_SPEC)
LOOM_DEFINE_OPERAND(loom_encoding_assume_spec_enc, 0)
LOOM_DEFINE_RESULT(loom_encoding_assume_spec_result, 0)
LOOM_DEFINE_ATTR_ENCODING(loom_encoding_assume_spec_spec, 0)
iree_status_t loom_encoding_assume_spec_build(
    loom_builder_t* builder,
    loom_value_id_t enc,
    uint16_t spec,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_assume_spec_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_assume_spec_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the encoding dialect.
const loom_op_vtable_t* const* loom_encoding_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the encoding dialect.
const loom_op_semantics_t* loom_encoding_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a encoding op kind, or empty metadata.
loom_op_semantics_t loom_encoding_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_OPS_H_
