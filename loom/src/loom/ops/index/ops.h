// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_INDEX_OPS_H_
#define LOOM_OPS_INDEX_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_INDEX_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 0),
  LOOM_OP_INDEX_CAST = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 1),
  LOOM_OP_INDEX_ASSUME = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 2),
  LOOM_OP_INDEX_ADD = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 3),
  LOOM_OP_INDEX_SUB = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 4),
  LOOM_OP_INDEX_MUL = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 5),
  LOOM_OP_INDEX_SCALE = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 6),
  LOOM_OP_INDEX_DIV = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 7),
  LOOM_OP_INDEX_REM = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 8),
  LOOM_OP_INDEX_MIN = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 9),
  LOOM_OP_INDEX_MAX = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 10),
  LOOM_OP_INDEX_MADD = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 11),
  LOOM_OP_INDEX_ANDI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 12),
  LOOM_OP_INDEX_ORI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 13),
  LOOM_OP_INDEX_XORI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 14),
  LOOM_OP_INDEX_SHLI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 15),
  LOOM_OP_INDEX_SHRSI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 16),
  LOOM_OP_INDEX_SHRUI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 17),
  LOOM_OP_INDEX_ROTLI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 18),
  LOOM_OP_INDEX_ROTRI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 19),
  LOOM_OP_INDEX_CTLZI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 20),
  LOOM_OP_INDEX_CTTZI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 21),
  LOOM_OP_INDEX_CTPOPI = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 22),
  LOOM_OP_INDEX_CMP = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 23),
  LOOM_OP_INDEX_COUNT_ = 24,
};

// Address-domain comparison predicates.
typedef enum loom_index_cmp_predicate_e {
  LOOM_INDEX_CMP_PREDICATE_EQ = 0,
  LOOM_INDEX_CMP_PREDICATE_NE = 1,
  LOOM_INDEX_CMP_PREDICATE_SLT = 2,
  LOOM_INDEX_CMP_PREDICATE_SLE = 3,
  LOOM_INDEX_CMP_PREDICATE_SGT = 4,
  LOOM_INDEX_CMP_PREDICATE_SGE = 5,
  LOOM_INDEX_CMP_PREDICATE_ULT = 6,
  LOOM_INDEX_CMP_PREDICATE_ULE = 7,
  LOOM_INDEX_CMP_PREDICATE_UGT = 8,
  LOOM_INDEX_CMP_PREDICATE_UGE = 9,
  LOOM_INDEX_CMP_PREDICATE_COUNT_ = 10,
} loom_index_cmp_predicate_t;

// LOOM_OP_INDEX_CONSTANT: Materialize a compile-time address-domain constant. The result type must be index for logical coordinates or offset for physical byte counts; fixed-width integer payload constants remain scalar.constant.
// %c0 = index.constant 0 : index
LOOM_DEFINE_ISA(loom_index_constant_isa, LOOM_OP_INDEX_CONSTANT)
LOOM_DEFINE_RESULT(loom_index_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_index_constant_value, 0)
iree_status_t loom_index_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CAST: Explicit conversion at an address boundary. At least one side must be index or offset; pure integer width changes use scalar.extsi, scalar.extui, or scalar.trunci.
// %i = index.cast %n : i64 to index
LOOM_DEFINE_ISA(loom_index_cast_isa, LOOM_OP_INDEX_CAST)
LOOM_DEFINE_OPERAND(loom_index_cast_input, 0)
LOOM_DEFINE_RESULT(loom_index_cast_result, 0)
iree_status_t loom_index_cast_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_cast_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_cast_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_index_cast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_INDEX_ASSUME: Identity with predicate constraints on index or offset results.
// %n2 = index.assume %n [mul(%n, 16)] : index
LOOM_DEFINE_ISA(loom_index_assume_isa, LOOM_OP_INDEX_ASSUME)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_index_assume_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_index_assume_results, 0)
iree_status_t loom_index_assume_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_assume_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_index_assume_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_INDEX_ADD: Address-domain addition. Operands and result must all be index or all offset.
// %r = index.add %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_add_isa, LOOM_OP_INDEX_ADD)
LOOM_DEFINE_OPERAND(loom_index_add_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_add_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_add_result, 0)
iree_status_t loom_index_add_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_add_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_add_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SUB: Address-domain subtraction. Operands and result must all be index or all offset.
// %r = index.sub %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_sub_isa, LOOM_OP_INDEX_SUB)
LOOM_DEFINE_OPERAND(loom_index_sub_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_sub_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_sub_result, 0)
iree_status_t loom_index_sub_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_sub_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_sub_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MUL: Logical coordinate multiplication. Offsets are physical byte counts and cannot be multiplied with this op.
// %r = index.mul %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_mul_isa, LOOM_OP_INDEX_MUL)
LOOM_DEFINE_OPERAND(loom_index_mul_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_mul_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_mul_result, 0)
iree_status_t loom_index_mul_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_mul_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_mul_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SCALE: Scale a logical coordinate by a physical byte stride to produce a physical byte offset. This is the address-domain boundary for packed payload layouts: logical indices stay index values, byte strides stay offset values, and the result feeds byte-addressed buffer/view bases.
// %bytes = index.scale %lane, %byte_stride : index, offset -> offset
LOOM_DEFINE_ISA(loom_index_scale_isa, LOOM_OP_INDEX_SCALE)
LOOM_DEFINE_OPERAND(loom_index_scale_index, 0)
LOOM_DEFINE_OPERAND(loom_index_scale_stride, 1)
LOOM_DEFINE_RESULT(loom_index_scale_result, 0)
iree_status_t loom_index_scale_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t index,
    loom_may_consume loom_value_id_t stride,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_scale_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_scale_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_DIV: Logical coordinate quotient for non-negative index values with a positive divisor. Offsets are physical byte counts and cannot be divided with this op; use an explicit layout or storage mapping before deriving physical address pieces.
// %q = index.div %lane, %group_size : index
LOOM_DEFINE_ISA(loom_index_div_isa, LOOM_OP_INDEX_DIV)
LOOM_DEFINE_OPERAND(loom_index_div_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_div_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_div_result, 0)
iree_status_t loom_index_div_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_div_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_div_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_REM: Logical coordinate remainder for non-negative index values with a positive divisor. Offsets are physical byte counts and cannot use remainder with this op; use an explicit layout or storage mapping before deriving physical address pieces.
// %r = index.rem %lane, %group_size : index
LOOM_DEFINE_ISA(loom_index_rem_isa, LOOM_OP_INDEX_REM)
LOOM_DEFINE_OPERAND(loom_index_rem_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_rem_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_rem_result, 0)
iree_status_t loom_index_rem_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_rem_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_rem_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MIN: Signed minimum of two logical coordinate values.
// %r = index.min %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_min_isa, LOOM_OP_INDEX_MIN)
LOOM_DEFINE_OPERAND(loom_index_min_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_min_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_min_result, 0)
iree_status_t loom_index_min_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_min_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_min_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MAX: Signed maximum of two logical coordinate values.
// %r = index.max %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_max_isa, LOOM_OP_INDEX_MAX)
LOOM_DEFINE_OPERAND(loom_index_max_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_max_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_max_result, 0)
iree_status_t loom_index_max_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_max_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_max_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MADD: Logical coordinate multiply-add: a*b + c. Offsets are physical byte counts and cannot be multiplied with this op.
// %r = index.madd %a, %b, %c : index
LOOM_DEFINE_ISA(loom_index_madd_isa, LOOM_OP_INDEX_MADD)
LOOM_DEFINE_OPERAND(loom_index_madd_a, 0)
LOOM_DEFINE_OPERAND(loom_index_madd_b, 1)
LOOM_DEFINE_OPERAND(loom_index_madd_c, 2)
LOOM_DEFINE_RESULT(loom_index_madd_result, 0)
iree_status_t loom_index_madd_build(
    loom_builder_t* builder,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_madd_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_madd_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_ANDI: Bitwise AND over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.andi %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_andi_isa, LOOM_OP_INDEX_ANDI)
LOOM_DEFINE_OPERAND(loom_index_andi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_andi_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_andi_result, 0)
iree_status_t loom_index_andi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_andi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_andi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_ORI: Bitwise OR over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.ori %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_ori_isa, LOOM_OP_INDEX_ORI)
LOOM_DEFINE_OPERAND(loom_index_ori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_ori_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_ori_result, 0)
iree_status_t loom_index_ori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_ori_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_ori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_XORI: Bitwise XOR over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.xori %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_xori_isa, LOOM_OP_INDEX_XORI)
LOOM_DEFINE_OPERAND(loom_index_xori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_xori_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_xori_result, 0)
iree_status_t loom_index_xori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_xori_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_xori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SHLI: Left shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.shli %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_shli_isa, LOOM_OP_INDEX_SHLI)
LOOM_DEFINE_OPERAND(loom_index_shli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_shli_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_shli_result, 0)
iree_status_t loom_index_shli_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_shli_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_shli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SHRSI: Arithmetic right shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.shrsi %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_shrsi_isa, LOOM_OP_INDEX_SHRSI)
LOOM_DEFINE_OPERAND(loom_index_shrsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_shrsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_shrsi_result, 0)
iree_status_t loom_index_shrsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_shrsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_shrsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SHRUI: Logical right shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.shrui %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_shrui_isa, LOOM_OP_INDEX_SHRUI)
LOOM_DEFINE_OPERAND(loom_index_shrui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_shrui_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_shrui_result, 0)
iree_status_t loom_index_shrui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_shrui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_shrui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_ROTLI: Left rotate over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.rotli %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_rotli_isa, LOOM_OP_INDEX_ROTLI)
LOOM_DEFINE_OPERAND(loom_index_rotli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_rotli_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_rotli_result, 0)
iree_status_t loom_index_rotli_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_rotli_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_rotli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_ROTRI: Right rotate over logical coordinate values. Offsets are physical byte counts and cannot use this op.
// %r = index.rotri %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_rotri_isa, LOOM_OP_INDEX_ROTRI)
LOOM_DEFINE_OPERAND(loom_index_rotri_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_rotri_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_rotri_result, 0)
iree_status_t loom_index_rotri_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_rotri_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_rotri_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CTLZI: Count leading zeros in a logical coordinate value.
// %r = index.ctlzi %input : index
LOOM_DEFINE_ISA(loom_index_ctlzi_isa, LOOM_OP_INDEX_CTLZI)
LOOM_DEFINE_OPERAND(loom_index_ctlzi_input, 0)
LOOM_DEFINE_RESULT(loom_index_ctlzi_result, 0)
iree_status_t loom_index_ctlzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_ctlzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CTTZI: Count trailing zeros in a logical coordinate value.
// %r = index.cttzi %input : index
LOOM_DEFINE_ISA(loom_index_cttzi_isa, LOOM_OP_INDEX_CTTZI)
LOOM_DEFINE_OPERAND(loom_index_cttzi_input, 0)
LOOM_DEFINE_RESULT(loom_index_cttzi_result, 0)
iree_status_t loom_index_cttzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_cttzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CTPOPI: Count set bits in a logical coordinate value.
// %r = index.ctpopi %input : index
LOOM_DEFINE_ISA(loom_index_ctpopi_isa, LOOM_OP_INDEX_CTPOPI)
LOOM_DEFINE_OPERAND(loom_index_ctpopi_input, 0)
LOOM_DEFINE_RESULT(loom_index_ctpopi_result, 0)
iree_status_t loom_index_ctpopi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_ctpopi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CMP: Address-domain comparison. Operands must both be index or both be offset.
// %p = index.cmp slt, %i, %n : index
LOOM_DEFINE_ISA(loom_index_cmp_isa, LOOM_OP_INDEX_CMP)
LOOM_DEFINE_OPERAND(loom_index_cmp_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_cmp_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_cmp_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_index_cmp_predicate, 0, loom_index_cmp_predicate_t)
iree_status_t loom_index_cmp_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_cmp_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_cmp_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Returns the vtable array for the index dialect.
const loom_op_vtable_t* const* loom_index_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the index dialect.
const loom_op_semantics_t* loom_index_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a index op kind, or empty metadata.
loom_op_semantics_t loom_index_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_INDEX_OPS_H_
