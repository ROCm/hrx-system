// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_TEST_OPS_H_
#define LOOM_OPS_TEST_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_TEST_ADDI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 0),
  LOOM_OP_TEST_NEG = LOOM_OP_KIND(LOOM_DIALECT_TEST, 1),
  LOOM_OP_TEST_CAST = LOOM_OP_KIND(LOOM_DIALECT_TEST, 2),
  LOOM_OP_TEST_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 3),
  LOOM_OP_TEST_USE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 4),
  LOOM_OP_TEST_CONVERGENT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 5),
  LOOM_OP_TEST_CMP = LOOM_OP_KIND(LOOM_DIALECT_TEST, 6),
  LOOM_OP_TEST_MAP = LOOM_OP_KIND(LOOM_DIALECT_TEST, 7),
  LOOM_OP_TEST_UPDATE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 8),
  LOOM_OP_TEST_INVOKE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 9),
  LOOM_OP_TEST_LOW_CALL = LOOM_OP_KIND(LOOM_DIALECT_TEST, 10),
  LOOM_OP_TEST_LOW_INVOKE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 11),
  LOOM_OP_TEST_SLICE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 12),
  LOOM_OP_TEST_LOOP = LOOM_OP_KIND(LOOM_DIALECT_TEST, 13),
  LOOM_OP_TEST_BLOCK_ARGS = LOOM_OP_KIND(LOOM_DIALECT_TEST, 14),
  LOOM_OP_TEST_BRANCH = LOOM_OP_KIND(LOOM_DIALECT_TEST, 15),
  LOOM_OP_TEST_OPTIONAL_REGION = LOOM_OP_KIND(LOOM_DIALECT_TEST, 16),
  LOOM_OP_TEST_IMPLICIT_YIELD = LOOM_OP_KIND(LOOM_DIALECT_TEST, 17),
  LOOM_OP_TEST_YIELD = LOOM_OP_KIND(LOOM_DIALECT_TEST, 18),
  LOOM_OP_TEST_BR = LOOM_OP_KIND(LOOM_DIALECT_TEST, 19),
  LOOM_OP_TEST_FUNC = LOOM_OP_KIND(LOOM_DIALECT_TEST, 20),
  LOOM_OP_TEST_SPLIT_FUNC = LOOM_OP_KIND(LOOM_DIALECT_TEST, 21),
  LOOM_OP_TEST_DECL = LOOM_OP_KIND(LOOM_DIALECT_TEST, 22),
  LOOM_OP_TEST_RECORD = LOOM_OP_KIND(LOOM_DIALECT_TEST, 23),
  LOOM_OP_TEST_ATTRS = LOOM_OP_KIND(LOOM_DIALECT_TEST, 24),
  LOOM_OP_TEST_OPERAND_DICT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 25),
  LOOM_OP_TEST_ATTR_TABLE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 26),
  LOOM_OP_TEST_REGION_TABLE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 27),
  LOOM_OP_TEST_DEFLATE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 28),
  LOOM_OP_TEST_ASSUME = LOOM_OP_KIND(LOOM_DIALECT_TEST, 29),
  LOOM_OP_TEST_CONVERT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 30),
  LOOM_OP_TEST_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 31),
  LOOM_OP_TEST_READ_RESOURCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 32),
  LOOM_OP_TEST_WRITE_RESOURCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 33),
  LOOM_OP_TEST_MUTATE_RESOURCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 34),
  LOOM_OP_TEST_ALLOC = LOOM_OP_KIND(LOOM_DIALECT_TEST, 35),
  LOOM_OP_TEST_ISOLATED_REGION = LOOM_OP_KIND(LOOM_DIALECT_TEST, 36),
  LOOM_OP_TEST_COUNTER = LOOM_OP_KIND(LOOM_DIALECT_TEST, 37),
  LOOM_OP_TEST_DIM = LOOM_OP_KIND(LOOM_DIALECT_TEST, 38),
  LOOM_OP_TEST_FACT_RANGE_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 39),
  LOOM_OP_TEST_FACT_RANGE_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 40),
  LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 41),
  LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 42),
  LOOM_OP_TEST_FACT_DIVISOR = LOOM_OP_KIND(LOOM_DIALECT_TEST, 43),
  LOOM_OP_TEST_FACT_NON_NEGATIVE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 44),
  LOOM_OP_TEST_FACT_NON_ZERO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 45),
  LOOM_OP_TEST_FACT_POSITIVE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 46),
  LOOM_OP_TEST_FACT_POWER_OF_TWO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 47),
  LOOM_OP_TEST_FACT_UNIFORM = LOOM_OP_KIND(LOOM_DIALECT_TEST, 48),
  LOOM_OP_TEST_FACT_LANE_VARYING = LOOM_OP_KIND(LOOM_DIALECT_TEST, 49),
  LOOM_OP_TEST_FACT_LANE_PREDICATE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 50),
  LOOM_OP_TEST_FACT_SUBGROUP_LANE_MASK = LOOM_OP_KIND(LOOM_DIALECT_TEST, 51),
  LOOM_OP_TEST_FACT_IS_VECTOR_IOTA = LOOM_OP_KIND(LOOM_DIALECT_TEST, 52),
  LOOM_OP_TEST_FACT_IS_VECTOR_PREFIX_MASK = LOOM_OP_KIND(LOOM_DIALECT_TEST, 53),
  LOOM_OP_TEST_FACT_ENCODING_LAYOUT_KIND = LOOM_OP_KIND(LOOM_DIALECT_TEST, 54),
  LOOM_OP_TEST_FACT_ENCODING_LAYOUT_STRIDE_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 55),
  LOOM_OP_TEST_FACT_ENCODING_MATRIX_FIELD = LOOM_OP_KIND(LOOM_DIALECT_TEST, 56),
  LOOM_OP_TEST_FACT_IS_BUFFER_REFERENCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 57),
  LOOM_OP_TEST_FACT_IS_VIEW_REFERENCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 58),
  LOOM_OP_TEST_FACT_BUFFER_MEMORY_SPACE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 59),
  LOOM_OP_TEST_FACT_VIEW_MEMORY_SPACE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 60),
  LOOM_OP_TEST_FACT_VIEW_ROOT_MATCHES = LOOM_OP_KIND(LOOM_DIALECT_TEST, 61),
  LOOM_OP_TEST_FACT_ALIAS_SCOPE_KNOWN = LOOM_OP_KIND(LOOM_DIALECT_TEST, 62),
  LOOM_OP_TEST_FACT_ALIAS_SCOPE_MATCHES = LOOM_OP_KIND(LOOM_DIALECT_TEST, 63),
  LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 64),
  LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 65),
  LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 66),
  LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 67),
  LOOM_OP_TEST_FACT_VIEW_MIN_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 68),
  LOOM_OP_TEST_FACT_BUFFER_MIN_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 69),
  LOOM_OP_TEST_FACT_VIEW_ROOT_MIN_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 70),
  LOOM_OP_TEST_FACT_VIEW_ELEMENT_BYTES = LOOM_OP_KIND(LOOM_DIALECT_TEST, 71),
  LOOM_OP_TEST_FACT_IS_STORAGE_REFERENCE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 72),
  LOOM_OP_TEST_FACT_STORAGE_SAME_BACKING = LOOM_OP_KIND(LOOM_DIALECT_TEST, 73),
  LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 74),
  LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 75),
  LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_DIVISOR = LOOM_OP_KIND(LOOM_DIALECT_TEST, 76),
  LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_LO = LOOM_OP_KIND(LOOM_DIALECT_TEST, 77),
  LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_HI = LOOM_OP_KIND(LOOM_DIALECT_TEST, 78),
  LOOM_OP_TEST_FACT_STORAGE_MIN_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 79),
  LOOM_OP_TEST_FACT_STORAGE_SPACE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 80),
  LOOM_OP_TEST_REGION_SYNTAX = LOOM_OP_KIND(LOOM_DIALECT_TEST, 81),
  LOOM_OP_TEST_LOW_ASM_REGION = LOOM_OP_KIND(LOOM_DIALECT_TEST, 82),
  LOOM_OP_TEST_CLAUSE_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_TEST, 83),
  LOOM_OP_TEST_CLAUSE_COPY = LOOM_OP_KIND(LOOM_DIALECT_TEST, 84),
  LOOM_OP_TEST_TYPED_USE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 85),
  LOOM_OP_TEST_SHAPE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 86),
  LOOM_OP_TEST_TARGET = LOOM_OP_KIND(LOOM_DIALECT_TEST, 87),
  LOOM_OP_TEST_RESOURCE_ALLOC = LOOM_OP_KIND(LOOM_DIALECT_TEST, 88),
  LOOM_OP_TEST_RESOURCE_BORROW = LOOM_OP_KIND(LOOM_DIALECT_TEST, 89),
  LOOM_OP_TEST_RESOURCE_BORROW_REF = LOOM_OP_KIND(LOOM_DIALECT_TEST, 90),
  LOOM_OP_TEST_RESOURCE_CONSUME = LOOM_OP_KIND(LOOM_DIALECT_TEST, 91),
  LOOM_OP_TEST_RESOURCE_RETAIN = LOOM_OP_KIND(LOOM_DIALECT_TEST, 92),
  LOOM_OP_TEST_RESOURCE_RELEASE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 93),
  LOOM_OP_TEST_RESOURCE_DISCARD = LOOM_OP_KIND(LOOM_DIALECT_TEST, 94),
  LOOM_OP_TEST_RESOURCE_ESCAPE = LOOM_OP_KIND(LOOM_DIALECT_TEST, 95),
  LOOM_OP_TEST_RESOURCE_ALIAS = LOOM_OP_KIND(LOOM_DIALECT_TEST, 96),
  LOOM_OP_TEST_RESOURCE_BORROWED = LOOM_OP_KIND(LOOM_DIALECT_TEST, 97),
  LOOM_OP_TEST_SEGMENTED = LOOM_OP_KIND(LOOM_DIALECT_TEST, 98),
  LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL = LOOM_OP_KIND(LOOM_DIALECT_TEST, 99),
  LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL_FLAGS = LOOM_OP_KIND(LOOM_DIALECT_TEST, 100),
  LOOM_OP_TEST_COUNT_ = 101,
};

// Synthetic flags for TemplateParamFlags parser/printer coverage.
#define LOOM_TEST_TEMPLATEFLAGS_DEBUG ((uint8_t)1)
#define LOOM_TEST_TEMPLATEFLAGS_TRACE ((uint8_t)2)

// Function visibility. Absent (0) means private.
typedef enum loom_test_visibility_e {
  LOOM_TEST_VISIBILITY_PUBLIC = 1,
  LOOM_TEST_VISIBILITY_COUNT_ = 2,
} loom_test_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_test_cc_e {
  LOOM_TEST_CC_HOST = 1,
  LOOM_TEST_CC_DEVICE = 2,
  LOOM_TEST_CC_INITIALIZER = 3,
  LOOM_TEST_CC_DEINITIALIZER = 4,
  LOOM_TEST_CC_COUNT_ = 5,
} loom_test_cc_t;

typedef enum loom_test_cmp_predicate_e {
  LOOM_TEST_CMP_PREDICATE_EQ = 0,
  LOOM_TEST_CMP_PREDICATE_NE = 1,
  LOOM_TEST_CMP_PREDICATE_LT = 2,
  LOOM_TEST_CMP_PREDICATE_LE = 3,
  LOOM_TEST_CMP_PREDICATE_GT = 4,
  LOOM_TEST_CMP_PREDICATE_GE = 5,
  LOOM_TEST_CMP_PREDICATE_COUNT_ = 6,
} loom_test_cmp_predicate_t;

// Synthetic record kind. Open for bytecode future-ordinal tests.
typedef uint8_t loom_test_record_kind_t;
typedef enum loom_test_record_kind_e {
  LOOM_TEST_RECORD_KIND_TARGET = 1,
  LOOM_TEST_RECORD_KIND_ARTIFACT = 2,
  LOOM_TEST_RECORD_KIND_COUNT_ = 3,
} loom_test_record_kind_e;

// Synthetic target kind for target-like interface tests.
typedef enum loom_test_target_kind_e {
  LOOM_TEST_TARGET_KIND_LOW_CORE = 1,
  LOOM_TEST_TARGET_KIND_QUIRKY = 2,
  LOOM_TEST_TARGET_KIND_COUNT_ = 3,
} loom_test_target_kind_t;

// LOOM_OP_TEST_ADDI: Test binary integer op.
// %result = test.addi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_test_addi_isa, LOOM_OP_TEST_ADDI)
LOOM_DEFINE_OPERAND(loom_test_addi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_test_addi_rhs, 1)
LOOM_DEFINE_RESULT(loom_test_addi_result, 0)
iree_status_t loom_test_addi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_test_addi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_test_addi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_NEG: Test unary float op.
// %result = test.neg %input : f32
LOOM_DEFINE_ISA(loom_test_neg_isa, LOOM_OP_TEST_NEG)
LOOM_DEFINE_OPERAND(loom_test_neg_input, 0)
LOOM_DEFINE_RESULT(loom_test_neg_result, 0)
iree_status_t loom_test_neg_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CAST: Test cast op.
// %result = test.cast %input : i32 to f32
LOOM_DEFINE_ISA(loom_test_cast_isa, LOOM_OP_TEST_CAST)
LOOM_DEFINE_OPERAND(loom_test_cast_input, 0)
LOOM_DEFINE_RESULT(loom_test_cast_result, 0)
iree_status_t loom_test_cast_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_TEST_CONSTANT: Test constant materialization.
// %c42 = test.constant 42 : i32
LOOM_DEFINE_ISA(loom_test_constant_isa, LOOM_OP_TEST_CONSTANT)
LOOM_DEFINE_RESULT(loom_test_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_test_constant_value, 0)
iree_status_t loom_test_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_test_constant_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEST_USE: Side-effecting sink that observes values without producing results. Not DCE-able. Use in tests to keep values alive for inspection.
// test.use %a : i32
LOOM_DEFINE_ISA(loom_test_use_isa, LOOM_OP_TEST_USE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_use_values, 0)
iree_status_t loom_test_use_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CONVERGENT: Pure value transform whose dynamic participant set is semantically observable.
// %result = test.convergent %input : i32
LOOM_DEFINE_ISA(loom_test_convergent_isa, LOOM_OP_TEST_CONVERGENT)
LOOM_DEFINE_OPERAND(loom_test_convergent_input, 0)
LOOM_DEFINE_RESULT(loom_test_convergent_result, 0)
iree_status_t loom_test_convergent_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CMP: Test comparison op.
// %result = test.cmp lt, %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_test_cmp_isa, LOOM_OP_TEST_CMP)
LOOM_DEFINE_OPERAND(loom_test_cmp_lhs, 0)
LOOM_DEFINE_OPERAND(loom_test_cmp_rhs, 1)
LOOM_DEFINE_RESULT(loom_test_cmp_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_cmp_predicate, 0, loom_test_cmp_predicate_t)
iree_status_t loom_test_cmp_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_TEST_MAP: Test region-capture elementwise op.
// %result = test.map(%element = %input : tile<4xf32>) {
//   %negated = test.neg %element : f32
//   test.yield %negated : f32
// } -> (tile<4xf32>)
LOOM_DEFINE_ISA(loom_test_map_isa, LOOM_OP_TEST_MAP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_map_inputs, 0)
LOOM_DEFINE_RESULT(loom_test_map_result, 0)
LOOM_DEFINE_REGION(loom_test_map_body, 0)
iree_status_t loom_test_map_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    loom_type_t result_type,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_UPDATE: Test tied result with index list.
// %result = test.update %tile, %tensor[%offset] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)
LOOM_DEFINE_ISA(loom_test_update_isa, LOOM_OP_TEST_UPDATE)
LOOM_DEFINE_OPERAND(loom_test_update_source, 0)
LOOM_DEFINE_OPERAND(loom_test_update_target, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_update_offsets, 2)
LOOM_DEFINE_RESULT(loom_test_update_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_test_update_static_offsets, 0)
iree_status_t loom_test_update_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t target,
    const loom_value_id_t* offsets,
    iree_host_size_t offsets_count,
    const int64_t* static_offsets,
    iree_host_size_t static_offsets_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_INVOKE: Test variadic call-like op with tied results. The verifier checks that the invoke signature matches the referenced function declaration or definition.
// %output, %count = test.invoke @callee(%weights, %input) : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)
LOOM_DEFINE_ISA(loom_test_invoke_isa, LOOM_OP_TEST_INVOKE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_invoke_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_invoke_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_invoke_callee, 0)
iree_status_t loom_test_invoke_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_call_like_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEST_LOW_CALL: Test call-like op classified like a target-low internal call.
// test.low_call @callee() : ()
LOOM_DEFINE_ISA(loom_test_low_call_isa, LOOM_OP_TEST_LOW_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_low_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_low_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_low_call_callee, 0)
iree_status_t loom_test_low_call_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_call_like_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEST_LOW_INVOKE: Test call-like op classified like an explicit target-low invocation.
// test.low_invoke @callee() : ()
LOOM_DEFINE_ISA(loom_test_low_invoke_isa, LOOM_OP_TEST_LOW_INVOKE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_low_invoke_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_low_invoke_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_low_invoke_callee, 0)
iree_status_t loom_test_low_invoke_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_call_like_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEST_SLICE: Test index list with mixed static/dynamic offsets.
// %subtile = test.slice %source[0, %offset] : tile<64x64xf16> -> (tile<16x16xf16>)
LOOM_DEFINE_ISA(loom_test_slice_isa, LOOM_OP_TEST_SLICE)
LOOM_DEFINE_OPERAND(loom_test_slice_source, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_slice_offsets, 1)
LOOM_DEFINE_RESULT(loom_test_slice_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_test_slice_static_offsets, 0)
iree_status_t loom_test_slice_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    const loom_value_id_t* offsets,
    iree_host_size_t offsets_count,
    const int64_t* static_offsets,
    iree_host_size_t static_offsets_count,
    loom_type_t result_type,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_LOOP: Test for-loop with iter_args and tied results.
// %result = test.loop %i = %c0 to %count step %c1 iter_args(%accumulator = %init : f32) -> (%init as f32) {
//   %next = test.neg %accumulator : f32
//   test.yield %next : f32
// }
LOOM_DEFINE_ISA(loom_test_loop_isa, LOOM_OP_TEST_LOOP)
LOOM_DEFINE_OPERAND(loom_test_loop_lower_bound, 0)
LOOM_DEFINE_OPERAND(loom_test_loop_upper_bound, 1)
LOOM_DEFINE_OPERAND(loom_test_loop_step, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_loop_iter_args, 3)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_loop_results, 0)
LOOM_DEFINE_REGION(loom_test_loop_body, 0)
iree_status_t loom_test_loop_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t lower_bound,
    loom_may_consume loom_value_id_t upper_bound,
    loom_may_consume loom_value_id_t step,
    loom_may_consume const loom_value_id_t* iter_args,
    iree_host_size_t iter_args_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_BLOCK_ARGS: Test op with explicit BlockArgs syntax for a region entry block.
// test.block_args %value : i32 do(%arg: i32) {
//   test.yield
// }
LOOM_DEFINE_ISA(loom_test_block_args_isa, LOOM_OP_TEST_BLOCK_ARGS)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_block_args_inputs, 0)
LOOM_DEFINE_REGION(loom_test_block_args_body, 0)
iree_status_t loom_test_block_args_build(
    loom_builder_t* builder,
    const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_BRANCH: Test if/else with both regions always present.
// %result = test.branch %condition -> (f32) {
//   test.yield %true_value : f32
// } else {
//   test.yield %false_value : f32
// }
LOOM_DEFINE_ISA(loom_test_branch_isa, LOOM_OP_TEST_BRANCH)
LOOM_DEFINE_OPERAND(loom_test_branch_condition, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_branch_results, 0)
LOOM_DEFINE_REGION(loom_test_branch_then_region, 0)
LOOM_DEFINE_REGION(loom_test_branch_else_region, 1)
iree_status_t loom_test_branch_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t condition,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_OPTIONAL_REGION: Test op with a required body and a trailing optional region.
// test.optional_region %condition {
//   test.yield
// }
LOOM_DEFINE_ISA(loom_test_optional_region_isa, LOOM_OP_TEST_OPTIONAL_REGION)
LOOM_DEFINE_OPERAND(loom_test_optional_region_condition, 0)
LOOM_DEFINE_REGION(loom_test_optional_region_body, 0)
LOOM_DEFINE_OPTIONAL_REGION(loom_test_optional_region_else_region, 1)
enum loom_test_optional_region_build_flag_bits_e {
  LOOM_TEST_OPTIONAL_REGION_BUILD_FLAG_HAS_ELSE_REGION = 1u << 0,
};
typedef uint32_t loom_test_optional_region_build_flags_t;
iree_status_t loom_test_optional_region_build(
    loom_builder_t* builder,
    loom_test_optional_region_build_flags_t build_flags,
    loom_value_id_t condition,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_IMPLICIT_YIELD: Dedicated zero-field implicit terminator synthesized for elidable test regions.
// test.implicit_yield
LOOM_DEFINE_ISA(loom_test_implicit_yield_isa, LOOM_OP_TEST_IMPLICIT_YIELD)
iree_status_t loom_test_implicit_yield_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_YIELD: Test yield terminator.
// test.yield
LOOM_DEFINE_ISA(loom_test_yield_isa, LOOM_OP_TEST_YIELD)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_yield_values, 0)
iree_status_t loom_test_yield_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_BR: Test CFG branch terminator with a semantic successor edge.
// test.br ^dest
LOOM_DEFINE_ISA(loom_test_br_isa, LOOM_OP_TEST_BR)
LOOM_DEFINE_SUCCESSOR(loom_test_br_dest, 0)
iree_status_t loom_test_br_build(
    loom_builder_t* builder,
    loom_block_t* dest,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_FUNC: Test function definition with body always present.
// test.func @identity(%input: f32) -> (f32) {
//   test.yield %input : f32
// }
LOOM_DEFINE_ISA(loom_test_func_isa, LOOM_OP_TEST_FUNC)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_func_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_func_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_func_visibility, 1, loom_test_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_func_cc, 2, loom_test_cc_t)
LOOM_DEFINE_REGION(loom_test_func_body, 0)
enum loom_test_func_build_flag_bits_e {
  LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEST_FUNC_BUILD_FLAG_HAS_CC = 1u << 1,
};
typedef uint32_t loom_test_func_build_flags_t;
iree_status_t loom_test_func_build(
    loom_builder_t* builder,
    loom_test_func_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_SPLIT_FUNC: Test function definition with projected signature args in a side region.
// test.split_func @projected(%arg: i32) {
//   test.yield
// } launch {
//   test.yield
// }
LOOM_DEFINE_ISA(loom_test_split_func_isa, LOOM_OP_TEST_SPLIT_FUNC)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_split_func_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_split_func_visibility, 1, loom_test_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_split_func_cc, 2, loom_test_cc_t)
LOOM_DEFINE_REGION(loom_test_split_func_config, 0)
LOOM_DEFINE_REGION(loom_test_split_func_body, 1)
enum loom_test_split_func_build_flag_bits_e {
  LOOM_TEST_SPLIT_FUNC_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEST_SPLIT_FUNC_BUILD_FLAG_HAS_CC = 1u << 1,
};
typedef uint32_t loom_test_split_func_build_flags_t;
iree_status_t loom_test_split_func_build(
    loom_builder_t* builder,
    loom_test_split_func_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_DECL: Test function declaration with no body and signature arguments stored as op operands.
// test.decl @identity(%input: f32) -> (%input as f32)
LOOM_DEFINE_ISA(loom_test_decl_isa, LOOM_OP_TEST_DECL)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_decl_visibility, 1, loom_test_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_decl_cc, 2, loom_test_cc_t)
enum loom_test_decl_build_flag_bits_e {
  LOOM_TEST_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEST_DECL_BUILD_FLAG_HAS_CC = 1u << 1,
};
typedef uint32_t loom_test_decl_build_flags_t;
iree_status_t loom_test_decl_build(
    loom_builder_t* builder,
    loom_test_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RECORD: Test named module record with generic symbol payload metadata.
// test.record target @target {arch = "gfx1100", lanes = 64}
LOOM_DEFINE_ISA(loom_test_record_isa, LOOM_OP_TEST_RECORD)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_record_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_record_kind, 1, loom_test_record_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_test_record_dict, 2)
enum loom_test_record_build_flag_bits_e {
  LOOM_TEST_RECORD_BUILD_FLAG_HAS_KIND = 1u << 0,
};
typedef uint32_t loom_test_record_build_flags_t;
iree_status_t loom_test_record_build(
    loom_builder_t* builder,
    loom_test_record_build_flags_t build_flags,
    loom_optional uint8_t kind,
    loom_symbol_ref_t symbol,
    loom_optional loom_named_attr_slice_t dict,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_ATTRS: Test op with attribute dictionary.
// %result = test.attrs %input {axis = 0, label = "foo"} : f32
LOOM_DEFINE_ISA(loom_test_attrs_isa, LOOM_OP_TEST_ATTRS)
LOOM_DEFINE_OPERAND(loom_test_attrs_input, 0)
LOOM_DEFINE_RESULT(loom_test_attrs_result, 0)
LOOM_DEFINE_ATTR_DICT(loom_test_attrs_dict, 0)
iree_status_t loom_test_attrs_build(
    loom_builder_t* builder,
    loom_value_id_t input,
    loom_optional loom_named_attr_slice_t dict,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_OPERAND_DICT: Test op with a keyed SSA operand dictionary.
// %result = test.operand_dict %input {alpha = %a : i32, beta = %b : f32} : f32
LOOM_DEFINE_ISA(loom_test_operand_dict_isa, LOOM_OP_TEST_OPERAND_DICT)
LOOM_DEFINE_OPERAND(loom_test_operand_dict_input, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_operand_dict_params, 1)
LOOM_DEFINE_RESULT(loom_test_operand_dict_result, 0)
LOOM_DEFINE_ATTR_DICT(loom_test_operand_dict_param_names, 0)
iree_status_t loom_test_operand_dict_build(
    loom_builder_t* builder,
    loom_value_id_t input,
    const loom_named_value_t* params,
    iree_host_size_t params_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_ATTR_TABLE: Test op with a static-attribute-keyed SSA value table.
// %a, %b = test.attr_table %selector {0 = (%a0, %b0), 1 = (%a1, %b1)} default(%ad, %bd) : i32, f32
LOOM_DEFINE_ISA(loom_test_attr_table_isa, LOOM_OP_TEST_ATTR_TABLE)
LOOM_DEFINE_OPERAND(loom_test_attr_table_selector, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_attr_table_values, 1)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_attr_table_results, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_test_attr_table_case_keys, 0)
iree_status_t loom_test_attr_table_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t selector,
    const int64_t* case_keys,
    iree_host_size_t case_keys_count,
    loom_may_consume const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_REGION_TABLE: Test op with a static-attribute-keyed region table.
// test.region_table %selector {
//   case 0 {
//     test.yield
//   }
//   default {
//     test.yield
//   }
// }
LOOM_DEFINE_ISA(loom_test_region_table_isa, LOOM_OP_TEST_REGION_TABLE)
LOOM_DEFINE_OPERAND(loom_test_region_table_selector, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_test_region_table_case_keys, 0)
LOOM_DEFINE_REGION(loom_test_region_table_default_region, 0)
LOOM_DEFINE_VARIADIC_REGIONS(loom_test_region_table_case_regions, 1)
iree_status_t loom_test_region_table_build(
    loom_builder_t* builder,
    loom_value_id_t selector,
    const int64_t* case_keys,
    iree_host_size_t case_keys_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_DEFLATE: Test op with result type referencing a co-result dim.
// %output, %length = test.deflate %input : tensor<[%M]xf32> -> (tensor<[%length]xf32>, index)
LOOM_DEFINE_ISA(loom_test_deflate_isa, LOOM_OP_TEST_DEFLATE)
LOOM_DEFINE_OPERAND(loom_test_deflate_input, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_deflate_results, 0)
iree_status_t loom_test_deflate_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t input,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_ASSUME: Test predicate-constrained identity (SSA assume).
// %M2 = test.assume %M [mul(%M, 16)] : index
LOOM_DEFINE_ISA(loom_test_assume_isa, LOOM_OP_TEST_ASSUME)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_assume_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_assume_results, 0)
iree_status_t loom_test_assume_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CONVERT: Test op with bare result type (no parentheses).
// %result = test.convert %input : i32 -> f32
LOOM_DEFINE_ISA(loom_test_convert_isa, LOOM_OP_TEST_CONVERT)
LOOM_DEFINE_OPERAND(loom_test_convert_input, 0)
LOOM_DEFINE_RESULT(loom_test_convert_result, 0)
iree_status_t loom_test_convert_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_REDUCE: Test variadic operands with SameType constraint across variadic and result.
// %sum = test.reduce %a, %b, %c : i32
LOOM_DEFINE_ISA(loom_test_reduce_isa, LOOM_OP_TEST_REDUCE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_reduce_inputs, 0)
LOOM_DEFINE_RESULT(loom_test_reduce_result, 0)
iree_status_t loom_test_reduce_build(
    loom_builder_t* builder,
    const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_READ_RESOURCE: Test op that reads from a resource operand.
// %tile = test.read_resource %pool : pool<[%BS]> -> tile<4xf32>
LOOM_DEFINE_ISA(loom_test_read_resource_isa, LOOM_OP_TEST_READ_RESOURCE)
LOOM_DEFINE_OPERAND(loom_test_read_resource_source, 0)
LOOM_DEFINE_RESULT(loom_test_read_resource_result, 0)
iree_status_t loom_test_read_resource_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_WRITE_RESOURCE: Test op that writes to a resource operand.
// test.write_resource %pool, %tile : pool<[%BS]>, tile<4xf32>
LOOM_DEFINE_ISA(loom_test_write_resource_isa, LOOM_OP_TEST_WRITE_RESOURCE)
LOOM_DEFINE_OPERAND(loom_test_write_resource_target, 0)
LOOM_DEFINE_OPERAND(loom_test_write_resource_data, 1)
iree_status_t loom_test_write_resource_build(
    loom_builder_t* builder,
    loom_value_id_t target,
    loom_value_id_t data,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_MUTATE_RESOURCE: Test op that atomically reads and writes a resource operand.
// %old = test.mutate_resource %pool, %delta : pool<[%BS]>, i32 -> i32
LOOM_DEFINE_ISA(loom_test_mutate_resource_isa, LOOM_OP_TEST_MUTATE_RESOURCE)
LOOM_DEFINE_OPERAND(loom_test_mutate_resource_target, 0)
LOOM_DEFINE_OPERAND(loom_test_mutate_resource_value, 1)
LOOM_DEFINE_RESULT(loom_test_mutate_resource_old_value, 0)
iree_status_t loom_test_mutate_resource_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t target,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_ALLOC: Test allocation op. Each execution produces a distinct identity even with identical operands. Prevents CSE but allows DCE when unused.
// %pool = test.alloc %sz : index -> pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_alloc_isa, LOOM_OP_TEST_ALLOC)
LOOM_DEFINE_OPERAND(loom_test_alloc_size, 0)
LOOM_DEFINE_RESULT(loom_test_alloc_result, 0)
iree_status_t loom_test_alloc_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t size,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_ISOLATED_REGION: Test op with an isolated single-block region. Values from the enclosing scope are not visible inside the body.
// %r = test.isolated_region -> (i32) {
//   %c = test.constant 42 : i32
//   test.yield %c : i32
// }
LOOM_DEFINE_ISA(loom_test_isolated_region_isa, LOOM_OP_TEST_ISOLATED_REGION)
LOOM_DEFINE_VARIADIC_RESULTS(loom_test_isolated_region_results, 0)
LOOM_DEFINE_REGION(loom_test_isolated_region_body, 0)
iree_status_t loom_test_isolated_region_build(
    loom_builder_t* builder,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_COUNTER: Test op for canonicalize multi-step and error path testing. Canonicalize: value < 0 returns error, value > 0 decrements, value == 0 is fixed point.
// %c = test.counter 3 : i32
LOOM_DEFINE_ISA(loom_test_counter_isa, LOOM_OP_TEST_COUNTER)
LOOM_DEFINE_RESULT(loom_test_counter_result, 0)
LOOM_DEFINE_ATTR_I64(loom_test_counter_value, 0)
iree_status_t loom_test_counter_build(
    loom_builder_t* builder,
    int64_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_counter_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_TEST_DIM: Test dimension query to exercise ATTR_IN_RANGE_RANK constraint.
// %d = test.dim %t[0] : tile<4xf32> -> index
LOOM_DEFINE_ISA(loom_test_dim_isa, LOOM_OP_TEST_DIM)
LOOM_DEFINE_OPERAND(loom_test_dim_source, 0)
LOOM_DEFINE_RESULT(loom_test_dim_result, 0)
LOOM_DEFINE_ATTR_I64(loom_test_dim_dim_index, 0)
iree_status_t loom_test_dim_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t dim_index,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_FACT_RANGE_LO: Exposes the analysis range lower bound as an i64 constant.
// %lo = test.fact_range_lo %x : index -> i64
LOOM_DEFINE_ISA(loom_test_fact_range_lo_isa, LOOM_OP_TEST_FACT_RANGE_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_range_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_range_lo_result, 0)
iree_status_t loom_test_fact_range_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_range_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_RANGE_HI: Exposes the analysis range upper bound as an i64 constant.
// %hi = test.fact_range_hi %x : index -> i64
LOOM_DEFINE_ISA(loom_test_fact_range_hi_isa, LOOM_OP_TEST_FACT_RANGE_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_range_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_range_hi_result, 0)
iree_status_t loom_test_fact_range_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_range_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_LO: Exposes the all-equal element range lower bound as an i64 constant.
// %lo = test.fact_all_equal_range_lo %x : reg<test.i32 x4> -> i64
LOOM_DEFINE_ISA(loom_test_fact_all_equal_range_lo_isa, LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_all_equal_range_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_all_equal_range_lo_result, 0)
iree_status_t loom_test_fact_all_equal_range_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_all_equal_range_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_HI: Exposes the all-equal element range upper bound as an i64 constant.
// %hi = test.fact_all_equal_range_hi %x : reg<test.i32 x4> -> i64
LOOM_DEFINE_ISA(loom_test_fact_all_equal_range_hi_isa, LOOM_OP_TEST_FACT_ALL_EQUAL_RANGE_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_all_equal_range_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_all_equal_range_hi_result, 0)
iree_status_t loom_test_fact_all_equal_range_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_all_equal_range_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_DIVISOR: Exposes the analysis known divisor as an i64 constant.
// %div = test.fact_divisor %x : index -> i64
LOOM_DEFINE_ISA(loom_test_fact_divisor_isa, LOOM_OP_TEST_FACT_DIVISOR)
LOOM_DEFINE_OPERAND(loom_test_fact_divisor_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_divisor_result, 0)
iree_status_t loom_test_fact_divisor_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_divisor_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_NON_NEGATIVE: Returns 1 if the input is provably non-negative, 0 otherwise.
// %nn = test.fact_non_negative %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_non_negative_isa, LOOM_OP_TEST_FACT_NON_NEGATIVE)
LOOM_DEFINE_OPERAND(loom_test_fact_non_negative_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_non_negative_result, 0)
iree_status_t loom_test_fact_non_negative_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_non_negative_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_NON_ZERO: Returns 1 if the input is provably non-zero, 0 otherwise.
// %nz = test.fact_non_zero %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_non_zero_isa, LOOM_OP_TEST_FACT_NON_ZERO)
LOOM_DEFINE_OPERAND(loom_test_fact_non_zero_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_non_zero_result, 0)
iree_status_t loom_test_fact_non_zero_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_non_zero_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_POSITIVE: Returns 1 if the input is provably positive (> 0), 0 otherwise.
// %pos = test.fact_positive %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_positive_isa, LOOM_OP_TEST_FACT_POSITIVE)
LOOM_DEFINE_OPERAND(loom_test_fact_positive_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_positive_result, 0)
iree_status_t loom_test_fact_positive_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_positive_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_POWER_OF_TWO: Returns 1 if the input is provably a power of two, 0 otherwise.
// %p2 = test.fact_power_of_two %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_power_of_two_isa, LOOM_OP_TEST_FACT_POWER_OF_TWO)
LOOM_DEFINE_OPERAND(loom_test_fact_power_of_two_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_power_of_two_result, 0)
iree_status_t loom_test_fact_power_of_two_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_power_of_two_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_UNIFORM: Returns 1 if the input is provably uniform across active lanes, 0 otherwise.
// %uniform = test.fact_uniform %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_uniform_isa, LOOM_OP_TEST_FACT_UNIFORM)
LOOM_DEFINE_OPERAND(loom_test_fact_uniform_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_uniform_result, 0)
iree_status_t loom_test_fact_uniform_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_uniform_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_LANE_VARYING: Returns 1 if the input may vary across active lanes, 0 otherwise.
// %varying = test.fact_lane_varying %x : index -> i1
LOOM_DEFINE_ISA(loom_test_fact_lane_varying_isa, LOOM_OP_TEST_FACT_LANE_VARYING)
LOOM_DEFINE_OPERAND(loom_test_fact_lane_varying_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_lane_varying_result, 0)
iree_status_t loom_test_fact_lane_varying_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_lane_varying_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_LANE_PREDICATE: Returns 1 if the input is an i1 lane predicate, 0 otherwise.
// %predicate = test.fact_lane_predicate %x : i1 -> i1
LOOM_DEFINE_ISA(loom_test_fact_lane_predicate_isa, LOOM_OP_TEST_FACT_LANE_PREDICATE)
LOOM_DEFINE_OPERAND(loom_test_fact_lane_predicate_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_lane_predicate_result, 0)
iree_status_t loom_test_fact_lane_predicate_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_lane_predicate_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_SUBGROUP_LANE_MASK: Returns 1 if the input is an integer subgroup lane mask, 0 otherwise.
// %mask = test.fact_subgroup_lane_mask %x : i64 -> i1
LOOM_DEFINE_ISA(loom_test_fact_subgroup_lane_mask_isa, LOOM_OP_TEST_FACT_SUBGROUP_LANE_MASK)
LOOM_DEFINE_OPERAND(loom_test_fact_subgroup_lane_mask_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_subgroup_lane_mask_result, 0)
iree_status_t loom_test_fact_subgroup_lane_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_subgroup_lane_mask_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_IS_VECTOR_IOTA: Returns 1 if the input has a vector.iota analysis summary, 0 otherwise.
// %is = test.fact_is_vector_iota %x : vector<[%n]xindex> -> i1
LOOM_DEFINE_ISA(loom_test_fact_is_vector_iota_isa, LOOM_OP_TEST_FACT_IS_VECTOR_IOTA)
LOOM_DEFINE_OPERAND(loom_test_fact_is_vector_iota_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_is_vector_iota_result, 0)
iree_status_t loom_test_fact_is_vector_iota_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_is_vector_iota_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_IS_VECTOR_PREFIX_MASK: Returns 1 if the input has a vector.mask.range analysis summary, 0 otherwise.
// %is = test.fact_is_vector_prefix_mask %x : vector<[%n]xi1> -> i1
LOOM_DEFINE_ISA(loom_test_fact_is_vector_prefix_mask_isa, LOOM_OP_TEST_FACT_IS_VECTOR_PREFIX_MASK)
LOOM_DEFINE_OPERAND(loom_test_fact_is_vector_prefix_mask_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_is_vector_prefix_mask_result, 0)
iree_status_t loom_test_fact_is_vector_prefix_mask_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_is_vector_prefix_mask_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ENCODING_LAYOUT_KIND: Exposes an encoding-summary address-layout kind as an i64 constant.
// %kind = test.fact_encoding_layout_kind %layout : encoding<layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_encoding_layout_kind_isa, LOOM_OP_TEST_FACT_ENCODING_LAYOUT_KIND)
LOOM_DEFINE_OPERAND(loom_test_fact_encoding_layout_kind_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_encoding_layout_kind_result, 0)
iree_status_t loom_test_fact_encoding_layout_kind_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_encoding_layout_kind_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ENCODING_LAYOUT_STRIDE_HI: Exposes an encoding-summary strided-layout stride upper bound as an i64 constant.
// %hi = test.fact_encoding_layout_stride_hi %layout[0] : encoding<layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_encoding_layout_stride_hi_isa, LOOM_OP_TEST_FACT_ENCODING_LAYOUT_STRIDE_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_encoding_layout_stride_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_encoding_layout_stride_hi_result, 0)
LOOM_DEFINE_ATTR_I64(loom_test_fact_encoding_layout_stride_hi_axis, 0)
iree_status_t loom_test_fact_encoding_layout_stride_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    int64_t axis,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_encoding_layout_stride_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ENCODING_MATRIX_FIELD: Exposes an encoded-operand storage-schema summary field as an i64 constant. Supported fields are element_format, payload_packing, scale_topology, scale_format, secondary_scale_format, affine, rounding, codebook, sparsity, payload_registers, payload_elements, scale_group_elements, scale_operands, zero_scale_fallback, and static_spec.
// %format = test.fact_encoding_matrix_field %schema["element_format"] : encoding<schema> -> i64
LOOM_DEFINE_ISA(loom_test_fact_encoding_matrix_field_isa, LOOM_OP_TEST_FACT_ENCODING_MATRIX_FIELD)
LOOM_DEFINE_OPERAND(loom_test_fact_encoding_matrix_field_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_encoding_matrix_field_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_test_fact_encoding_matrix_field_field, 0)
iree_status_t loom_test_fact_encoding_matrix_field_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_string_id_t field,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_encoding_matrix_field_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_IS_BUFFER_REFERENCE: Returns 1 if the input has a buffer-reference analysis summary, 0 otherwise.
// %is = test.fact_is_buffer_reference %buffer : buffer -> i1
LOOM_DEFINE_ISA(loom_test_fact_is_buffer_reference_isa, LOOM_OP_TEST_FACT_IS_BUFFER_REFERENCE)
LOOM_DEFINE_OPERAND(loom_test_fact_is_buffer_reference_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_is_buffer_reference_result, 0)
iree_status_t loom_test_fact_is_buffer_reference_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_is_buffer_reference_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_IS_VIEW_REFERENCE: Returns 1 if the input has a view-reference analysis summary, 0 otherwise.
// %is = test.fact_is_view_reference %view : view<4xf32, %layout> -> i1
LOOM_DEFINE_ISA(loom_test_fact_is_view_reference_isa, LOOM_OP_TEST_FACT_IS_VIEW_REFERENCE)
LOOM_DEFINE_OPERAND(loom_test_fact_is_view_reference_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_is_view_reference_result, 0)
iree_status_t loom_test_fact_is_view_reference_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_is_view_reference_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_BUFFER_MEMORY_SPACE: Exposes a buffer-reference memory-space enum value as an i64 constant, or -1 when unknown.
// %space = test.fact_buffer_memory_space %buffer : buffer -> i64
LOOM_DEFINE_ISA(loom_test_fact_buffer_memory_space_isa, LOOM_OP_TEST_FACT_BUFFER_MEMORY_SPACE)
LOOM_DEFINE_OPERAND(loom_test_fact_buffer_memory_space_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_buffer_memory_space_result, 0)
iree_status_t loom_test_fact_buffer_memory_space_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_buffer_memory_space_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_MEMORY_SPACE: Exposes a view-reference memory-space enum value as an i64 constant, or -1 when unknown.
// %space = test.fact_view_memory_space %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_memory_space_isa, LOOM_OP_TEST_FACT_VIEW_MEMORY_SPACE)
LOOM_DEFINE_OPERAND(loom_test_fact_view_memory_space_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_memory_space_result, 0)
iree_status_t loom_test_fact_view_memory_space_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_memory_space_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_ROOT_MATCHES: Returns 1 if a view reference and another reference share the same root identity.
// %same = test.fact_view_root_matches %view, %buffer : view<4xf32, %layout>, buffer -> i1
LOOM_DEFINE_ISA(loom_test_fact_view_root_matches_isa, LOOM_OP_TEST_FACT_VIEW_ROOT_MATCHES)
LOOM_DEFINE_OPERAND(loom_test_fact_view_root_matches_view, 0)
LOOM_DEFINE_OPERAND(loom_test_fact_view_root_matches_root, 1)
LOOM_DEFINE_RESULT(loom_test_fact_view_root_matches_result, 0)
iree_status_t loom_test_fact_view_root_matches_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t view,
    loom_may_consume loom_value_id_t root,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_root_matches_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ALIAS_SCOPE_KNOWN: Returns 1 if the input has a comparable storage alias scope, 0 otherwise.
// %known = test.fact_alias_scope_known %buffer : buffer -> i1
LOOM_DEFINE_ISA(loom_test_fact_alias_scope_known_isa, LOOM_OP_TEST_FACT_ALIAS_SCOPE_KNOWN)
LOOM_DEFINE_OPERAND(loom_test_fact_alias_scope_known_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_alias_scope_known_result, 0)
iree_status_t loom_test_fact_alias_scope_known_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_alias_scope_known_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_ALIAS_SCOPE_MATCHES: Returns 1 if both inputs have the same comparable storage alias scope, 0 otherwise.
// %same = test.fact_alias_scope_matches %lhs, %rhs : buffer, buffer -> i1
LOOM_DEFINE_ISA(loom_test_fact_alias_scope_matches_isa, LOOM_OP_TEST_FACT_ALIAS_SCOPE_MATCHES)
LOOM_DEFINE_OPERAND(loom_test_fact_alias_scope_matches_lhs, 0)
LOOM_DEFINE_OPERAND(loom_test_fact_alias_scope_matches_rhs, 1)
LOOM_DEFINE_RESULT(loom_test_fact_alias_scope_matches_result, 0)
iree_status_t loom_test_fact_alias_scope_matches_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_test_fact_alias_scope_matches_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_LO: Exposes a view-reference byte-offset lower bound as an i64 constant.
// %lo = test.fact_view_byte_offset_lo %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_byte_offset_lo_isa, LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_view_byte_offset_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_byte_offset_lo_result, 0)
iree_status_t loom_test_fact_view_byte_offset_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_byte_offset_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_HI: Exposes a view-reference byte-offset upper bound as an i64 constant.
// %hi = test.fact_view_byte_offset_hi %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_byte_offset_hi_isa, LOOM_OP_TEST_FACT_VIEW_BYTE_OFFSET_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_view_byte_offset_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_byte_offset_hi_result, 0)
iree_status_t loom_test_fact_view_byte_offset_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_byte_offset_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_LO: Exposes a view-reference footprint byte-length lower bound as an i64 constant.
// %lo = test.fact_view_byte_length_lo %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_byte_length_lo_isa, LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_view_byte_length_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_byte_length_lo_result, 0)
iree_status_t loom_test_fact_view_byte_length_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_byte_length_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_HI: Exposes a view-reference footprint byte-length upper bound as an i64 constant.
// %hi = test.fact_view_byte_length_hi %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_byte_length_hi_isa, LOOM_OP_TEST_FACT_VIEW_BYTE_LENGTH_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_view_byte_length_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_byte_length_hi_result, 0)
iree_status_t loom_test_fact_view_byte_length_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_byte_length_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_MIN_ALIGNMENT: Exposes the minimum provable view byte-offset alignment as an i64 constant.
// %align = test.fact_view_min_alignment %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_min_alignment_isa, LOOM_OP_TEST_FACT_VIEW_MIN_ALIGNMENT)
LOOM_DEFINE_OPERAND(loom_test_fact_view_min_alignment_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_min_alignment_result, 0)
iree_status_t loom_test_fact_view_min_alignment_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_min_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_BUFFER_MIN_ALIGNMENT: Exposes the minimum provable buffer root byte alignment as an i64 constant.
// %align = test.fact_buffer_min_alignment %buffer : buffer -> i64
LOOM_DEFINE_ISA(loom_test_fact_buffer_min_alignment_isa, LOOM_OP_TEST_FACT_BUFFER_MIN_ALIGNMENT)
LOOM_DEFINE_OPERAND(loom_test_fact_buffer_min_alignment_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_buffer_min_alignment_result, 0)
iree_status_t loom_test_fact_buffer_min_alignment_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_buffer_min_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_ROOT_MIN_ALIGNMENT: Exposes the minimum provable view root byte alignment as an i64 constant.
// %align = test.fact_view_root_min_alignment %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_root_min_alignment_isa, LOOM_OP_TEST_FACT_VIEW_ROOT_MIN_ALIGNMENT)
LOOM_DEFINE_OPERAND(loom_test_fact_view_root_min_alignment_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_root_min_alignment_result, 0)
iree_status_t loom_test_fact_view_root_min_alignment_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_root_min_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_VIEW_ELEMENT_BYTES: Exposes the static addressed element byte count, or -1 when unknown.
// %bytes = test.fact_view_element_bytes %view : view<4xf32, %layout> -> i64
LOOM_DEFINE_ISA(loom_test_fact_view_element_bytes_isa, LOOM_OP_TEST_FACT_VIEW_ELEMENT_BYTES)
LOOM_DEFINE_OPERAND(loom_test_fact_view_element_bytes_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_view_element_bytes_result, 0)
iree_status_t loom_test_fact_view_element_bytes_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_view_element_bytes_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_IS_STORAGE_REFERENCE: Returns 1 if the input has a storage reference analysis summary, 0 otherwise.
// %is = test.fact_is_storage_reference %storage : low.storage<workgroup> -> i1
LOOM_DEFINE_ISA(loom_test_fact_is_storage_reference_isa, LOOM_OP_TEST_FACT_IS_STORAGE_REFERENCE)
LOOM_DEFINE_OPERAND(loom_test_fact_is_storage_reference_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_is_storage_reference_result, 0)
iree_status_t loom_test_fact_is_storage_reference_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_is_storage_reference_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_SAME_BACKING: Returns 1 if two storage references share the same backing reservation.
// %same = test.fact_storage_same_backing %lhs, %rhs : low.storage<workgroup>, low.storage<workgroup> -> i1
LOOM_DEFINE_ISA(loom_test_fact_storage_same_backing_isa, LOOM_OP_TEST_FACT_STORAGE_SAME_BACKING)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_same_backing_lhs, 0)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_same_backing_rhs, 1)
LOOM_DEFINE_RESULT(loom_test_fact_storage_same_backing_result, 0)
iree_status_t loom_test_fact_storage_same_backing_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_test_fact_storage_same_backing_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_LO: Exposes a storage reference byte-offset lower bound as an i64 constant.
// %lo = test.fact_storage_byte_offset_lo %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_byte_offset_lo_isa, LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_byte_offset_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_byte_offset_lo_result, 0)
iree_status_t loom_test_fact_storage_byte_offset_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_byte_offset_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_HI: Exposes a storage reference byte-offset upper bound as an i64 constant.
// %hi = test.fact_storage_byte_offset_hi %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_byte_offset_hi_isa, LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_byte_offset_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_byte_offset_hi_result, 0)
iree_status_t loom_test_fact_storage_byte_offset_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_byte_offset_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_DIVISOR: Exposes a storage reference byte-offset divisor as an i64 constant.
// %div = test.fact_storage_byte_offset_divisor %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_byte_offset_divisor_isa, LOOM_OP_TEST_FACT_STORAGE_BYTE_OFFSET_DIVISOR)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_byte_offset_divisor_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_byte_offset_divisor_result, 0)
iree_status_t loom_test_fact_storage_byte_offset_divisor_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_byte_offset_divisor_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_LO: Exposes a storage reference byte-length lower bound as an i64 constant.
// %lo = test.fact_storage_byte_length_lo %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_byte_length_lo_isa, LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_LO)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_byte_length_lo_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_byte_length_lo_result, 0)
iree_status_t loom_test_fact_storage_byte_length_lo_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_byte_length_lo_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_HI: Exposes a storage reference byte-length upper bound as an i64 constant.
// %hi = test.fact_storage_byte_length_hi %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_byte_length_hi_isa, LOOM_OP_TEST_FACT_STORAGE_BYTE_LENGTH_HI)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_byte_length_hi_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_byte_length_hi_result, 0)
iree_status_t loom_test_fact_storage_byte_length_hi_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_byte_length_hi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_MIN_ALIGNMENT: Exposes a storage reference minimum byte alignment as an i64 constant.
// %align = test.fact_storage_min_alignment %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_min_alignment_isa, LOOM_OP_TEST_FACT_STORAGE_MIN_ALIGNMENT)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_min_alignment_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_min_alignment_result, 0)
iree_status_t loom_test_fact_storage_min_alignment_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_min_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_FACT_STORAGE_SPACE: Exposes a storage reference storage-space enum value as an i64 constant, or -1 when unknown.
// %space = test.fact_storage_space %storage : low.storage<workgroup> -> i64
LOOM_DEFINE_ISA(loom_test_fact_storage_space_isa, LOOM_OP_TEST_FACT_STORAGE_SPACE)
LOOM_DEFINE_OPERAND(loom_test_fact_storage_space_value, 0)
LOOM_DEFINE_RESULT(loom_test_fact_storage_space_result, 0)
iree_status_t loom_test_fact_storage_space_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_test_fact_storage_space_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_TEST_REGION_SYNTAX: Test op whose body uses an alternate declarative region syntax while preserving ordinary region storage.
// test.region_syntax do {
//   test.yield
// }
LOOM_DEFINE_ISA(loom_test_region_syntax_isa, LOOM_OP_TEST_REGION_SYNTAX)
LOOM_DEFINE_REGION(loom_test_region_syntax_body, 0)
iree_status_t loom_test_region_syntax_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_LOW_ASM_REGION: Test op whose body uses descriptor-backed target-low assembly syntax while preserving ordinary region storage.
// test.low_asm_region asm<test.low.core> {
//   return
// }
LOOM_DEFINE_ISA(loom_test_low_asm_region_isa, LOOM_OP_TEST_LOW_ASM_REGION)
LOOM_DEFINE_REGION(loom_test_low_asm_region_body, 0)
iree_status_t loom_test_low_asm_region_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CLAUSE_CONSTANT: Test constant materialization using a named value clause.
// %c42 = test.clause_constant value(42) : i32
LOOM_DEFINE_ISA(loom_test_clause_constant_isa, LOOM_OP_TEST_CLAUSE_CONSTANT)
LOOM_DEFINE_RESULT(loom_test_clause_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_test_clause_constant_value, 0)
iree_status_t loom_test_clause_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_CLAUSE_COPY: Test dynamic operand clauses that model source/target-style syntax.
// test.clause_copy source(%src) target(%dst) : i32
LOOM_DEFINE_ISA(loom_test_clause_copy_isa, LOOM_OP_TEST_CLAUSE_COPY)
LOOM_DEFINE_OPERAND(loom_test_clause_copy_source, 0)
LOOM_DEFINE_OPERAND(loom_test_clause_copy_target, 1)
iree_status_t loom_test_clause_copy_build(
    loom_builder_t* builder,
    loom_value_id_t source,
    loom_value_id_t target,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_TYPED_USE: Side-effecting sink with adjacent SSA type annotations in its format.
// test.typed_use %a: i32
LOOM_DEFINE_ISA(loom_test_typed_use_isa, LOOM_OP_TEST_TYPED_USE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_typed_use_values, 0)
iree_status_t loom_test_typed_use_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_SHAPE: Test named index-list clause that keeps a space before '['.
// test.shape %value shape [%m, 4] : tile<[%m]x4xf32>
LOOM_DEFINE_ISA(loom_test_shape_isa, LOOM_OP_TEST_SHAPE)
LOOM_DEFINE_OPERAND(loom_test_shape_value, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_test_shape_dims, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_test_shape_static_dims, 0)
iree_status_t loom_test_shape_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    const loom_value_id_t* dims,
    iree_host_size_t dims_count,
    const int64_t* static_dims,
    iree_host_size_t static_dims_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_TARGET: Test target-like module record with structural interface metadata.
// test.target<low_core> @target {subgroup_size = 64}
LOOM_DEFINE_ISA(loom_test_target_isa, LOOM_OP_TEST_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_target_kind, 1, loom_test_target_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_target_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_target_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_test_target_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_test_target_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_test_target_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_test_target_subgroup_size, 11)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_grid_size_x, 12)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_grid_size_y, 13)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_grid_size_z, 14)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_flat_grid_size, 15)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_count_x, 16)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_count_y, 17)
LOOM_DEFINE_ATTR_I64(loom_test_target_max_workgroup_count_z, 18)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_generic, 19)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_global, 20)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_workgroup, 21)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_constant, 22)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_private, 23)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_host, 24)
LOOM_DEFINE_ATTR_I64(loom_test_target_memory_space_descriptor, 25)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_target_abi, 26, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_test_target_export_symbol, 27)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_target_linkage, 28, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_test_target_hal_buffer_resource_flags, 29)
LOOM_DEFINE_ATTR_STRING(loom_test_target_contract_set_key, 30)
LOOM_DEFINE_ATTR_I64(loom_test_target_contract_feature_bits, 31)
enum loom_test_target_build_flag_bits_e {
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT = 1u << 0,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT = 1u << 1,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH = 1u << 2,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH = 1u << 3,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH = 1u << 4,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X = 1u << 5,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y = 1u << 6,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z = 1u << 7,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE = 1u << 8,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE = 1u << 9,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X = 1u << 10,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y = 1u << 11,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z = 1u << 12,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE = 1u << 13,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X = 1u << 14,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y = 1u << 15,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z = 1u << 16,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC = 1u << 17,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL = 1u << 18,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP = 1u << 19,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT = 1u << 20,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE = 1u << 21,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST = 1u << 22,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR = 1u << 23,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_ABI = 1u << 24,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 25,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_LINKAGE = 1u << 26,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS = 1u << 27,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY = 1u << 28,
  LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS = 1u << 29,
};
typedef uint32_t loom_test_target_build_flags_t;
iree_status_t loom_test_target_build(
    loom_builder_t* builder,
    loom_test_target_build_flags_t build_flags,
    loom_test_target_kind_t kind,
    loom_symbol_ref_t symbol,
    loom_optional uint8_t codegen_format,
    loom_optional uint8_t artifact_format,
    loom_optional int64_t default_pointer_bitwidth,
    loom_optional int64_t index_bitwidth,
    loom_optional int64_t offset_bitwidth,
    loom_optional int64_t max_workgroup_size_x,
    loom_optional int64_t max_workgroup_size_y,
    loom_optional int64_t max_workgroup_size_z,
    loom_optional int64_t max_flat_workgroup_size,
    loom_optional int64_t subgroup_size,
    loom_optional int64_t max_grid_size_x,
    loom_optional int64_t max_grid_size_y,
    loom_optional int64_t max_grid_size_z,
    loom_optional int64_t max_flat_grid_size,
    loom_optional int64_t max_workgroup_count_x,
    loom_optional int64_t max_workgroup_count_y,
    loom_optional int64_t max_workgroup_count_z,
    loom_optional int64_t memory_space_generic,
    loom_optional int64_t memory_space_global,
    loom_optional int64_t memory_space_workgroup,
    loom_optional int64_t memory_space_constant,
    loom_optional int64_t memory_space_private,
    loom_optional int64_t memory_space_host,
    loom_optional int64_t memory_space_descriptor,
    loom_optional uint8_t abi,
    loom_optional loom_string_id_t export_symbol,
    loom_optional uint8_t linkage,
    loom_optional int64_t hal_buffer_resource_flags,
    loom_optional loom_string_id_t contract_set_key,
    loom_optional int64_t contract_feature_bits,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEST_RESOURCE_ALLOC: Test owned-resource allocation.
// %resource = test.resource.alloc %sz : index -> pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_alloc_isa, LOOM_OP_TEST_RESOURCE_ALLOC)
LOOM_DEFINE_OPERAND(loom_test_resource_alloc_size, 0)
LOOM_DEFINE_RESULT(loom_test_resource_alloc_result, 0)
iree_status_t loom_test_resource_alloc_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t size,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_BORROW: Test borrowed use of an owned resource.
// test.resource.borrow %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_borrow_isa, LOOM_OP_TEST_RESOURCE_BORROW)
LOOM_DEFINE_OPERAND(loom_test_resource_borrow_resource, 0)
iree_status_t loom_test_resource_borrow_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_BORROW_REF: Test borrowed by-reference use of an owned resource carrier.
// test.resource.borrow_ref %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_borrow_ref_isa, LOOM_OP_TEST_RESOURCE_BORROW_REF)
LOOM_DEFINE_OPERAND(loom_test_resource_borrow_ref_resource, 0)
iree_status_t loom_test_resource_borrow_ref_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_CONSUME: Test consuming use of an owned resource.
// test.resource.consume %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_consume_isa, LOOM_OP_TEST_RESOURCE_CONSUME)
LOOM_DEFINE_OPERAND(loom_test_resource_consume_resource, 0)
iree_status_t loom_test_resource_consume_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_RETAIN: Test retaining an additional owned reference to a resource.
// %retained = test.resource.retain %resource : pool<[%BS]> -> pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_retain_isa, LOOM_OP_TEST_RESOURCE_RETAIN)
LOOM_DEFINE_OPERAND(loom_test_resource_retain_resource, 0)
LOOM_DEFINE_RESULT(loom_test_resource_retain_result, 0)
iree_status_t loom_test_resource_retain_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t resource,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_RELEASE: Test releasing an owned resource.
// test.resource.release %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_release_isa, LOOM_OP_TEST_RESOURCE_RELEASE)
LOOM_DEFINE_OPERAND(loom_test_resource_release_resource, 0)
iree_status_t loom_test_resource_release_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_DISCARD: Test discarding compiler ownership without releasing the resource.
// test.resource.discard %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_discard_isa, LOOM_OP_TEST_RESOURCE_DISCARD)
LOOM_DEFINE_OPERAND(loom_test_resource_discard_resource, 0)
iree_status_t loom_test_resource_discard_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_ESCAPE: Test transferring a resource to an untracked owner.
// test.resource.escape %resource : pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_escape_isa, LOOM_OP_TEST_RESOURCE_ESCAPE)
LOOM_DEFINE_OPERAND(loom_test_resource_escape_resource, 0)
iree_status_t loom_test_resource_escape_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_ALIAS: Test producing a borrowed alias of a resource.
// %alias = test.resource.alias %resource : pool<[%BS]> -> pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_alias_isa, LOOM_OP_TEST_RESOURCE_ALIAS)
LOOM_DEFINE_OPERAND(loom_test_resource_alias_resource, 0)
LOOM_DEFINE_RESULT(loom_test_resource_alias_result, 0)
iree_status_t loom_test_resource_alias_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t resource,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_RESOURCE_BORROWED: Test materializing a borrowed resource result.
// %borrowed = test.resource.borrowed %resource : pool<[%BS]> -> pool<[%BS]>
LOOM_DEFINE_ISA(loom_test_resource_borrowed_isa, LOOM_OP_TEST_RESOURCE_BORROWED)
LOOM_DEFINE_OPERAND(loom_test_resource_borrowed_resource, 0)
LOOM_DEFINE_RESULT(loom_test_resource_borrowed_result, 0)
iree_status_t loom_test_resource_borrowed_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t resource,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_SEGMENTED: Pure test op with independent operand spans sharing one flat operand array.
// %result = test.segmented %root base %guard values %lhs0, %lhs1 expected %rhs : i32 -> i32
LOOM_DEFINE_ISA(loom_test_segmented_isa, LOOM_OP_TEST_SEGMENTED)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_test_segmented_root, 0)
LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND(loom_test_segmented_guard, 1)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_test_segmented_lhs, 2)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_test_segmented_rhs, 3)
LOOM_DEFINE_RESULT(loom_test_segmented_result, 0)
enum loom_test_segmented_build_flag_bits_e {
  LOOM_TEST_SEGMENTED_BUILD_FLAG_HAS_GUARD = 1u << 0,
};
typedef uint32_t loom_test_segmented_build_flags_t;
iree_status_t loom_test_segmented_build(
    loom_builder_t* builder,
    loom_test_segmented_build_flags_t build_flags,
    loom_may_consume loom_value_id_t root,
    loom_optional loom_may_consume loom_value_id_t guard,
    loom_may_consume const loom_value_id_t* lhs,
    iree_host_size_t lhs_count,
    loom_may_consume const loom_value_id_t* rhs,
    iree_host_size_t rhs_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL: Test op with a real symbol reference spelled as an angle parameter.
// test.template_param_symbol<@target>
LOOM_DEFINE_ISA(loom_test_template_param_symbol_isa, LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_template_param_symbol_target, 0)
iree_status_t loom_test_template_param_symbol_build(
    loom_builder_t* builder,
    loom_symbol_ref_t target,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL_FLAGS: Test op with a symbol angle parameter followed by instance flags.
// test.template_param_symbol_flags<@target, debug|trace>
LOOM_DEFINE_ISA(loom_test_template_param_symbol_flags_isa, LOOM_OP_TEST_TEMPLATE_PARAM_SYMBOL_FLAGS)
LOOM_DEFINE_ATTR_SYMBOL(loom_test_template_param_symbol_flags_target, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_test_template_param_symbol_flags_flags)
iree_status_t loom_test_template_param_symbol_flags_build(
    loom_builder_t* builder,
    loom_symbol_ref_t target,
    uint8_t instance_flags,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the test dialect.
const loom_op_vtable_t* const* loom_test_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the test dialect.
const loom_op_semantics_t* loom_test_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a test op kind, or empty metadata.
loom_op_semantics_t loom_test_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TEST_OPS_H_
