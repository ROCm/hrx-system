// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_CHECK_OPS_H_
#define LOOM_OPS_CHECK_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_CHECK_CASE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 0),
  LOOM_OP_CHECK_RETURN = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 1),
  LOOM_OP_CHECK_REQUIRES = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 2),
  LOOM_OP_CHECK_SKIP_IF = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 3),
  LOOM_OP_CHECK_PARAM_RANGE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 4),
  LOOM_OP_CHECK_PARAM_CHOICE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 5),
  LOOM_OP_CHECK_PARAM_SEED = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 6),
  LOOM_OP_CHECK_LITERAL = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 7),
  LOOM_OP_CHECK_GENERATE_IOTA = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 8),
  LOOM_OP_CHECK_GENERATE_FILL = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 9),
  LOOM_OP_CHECK_GENERATE_RANDOM_UNIFORM = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 10),
  LOOM_OP_CHECK_FILE_READ_NPY = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 11),
  LOOM_OP_CHECK_FILE_WRITE_NPY = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 12),
  LOOM_OP_CHECK_ORACLE_CALL = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 13),
  LOOM_OP_CHECK_EXPECT_EQUAL = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 14),
  LOOM_OP_CHECK_EXPECT_BITWISE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 15),
  LOOM_OP_CHECK_EXPECT_CLOSE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 16),
  LOOM_OP_CHECK_EXPECT_SHAPE = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 17),
  LOOM_OP_CHECK_EXPECT = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 18),
  LOOM_OP_CHECK_EXPECT_EVENT = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 19),
  LOOM_OP_CHECK_BENCHMARK = LOOM_OP_KIND(LOOM_DIALECT_CHECK, 20),
  LOOM_OP_CHECK_COUNT_ = 21,
};

// Check symbol visibility. Absent (0) means private.
typedef enum loom_check_case_visibility_e {
  LOOM_CHECK_CASE_VISIBILITY_PUBLIC = 1,
  LOOM_CHECK_CASE_VISIBILITY_COUNT_ = 2,
} loom_check_case_visibility_t;

// Deterministic scalar range sampling policy.
typedef enum loom_check_param_range_policy_e {
  LOOM_CHECK_PARAM_RANGE_POLICY_LINEAR = 1,
  LOOM_CHECK_PARAM_RANGE_POLICY_PO2 = 2,
  LOOM_CHECK_PARAM_RANGE_POLICY_COUNT_ = 3,
} loom_check_param_range_policy_t;

// Fixture output policy for file writes.
typedef enum loom_check_file_write_npy_mode_e {
  LOOM_CHECK_FILE_WRITE_NPY_MODE_ALWAYS = 1,
  LOOM_CHECK_FILE_WRITE_NPY_MODE_ON_FAILURE = 2,
  LOOM_CHECK_FILE_WRITE_NPY_MODE_COUNT_ = 3,
} loom_check_file_write_npy_mode_t;

// NaN comparison policy for approximate expectations.
typedef enum loom_check_expect_close_nan_e {
  LOOM_CHECK_EXPECT_CLOSE_NAN_SAME = 1,
  LOOM_CHECK_EXPECT_CLOSE_NAN_DIFFERENT = 2,
  LOOM_CHECK_EXPECT_CLOSE_NAN_COUNT_ = 3,
} loom_check_expect_close_nan_t;

// LOOM_OP_CHECK_CASE: Named correctness harness.
// check.case @empty {
//   check.return
// }
LOOM_DEFINE_ISA(loom_check_case_isa, LOOM_OP_CHECK_CASE)
LOOM_DEFINE_ATTR_SYMBOL(loom_check_case_case_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_check_case_visibility, 1, loom_check_case_visibility_t)
LOOM_DEFINE_REGION(loom_check_case_body, 0)
enum loom_check_case_build_flag_bits_e {
  LOOM_CHECK_CASE_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
};
typedef uint32_t loom_check_case_build_flags_t;
iree_status_t loom_check_case_build(
    loom_builder_t* builder,
    loom_check_case_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_symbol_ref_t case_symbol,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_RETURN: Terminates a check.case body.
// check.return
LOOM_DEFINE_ISA(loom_check_return_isa, LOOM_OP_CHECK_RETURN)
iree_status_t loom_check_return_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_REQUIRES: Declares a provider requirement; unmet requirements skip the case.
// check.requires<target.feature> {feature = "amdgpu.gfx11"}
LOOM_DEFINE_ISA(loom_check_requires_isa, LOOM_OP_CHECK_REQUIRES)
LOOM_DEFINE_ATTR_STRING(loom_check_requires_provider, 0)
LOOM_DEFINE_ATTR_DICT(loom_check_requires_attrs, 1)
iree_status_t loom_check_requires_build(
    loom_builder_t* builder,
    loom_string_id_t provider,
    loom_named_attr_slice_t attrs,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_SKIP_IF: Declares a provider skip predicate for exceptional environments.
// check.skip_if<device.memory> {max_bytes = 1073741824} reason("fixture too large")
LOOM_DEFINE_ISA(loom_check_skip_if_isa, LOOM_OP_CHECK_SKIP_IF)
LOOM_DEFINE_ATTR_STRING(loom_check_skip_if_provider, 0)
LOOM_DEFINE_ATTR_DICT(loom_check_skip_if_attrs, 1)
LOOM_DEFINE_ATTR_STRING(loom_check_skip_if_reason, 2)
enum loom_check_skip_if_build_flag_bits_e {
  LOOM_CHECK_SKIP_IF_BUILD_FLAG_HAS_REASON = 1u << 0,
};
typedef uint32_t loom_check_skip_if_build_flags_t;
iree_status_t loom_check_skip_if_build(
    loom_builder_t* builder,
    loom_check_skip_if_build_flags_t build_flags,
    loom_string_id_t provider,
    loom_named_attr_slice_t attrs,
    loom_optional loom_string_id_t reason,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_PARAM_RANGE: Produces one sampled scalar parameter from a static interval.
// %m = check.param.range po2 bounds(1 to 64) : index
LOOM_DEFINE_ISA(loom_check_param_range_isa, LOOM_OP_CHECK_PARAM_RANGE)
LOOM_DEFINE_RESULT(loom_check_param_range_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_check_param_range_policy, 0, loom_check_param_range_policy_t)
LOOM_DEFINE_ATTR_ANY(loom_check_param_range_lower, 1)
LOOM_DEFINE_ATTR_ANY(loom_check_param_range_upper, 2)
LOOM_DEFINE_ATTR_ANY(loom_check_param_range_step, 3)
LOOM_DEFINE_ATTR_STRING(loom_check_param_range_param_name, 4)
enum loom_check_param_range_build_flag_bits_e {
  LOOM_CHECK_PARAM_RANGE_BUILD_FLAG_HAS_PARAM_NAME = 1u << 0,
};
typedef uint32_t loom_check_param_range_build_flags_t;
iree_status_t loom_check_param_range_build(
    loom_builder_t* builder,
    loom_check_param_range_build_flags_t build_flags,
    loom_check_param_range_policy_t policy,
    loom_attribute_t lower,
    loom_attribute_t upper,
    loom_optional loom_attribute_t step,
    loom_optional loom_string_id_t param_name,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_PARAM_CHOICE: Produces one sampled integer/index parameter from an explicit choice set.
// %k = check.param.choice values([16, 24, 32, 64]) : index
LOOM_DEFINE_ISA(loom_check_param_choice_isa, LOOM_OP_CHECK_PARAM_CHOICE)
LOOM_DEFINE_RESULT(loom_check_param_choice_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_check_param_choice_values, 0)
LOOM_DEFINE_ATTR_STRING(loom_check_param_choice_param_name, 1)
enum loom_check_param_choice_build_flag_bits_e {
  LOOM_CHECK_PARAM_CHOICE_BUILD_FLAG_HAS_PARAM_NAME = 1u << 0,
};
typedef uint32_t loom_check_param_choice_build_flags_t;
iree_status_t loom_check_param_choice_build(
    loom_builder_t* builder,
    loom_check_param_choice_build_flags_t build_flags,
    const int64_t* values,
    iree_host_size_t values_count,
    loom_optional loom_string_id_t param_name,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_PARAM_SEED: Produces deterministic i64 seeds for randomized generators.
// %seed = check.param.seed base(0x4c6f6f6d) count(32) : i64
LOOM_DEFINE_ISA(loom_check_param_seed_isa, LOOM_OP_CHECK_PARAM_SEED)
LOOM_DEFINE_RESULT(loom_check_param_seed_result, 0)
LOOM_DEFINE_ATTR_I64(loom_check_param_seed_base, 0)
LOOM_DEFINE_ATTR_I64(loom_check_param_seed_count, 1)
LOOM_DEFINE_ATTR_STRING(loom_check_param_seed_param_name, 2)
enum loom_check_param_seed_build_flag_bits_e {
  LOOM_CHECK_PARAM_SEED_BUILD_FLAG_HAS_PARAM_NAME = 1u << 0,
};
typedef uint32_t loom_check_param_seed_build_flags_t;
iree_status_t loom_check_param_seed_build(
    loom_builder_t* builder,
    loom_check_param_seed_build_flags_t build_flags,
    int64_t base,
    int64_t count,
    loom_optional loom_string_id_t param_name,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_LITERAL: Materializes a typed scalar literal for reproducer inputs or expected values.
// %scalar = check.literal value(42) : i32
LOOM_DEFINE_ISA(loom_check_literal_isa, LOOM_OP_CHECK_LITERAL)
LOOM_DEFINE_RESULT(loom_check_literal_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_literal_value, 0)
iree_status_t loom_check_literal_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_GENERATE_IOTA: Generates a deterministic iota-shaped value.
// %lhs = check.generate.iota offset(0) step(1) : tensor<[%m]x[%n]xi32>
LOOM_DEFINE_ISA(loom_check_generate_iota_isa, LOOM_OP_CHECK_GENERATE_IOTA)
LOOM_DEFINE_RESULT(loom_check_generate_iota_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_generate_iota_offset, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_generate_iota_step, 1)
iree_status_t loom_check_generate_iota_build(
    loom_builder_t* builder,
    loom_attribute_t offset,
    loom_attribute_t step,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_GENERATE_FILL: Generates a value filled with one static scalar payload.
// %rhs = check.generate.fill value(17) : tensor<[%m]x[%n]xi32>
LOOM_DEFINE_ISA(loom_check_generate_fill_isa, LOOM_OP_CHECK_GENERATE_FILL)
LOOM_DEFINE_RESULT(loom_check_generate_fill_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_generate_fill_value, 0)
iree_status_t loom_check_generate_fill_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_GENERATE_RANDOM_UNIFORM: Generates a deterministic random-uniform value from a sampled seed.
// %input = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<[%m]xf32>
LOOM_DEFINE_ISA(loom_check_generate_random_uniform_isa, LOOM_OP_CHECK_GENERATE_RANDOM_UNIFORM)
LOOM_DEFINE_OPERAND(loom_check_generate_random_uniform_seed, 0)
LOOM_DEFINE_RESULT(loom_check_generate_random_uniform_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_generate_random_uniform_lower, 0)
LOOM_DEFINE_ATTR_ANY(loom_check_generate_random_uniform_upper, 1)
iree_status_t loom_check_generate_random_uniform_build(
    loom_builder_t* builder,
    loom_value_id_t seed,
    loom_attribute_t lower,
    loom_attribute_t upper,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_FILE_READ_NPY: Reads a typed value from an NPY fixture file.
// %input = check.file.read.npy path("fixtures/layer_norm/input.npy") : tensor<1024xf32>
LOOM_DEFINE_ISA(loom_check_file_read_npy_isa, LOOM_OP_CHECK_FILE_READ_NPY)
LOOM_DEFINE_RESULT(loom_check_file_read_npy_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_check_file_read_npy_path, 0)
iree_status_t loom_check_file_read_npy_build(
    loom_builder_t* builder,
    loom_string_id_t path,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_FILE_WRITE_NPY: Writes a typed value to an NPY file according to a static output policy.
// check.file.write.npy value(%actual) path("outputs/layer_norm.actual.npy") mode(on_failure) : tensor<1024xf32>
LOOM_DEFINE_ISA(loom_check_file_write_npy_isa, LOOM_OP_CHECK_FILE_WRITE_NPY)
LOOM_DEFINE_OPERAND(loom_check_file_write_npy_value, 0)
LOOM_DEFINE_ATTR_STRING(loom_check_file_write_npy_path, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_check_file_write_npy_mode, 1, loom_check_file_write_npy_mode_t)
enum loom_check_file_write_npy_build_flag_bits_e {
  LOOM_CHECK_FILE_WRITE_NPY_BUILD_FLAG_HAS_MODE = 1u << 0,
};
typedef uint32_t loom_check_file_write_npy_build_flags_t;
iree_status_t loom_check_file_write_npy_build(
    loom_builder_t* builder,
    loom_check_file_write_npy_build_flags_t build_flags,
    loom_value_id_t value,
    loom_string_id_t path,
    loom_optional uint8_t mode,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_ORACLE_CALL: Calls a pluggable oracle provider for expected values.
// %expected = check.oracle.call<reference.scalar> callee(@gemv_f32) inputs(%lhs, %rhs) : (tensor<[%m]x[%n]xf32>, tensor<[%n]xf32>) -> (tensor<[%m]xf32>)
LOOM_DEFINE_ISA(loom_check_oracle_call_isa, LOOM_OP_CHECK_ORACLE_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_check_oracle_call_inputs, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_check_oracle_call_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_check_oracle_call_provider, 0)
LOOM_DEFINE_ATTR_DICT(loom_check_oracle_call_attrs, 1)
LOOM_DEFINE_ATTR_SYMBOL(loom_check_oracle_call_callee, 2)
iree_status_t loom_check_oracle_call_build(
    loom_builder_t* builder,
    loom_string_id_t provider,
    loom_optional loom_named_attr_slice_t attrs,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT_EQUAL: Requires actual and expected values to compare equal.
// check.expect.equal actual(%actual) expected(%expected) : tensor<[%m]xi32>
LOOM_DEFINE_ISA(loom_check_expect_equal_isa, LOOM_OP_CHECK_EXPECT_EQUAL)
LOOM_DEFINE_OPERAND(loom_check_expect_equal_actual, 0)
LOOM_DEFINE_OPERAND(loom_check_expect_equal_expected, 1)
iree_status_t loom_check_expect_equal_build(
    loom_builder_t* builder,
    loom_value_id_t actual,
    loom_value_id_t expected,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT_BITWISE: Requires actual and expected values to match bit-for-bit.
// check.expect.bitwise actual(%actual) expected(%expected) : tensor<1024xf32>
LOOM_DEFINE_ISA(loom_check_expect_bitwise_isa, LOOM_OP_CHECK_EXPECT_BITWISE)
LOOM_DEFINE_OPERAND(loom_check_expect_bitwise_actual, 0)
LOOM_DEFINE_OPERAND(loom_check_expect_bitwise_expected, 1)
iree_status_t loom_check_expect_bitwise_build(
    loom_builder_t* builder,
    loom_value_id_t actual,
    loom_value_id_t expected,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT_CLOSE: Requires actual and expected floating-point values to be approximately equal.
// check.expect.close actual(%actual) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<[%m]xf32>
LOOM_DEFINE_ISA(loom_check_expect_close_isa, LOOM_OP_CHECK_EXPECT_CLOSE)
LOOM_DEFINE_OPERAND(loom_check_expect_close_actual, 0)
LOOM_DEFINE_OPERAND(loom_check_expect_close_expected, 1)
LOOM_DEFINE_ATTR_F64(loom_check_expect_close_atol, 0)
LOOM_DEFINE_ATTR_F64(loom_check_expect_close_rtol, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_check_expect_close_nan, 2, loom_check_expect_close_nan_t)
iree_status_t loom_check_expect_close_build(
    loom_builder_t* builder,
    loom_value_id_t actual,
    loom_value_id_t expected,
    double atol,
    double rtol,
    loom_check_expect_close_nan_t nan,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT_SHAPE: Requires a shaped value to have the expected dynamic/static shape.
// check.expect.shape value(%actual) shape([%m, %n, 4]) : tensor<[%m]x[%n]x4xf32>
LOOM_DEFINE_ISA(loom_check_expect_shape_isa, LOOM_OP_CHECK_EXPECT_SHAPE)
LOOM_DEFINE_OPERAND(loom_check_expect_shape_value, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_check_expect_shape_dims, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_check_expect_shape_static_dims, 0)
iree_status_t loom_check_expect_shape_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    const loom_value_id_t* dims,
    iree_host_size_t dims_count,
    const int64_t* static_dims,
    iree_host_size_t static_dims_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT: Runs a pluggable custom validator over actual and expected values.
// check.expect<topk.equal> actual(%actual) expected(%expected) {k = 5} : tensor<1000xf32>
LOOM_DEFINE_ISA(loom_check_expect_isa, LOOM_OP_CHECK_EXPECT)
LOOM_DEFINE_OPERAND(loom_check_expect_actual, 0)
LOOM_DEFINE_OPERAND(loom_check_expect_expected, 1)
LOOM_DEFINE_ATTR_STRING(loom_check_expect_provider, 0)
LOOM_DEFINE_ATTR_DICT(loom_check_expect_attrs, 1)
iree_status_t loom_check_expect_build(
    loom_builder_t* builder,
    loom_string_id_t provider,
    loom_value_id_t actual,
    loom_value_id_t expected,
    loom_optional loom_named_attr_slice_t attrs,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_EXPECT_EVENT: Requires a structured runtime event matching a provider-specific subset pattern.
// check.expect.event<device> {type = "asan_report", count = 1}
LOOM_DEFINE_ISA(loom_check_expect_event_isa, LOOM_OP_CHECK_EXPECT_EVENT)
LOOM_DEFINE_ATTR_STRING(loom_check_expect_event_provider, 0)
LOOM_DEFINE_ATTR_DICT(loom_check_expect_event_attrs, 1)
iree_status_t loom_check_expect_event_build(
    loom_builder_t* builder,
    loom_string_id_t provider,
    loom_optional loom_named_attr_slice_t attrs,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_CHECK_BENCHMARK: Declares a benchmark slice over a check.case.
// check.benchmark<@gemv_sweep> @gemv_latency {m = 8, n = 96}
LOOM_DEFINE_ISA(loom_check_benchmark_isa, LOOM_OP_CHECK_BENCHMARK)
LOOM_DEFINE_ATTR_SYMBOL(loom_check_benchmark_benchmark, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_check_benchmark_case_ref, 1)
LOOM_DEFINE_ATTR_DICT(loom_check_benchmark_attrs, 2)
enum loom_check_benchmark_build_flag_bits_e {
  LOOM_CHECK_BENCHMARK_BUILD_FLAG_HAS_BENCHMARK = 1u << 0,
};
typedef uint32_t loom_check_benchmark_build_flags_t;
iree_status_t loom_check_benchmark_build(
    loom_builder_t* builder,
    loom_check_benchmark_build_flags_t build_flags,
    loom_symbol_ref_t case_ref,
    loom_optional loom_symbol_ref_t benchmark,
    loom_optional loom_named_attr_slice_t attrs,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the check dialect.
const loom_op_vtable_t* const* loom_check_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the check dialect.
const loom_op_semantics_t* loom_check_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a check op kind, or empty metadata.
loom_op_semantics_t loom_check_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_CHECK_OPS_H_
